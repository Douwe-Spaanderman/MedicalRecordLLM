import subprocess
import argparse
import json
import os
import yaml
import logging
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict

try:
    project_root = Path(__file__).resolve().parents[1]
except NameError:
    project_root = Path.cwd().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ExperimentRunner:
    def __init__(
        self,
        data_path: Path,
        output_dir: Path,
        prompt_config_path: Path,
        model_configs: List[Path],
        prompt_methods: List[str],
        input_format: str,
        patient_id_col: str,
        gpus: int,
        default_timeout: int,
        default_max_concurrent: int,
        override_config_path: Optional[Path] = None,
        vllm_server: bool = False,
        base_url: str = "http://localhost:8000/v1/",
        dry_run: bool = False,
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.prompt_config_path = prompt_config_path
        self.model_configs = model_configs
        self.prompt_methods = prompt_methods
        self.input_format = input_format
        self.patient_id_col = patient_id_col
        self.gpus = gpus
        self.default_timeout = default_timeout
        self.default_max_concurrent = default_max_concurrent
        self.override_config_path = override_config_path
        self.vllm_server = vllm_server
        self.base_url = base_url
        self.dry_run = dry_run

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.performance_files = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self.load_overrides()

    def load_overrides(self):
        if not self.override_config_path:
            self.override_config = {}
            return
        
        self.override_config = self.read_config(self.override_config_path)

    def get_timeout(self, model_name: str, prompt_method: str) -> int:
        model_cfg = self.override_config.get(model_name, {})
        return (
            model_cfg.get(prompt_method, {}).get("timeout") or
            self.default_timeout
        )

    def get_max_concurrent(self, model_name: str, prompt_method: str) -> int:
        model_cfg = self.override_config.get(model_name, {})
        return (
            model_cfg.get(prompt_method, {}).get("max_concurrent") or
            self.default_max_concurrent
        )
    
    def read_config(self, config_path) -> Dict[str, Any]:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def run(self):
        for model_config_path in self.model_configs:
            model_config = self.read_config(model_config_path)
            if self.vllm_server:
                vllm_process = self.start_vllm_server(model_config)

            try:
                self.wait_for_vllm_ready()
            except TimeoutError as e:
                self.logger.error(f"[Error] vLLM server did not start in time for model {model_config("model", model_config_path.stem)}")
                if vllm_process:
                    self.kill_vllm_server(vllm_process)
                continue

            for prompt_method in self.prompt_methods:
                self.run_single_experiment(prompt_method, model_config_path, model_config)

            if vllm_process or self.dry_run:
                self.kill_vllm_server(vllm_process)

    def start_vllm_server(self, model_config: Dict[str, Any]) -> Optional[subprocess.Popen]:
        if not model_config.get("model", False):
            self.logger.info(f"No model name found in {model_config}. Skipping vLLM server start.")
            return None
        
        command = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_config["model"],
            "--tensor-parallel-size", str(self.gpus),
            "--dtype", "bfloat16",
        ]
        if self.dry_run:
            self.logger.info("[Dry Run] Starting vLLM server with command: " + " ".join(command))
            return None
        
        self.logger.info(f"[Starting vLLM] {model_config['model']}")
        return subprocess.Popen(command)

    def wait_for_vllm_ready(self, timeout=600, interval=1):
        url = self.base_url + "models"
        start_time = time.time()
        if self.dry_run:
           self.logger.info("[Dry Run] Would ping vLLM server to see if up")
           return True 

        self.logger.info("[Waiting] for vLLM server to become ready...")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    self.logger.info("[Ready] vLLM server is up")
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(interval)

        raise TimeoutError("vLLM server did not become ready within timeout.")

    def kill_vllm_server(self, process):
        if self.dry_run:
            self.logger.info("[Dry Run] Would kill vLLM server")
            return
        self.logger.info("[Killing vLLM] ...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        self.logger.info("[Killed] vLLM server")

    def run_single_experiment(self, prompt_method: str, model_config_path: Path, model_config: Dict[str, Any]):
        model_name = model_config.get("model", model_config_path.stem).split("/")[-1]
        output_file = self.output_dir / model_name / f"{prompt_method}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        timeout = self.get_timeout(model_name, prompt_method)
        max_concurrent = self.get_max_concurrent(model_name, prompt_method)

        command = [
            "python", str(project_root / "run.py"),
            "-i", str(self.data_path),
            "-o", str(output_file),
            "-pm", prompt_method,
            "-pc", str(self.prompt_config_path),
            "-pa", str(model_config_path),
            "-f", self.input_format,
            "--patient-id-col", self.patient_id_col,
            "--timeout", str(timeout),
            "-mc", str(max_concurrent),
        ]

        if self.dry_run:
            self.logger.info("[Dry Run] Would run experiment with command: " + " ".join(command))
            return

        try:
            subprocess.run(command, check=True)
            self.logger.info(f"[Success] Experiment completed for {prompt_method} + {model_name}")
            self.logger.info(f"[Output] Results saved to {output_file}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[Error] Failed to run experiment for {prompt_method} + {model_name}")
            self.logger.error(e)
            return

        try:
            self.run_single_calculation(output_file, prompt_method, model_name)
        except Exception as e:
            self.logger.error(f"[Error] Performance calculation failed for {output_file}")
            self.logger.error(e)
            return

    def run_single_calculation(self, llm_output_path: Path, prompt_method: str, model_name: str):
        perf_output_path = llm_output_path.with_suffix(".performance.csv")

        command = [
            "python", str(project_root / "evaluation" / "calculate_performance.py"),
            "-l", str(llm_output_path),
            "-p", str(self.prompt_config_path),
            "-o", str(perf_output_path),
        ]

        if self.dry_run:
            self.logger.info("[Dry Run] Would calculate performance with command: " + " ".join(command))
            return

        try:
            subprocess.run(command, check=True)
            self.logger.info(f"[Calculated] Performance for {llm_output_path}")
            self.performance_files[model_name].append((prompt_method, perf_output_path))
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[Error] Failed to calculate performance for {llm_output_path}")
            self.logger.error(e)

    def run_visualization(self):
        if self.dry_run:
           self.logger.info("[Dry Run] Would visualize performance")
           return

        if not self.performance_files:
            self.logger.info("No performance files found to visualize.")
            return
        
        self.logger.info("Starting visualization of performance results...")

        # Per-model comparison of prompt methods
        for model_name, method_files in self.performance_files.items():
            methods = [m for m, _ in method_files]
            files = [str(f) for _, f in method_files]
            labels = methods
            out_file = self.output_dir / model_name / f"all_results.png"
            self.visualize(files, out_file, labels)

    def visualize(self, input_files: List[str], output_file: Path, labels: List[str]):
        command = [
            "python", str(project_root / "evaluation" / "visualize_performance.py"),
            "-i"
        ] + input_files + [
            "-o", str(output_file),
            "-l"
        ] + labels

        if self.dry_run:
            self.logger.info("[Dry Run] Would visualize with command: " + " ".join(command))
            return

        try:
            subprocess.run(command, check=True)
            self.logger.info(f"[Visualized] {output_file}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[Error] Failed to visualize performance results for {output_file}")
            self.logger.error(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all LLM experiments.")
    parser.add_argument(
        "--data-path", required=True, type=Path, help="Path to input CSV or JSON file."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to save output files."
    )
    parser.add_argument(
        "--prompt-config",
        required=True,
        type=Path,
        help="YAML file with prompt templates.",
    )
    parser.add_argument(
        "--model-configs",
        required=True,
        type=Path,
        nargs="+",
        help="List of YAML model config files.",
    )
    parser.add_argument(
        "--prompt-methods",
        nargs="+",
        required=False,
        default=[
            "ZeroShot",
            "OneShot",
            "FewShot",
            "CoT",
            "SelfConsistency",
            "PromptChain",
        ],
        help="Prompting methods to try.",
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], required=True, help="Input file format."
    )
    parser.add_argument(
        "--patient-id-col", default="Patient-ID", help="Patient ID column name."
    )
    parser.add_argument(
        "--gpus", type=int, default=4, help="How many gpus to use for VLLM tensor parallel."
    )
    parser.add_argument(
        "--timeout", type=int, default=240, help="Timeout for each request in seconds."
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=64, help="Max concurrent requests."
    )
    parser.add_argument(
        "--overrides-config",
        type=Path,
        help="YAML file with timeout and concurrent overrides for models and methods.",
    )
    parser.add_argument(
        "--vllm-server",
        action="store_true",
        help="Run vLLM server for each model configuration.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them."
    )

    args = parser.parse_args()

    runner = ExperimentRunner(
        data_path=args.data_path,
        output_dir=args.output_dir,
        prompt_config_path=args.prompt_config,
        model_configs=args.model_configs,
        prompt_methods=args.prompt_methods,
        input_format=args.format,
        patient_id_col=args.patient_id_col,
        gpus=args.gpus,
        default_timeout=args.timeout,
        default_max_concurrent=args.max_concurrent,
        override_config_path=args.overrides_config,
        vllm_server=args.vllm_server,
        dry_run=args.dry_run,
    )
    runner.run()
    runner.run_visualization()
