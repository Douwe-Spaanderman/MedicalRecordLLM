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
        default_timeout: int,
        default_max_concurrent: int,
        override_config_path: Optional[Path] = None,
        vllm_server: bool = False,
        gpu_parallelization: int = 1,
        node_parallelization: int = 1,
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
        self.default_timeout = default_timeout
        self.default_max_concurrent = default_max_concurrent
        self.override_config_path = override_config_path
        self.vllm_server = vllm_server
        self.gpu_parallelization = gpu_parallelization
        self.node_parallelization = node_parallelization
        self.base_url = base_url
        self.dry_run = dry_run

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.performance_files = defaultdict(list)
        self.ranked_results = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self.load_overrides()

    def load_overrides(self):
        """
        Load overrides from the provided YAML file if it exists.
        """
        if not self.override_config_path:
            self.override_config = {}
            return
        
        self.override_config = self.read_config(self.override_config_path)

    def get_timeout(self, model_name: str, prompt_method: str) -> int:
        """
        Get the timeout for a specific model and prompt method from the overrides.
        If not specified, return the default timeout.
        
        Args:
            model_name (str): Name of the model.
            prompt_method (str): Prompt method being used.

        Returns:
            int: Timeout in seconds.
        """
        model_cfg = self.override_config.get(model_name, {})
        return (
            model_cfg.get(prompt_method, {}).get("timeout") or
            self.default_timeout
        )

    def get_max_concurrent(self, model_name: str, prompt_method: str) -> int:
        """
        Get the max concurrent requests for a specific model and prompt method from the overrides.
        If not specified, return the default max concurrent requests.

        Args:
            model_name (str): Name of the model.
            prompt_method (str): Prompt method being used.

        Returns:
            int: Max concurrent requests.
        """
        model_cfg = self.override_config.get(model_name, {})
        return (
            model_cfg.get(prompt_method, {}).get("max_concurrent") or
            self.default_max_concurrent
        )
    
    def read_config(self, config_path) -> Dict[str, Any]:
        """
        Read a YAML configuration file and return its contents as a dictionary.

        Args:
            config_path (Path): Path to the YAML configuration file.

        Returns:
            Dict[str, Any]: Parsed configuration as a dictionary.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def run(self):
        """
        Run the entire experiment workflow:
        1. Start vLLM server for each model configuration if specified.
        2. Run experiments for each prompt method and model configuration.
        3. Calculate performance metrics for each experiment.
        """
        for model_config_path in self.model_configs:
            model_config = self.read_config(model_config_path)
            model_name = model_config.get("model", model_config_path.stem).split("/")[-1]
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

            self.logger.info(f"[Completed] All experiments for model {model_config.get('model', model_config_path.stem)}")

    def start_vllm_server(self, model_config: Dict[str, Any]) -> Optional[subprocess.Popen]:
        """
        Start the vLLM server for the given model configuration.

        Args:
            model_config (Dict[str, Any]): Model configuration dictionary.

        Returns:
            subprocess.Popen: Process handle for the vLLM server.
        """
        if not model_config.get("model", False):
            self.logger.info(f"No model name found in {model_config}. Skipping vLLM server start.")
            return None
        
        command = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_config["model"],
            "--tensor-parallel-size", str(self.gpu_parallelization),
            "--pipeline_parallel_size", str(self.node_parallelization),
            "--dtype", "bfloat16",
            "--trust-remote-code",
        ]
        if self.dry_run:
            self.logger.info("[Dry Run] Starting vLLM server with command: " + " ".join(command))
            return None
        
        self.logger.info(f"[Starting vLLM] {model_config['model']}")
        return subprocess.Popen(command)

    def wait_for_vllm_ready(self, timeout=600, interval=1):
        """
        Wait for the vLLM server to become ready by pinging the API endpoint.

        Args:
            timeout (int): Maximum time to wait for the server to become ready.
            interval (int): Time to wait between pings.

        Raises:
            TimeoutError: If the server does not become ready within the timeout period.
        """
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
        """
        Kill the vLLM server process.

        Args:
            process (subprocess.Popen): Process handle for the vLLM server.
        """
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
        """
        Run a single experiment for a given prompt method and model configuration.

        Args:
            prompt_method (str): The prompt method to use.
            model_config_path (Path): Path to the model configuration file.
            model_config (Dict[str, Any]): Model configuration dictionary.
        """
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
        """
        Calculate performance metrics for a single LLM output file.

        Args:
            llm_output_path (Path): Path to the LLM output file.
            prompt_method (str): The prompt method used.
            model_name (str): Name of the model.
        """
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
        """
        Visualize the performance results from the LLM output files.

        This function will generate plots comparing the performance of different prompt methods
        across the models used in the experiments.
        """
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
            if self.ranked_results:
                ranked_file = self.ranked_results.get(model_name)
            else:
                ranked_file = None

            self.visualize(files, out_file, labels, ranked_file)

    def visualize(self, input_files: List[str], output_file: Path, labels: List[str]):
        """
        Visualize performance results from LLM output files.

        Args:
            input_files (List[str]): List of paths to the LLM output files.
            output_file (Path): Path to save the visualization.
            labels (List[str]): Optional labels for each input file.
        """
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

    def run_ranking(self):
        """
        Run rank aggregation on the collected performance files.
        
        """

        if self.dry_run:
           self.logger.info("[Dry Run] Would run rank aggregation")
           return

        if not self.performance_files:
            self.logger.info("No performance files found to rank.")
            return
        
        self.logger.info("Starting rank aggregation of performance results...")

        # Per-model comparison of prompt methods
        for model_name, method_files in self.performance_files.items():
            files = [str(f) for _, f in method_files]
            out_file = self.output_dir / model_name / "ranked_results.csv"
            self.rank(files, out_file, method="kemeny")

        self.logger.info("Rank aggregation completed.")

    def rank(self, input_files: List[str], output_file: Optional[Path] = None, method: str = "kemeny"):
        """
        Rank aggregation of multiple LLM performance files.

        Args:
            input_files (List[str]): List of paths to the LLM performance files.
            output_file (Optional[Path]): Path to save the aggregated results. Defaults to None.
            method (str): Method for rank aggregation. Defaults to "kemeny".
                Options are "borda", "kemeny", or "ranked_pairs".
        """
        command = [
            "python", str(project_root / "evaluation" / "rank_aggregation.py"),
            "-i"
        ] + input_files + [
            "-o", str(output_file),
            "-m", method
        ]

        if self.dry_run:
            self.logger.info("[Dry Run] Would run rank aggregation with command: " + " ".join(command))
            return

        try:
            subprocess.run(command, check=True)
            self.logger.info(f"[Ranked] Results saved to {output_file}")
            self.ranked_results[method].append(output_file)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[Error] Failed to run rank aggregation for {input_files}")
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
            "PromptGraph",
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
        "--gpu-parallelization",
        type=int,
        default=1,
        help="Number of GPUs to use for parallelization.",
    )
    parser.add_argument(
        "--node-parallelization",
        type=int,
        default=1,
        help="Number of nodes to use for parallelization.",
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
        default_timeout=args.timeout,
        default_max_concurrent=args.max_concurrent,
        override_config_path=args.overrides_config,
        vllm_server=args.vllm_server,
        gpu_parallelization=args.gpu_parallelization,
        node_parallelization=args.node_parallelization,
        dry_run=args.dry_run,
    )
    runner.run()
    runner.run_ranking()
    runner.run_visualization()
