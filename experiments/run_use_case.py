import subprocess
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

file_path = Path(os.path.realpath(__file__)).parent

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
        timeout_overrides: Dict[str, int],
        concurrent_overrides: Dict[str, int],
        method_timeout_overrides: Dict[str, int],
        method_concurrent_overrides: Dict[str, int],
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
        self.timeout_overrides = timeout_overrides
        self.concurrent_overrides = concurrent_overrides
        self.method_timeout_overrides = method_timeout_overrides
        self.method_concurrent_overrides = method_concurrent_overrides
        self.dry_run = dry_run

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.performance_files = defaultdict(list)  # for visualization

    def get_timeout(self, model_name: str, prompt_method: str) -> int:
        return self.method_timeout_overrides.get(
            prompt_method,
            self.timeout_overrides.get(model_name, self.default_timeout)
        )

    def get_max_concurrent(self, model_name: str, prompt_method: str) -> int:
        return self.method_concurrent_overrides.get(
            prompt_method,
            self.concurrent_overrides.get(model_name, self.default_max_concurrent)
        )

    def run(self):
        for prompt_method in self.prompt_methods:
            for model_config in self.model_configs:
                self.run_single_experiment(prompt_method, model_config)

    def run_single_experiment(self, prompt_method: str, model_config: Path):
        model_name = model_config.stem
        output_file = self.output_dir / model_name / f"{prompt_method}.csv"

        timeout = self.get_timeout(model_name, prompt_method)
        max_concurrent = self.get_max_concurrent(model_name, prompt_method)

        command = [
            "python", file_path / "run.py",
            "-i", str(self.data_path),
            "-o", str(output_file),
            "-pm", prompt_method,
            "-pc", str(self.prompt_config_path),
            "-pa", str(model_config),
            "-f", self.input_format,
            "--patient-id-col", self.patient_id_col,
            "--timeout", str(timeout),
            "-mc", str(max_concurrent),
        ]

        if self.dry_run:
            print("[Dry Run] Command:", " ".join(command))
            return

        try:
            subprocess.run(command, check=True)
            print(f"[Success] Output saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Failed: {prompt_method} + {model_name}")
            print(e)
            return

        try:
            self.run_single_calculation(output_file, prompt_method, model_name)
        except Exception as e:
            print(f"[Error] Performance calculation failed for {output_file}")
            print(e)
            return

    def run_single_calculation(self, llm_output_path: Path, prompt_method: str, model_name: str):
        perf_output_path = llm_output_path.with_suffix(".performance.csv")

        command = [
            "python", file_path / "evaluation" / "calculate_performance.py",
            "-l", str(llm_output_path),
            "-p", str(self.prompt_config_path),
            "-o", str(perf_output_path),
        ]

        if self.dry_run:
            print("[Dry Run] Would calculate performance:", " ".join(command))
            return

        try:
            subprocess.run(command, check=True)
            print(f"[Calculated] Performance saved to {perf_output_path}")
            self.performance_files[model_name].append((prompt_method, perf_output_path))
        except subprocess.CalledProcessError as e:
            print(f"[Error] Failed to calculate performance for {llm_output_path}")
            print(e)

    def run_visualization(self):
        print("\n[Visualizing] Generating performance plots...")

        # Per-model comparison of prompt methods
        for model_name, method_files in self.performance_files.items():
            methods = [m for m, _ in method_files]
            files = [str(f) for _, f in method_files]
            labels = methods
            out_file = self.output_dir / model_name / f"all_results.png"
            self.visualize(files, out_file, labels)

    def visualize(self, input_files: List[str], output_file: Path, labels: List[str]):
        command = [
            "python", file_path / "evaluation" / "visualize_performance.py",
            "-i"
        ] + input_files + [
            "-o", str(output_file),
            "-l"
        ] + labels

        if self.dry_run:
            print("[Dry Run] Would visualize:", " ".join(command))
            return

        try:
            subprocess.run(command, check=True)
            print(f"[Visualized] {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Visualization failed for {output_file}")
            print(e)

def load_overrides(arg: Optional[str]) -> Dict[str, int]:
    if not arg:
        return {}
    path = Path(arg)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return json.loads(arg)

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
        "--timeout", type=int, default=240, help="Timeout for each request in seconds."
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=64, help="Max concurrent requests."
    )
    parser.add_argument(
        "--timeout-overrides",
        type=str,
        help="JSON string or path to JSON file with per-model timeout overrides.",
    )
    parser.add_argument(
        "--concurrent-overrides",
        type=str,
        help="JSON string or path to JSON file with per-model concurrency overrides.",
    )
    parser.add_argument(
        "--method-timeout-overrides",
        type=str,
        help="JSON string or path to JSON file with per-method timeout overrides.",
    )
    parser.add_argument(
        "--method-concurrent-overrides",
        type=str,
        help="JSON string or path to JSON file with per-method concurrency overrides.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them."
    )

    args = parser.parse.args()

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
        timeout_overrides=load_overrides(args.timeout_overrides),
        concurrent_overrides=load_overrides(args.concurrent_overrides),
        method_timeout_overrides=load_overrides(args.method_timeout_overrides),
        method_concurrent_overrides=load_overrides(args.method_concurrent_overrides),
        dry_run=args.dry_run,
    )
    runner.run()
    runner.run_visualization()
