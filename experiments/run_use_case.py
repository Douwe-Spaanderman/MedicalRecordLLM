import subprocess
import argparse
from pathlib import Path
from typing import List


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
        timeout: int,
        max_concurrent: int,
        dry_run: bool = False,
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.prompt_config_path = prompt_config_path
        self.model_configs = model_configs
        self.prompt_methods = prompt_methods
        self.input_format = input_format
        self.patient_id_col = patient_id_col
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.dry_run = dry_run

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        for prompt_method in self.prompt_methods:
            for model_config in self.model_configs:
                self.run_single_experiment(prompt_method, model_config)

    def run_single_experiment(self, prompt_method: str, model_config: Path):
        model_name = model_config.stem
        output_file = self.output_dir / {model_name} / f"{prompt_method}.csv"

        command = [
            "python", "run.py",
            "-i", str(self.data_path),
            "-o", str(output_file),
            "-pm", prompt_method,
            "-pc", str(self.prompt_config_path),
            "-pa", str(model_config),
            "-f", self.input_format,
            "--patient-id-col", self.patient_id_col,
            "--timeout", str(self.timeout),
            "-mc", str(self.max_concurrent),
        ]

        print(f"\n[Running] {prompt_method} with {model_name}")
        if self.dry_run:
            print("[Dry Run] Command:", " ".join(command))
            return

        try:
            subprocess.run(command, check=True)
            print(f"[Success] Output saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Failed: {prompt_method} + {model_name}")
            print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all LLM experiments.")
    parser.add_argument("--data-path", required=True, type=Path, help="Path to input CSV or JSON file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save output files.")
    parser.add_argument("--prompt-config", required=True, type=Path, help="YAML file with prompt templates.")
    parser.add_argument("--model-configs", required=True, type=Path, nargs="+", help="List of YAML model config files.")
    parser.add_argument("--prompt-methods", nargs="+", required=False, default=["ZeroShot", "OneShot", "FewShot", "CoT", "SelfConsistency", "PromptChain"], help="Prompting methods to try.")
    parser.add_argument("--format", choices=["csv", "json"], required=True, help="Input file format.")
    parser.add_argument("--patient-id-col", default="Patient-ID", help="Patient ID column name.")
    parser.add_argument("--timeout", type=int, default=240, help="Timeout for each request in seconds.")
    parser.add_argument("--max-concurrent", type=int, default=64, help="Max concurrent requests.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")

    parser.parse.args()

    runner = ExperimentRunner(
        data_path=args.data_path,
        output_dir=args.output_dir,
        prompt_config_path=args.prompt_config,
        model_configs=args.model_configs,
        prompt_methods=args.prompt_methods,
        input_format=args.format,
        patient_id_col=args.patient_id_col,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
    )
    runner.run()

