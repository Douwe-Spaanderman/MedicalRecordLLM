import argparse
import pandas as pd
from pathlib import Path
from parser import VLLMReportParser
from adapters import DataFrameAdapter, JsonAdapter
import yaml
import json

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(
        description="LLM report parser with configurable input adapters",
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to input file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Output file path"
    )
    parser.add_argument(
        "-pm",
        "--prompt-method",
        required=True,
        choices=["ZeroShot", "OneShot", "FewShot", "CoT", "SelfConsistency", "PromptGraph"],
        type=str,
        help="prompting method"
    )
    parser.add_argument(
        "-pc",
        "--prompt-config",
        required=True,
        type=str,
        help="Path to YAML config for prompt definitions"
    )
    parser.add_argument(
        "-pa",
        "--params-config",
        default="config_parameters.yaml",
        type=str,
        help="Path to YAML config for model parameters"
    )
    parser.add_argument(
        "-f", "--format",
        required=True,
        type=str,
        choices=["csv", "json"],
        help="Input file format"
    )
    parser.add_argument(
        "-u", 
        "--base-url",
        default="http://localhost:8000/v1",
        type=str,
        help="Base URL for the LLM API"
    )
    parser.add_argument(
        "--api-key",
        default="DummyAPIKey",
        type=str,
        help="API key for the LLM service (if required)"
    )
    parser.add_argument(
        "--text-key",
        default="Text",
        help="Key containing text in JSON (default: 'Text')"
    )
    parser.add_argument(
        "--report-type-key",
        default="reportType",
        help="Key containing report type in JSON (default: 'reportType')"
    )
    parser.add_argument(
        "--text-col",
        default="Text",
        help="Text column in CSV (default: 'Text')"
    )
    parser.add_argument(
        "--patient-id-col",
        default="patientID",
        help="Patient ID column in CSV (default: 'patientID')"
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw model output to a file",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        help="Optional internal batch size for processing reports (default: None)"
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=60,
        help="Timeout for each request in seconds (default: 60)"
    )
    parser.add_argument(
        "-mc",
        "--max-concurrent",
        type=int,
        default=32,
        help="Maximum number of concurrent requests (default: 32)"
    )
    parser.add_argument(
        "-se",
        "--select_example",
        type=int,
        default=None,
        help="1-based index of the example to use (only for example-based prompt methods) (default: None)"
    )
    parser.add_argument(
        "-r",
        "--regex",
        required=False,
        type=str,
        default=None,
        help="Path to the regex patterns to extract (should be .json file). This can be used to use existing structured data in the reports",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Do you want to print intermediates, such as raw prompts etc. (Nice for debugging but slows down workflow quite a bit)"
    )

    args = parser.parse_args()
    
    # Load model parameters from config file
    params_config = load_config(args.params_config)

    # Load prompt configuration
    prompt_config = load_config(args.prompt_config)

    # Initialize parser
    report_parser = VLLMReportParser(
        prompt_config=prompt_config,
        params_config=params_config,
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_method=args.prompt_method,
        batch_size=args.batch_size,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        select_example=args.select_example,
        patterns_path=args.regex,
        save_raw_output=args.save_raw,
        verbose=args.verbose,
    )

    # Initialize appropriate adapter
    if args.format == "csv":
        df = pd.read_csv(args.input)
        adapter = DataFrameAdapter(
            df=df,
            report_type_column=args.report_type_key,
            text_column=args.text_col,
            patient_id_column=args.patient_id_col
        )
    else:  # json
        adapter = JsonAdapter(
            input_path=args.input,
            text_key=args.text_key,
            id_key=args.patient_id_col
        )
    
    # Process reports
    result = report_parser.process_with_adapter(adapter)
    
    # Save output
    output_path = Path(args.output)
    if args.format == "csv":
        result.to_csv(output_path, index=False)
    else:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()