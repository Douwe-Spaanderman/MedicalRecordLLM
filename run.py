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
        "--query-config",
        required=True,
        type=str,
        help="Path to YAML config for query definitions"
    )
    parser.add_argument(
        "--params-config",
        default="config_parameters.yaml",
        type=str,
        help="Path to YAML config for model parameters"
    )
    parser.add_argument(
        "-g",
        "--gpus",
        required=False,
        type=int,
        default=2,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "-f", "--format",
        required=True,
        type=str,
        choices=["csv", "json"],
        help="Input file format"
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Key containing text in JSON (default: 'text')"
    )
    parser.add_argument(
        "--report-type-key",
        default="reportType",
        help="Key containing report type in JSON (default: 'reportType')"
    )
    parser.add_argument(
        "--text-col",
        default="presentedForm_data",
        help="Text column in CSV (default: 'presentedForm_data')"
    )
    parser.add_argument(
        "-r",
        "--regex",
        required=False,
        type=str,
        default=None,
        help="Path to the regex patterns to extract (should be .json file). This can be used to use existing structured data in the reports",
    )

    args = parser.parse_args()
    
    # Load model parameters from config file
    params_config = load_config(args.params_config)
    
    # Model full path from config
    model_path = params_config['model']

    # Load query configuration
    query_config = load_config(args.query_config)

    # Initialize parser
    report_parser = VLLMReportParser(
        model=model_path,
        query_config=query_config,
        gpus=args.gpus,
        patterns_path=args.regex,
        max_model_len=params_config['max_model_len'],
        max_tokens=params_config['max_tokens'],
        temperature=params_config['temperature'],
        top_p=params_config['top_p'],
        repetition_penalty=params_config['repetition_penalty'],
        max_attempts=params_config['max_attempts'],
        update_config=params_config.get('update_config'),
    )

    # Initialize appropriate adapter
    if args.format == "csv":
        df = pd.read_csv(args.input)
        adapter = DataFrameAdapter(
            df=df,
            report_type_column=args.report_type_key,
            text_column=args.text_col,
            report_type_filter=report_parser.report_type
        )
    else:  # json
        adapter = JsonAdapter(
            input_path=args.input,
            text_key=args.text_key
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