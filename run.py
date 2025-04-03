import argparse
import pandas as pd
from pathlib import Path
from parser import VLLMReportParser
from adapters import DataFrameAdapter, JsonAdapter

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
        "-c", "--config",
        required=True,
        type=str,
        help="Path to YAML config"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        type=str,
        choices=["V3", "R1", "R1-Distill"],
        help="Model to use"
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
        "--report-type-col",
        default="reportType",
        help="Report type column in CSV (default: 'reportType')"
    )
    parser.add_argument(
        "--text-col",
        default="presentedForm_data",
        help="Text column in CSV (default: 'presentedForm_data')"
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
        "-a",
        "--attempts",
        required=False,
        type=int,
        default=3,
        help="Maximum number of LLM retry attempts",
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
    model = f"deepseek-ai/DeepSeek-{args.model}"
    if args.model == "R1-Distill":
        model += "-Qwen-32B"
    
    # Initialize parser
    report_parser = VLLMReportParser(
        model=f"deepseek-ai/DeepSeek-{args.model}",
        yaml_config=args.config,
        patterns_path=args.regex,
        max_attempts=args.attempts,
    )
    
    # Initialize appropriate adapter
    if args.format == "csv":
        df = pd.read_csv(args.input)
        adapter = DataFrameAdapter(
            df=df,
            report_type_column=args.report_type_col,
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