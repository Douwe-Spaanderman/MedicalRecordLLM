import json
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RawOutputAnalyzer:
    def __init__(self, csv_path: str, config_path: str):
        """
        Initialize the analyzer with CSV path and config path
        
        Args:
            csv_path: Path to CSV containing raw outputs
            config_path: Path to YAML config used for parsing
        """
        self.csv_path = Path(csv_path)
        self.config_path = Path(config_path)
        self.required_fields = []
        self.optional_fields = []
        self.field_config = []
        
        self._load_config()
        self._validate_csv()
        
    def _load_config(self):
        """Load the YAML config to understand field requirements"""
        import yaml
        
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
            
        self.field_config = config['field_instructions']
        self.required_fields = [field['name'] for field in self.field_config if field.get('required', False)]
        self.optional_fields = [field['name'] for field in self.field_config if not field.get('required', False)]
        
    def _validate_csv(self):
        """Check if CSV exists and has required columns"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        # Check for at least one raw output column
        df = pd.read_csv(self.csv_path, nrows=1)
        if 'raw_output_initial' not in df.columns:
            raise ValueError("CSV does not contain raw_output_initial column")

    def analyze(self, show_examples: bool = False) -> Dict[str, Dict]:
        """
        Analyze raw outputs to identify failure patterns
        
        Args:
            show_examples: Whether to collect full examples (can be memory intensive)
            
        Returns:
            Dictionary with analysis results
        """
        df = pd.read_csv(self.csv_path)
        
        results = {
            'initial_attempt_stats': self._analyze_initial_attempt(df, show_examples),
            'required_attempt_counts': self._count_attempts(df, 'required'),
            'optional_attempt_counts': self._count_attempts(df, 'optional'),
        }
        
        if show_examples:
            results['initial_failure_examples'] = self._collect_initial_failures(df)
        
        # Calculate derived statistics
        total_reports = results['initial_attempt_stats']['total_reports']
        results['needed_required_attempts'] = sum(results['required_attempt_counts'].values())
        results['needed_optional_attempts'] = sum(results['optional_attempt_counts'].values())
        
        return results
        
    def _analyze_initial_attempt(self, df: pd.DataFrame, show_examples: bool) -> Dict:
        """Analyze just the initial greedy search attempt"""
        stats = {
            'total_reports': len(df),
            'missing_json': 0,
            'json_parse_errors': 0,
            'missing_any_required': 0,  # Reports missing at least one required field
            'missing_required_fields': defaultdict(int),  # Count per field
            'invalid_field_values': defaultdict(int),
            'complete_reports': 0  # Reports with all required fields present
        }
        
        for _, row in df.iterrows():
            if pd.isna(row['raw_output_initial']):
                continue
                
            raw_output = row['raw_output_initial']
            missing_fields = []
            
            # Check for JSON markers
            if '```json' not in raw_output:
                stats['missing_json'] += 1
                continue
            
            # Try to parse JSON
            try:
                parsed = self._extract_json(raw_output)
                if not parsed:
                    stats['json_parse_errors'] += 1
                    continue
                    
                # Check field completeness
                missing_fields = [
                    field for field in self.required_fields 
                    if field not in parsed or parsed[field] in ['', 'Not specified', None]
                ]
                
                if missing_fields:
                    stats['missing_any_required'] += 1
                    for field in missing_fields:
                        stats['missing_required_fields'][field] += 1
                else:
                    stats['complete_reports'] += 1
                
                # Check field validity
                for field_spec in self.field_config:
                    field = field_spec['name']
                    if field in parsed and 'options' in field_spec and parsed[field] not in field_spec['options']:
                        stats['invalid_field_values'][field] += 1
                
            except Exception as e:
                logging.warning(f"Error processing initial output: {e}")
                stats['json_parse_errors'] += 1
                continue
        
        # Calculate percentages
        stats['missing_json_pct'] = (stats['missing_json'] / stats['total_reports']) * 100
        stats['json_parse_errors_pct'] = (stats['json_parse_errors'] / stats['total_reports']) * 100
        stats['missing_any_required_pct'] = (stats['missing_any_required'] / stats['total_reports']) * 100
        stats['complete_reports_pct'] = (stats['complete_reports'] / stats['total_reports']) * 100
        
        # Convert defaultdicts to regular dicts
        stats['missing_required_fields'] = dict(stats['missing_required_fields'])
        stats['invalid_field_values'] = dict(stats['invalid_field_values'])
        
        return stats
        
    def _count_attempts(self, df: pd.DataFrame, attempt_type: str) -> Dict[int, int]:
        """Count how many reports needed each attempt (1st, 2nd, etc.)"""
        attempt_cols = [col for col in df.columns if col.startswith(f'raw_output_{attempt_type}')]
        attempt_counts = defaultdict(int)
        
        for _, row in df.iterrows():
            last_attempt = 0
            for col in sorted(attempt_cols):
                if pd.notna(row[col]):
                    last_attempt = int(col.split('_')[-1])  # Get the attempt number
                    
            if last_attempt > 0:
                attempt_counts[last_attempt] += 1
                
        return dict(attempt_counts)
        
    def _collect_initial_failures(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Collect examples of different failure types from initial attempt"""
        examples = {
            'missing_json': [],
            'json_parse_errors': [],
            'missing_required_fields': defaultdict(list),
            'invalid_field_values': defaultdict(list)
        }
        
        for _, row in df.iterrows():
            if pd.isna(row['raw_output_initial']):
                continue
                
            raw_output = row['raw_output_initial']
            
            # Check for JSON markers
            if '```json' not in raw_output:
                examples['missing_json'].append(raw_output)
                continue
            
            # Try to parse JSON
            try:
                parsed = self._extract_json(raw_output)
                if not parsed:
                    examples['json_parse_errors'].append(raw_output)
                    continue
                    
                # Check for missing required fields
                for field in self.required_fields:
                    if field not in parsed or parsed[field] in ['', 'Not specified', None]:
                        examples['missing_required_fields'][field].append({
                            'output': raw_output,
                            'parsed': parsed
                        })
                
                # Check field validity
                for field_spec in self.field_config:
                    field = field_spec['name']
                    if field in parsed and 'options' in field_spec and parsed[field] not in field_spec['options']:
                        examples['invalid_field_values'][field].append({
                            'value': parsed[field],
                            'output': raw_output,
                            'expected': field_spec['options']
                        })
                
            except Exception:
                examples['json_parse_errors'].append(raw_output)
                continue
        
        # Convert defaultdicts to regular dicts
        examples['missing_required_fields'] = dict(examples['missing_required_fields'])
        examples['invalid_field_values'] = dict(examples['invalid_field_values'])
        
        return examples
        
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from between ```json markers"""
        try:
            json_matches = list(re.finditer(r'```json\s*(?P<json>{.*?})\s*```', text, re.DOTALL))
            for match in reversed(json_matches):
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        except Exception:
            return None
        return None
    
    def generate_report(self, results: Dict, output_file: Optional[str] = None, show_examples: bool = False):
        """Generate a human-readable analysis report"""
        stats = results['initial_attempt_stats']
        
        report = [
            "="*80,
            "LLM OUTPUT ANALYSIS REPORT",
            "="*80,
            f"Total reports processed: {stats['total_reports']}",
            "",
            "INITIAL GREEDY SEARCH RESULTS:",
            "-"*40,
            f"Complete reports (all required fields present): {stats['complete_reports']} ({stats['complete_reports_pct']:.1f}%)",
            f"Reports missing JSON markers: {stats['missing_json']} ({stats['missing_json_pct']:.1f}%)",
            f"Reports with JSON parse errors: {stats['json_parse_errors']} ({stats['json_parse_errors_pct']:.1f}%)",
            f"Reports missing at least one required field: {stats['missing_any_required']} ({stats['missing_any_required_pct']:.1f}%)",
            "",
            "MISSING REQUIRED FIELDS (INITIAL ATTEMPT):",
            "-"*40
        ]
        
        for field, count in sorted(stats['missing_required_fields'].items(), key=lambda x: -x[1]):
            report.append(f"- {field}: {count} ({(count/stats['total_reports'])*100:.1f}%)")
        
        report.extend([
            "",
            "REQUIRED FIELD ATTEMPTS:",
            "-"*40,
            f"Reports needing required field attempts: {results['needed_required_attempts']}",
            f"Breakdown by number of attempts needed:"
        ])
        
        for attempt, count in sorted(results['required_attempt_counts'].items()):
            report.append(f"- {attempt} attempt(s): {count} reports")
        
        report.extend([
            "",
            "OPTIONAL FIELD ATTEMPTS:",
            "-"*40,
            f"Reports needing optional field attempts: {results['needed_optional_attempts']}",
            f"Breakdown by number of attempts needed:"
        ])
        
        for attempt, count in sorted(results['optional_attempt_counts'].items()):
            report.append(f"- {attempt} attempt(s): {count} reports")
        
        # Add examples section if requested
        if show_examples and 'initial_failure_examples' in results:
            report.extend(self._generate_examples_section(results['initial_failure_examples']))
        
        # Write to file or print
        report_text = "\n".join(report)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logging.info(f"Report saved to {output_file}")
        else:
            print(report_text)
    
    def _generate_examples_section(self, examples: Dict) -> List[str]:
        """Generate the examples section of the report"""
        section = [
            "",
            "FAILURE EXAMPLES (INITIAL ATTEMPT):",
            "-"*40,
            "MISSING JSON MARKERS:",
            *examples['missing_json'][:3],
            "",
            "JSON PARSE ERRORS:",
            *examples['json_parse_errors'][:3],
            "",
            "MOST COMMON MISSING REQUIRED FIELDS:"
        ]
        
        # Show examples for top 3 missing fields
        top_missing = sorted(
            examples['missing_required_fields'].items(),
            key=lambda x: -len(x[1])
        )[:3]
        
        for field, field_examples in top_missing:
            section.extend([
                f"Field: {field}",
                "Full output:",
                field_examples[0]['output'],
                "",
                "Parsed JSON (partial):",
                json.dumps({k: v for k, v in field_examples[0]['parsed'].items() if k in self.required_fields}, indent=2),
                ""
            ])
        
        section.extend([
            "INVALID FIELD VALUES:"
        ])
        
        # Show examples for top 3 invalid fields
        top_invalid = sorted(
            examples['invalid_field_values'].items(),
            key=lambda x: -len(x[1])
        )[:3]
        
        for field, field_examples in top_invalid:
            section.extend([
                f"Field: {field}",
                f"Invalid value: {field_examples[0]['value']}",
                f"Expected one of: {field_examples[0]['expected']}",
                "Full output:",
                field_examples[0]['output'],
                ""
            ])
        
        return section

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze raw outputs from VLLMReportParser")
    parser.add_argument("csv_path", help="Path to CSV containing raw outputs")
    parser.add_argument("config_path", help="Path to YAML config used for parsing")
    parser.add_argument("--output", help="Path to save analysis report", default=None)
    parser.add_argument("--show-examples", help="Include failure examples in report", action="store_true")
    args = parser.parse_args()
    
    analyzer = RawOutputAnalyzer(args.csv_path, args.config_path)
    results = analyzer.analyze(show_examples=args.show_examples)
    analyzer.generate_report(results, args.output, show_examples=args.show_examples)