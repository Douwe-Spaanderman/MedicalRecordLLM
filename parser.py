from pathlib import Path
import json
import re
import yaml
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VLLMReportParser:
    def __init__(
        self,
        model: str,
        query_config: dict,
        gpus: int = 1,
        patterns_path: Optional[str] = None,
        max_model_len: int = 32768,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        max_attempts: int = 3,
        update_config: Optional[List[Dict[str, Any]]] = None,
        save_raw_output: bool = False,
    ):
        """
        Initialize the parser with vLLM configuration
        
        Args:
            model: Model name/path for vLLM
            query_config: query configuration dictionary
            gpus: Number of GPUs for tensor parallelism
            patterns_path: Path to JSON file with optional extraction patterns
            max_model_len: Max sequence length for model
            max_tokens: Max tokens for sampling
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            repetition_penalty: Repetition penalty factor
            max_attempts: Maximum retry attempts
            update_config: Optional config to update the default sampling params each attempt
            save_raw_output: Save raw model outputs to data
        """
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=gpus
        )
        self.max_model_len = max_model_len
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        self.update_config = update_config
        
        self.report_type = query_config['report_type']
        self.field_config = query_config['field_instructions']
        self.required_fields = [field['name'] for field in self.field_config]
        logging.info(f"Extracted required fields: {self.required_fields}")
        
        self.templates = {
            'system': query_config['system_instruction'],
            'fields': self._parse_field_instructions(self.field_config),
            'task': query_config['task'],
            'example': query_config['example']
        }

        self.max_attempts = max_attempts
        self.patterns = self._load_patterns(patterns_path) if patterns_path else None
        self.save_raw_output = save_raw_output

    def _load_patterns(self, patterns_path: str) -> Optional[Dict]:
        """Load and compile regex patterns from JSON file"""
        try:
            with open(patterns_path) as f:
                patterns = json.load(f)
            return self._compile_patterns(patterns)
        except Exception as e:
            logging.error(f"Error loading patterns: {e}")
            return None

    def _compile_patterns(self, patterns: Dict) -> Dict:
        """Compile regex patterns with flags"""
        compiled = {}
        for name, pat in patterns.items():
            try:
                flags = 0
                if 'flags' in pat:
                    for flag in pat['flags'].split('|'):
                        flags |= getattr(re, flag.strip(), 0)
                
                compiled[name] = {
                    'pattern': re.compile(pat['pattern'], flags),
                    'start': pat.get('start', 0)
                }
            except Exception as e:
                logging.error(f"Error compiling pattern {name}: {e}")
        return compiled

    def _parse_field_instructions(self, fields: list) -> str:
        """Convert YAML field config into numbered instruction format"""
        instructions = []
        
        for idx, field in enumerate(fields, start=1):
            parts = [f'{idx}. "{field["name"]}":']
            
            # Add constraints
            if 'constraints' in field:
                if isinstance(field['constraints'], list):
                    for constraint in field['constraints']:
                        parts.append(f"   - {constraint}")
                else:
                    for line in str(field['constraints']).split('\n'):
                        if line.strip():
                            parts.append(f"   - {line.strip()}")
            
            # Add options for choice fields
            if 'options' in field:
                opts = ", ".join(f'"{opt}"' for opt in field['options'])
                parts.append(f"   - Options: {opts}")
            
            # Handle nested structures
            if field['type'] == 'nested':
                structure_desc = "Nested structure: {" + ", ".join(
                    f'"{sf["key"]}"' for sf in field['structure']
                ) + "}"
                parts.append(f"   - {structure_desc}")
                
                for subfield in field['structure']:
                    if 'constraints' in subfield:
                        parts.append(f"     - {subfield['key']}: {subfield['constraints']}")
            
            # Handle list structures
            elif field['type'] == 'list' and field.get('item_type') == 'dict':
                req_keys = ", ".join(f'"{k}"' for k in field.get('required_keys', []))
                parts.append(f"   - List of dictionaries with keys: {req_keys}")
            
            # Add notes if present
            if 'notes' in field:
                parts.append(f"   - Note: {field['notes']}")
            
            # Add default value
            default_value = field.get('default', 'Not specified')
            if isinstance(default_value, (dict, list)):
                default_str = json.dumps(default_value, indent=4).replace('\n', '\n      ')
                parts.append(f"   - Default: {default_str}")
            else:
                parts.append(f'   - Default: "{default_value}"')
            
            instructions.append("\n".join(parts))
        
        return "[FIELD INSTRUCTIONS]\n" + "\n\n".join(instructions)

    def _extract_text(self, text: str) -> str:
        """Apply pattern extraction if patterns are configured"""
        if not self.patterns or not text or not isinstance(text, str):
            return text
        
        extracted = []
        for pattern in self.patterns.values():
            try:
                match = pattern['pattern'].search(text)
                if match:
                    start = max(match.start(), pattern['start'])
                    extracted.append(text[start:match.end()])
            except Exception as e:
                logging.error(f"Error applying pattern {pattern['pattern']}: {e}")
        
        return "\n".join(extracted) if extracted else text

    def _generate_query(self, report: str, attempt: int = 1) -> str:
        """Generate query for a single report"""
        processed_report = self._extract_text(report)
        return f"""
            {self.templates['system']}

            {self.templates['fields']}

            [file name]: report_attempt_{attempt}
            [file content begin]
            {processed_report}
            [file content end]

            {self.templates['task']}

            {self.templates['example']}

            Begin your response with: ```json
        """

    def _generate_followup_query(self, report: str, missing_fields: List[str]) -> str:
        """Generate targeted follow-up query for missing fields"""
        processed_report = self._extract_text(report)
        
        # Get specs for missing fields
        field_specs = [
            fs for fs in self.field_config 
            if fs['name'] in missing_fields
        ]
        
        # Build detailed instructions
        instructions = []
        for spec in field_specs:
            desc = [f"{spec['name']}:"]
            
            if 'constraints' in spec:
                if isinstance(spec['constraints'], str):
                    desc.append(f"  - {spec['constraints']}")
                else:
                    for c in spec['constraints']:
                        desc.append(f"  - {c}")
            
            if spec['type'] == 'nested':
                subfields = ", ".join(
                    f"{sf['key']} ({sf.get('constraints', '')}" 
                    for sf in spec['structure']
                )
                desc.append(f"  - Nested structure: {{{subfields}}}")
            
            elif spec['type'] == 'list':
                if spec.get('item_type') == 'dict':
                    keys = ", ".join(spec['required_keys'])
                    desc.append(f"  - List of dicts with keys: {keys}")
                else:
                    desc.append("  - List of text items")
            
            instructions.append("\n".join(desc))
        
        # Build JSON template
        template = {}
        for spec in field_specs:
            field = spec['name']
            
            if spec['type'] == 'nested':
                template[field] = {
                    sf['key']: "" 
                    for sf in spec['structure']
                }
            elif spec['type'] == 'list':
                if spec.get('item_type') == 'dict':
                    template[field] = [{}]  # Example with one empty dict
                else:
                    template[field] = ["item1", "item2"]  # Example string list
            else:
                template[field] = ""
        
        return f"""
        {self.templates['system']}
        [IMPORTANT] Please ONLY provide these missing fields with EXACT formatting:
        
        {chr(10).join(instructions)}
        
        [file content begin]
        {processed_report}
        [file content end]
        
        [TASK]
        Extract information into this exact JSON structure:
        ```json
        {json.dumps(template, indent=4)}
        ```

        {self.templates['example']}

        Begin your response with: ```json
        """

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response"""
        try:
            json_match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def find_missing_fields(self, data: Dict[str, Any]) -> List[str]:
        """Identify missing or invalid fields based on the YAML config"""
        missing = []
        
        for field_spec in self.field_config:
            field_name = field_spec['name']
            
            # Field is completely missing
            if field_name not in data:
                missing.append(field_name)
                continue
                
            value = data[field_name]
            
            # Check nested structures
            if field_spec['type'] == 'nested':
                if not isinstance(value, dict):
                    missing.append(field_name)
                else:
                    # Check all required subfields exist
                    for subfield in field_spec.get('structure', []):
                        if subfield['key'] not in value:
                            missing.append(field_name)
                            break

            elif field_spec['type'] == 'dynamic':
                if not isinstance(value, dict):
                    missing.append(field_name)
            
            # Check list structures
            elif field_spec['type'] == 'list':
                if not isinstance(value, list):
                    missing.append(field_name)
                else:
                    # For lists of dictionaries
                    if field_spec.get('item_type') == 'dict':
                        required_keys = field_spec.get('required_keys', [])
                        for item in value:
                            if not isinstance(item, dict) or not all(k in item for k in required_keys):
                                missing.append(field_name)
                                break
                    # For simple lists (like differential diagnosis)
                    # No additional validation needed beyond being a list
            
            # Check empty values for simple fields
            elif not value and value != 0:  # 0 is a valid value
                missing.append(field_name)
        
        return missing

    def process_reports(self, reports: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of reports and extract information"""
        results = [{} for _ in reports]
        active_indices = list(range(len(reports)))
        
        for attempt in range(self.max_attempts):
            # Generate queries
            if attempt == 0:
                queries = [self._generate_query(reports[i]) for i in active_indices]
            else:
                retry_indices = []
                retry_queries = []
                for idx in active_indices:
                    missing = self.find_missing_fields(results[idx])
                    if missing:
                        retry_indices.append(idx)
                        retry_queries.append(
                            self._generate_followup_query(reports[idx], missing)
                        )

                if not retry_queries:
                    break
                    
                queries = retry_queries
                active_indices = retry_indices

                if self.update_config:
                    logging.info(f"Updating sampling params for attempt {attempt + 1}")
                    self.sampling_params = SamplingParams(**self.update_config[attempt - 1])
            
            # Process batch
            responses = self.llm.generate(queries, self.sampling_params)
            
            # Update results
            for i, resp in zip(active_indices, responses):
                if parsed := self._parse_response(resp.outputs[0].text):
                    results[i].update(parsed)

            if self.save_raw_output:
                for i, resp in zip(active_indices, responses):
                    results[i][f"raw_output_attempt_{attempt}"] = resp.outputs[0].text
        
        # Final validation
        for res in results:
            for field_spec in self.field_config:
                field = field_spec['name']
                if field not in res:
                    res[field] = field_spec.get('default', 'Not specified')
                elif field_spec['type'] == 'nested':
                    if not isinstance(res[field], dict):
                        res[field] = field_spec['default']
                    else:
                        for subfield in field_spec['structure']:
                            if subfield['key'] not in res[field]:
                                res[field][subfield['key']] = subfield.get('default', 'Not specified')
                elif field_spec['type'] == 'list':
                    if not isinstance(res[field], list):
                        res[field] = field_spec['default']
        
        # Additional pass for required fields
        required_fields = [
            field['name'] for field in self.field_config 
            if field.get('required', False)
        ]
        
        if required_fields:
            # Identify reports with missing required fields
            required_indices = []
            required_queries = []
            for idx, res in enumerate(results):
                missing_required = [
                    field for field in required_fields
                    if res.get(field, 'Not specified') == 'Not specified'
                ]
                if missing_required:
                    required_indices.append(idx)
                    required_queries.append(
                        self._generate_followup_query(reports[idx], missing_required)
                    )
            
            # Process required fields follow-up if needed
            if required_queries:
                responses = self.llm.generate(required_queries, self.sampling_params)
                
                # Update results with required fields
                for i, resp in zip(required_indices, responses):
                    if parsed := self._parse_response(resp.outputs[0].text):
                        # Only update the required fields, preserve other fields
                        for field in required_fields:
                            if field in parsed:
                                results[i][field] = parsed[field]
                
                if self.save_raw_output:
                    for i, resp in zip(required_indices, responses):
                        results[i]["raw_output_required_fields"] = resp.outputs[0].text
        
        return results

    def process_with_adapter(self, adapter: BaseAdapter) -> Any:
        """
        Process reports using the specified adapter
        
        Args:
            adapter: Configured adapter instance
            
        Returns:
            Processed results in adapter's output format
        """
        texts = adapter.prepare_inputs()
        results = self.process_reports(texts)
        return adapter.format_outputs(results)