from pathlib import Path
import json
import re
import yaml
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from vllm import LLM, SamplingParams
import logging
from adapters.base_adapter import BaseAdapter

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
        
        self.templates = {
            'system': query_config['system_instruction'],
            'fields': self._parse_field_instructions(self.field_config),
            'task': query_config['task']
        }

        # Handle both single example and multiple examples format
        if 'examples' in query_config:
            self.templates['examples'] = query_config['examples']
        elif 'example' in query_config:
            self.templates['examples'] = [query_config['example']]
        else:
            self.templates['examples'] = []

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
            lines = [f'{idx}. "{field["name"]}":']
            
            # Only add type for complex fields
            if field["type"] in ("nested", "list") or "options" in field:
                lines.append(f'   - Type: {field["type"].capitalize()}')
            
            # Handle constraints
            if 'constraints' in field:
                constraints = field['constraints']
                if isinstance(constraints, list):
                    lines.append('   - Constraints:')
                    for item in constraints:
                        lines.append(f'     • {item}')
                else:
                    for line in str(constraints).split('\n'):
                        if line.strip():
                            if line.strip().startswith('- '):
                                lines.append(f'     • {line.strip()[2:]}')
                            else:
                                lines.append(f'   - {line.strip()}')
            
            # Handle options with strict wording
            if 'options' in field:
                opts = ', '.join(f'"{opt}"' for opt in field['options'])
                lines.extend([
                    f'   - Must be EXACTLY one of: [{opts}]',
                    '      No variations or additions allowed'
                ])
            
            # Handle nested structures
            if field['type'] == 'nested' and 'structure' in field:
                subfields = ', '.join(f'"{sf["key"]}"' for sf in field['structure'])
                lines.append(f'   - Structure: {{{subfields}}}')
                for subfield in field['structure']:
                    if 'constraints' in subfield:
                        lines.append(f'     • {subfield["key"]}: {subfield["constraints"]}')
            
            # Handle list of dictionaries
            elif field['type'] == 'list' and field.get('item_type') == 'dict':
                if 'required_keys' in field:
                    keys = ', '.join(f'"{k}"' for k in field['required_keys'])
                    lines.append(f'   - Required keys: {keys}')
            
            # Handle default values
            default_value = field.get('default', 'Not specified')
            if isinstance(default_value, (dict, list)):
                if isinstance(default_value, list) and not default_value:
                    lines.append('   - Default: []')
                else:
                    default_str = json.dumps(default_value, indent=2)
                    if '\n' in default_str:
                        default_str = default_str.replace('\n', '\n      ')
                        lines.append(f'   - Default: {default_str}')
                    else:
                        lines.append(f'   - Default: {default_str}')
            else:
                lines.append(f'   - Default: "{default_value}"')
            
            instructions.append('\n'.join(lines))
        
        return instructions

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

    def _generate_query(self, report: str, patient: str, attempt: int = 1) -> str:
        """Generate query for a single report with improved formatting"""
        processed_report = self._extract_text(report)
        prompt_parts = [
            "[SYSTEM INSTRUCTION]",
            self.templates['system'].strip(),
            "",
            "[FIELD INSTRUCTIONS]",
            "\n".join(self.templates['fields']),
            "",
            "[TASK INSTRUCTION]",
            self.templates['task'].strip(),
        ]

        # Add examples section if examples exist
        if self.templates['examples']:
            prompt_parts.extend([
                "",
                "[EXAMPLES]"
            ])
            
            for example in self.templates['examples']:
                prompt_parts.extend([
                    "---",
                    "[EXAMPLE INPUT REPORT]",
                    example['input'].strip(),
                    "",
                    "[EXAMPLE EXPECTED OUTPUT]",
                    example['output'].strip()
                ])

        prompt_parts.extend([
            "",
            "[BEGIN FILE CONTENT]",
            f"Patient ID: {patient}",
            f"Attempt: {attempt}",
            processed_report.strip(),
            "[END FILE CONTENT]",
            "",
            "Begin your extraction now. Your response MUST start with: ```json"
        ])

        return "\n".join(prompt_parts)

    def _generate_followup_query(self, report: str, missing_fields: List[str], patient, attempt: int = 1) -> str:
        """Generate targeted follow-up query for missing fields using consistent template"""
        processed_report = self._extract_text(report)
        
        # Filter field config to only include missing fields
        missing_config = [
            field for field in self.field_config 
            if field['name'] in missing_fields
        ]

        # Generate field instructions only for missing fields
        missing_instructions = self._parse_field_instructions(missing_config)

        # Create JSON template with only missing fields
        json_template = {field['name']: "" for field in missing_config}

        # Collect example outputs for missing fields from all examples
        example_outputs = []
        for example in self.templates['examples']:
            try:
                # Extract JSON from between the ```json markers
                example_json_str = re.search(r'```json\s*({.*?})\s*```', 
                                          example['output'], 
                                          re.DOTALL).group(1)
                full_example_output = json.loads(example_json_str)
                filtered_output = {
                    field['name']: full_example_output[field['name']]
                    for field in missing_config
                    if field['name'] in full_example_output
                }
                if filtered_output:
                    example_outputs.append({
                        'input': example['input'],
                        'output': filtered_output
                    })
            except (AttributeError, json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Could not parse example output: {e}")
                continue

        prompt_parts = [
            "[SYSTEM INSTRUCTION]",
            self.templates['system'].strip(),
            "",
            "[FIELD INSTRUCTIONS]",
            "\n".join(missing_instructions),
            "",
            "[TASK INSTRUCTION]",
            "Extract ONLY these fields in EXACTLY this structure:",
            "```json",
            json.dumps(json_template, indent=4) + "```",
        ]

        # Add examples section if we have relevant examples
        if example_outputs:
            prompt_parts.extend([
                "",
                "[EXAMPLES]"
            ])
            
            for example in example_outputs:
                prompt_parts.extend([
                    "---",
                    "[EXAMPLE INPUT REPORT]",
                    example['input'].strip(),
                    "",
                    "[EXAMPLE EXPECTED OUTPUT]",
                    "```json",
                    json.dumps(example['output'], indent=4) + "```"
                ])

        prompt_parts.extend([
            "",
            "[BEGIN FILE CONTENT]",
            f"Patient ID: {patient}",
            f"Attempt: {attempt}",
            processed_report.strip(),
            "[END FILE CONTENT]",
            "",
            "Extract ONLY the missing fields listed above. Your response MUST start with: ```json"
        ])

        return "\n".join(prompt_parts)

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response"""
        try:
            json_matches = list(re.finditer(r'```json\s*(?P<json>{.*?})\s*```', response, re.DOTALL))
            for match in reversed(json_matches):
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue  # Try the next candidate      
        except Exception as e:
            logging.warning(f"JSON extraction failed: {e}")
            
        return None

    def process_reports(self, reports: List[str], patients: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of reports with refined workflow:
        1. First attempt to get all fields (naive search)
        2. Then focus on missing required fields (max_attempts-1 attempts)
        3. Finally focus on missing optional fields (max_attempts-1 attempts)
        """
        # Initialize empty results
        results = [{} for _ in reports]
        required_fields = [field['name'] for field in self.field_config if field.get('required', False)]
        optional_fields = [field['name'] for field in self.field_config if not field.get('required', False)]
        
        # Phase 1: Initial naive search for all fields
        logging.info("Starting initial naive search for all fields")
        queries = [self._generate_query(report, patient) for report, patient in zip(reports, patients)]
        responses = self.llm.generate(queries, self.sampling_params)
        
        for idx, resp in enumerate(responses):
            if parsed := self._parse_response(resp.outputs[0].text):
                for field in self.field_config:
                    field_name = field['name']
                    if field_name in parsed:
                        results[idx][field_name] = parsed[field_name]

                if self.save_raw_output:
                    results[idx]["raw_output_initial"] = resp.outputs[0].text
            
        # Phase 2: Focus on missing required fields
        remaining_attempts = self.max_attempts - 1
        if remaining_attempts > 0:
            active_indices = [
                i for i, res in enumerate(results)
                if any(f not in res for f in required_fields)
            ]
            logging.info(f"Processing {len(active_indices)} reports with missing required fields")
            
            for attempt in range(remaining_attempts):
                if not active_indices:
                    break
                    
                # Generate follow-up queries for missing required fields
                queries = []
                new_active_indices = []
                
                for idx in active_indices:
                    missing_required = [f for f in required_fields if f not in results[idx]]
                    if missing_required:
                        new_active_indices.append(idx)
                        queries.append(
                            self._generate_followup_query(reports[idx], missing_required, patients[idx], attempt)
                        )
                
                if not queries:
                    break
                    
                active_indices = new_active_indices

                if self.update_config:
                    logging.info(f"Updating sampling params for attempt {attempt + 1}")
                    self.sampling_params = SamplingParams(**self.update_config[attempt])
                
                # Process batch
                responses = self.llm.generate(queries, self.sampling_params)
                
                # Update only required fields
                for i, resp in zip(active_indices, responses):
                    if parsed := self._parse_response(resp.outputs[0].text):
                        for field in required_fields:
                            if field in parsed:
                                results[i][field] = parsed[field]
                        
                        if self.save_raw_output:
                            results[i][f"raw_output_required_{attempt+1}"] = resp.outputs[0].text
        
        # Phase 3: Focus on missing optional fields (only for reports with all required fields)
        remaining_attempts = self.max_attempts - 1
        if remaining_attempts > 0:
            active_indices = [
                i for i, res in enumerate(results)
                if all(f in res for f in required_fields) and 
                any(f not in res for f in optional_fields)
            ]
            logging.info(f"Processing {len(active_indices)} reports with missing optional fields")
            
            for attempt in range(remaining_attempts):
                if not active_indices:
                    break
                    
                # Generate follow-up queries for missing optional fields
                queries = []
                new_active_indices = []
                
                for idx in active_indices:
                    missing_optional = [f for f in optional_fields if f not in results[idx]]
                    if missing_optional:
                        new_active_indices.append(idx)
                        queries.append(
                            self._generate_followup_query(reports[idx], missing_optional, patients[idx], attempt)
                        )
                
                if not queries:
                    break
                    
                active_indices = new_active_indices

                if self.update_config:
                    logging.info(f"Updating sampling params for attempt {attempt + 1}")
                    self.sampling_params = SamplingParams(**self.update_config[attempt])
                
                # Process batch
                responses = self.llm.generate(queries, self.sampling_params)
                
                # Update only optional fields
                for i, resp in zip(active_indices, responses):
                    if parsed := self._parse_response(resp.outputs[0].text):
                        for field in optional_fields:
                            if field in parsed:
                                results[i][field] = parsed[field]
                        
                        if self.save_raw_output:
                            results[i][f"raw_output_optional_{attempt+1}"] = resp.outputs[0].text
        
        # Apply defaults for any remaining missing fields
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
        
        return results

    def process_with_adapter(self, adapter: BaseAdapter) -> Any:
        """
        Process reports using the specified adapter
        
        Args:
            adapter: Configured adapter instance
            
        Returns:
            Processed results in adapter's output format
        """
        texts, patients = adapter.prepare_inputs()
        results = self.process_reports(texts, patients)
        return adapter.format_outputs(results)