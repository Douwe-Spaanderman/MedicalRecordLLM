import json
import re
from typing import Dict, List, Any, Optional
from vllm import SamplingParams
import logging
from adapters.base_adapter import BaseAdapter
from openai import OpenAI
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VLLMReportParser:
    def __init__(
        self,
        params_config: dict,
        query_config: dict,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = "DummyAPIKey",
        prompt_method: str = "ZeroShot",
        patterns_path: Optional[str] = None,
        save_raw_output: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the parser with vLLM configuration
        
        Args:
            params_config: Model parameters configuration dictionary
            query_config: Configuration for the query, including field and tasks specifications
            base_url: Base URL for vLLM API
            api_key: API key for vLLM (often not required locally)
            prompt_method: Method for generating prompts (e.g., "ZeroShot")
            patterns_path: Path to JSON file with optional extraction patterns
            save_raw_output: Save raw model outputs to data
            verbose: Enable verbose logging for debugging
        """
        # Initialize OpenAI client
        self.openai = OpenAI(api_key = api_key, base_url = base_url)

        # Load model parameters
        self.model_name = params_config.get('model')
        self.max_model_len = params_config.get('max_model_len', 2048)
        self.sampling_params = SamplingParams(
            max_tokens=params_config.get('max_tokens', 1024),
            temperature=params_config.get('temperature', 0.3),
            top_p=params_config.get('top_p', 0.9),
            repetition_penalty=params_config.get('repetition_penalty', 1.0)
        )
        self.self_consistency_sampling = params_config.get('self_consistency_sampling', {"num_samples": 1, "temperature": 0.3})
    
        # Load prompt configuration and method
        self.query_config = query_config
        self.prompt_method = prompt_method
        self.chains = self._detect_chains()

        # Additional configurations
        self.patterns = self._load_patterns(patterns_path) if patterns_path else None
        self.save_raw_output = save_raw_output
        self.verbose = verbose

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

    def _parse_json(self, data: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from string"""
        try:
            json_matches = list(re.finditer(r'```json\s*(?P<json>{.*?})\s*```', data, re.DOTALL))
            for match in reversed(json_matches):
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue  # Try the next candidate      
        except Exception as e:
            logging.warning(f"JSON extraction failed: {e}")
            
        return None

    def _validate_example(self, example: Dict) -> None:
        """Validate that an example has the correct structure"""
        if not isinstance(example, dict):
            raise ValueError("Example must be a dictionary")
        if 'input' not in example or 'output' not in example:
            raise ValueError("Example must contain both 'input' and 'output'")
        if not isinstance(example['input'], str) or not isinstance(example['output'], str):
            raise ValueError("Example input and output must be strings")

    def _detect_chains(self) -> Optional[List[int]]:
        """
        Detect if the query configuration contains chains for PromptChain method

        Returns:
            List of chain indices if chains are defined, otherwise None
        """
        # Chains are defined in the query_config under 'field_instructions'
        chains = []
        for field in self.query_config['field_instructions']:
            if chain_order := field.get('chain_order'):
                chains.append(chain_order)

        # Now check if the length of task, example, reasoning if present and output match the chain order
        if len(chains) != len(self._parse_json(self.query_config.get('task', ''))):
            raise ValueError("Chain order does not match the number of tasks defined in the YAML configuration. This will result in prompt generation errors.")
        
        # Same validation of example if present
        if 'example' in self.query_config:
            example = self.query_config['example']
            if reasoning := example.get('reasoning'):
                # Parse reasoning as JSON if it exists
                reasoning = reasoning.strip().split("\n")
                if len(chains) != len(reasoning):
                    raise ValueError("Chain order does not match the number of reasoning steps defined in the example. This will result in prompt generation errors.")
            if 'output' in example and len(self._parse_json(example.get('output', ''))) != len(chains):
                raise ValueError("Chain order does not match the number of output fields defined in the example. This will result in prompt generation errors.")
                
        if not chains:
            return None
        else:
            return list(set(chains))  # Return unique chain orders

    def _format_field_instruction(self, field: Dict[str, str], index: int, chain: Optional[int]) -> str:
        """Convert YAML field config into numbered instruction format

        Args:
            field: Field configuration dictionary
            index: Index for numbering the field
            chain: Optional chain order to filter fields
        """
        # Skip field instruction not in chain if chain is specified
        if chain is not None and field.get('chain_order') != chain:
            return None

        instruction = [f'{index}. "{field["name"]}":']
        
        # Only add type when non-obvious
        instruction.append(f'   - Type: {field["type"]}')
        
        # Add constraints if present
        if 'constraints' in field:
            constraints = field['constraints']
            if isinstance(constraints, list):
                # Handle list constraints
                instruction.append(f'   - Constraints:')
                for item in constraints:
                    instruction.append(f'     • {item}')
            else:
                # Handle multi-line string constraints
                for line in constraints.split('\n'):
                    if line.strip():
                        if line.strip().startswith('- '):
                            # Convert list items to bullet points
                            instruction.append(f'     • {line.strip()[2:]}')
                        else:
                            instruction.append(f'   - {line.strip()}')
        
        # Add options if present
        if 'options' in field:
            opts = ', '.join(f'"{opt}"' for opt in field['options'])
            instruction.append(
                f'   - Must be EXACTLY one of the following options: [{opts}]. Please enter the value exactly as shown, without any variations.'
            )
        
        # Handle dict structures
        if field['type'] in ['dictionary', 'list of dictionaries'] and 'structure' in field:
            subfields = ', '.join(f'"{sf["key"]}"' for sf in field['structure'])
            if field['type'] == 'dictionary':
                instruction.append(f'   - Expected structure keys: {{ {subfields} }}')
            elif field['type'] == 'list of dictionaries':
                instruction.append(f'   - Each dictionary must contain the keys: {{ {subfields} }}')

            for subfield in field['structure']:
                key = subfield["key"]
                typ = subfield.get("type", "unknown")
                parts = [f'Type: {typ}']
                if 'options' in subfield:
                    opts = ', '.join(f'"{opt}"' for opt in subfield['options'])
                    parts.append(f'Must be EXACTLY one of the following options: [{opts}]. Please enter the value exactly as shown, without any variations.')
                if 'constraints' in subfield:
                    parts.append(subfield['constraints'].strip())
                details = '; '.join(parts)
                instruction.append(f'     - Key "{key}": {details}')
        
        # Handle default values
        default_value = field.get('default', 'Not specified')
        if isinstance(default_value, (dict, list)):
            # Compact formatting for empty lists
            if isinstance(default_value, list) and not default_value:
                instruction.append('   - Default: []')
            else:
                # Smart JSON formatting
                default_str = json.dumps(default_value, indent=2)
                if '\n' in default_str:
                    # For multi-line JSON, align with field instruction
                    default_str = default_str.replace('\n', '\n      ')
                    instruction.append(f'   - Default: {default_str}')
                else:
                    # Single-line for simple values
                    instruction.append(f'   - Default: {default_str}')
        else:
            instruction.append(f'   - Default: "{default_value}"')
        
        return '\n'.join(instruction)

    def _format_promptchain_instructions(self, instructions: str, chain_indexes: List[int]) -> str:
        """
        Extract and filter instructions for PromptChain method based on chain indexes.

        Args:
            instructions: Full instructions string containing JSON block
            chain_indexes: List of indexes to extract from the JSON block
        """
        # Extract everything before the ```json block
        preamble_match = re.split(r'```json', instructions)
        preamble = preamble_match[0].strip() if preamble_match else ""

        # Extract the JSON block using your existing function
        json_part = self._parse_json(instructions)

        # Filter the fields based on chain_indexes
        keys = list(json_part.keys())
        selected_keys = [keys[i] for i in chain_indexes if i < len(keys)]

        # Use OrderedDict to preserve the order
        filtered_json = OrderedDict()
        for key in selected_keys:
            filtered_json[key] = json_part[key]

        # Format back to task instruction style
        instructions = f"{preamble}\n```json\n{json.dumps(filtered_json, indent=2)}\n```"

        return instructions

    def _generate_query(self, report: str, patient: str, chain: Optional[int] = None) -> str:
        """Generate query for a single report with improved formatting"""
        # Build system instructions. Either use the config or default to a generic instruction
        if 'system_instruction' in self.query_config:
            print("WARNING: using system instruction from config file. This is not recommended.")
            system_instruction = self.query_config['system_instruction'].strip()
        else:
            print("Building system instruction based on prompt method, which is recommended")
            if self.prompt_method in ["ZeroShot", "OneShot"]:
                system_instruction = [
                    "You are a medical data extraction system that ONLY outputs valid JSON. Maintain strict compliance with these rules:",
                    "1. ALWAYS begin and end your response with ```json markers",
                    "2. Use EXACT field names and structure provided",
                    "3. If a value is missing or not mentioned, use the specified default for that field.",
                    "4. NEVER add commentary, explanations, or deviate from the output structure"
                ]
            elif self.prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
                system_instruction = [
                    "You are a medical data extraction system that performs structured reasoning before producing output. Follow these strict rules:",
                    "1. First, reason step-by-step to identify and justify each extracted field.",
                    "2. After reasoning, output ONLY valid JSON in the exact structure provided.",
                    "3. ALWAYS begin and end the final output with ```json markers — do not include reasoning within these markers.",
                    "4. Use EXACT field names and structure as specified.",
                    "5. If a value is missing or not mentioned, use the specified default for that field.",
                    "6. NEVER include commentary, explanations, or deviate from the specified format in the final JSON."
                ]

            system_instruction = "\n".join(system_instruction)

        # Build field instructions
        field_instructions = []
        chain_indexes = []
        for idx, field in enumerate(self.query_config['field_instructions'], start=1):
            field_instruction = self._format_field_instruction(field, idx, chain)
            if field_instruction is None:
                # Skip fields not in the specified chain
                continue
            else:
                field_instructions.append(field_instruction)
                chain_indexes.append(idx-1)
        
        # Build task instruction
        task_instructions = self.query_config['task'].strip()
        if self.prompt_method == "PromptChain":
            task_instructions = self._format_promptchain_instructions(task_instructions, chain_indexes)

        # Construct the prompt
        prompt_parts = [
            "[SYSTEM INSTRUCTION]",
            system_instruction,
            "",
            "[FIELD INSTRUCTIONS]",
            "\n".join(field_instructions),
            "",
            "[TASK INSTRUCTION]",
            task_instructions,
        ]

        # Add examples section
        if self.prompt_method != 'ZeroShot':
            if 'example' not in self.query_config:
                raise ValueError("Examples are required for prompt methods other than ZeroShot")
            
            example = self.query_config["example"]
            self.validate_example(example)
            prompt_parts.extend([
                "",
                "[EXAMPLE]"
            ])
            prompt_parts.extend([
                "---",
                "[EXAMPLE INPUT REPORT]",
                example['input'].strip(),
                "",
            ])

            if self.prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
                if 'reasoning' not in example:
                    raise ValueError("Examples for CoT, SelfConsistency, and PromptChain must include 'reasoning'")
                
                reasoning_instructions = example['reasoning'].strip()
                if self.prompt_method == "PromptChain":
                    reasoning_instructions = reasoning_instructions.split("\n")
                    reasoning_instructions = "\n".join([reasoning_instructions[i] for i in chain_indexes if i < len(reasoning_instructions)])

                prompt_parts.extend([
                    "[EXAMPLE THINKING]",
                    reasoning_instructions,
                    "",
                ])

            example_instructions = example['output'].strip()
            if self.prompt_method == "PromptChain":
                example_instructions = self._format_promptchain_instructions(example_instructions, chain_indexes)

            prompt_parts.extend([
                "[EXAMPLE EXPECTED OUTPUT]",
                example_instructions
            ])

        # Add the actual report content
        prompt_parts.extend([
            "",
            "[BEGIN FILE CONTENT]",
            f"[file name]: {patient}",
            report.strip(),
            "[END FILE CONTENT]",
            "",
        ])

        if self.prompt_method in ["ZeroShot", "OneShot"]:
            prompt_parts.append("Begin your extraction now. Your response MUST start with: ```json")
        elif self.prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
            prompt_parts.append("Begin your extraction now. First, reason step-by-step to identify each required field. After reasoning, output ONLY the final structured data and ensure your response starts with: ```json")

        return "\n".join(prompt_parts)

    def _generate_prompt_self_consistency(self, report: str, patient: str, responses: List[Dict[str, str]]) -> str:
        """
        Generate a prompt specifically for Self-Consistency method

        Args:
            report: The medical report text
            patient: Patient identifier or name
            responses: Candidate outputs from the model for Self-Consistency

        Returns:    
            The complete formatted prompt for Self-Consistency
        """
        system_instruction = "\n".join([
            "You are a medical data extraction system that performs structured reasoning across multiple reasoning paths before producing output. Follow these strict rules:",
            "1. First, review all candidate outputs and reason step-by-step to identify the most consistent and well-supported values for each field.",
            "2. After reasoning, output ONLY valid JSON in the exact structure provided.",
            "3. ALWAYS begin and end the final output with ```json markers — do not include reasoning within these markers.",
            "4. Use EXACT field names and structure as specified.",
            "5. If a field is missing from all reasoning paths, use the specified default for that field.",
            "6. NEVER include commentary, explanations, or deviate from the specified format in the final JSON."
        ])

        # Build field instructions
        field_instructions = []
        for idx, field in enumerate(self.query_config['field_instructions'], start=1):
            field_instructions.append(self._format_field_instruction(field, idx))

        # Construct the prompt
        prompt_parts = [
            "[SYSTEM INSTRUCTION]",
            system_instruction,
            "",
            "[FIELD INSTRUCTIONS]",
            "\n".join(field_instructions),
            "",
            "[TASK INSTRUCTION]",
            self.query_config['task'].strip(),
            "",
            "[BEGIN FILE CONTENT]",
            f"[file name]: {patient}",
            report.strip(),
            "[END FILE CONTENT]",
            "",
            "[BEGIN SELF-CONSISTENCY CANDIDATE OUTPUTS]",
            ""
        ]

        for i, response in enumerate(responses, start=1):
            prompt_parts.extend([
                f"[CANDIDATE OUTPUT {i}]",
                "[Reasoning]",
                response['reasoning'].strip(),
                "[Structured Output]",
                response['output'].strip(),
                ""
            ])

        prompt_parts += [
            "[END SELF-CONSISTENCY CANDIDATE OUTPUTS]",
            "Begin reconciliation now. First, review all reasoning paths and identify the most consistent and well-supported value for each required field. After reasoning, output ONLY the final structured data and ensure your response starts with: ```json"
        ]
        return "\n".join(prompt_parts)

    def query_model(self, query: str) -> str:
        """Query the LLM model using openAI"""
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": query}],
            temperature=self.sampling_params.temperature,
            max_tokens=self.sampling_params.max_tokens,
            top_p=self.sampling_params.top_p,
            repetition_penalty=self.sampling_params.repetition_penalty
        )
        return response.choices[0].message.content

    def process_reports(self, reports: List[str], patients: List[str]) -> List[Dict[str, Any]]:
        """Process all reports
        """       
        logging.info("Starting report processing...")
        response = []
        for i, (report, patient) in enumerate(zip(reports, patients)):
            logging.info(f"Processing report {i+1}/{len(reports)} for patient {patient}")
            if self.prompt_method == "SelfConsistency":
                print('Not implemented yet')
            elif self.prompt_method == "PromptChain":
                if self.chains is None:
                    raise ValueError("PromptChain method requires chain definitions in the query configuration")
                # Generate query for each chain
                query = []
                for chain in self.chains:
                    query.append(self._generate_query(report, patient, chain=chain))
            else:
                query =  self._generate_query(report, patient, chain=None)

            if self.verbose:
                logging.info(f"Raw query for patient {patient}: {query}")
                
            if self.prompt_method == "SelfConsistency":
                # TODO correctly implement number of samples generated and temperature changes
                responses = []
                for _ in range(self.self_consistency_sampling['num_samples']):
                    responses.append(self.query_model(query))
                
                query = self._generate_prompt_self_consistency(report, patient, response)
                response = self.query_model(query)
            else:
                response = self.query_model(query)

            if self.verbose:
                logging.info(f"Raw response for patient {patient}: {response}")

            # We should probably parse directly using text_format
            response = self._parse_json(response)

            for field_spec in self.field_config:
                field = field_spec['name']
                if field not in response:
                    response[field] = field_spec.get('default', 'Not specified')
                elif field_spec['type'] == 'nested':
                    if not isinstance(response[field], dict):
                        response[field] = field_spec['default']
                    else:
                        for subfield in field_spec['structure']:
                            if subfield['key'] not in response[field]:
                                response[field][subfield['key']] = subfield.get('default', 'Not specified')
                elif field_spec['type'] == 'list':
                    if not isinstance(response[field], list):
                        response[field] = field_spec['default']

            response.append(response)
        
        return response

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