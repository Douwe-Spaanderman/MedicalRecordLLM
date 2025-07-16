import json
import re
from typing import Dict, List, Any, Optional
import logging
from adapters.base_adapter import BaseAdapter
from collections import OrderedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import create_model, BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VLLMReportParser:
    def __init__(
        self,
        params_config: dict,
        prompt_config: dict,
        base_url: str = "http://localhost:8000/v1/",
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
            prompt_config: Configuration for the prompt, including field and tasks specifications
            base_url: Base URL for vLLM API
            api_key: API key for vLLM (often not required locally)
            prompt_method: Method for generating prompts (e.g., "ZeroShot")
            patterns_path: Path to JSON file with optional extraction patterns
            save_raw_output: Save raw model outputs to data
            verbose: Enable verbose logging for debugging
        """
        # Initialize OpenAI client
        self.llm = ChatOpenAI(
            model_name=params_config.get('model'),
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=params_config.get('temperature', 0.3),
            top_p=params_config.get('top_p', 0.9),
            #max_tokens=params_config.get('max_tokens', 1024),
            #max_model_len=params_config.get('max_model_len', 2048),
            #model_kwargs={
            #    "repetition_penalty": params_config.get('repetition_penalty', 1.0),
            #},
        )
        self.self_consistency_sampling = params_config.get('self_consistency_sampling', {"num_samples": 3, "temperature": [0.1, 0.3, 0.5]})

        # Load prompt configuration and method
        self.prompt_config = prompt_config
        self.prompt_method = prompt_method
        self.chains = self._detect_chains()

        # Additional configurations
        self.patterns = self._load_patterns(patterns_path) if patterns_path else None
        self.save_raw_output = save_raw_output
        self.verbose = verbose
        if self.verbose:
            import langchain
            langchain.verbose = True

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
        Detect if the prompt configuration contains chains for PromptChain method

        Returns:
            List of chain indices if chains are defined, otherwise None
        """
        # Chains are defined in the prompt_config under 'field_instructions'
        chains = []
        for field in self.prompt_config['field_instructions']:
            if chain_order := field.get('chain_order'):
                chains.append(chain_order)

        # Now check if the length of task, example, reasoning if present and output match the chain order
        if len(chains) != len(self._parse_json(self.prompt_config.get('task', ''))):
            raise ValueError("Chain order does not match the number of tasks defined in the YAML configuration. This will result in prompt generation errors.")
        
        # Same validation of example if present
        if 'example' in self.prompt_config:
            example = self.prompt_config['example']
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

    def _format_field_instruction(self, field: Dict[str, str], index: int, chain: Optional[int] = None) -> str:
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
                instruction.append(f'   - Expected structure keys: {subfields}')
            elif field['type'] == 'list of dictionaries':
                instruction.append(f'   - Each dictionary must contain the keys: {subfields}')

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

    def _format_promptchain_instructions(self, instructions: str, chain_indexes: List[int], variable_name:str) -> str:
        """
        Extract and filter instructions for PromptChain method based on chain indexes.

        Args:
            instructions: Full instructions string containing JSON block
            chain_indexes: List of indexes to extract from the JSON block
        """
        # Extract everything before the ```json block
        preamble_match = re.split(r'```json', instructions)
        preamble = preamble_match[0].strip() + "\n" if preamble_match else ""

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
        variable_placeholder = f"{{{variable_name}}}"
        instructions = "\n".join([
            f"{preamble}",
            "```json",
            variable_placeholder,
            "```"
        ])

        return instructions, json.dumps(filtered_json, indent=2)

    def _format_json_instructions(self, instructions:str, variable_name:str) -> str:
        """
        Convert instructions containing a JSON block into a properly formatted string

        Args:
            instructions: Full instructions string containing JSON block

        Returns:
            Formatted instructions string with JSON block
        """
        # Extract everything before the ```json block
        preamble_match = re.split(r'```json', instructions)
        preamble = preamble_match[0].strip() + "\n" if preamble_match else ""

        # Extract the JSON block using your existing function
        json_part = self._parse_json(instructions)

        # Format back to task instruction style
        variable_placeholder = f"{{{variable_name}}}"
        instructions = "\n".join([
            f"{preamble}",
            "```json",
            variable_placeholder,
            "```"
        ])

        return instructions, json.dumps(json_part, indent=2)

    def _generate_prompt(self, chain: Optional[int] = None) -> List[str]:
        """
        Generate the prompt string based on the prompt method and configuration

        Args:
            chain: Optional chain index for PromptChain method

        Returns:
            A list of strings representing the prompt components
        """
        # Initialize prompt variables for downstream chain construction - necessary for ChatPromptTemplate 
        prompt_variables = {}
        # Build system instructions. Either use the config or default to a generic instruction
        if 'system_instruction' in self.prompt_config:
            print("WARNING: using system instruction from config file. This is not recommended.")
            system_instructions = self.prompt_config['system_instruction'].strip()
        else:
            print("Building system instruction based on prompt method, which is recommended")
            if self.prompt_method in ["ZeroShot", "OneShot"]:
                system_instructions = [
                    "You are a medical data extraction system that ONLY outputs valid JSON. Maintain strict compliance with these rules:",
                    "1. ALWAYS begin and end your response with ```json markers",
                    "2. Use EXACT field names and structure provided",
                    "3. If a value is missing or not mentioned, use the specified default for that field.",
                    "4. NEVER add commentary, explanations, or deviate from the output structure"
                ]
            elif self.prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
                system_instructions = [
                    "You are a medical data extraction system that performs structured reasoning before producing output. Follow these strict rules:",
                    "1. First, reason step-by-step to identify and justify each extracted field.",
                    "2. After reasoning, output ONLY valid JSON in the exact structure provided.",
                    "3. ALWAYS begin and end the final output with ```json markers — do not include reasoning within these markers.",
                    "4. Use EXACT field names and structure as specified.",
                    "5. If a value is missing or not mentioned, use the specified default for that field.",
                    "6. NEVER include commentary, explanations, or deviate from the specified format in the final JSON."
                ]

            system_instructions = "\n".join(system_instructions)

        # Build field instructions
        field_instructions = []
        chain_indexes = []
        for idx, field in enumerate(self.prompt_config['field_instructions'], start=1):
            field_instruction = self._format_field_instruction(field, idx, chain)
            if field_instruction is None:
                # Skip fields not in the specified chain
                continue
            else:
                field_instructions.append(field_instruction)
                chain_indexes.append(idx-1)

        field_instructions = "\n".join(field_instructions)
        
        # Build task instruction
        task_instructions = self.prompt_config['task'].strip()
        if self.prompt_method == "PromptChain":
            task_instructions, task_variable = self._format_promptchain_instructions(task_instructions, chain_indexes, "task_variable")
        else:
            task_instructions, task_variable = self._format_json_instructions(task_instructions, "task_variable")
        
        # Store task instructions in prompt variables for downstream use
        prompt_variables['task_variable'] = task_variable

        task_instructions = "\n".join([task_instructions])

        # Add examples section
        if self.prompt_method != 'ZeroShot':
            example_instructions = {}
            if 'example' not in self.prompt_config:
                raise ValueError("Examples are required for prompt methods other than ZeroShot")
            
            example = self.prompt_config["example"]
            self._validate_example(example)
            example_instructions["user"] = "\n".join([example['input'].strip()])

            if self.prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
                if 'reasoning' not in example:
                    raise ValueError("Examples for CoT, SelfConsistency, and PromptChain must include 'reasoning'")
                
                reasoning_instructions = example['reasoning'].strip()
                if self.prompt_method == "PromptChain":
                    reasoning_instructions = reasoning_instructions.split("\n")
                    example_instructions["assistant_reasoning"] = "\n".join([reasoning_instructions[i] for i in chain_indexes if i < len(reasoning_instructions)])

            example_outcome = example['output'].strip()
            if self.prompt_method == "PromptChain":
                example_outcome, example_output_variable = self._format_promptchain_instructions(example_outcome, chain_indexes, "example_output_variable")
            else:
                example_outcome, example_output_variable = self._format_json_instructions(example_outcome, "example_output_variable")

            example_instructions["assistant_output"] = "\n".join([example_outcome])
            # Store example output variable in prompt variables for downstream use
            prompt_variables["example_output_variable"] = example_output_variable
        else:
            example_instructions = None

        # Add the actual report content
        report_instructions = "\n".join([
            "[file name]: {patient}",
            "{report}",
        ])

        if self.prompt_method in ["ZeroShot", "OneShot"]:
            final_instructions = "Begin your extraction now. Your response MUST only contain valid JSON and no thinking starting with: ```json"
        elif self.prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
            final_instructions = "Begin your extraction now. First, reason step-by-step to identify each required field. After reasoning, output ONLY the final structured data and ensure your response contains valid JSON starting with: ```json"

        return system_instructions, field_instructions, task_instructions, example_instructions, report_instructions, final_instructions, prompt_variables

    def _generate_prompt_self_consistency(self) -> List[str]:
        """
        Generate a prompt specifically for Self-Consistency method

        Returns:    
            Formatted prompt string for Self-Consistency
        """
        # Initialize prompt variables for downstream chain construction - necessary for ChatPromptTemplate
        prompt_variables = {}

        # Build system instructions
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
        for idx, field in enumerate(self.prompt_config['field_instructions'], start=1):
            field_instructions.append(self._format_field_instruction(field, idx))

        field_instructions = "\n".join(field_instructions)

        # Build task instruction
        task_instructions = self.prompt_config['task'].strip()
        task_instructions, task_variable = self._format_json_instructions(task_instructions, "task_variable")
        
        # Store task instructions in prompt variables for downstream use
        prompt_variables['task_variable'] = task_variable

        task_instructions = "\n".join([task_instructions])

        # Construct the prompt
        report_instructions = "\n".join([
            "[file name]: {patient}",
            "{report}",
        ])

        candidate_instructions = {}
        num_samples = self.self_consistency_sampling.get('num_samples', 3)
        # run through num_samples
        for i in range(num_samples):
            reasoning_placeholder = f"{{reasoning_{i+1}}}"
            response_placeholder = f"{{response_{i+1}}}"
            candidate_instructions[i] = {
                "reasoning": reasoning_placeholder,
                "response": response_placeholder
            }

        final_instructions = "Begin reconciliation now. First, review all reasoning paths and identify the most consistent and well-supported value for each required field. After reasoning, output ONLY the final structured data and ensure your response starts with: ```json"

        return system_instruction, field_instructions, task_instructions, report_instructions, candidate_instructions, final_instructions, prompt_variables

    def process_reports(self, reports: List[str], patients: List[str]) -> List[Dict[str, Any]]:
        """Process all reports

        Args:
            reports: List of report strings to process
            patients: List of patient identifiers corresponding to each report

        Returns:
            List of dictionaries with extracted data from each report
        """       
        logging.info(f"Stating generating prompt using {self.prompt_method} method")
        if self.prompt_method == "PromptChain":
            if self.chains is None:
                raise ValueError("PromptChain method requires chain definitions in the prompt configuration")
            
            # Generate prompts for each chain
            prompt_templates = []
            for chain in self.chains:
                system_instructions, field_instructions, task_instructions, example_instructions, report_instructions, final_instructions, prompt_variables = self._generate_prompt(chain=chain)
                prompt = [
                    SystemMessage(system_instructions, name="system_instructions"),
                    HumanMessage(field_instructions, name="field_instructions"),
                    HumanMessage(task_instructions, name="task_instructions"),
                ]

                if example_instructions:
                    prompt.extend([
                        HumanMessage(example_instructions["user"], name="example_user"),
                    ])
                    if "assistant_reasoning" in example_instructions:
                        prompt.extend([
                            AIMessage(example_instructions["assistant_reasoning"], name="example_assistant_reasoning"),
                        ])
                    prompt.extend([
                        AIMessage(example_instructions["assistant_output"], name="example_assistant_output")
                    ])

                prompt.extend([
                    HumanMessage(report_instructions, name="report_instructions"),
                    HumanMessage(final_instructions, name="final_instructions")
                ])
                prompt_template = ChatPromptTemplate(prompt)
                prompt_template = prompt_template.partial(**prompt_variables)

                prompt_templates.append(prompt_template)
            
            # TODO: Implement chain processing for PromptChain method with memory
        else:
            system_instructions, field_instructions, task_instructions, example_instructions, report_instructions, final_instructions, prompt_variables = self._generate_prompt(chain=None)
            prompt = [
                SystemMessage(system_instructions),
                HumanMessage(field_instructions, name="field_instructions"),
                HumanMessage(task_instructions, name="task_instructions"),
            ]

            if example_instructions:
                prompt.extend([
                    HumanMessage(example_instructions["user"], name="example_user"),
                ])
                if "assistant_reasoning" in example_instructions:
                    prompt.extend([
                        AIMessage(example_instructions["assistant_reasoning"], name="example_assistant_reasoning"),
                    ])
                prompt.extend([
                    AIMessage(example_instructions["assistant_output"], name="example_assistant_output")
                ])

            prompt.extend([
                HumanMessage(report_instructions, name="report_instructions"),
                HumanMessage(final_instructions, name="final_instructions")
            ])
            prompt_template = ChatPromptTemplate(prompt)
            prompt_template = prompt_template.partial(**prompt_variables)
            chain = prompt_template | self.llm

            if self.prompt_method == "SelfConsistency":
                # For SelfConsistency, we will generate a follow-up prompt
                system_instruction, field_instructions, task_instructions, report_instructions, candidate_instructions, final_instructions, prompt_variables = self._generate_prompt_self_consistency()
                follow_up_prompt_template = ChatPromptTemplate([
                    SystemMessage(system_instruction, name="system_instructions"),
                    HumanMessage(field_instructions, name="field_instructions"),
                    HumanMessage(task_instructions, name="task_instructions"),
                    HumanMessage(report_instructions, name="report_instructions"),
                ])

                for key, value in candidate_instructions.items():
                    follow_up_prompt_template.extend([
                        AIMessage(value['reasoning'], name=f"candidate_reasoning_{key}"),
                        AIMessage(value['response'], name=f"candidate_response_{key}")
                    ])

                follow_up_prompt_template.extend([
                    HumanMessage(final_instructions, name="final_instructions")
                ])
                follow_up_prompt_template = follow_up_prompt_template.partial(**prompt_variables)
                ensemble_chain = follow_up_prompt_template | self.llm

        logging.info("Starting report processing...")
        responses = []
        for i, (report, patient) in enumerate(zip(reports, patients)):
            logging.info(f"Processing report {i+1}/{len(reports)} for patient {patient}")

            if self.prompt_method == "PromptChain":
                raise NotImplementedError("PromptChain method is not yet implemented for processing multiple reports in a single call.")
            elif self.prompt_method == "SelfConsistency":
                raise NotImplementedError("SelfConsistency method is not yet implemented for processing multiple reports in a single call.")
                # For SelfConsistency, we need to generate multiple candidate outputs
                candidate_outputs = []
                for j in range(self.self_consistency_sampling.get('num_samples', 3)):
                    logging.info(f"Generating candidate output {j+1} for report {i+1}")
                    self.llm = self.self_consistency_sampling.get('temperature', [0.3, 0.5, 0.7])[j % len(self.self_consistency_sampling['temperature'])]  # Cycle through temperatures
                    response = chain.invoke({"report": report, "patient": patient})
                    candidate_outputs.append(response)

                response = ensemble_chain.invoke({"report": report, "patient": patient})   
            else:
                response = chain.invoke({"report": report, "patient": patient})
                import ipdb; ipdb.set_trace()

        return responses

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