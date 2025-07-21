import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Iterable
import logging
import asyncio
import time
from math import ceil
from adapters.base_adapter import BaseAdapter
from collections import OrderedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseOutputParser
from pydantic import create_model, Field
from functools import wraps
import backoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def backoff_except_timeout(max_tries: int = 3):
    """
    Decorator to apply exponential backoff on exceptions, excluding asyncio.TimeoutError.
    This is useful for retrying operations that may fail due to transient issues,
    """
    def decorator(func):
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=max_tries,
            jitter=backoff.full_jitter,
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except asyncio.TimeoutError:
                raise
        return wrapper
    return decorator

class ReasoningAndDynamicJSONParser(BaseOutputParser):
    def __init__(self, output_format: Dict[str, Any]):
        # Build pydantic fields dynamically from output_format
        fields = {}
        for key, val in output_format.items():
            field_type = val["type"]
            default = val.get("default", None)

            if field_type == "string":
                typ = Optional[str]
            elif field_type == "list":
                typ = Optional[List[str]]
            elif field_type == "int":
                typ = Optional[int]
            else:
                typ = Optional[Any]

            fields[key] = (typ, Field(default=default, description=str(val.get("options", ""))))

        self._output_model = create_model("ExtractedData", **fields)
        self._logger = logging.getLogger(__name__)

    def parse(self, text: str) -> Dict[str, Any]:
        # Extract <think>...</think>
        reasoning_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        # Extract all ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(?P<json>{.*?})\s*```', text, re.DOTALL)

        # Fallback to any { ... } block
        if not json_blocks:
            fallback_blocks = re.findall(r'(?P<json>{.*?})', text, re.DOTALL)
            json_blocks = fallback_blocks

        extracted_data = None
        for block in reversed(json_blocks):
            try:
                data = json.loads(block.strip())
                extracted_data = self._output_model(**data).dict()
                break
            except Exception:
                continue

        if extracted_data is None:
            self._logger.error("Failed to parse JSON block or no valid JSON found.")

        return {
            "reasoning": reasoning,
            "extracted_data": extracted_data,
        }

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
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            import langchain
            langchain.verbose = True
            self.logger.setLevel(logging.DEBUG)

        # Generate the chat chains based on the prompt method
        self._generate_chat_chain()

    def _load_patterns(self, patterns_path: str) -> Optional[Dict]:
        """Load and compile regex patterns from JSON file"""
        try:
            with open(patterns_path) as f:
                patterns = json.load(f)
            return self._compile_patterns(patterns)
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
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
                self.logger.error(f"Error compiling pattern {name}: {e}")
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
                self.logger.error(f"Error applying pattern {pattern['pattern']}: {e}")
        
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
            self.logger.warning(f"JSON extraction failed: {e}")
            
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

    def _format_field_instruction(self, field: Dict[str, str], index: int, chain: Optional[int] = None, output_format: Dict[str, str] = {}) -> Tuple[str, dict]:
        """Convert YAML field config into numbered instruction format

        Args:
            field: Field configuration dictionary
            index: Index for numbering the field
            chain: Optional chain order to filter fields

        Returns:
            Formatted field instruction string
        """
        # Skip field instruction not in chain if chain is specified
        if chain is not None and field.get('chain_order') != chain:
            return None

        instruction = [f'{index}. "{field["name"]}":']
        output_format[field['name']] = {}
        
        # Add type
        instruction.append(f'   - Type: {field["type"]}')
        output_format[field['name']]['type'] = field['type']
        
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
            output_format[field['name']]['options'] = field['options']
        
        # Handle dict structures
        if field['type'] in ['dictionary', 'list of dictionaries'] and 'structure' in field:
            # TODO this is not correctly formatted now in output_format I think
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
        output_format[field['name']]['default'] = default_value
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
        
        return '\n'.join(instruction), output_format

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
            self.logger.warning("System instruction found in config file. This is not recommended for production use.")
            system_instructions = self.prompt_config['system_instruction'].strip()
        else:
            self.logger.debug("System instruction not found in config file. Using default system instruction, which is recommended")
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
        output_format = {}
        for idx, field in enumerate(self.prompt_config['field_instructions'], start=1):
            field_instruction, output_format = self._format_field_instruction(field, idx, chain, output_format)
            if field_instruction is None:
                # Skip fields not in the specified chain
                continue
            else:
                field_instructions.append(field_instruction)
                chain_indexes.append(idx-1)

        field_instructions = "\n".join(field_instructions)
        
        # Build task instruction # TODO should also be easily possible to construct this from output_format
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
            final_instructions = (
                "Begin the extraction now. Your response must contain only a single valid JSON block, "
                "enclosed in triple backticks and prefixed with `json`, like this: ```json ... ```"
            )
        elif self.prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
            final_instructions = (
                "Begin the extraction now. First, reason step-by-step to identify and justify the value for each required field, "
                "enclosed within <think>...</think> tags. Then, output only the final structured data as a single valid JSON block, "
                "starting with ```json and ending with ```."
            )
        return system_instructions, field_instructions, task_instructions, example_instructions, report_instructions, final_instructions, prompt_variables, output_format

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
        output_format = {}
        for idx, field in enumerate(self.prompt_config['field_instructions'], start=1):
            field_instruction, output_format = self._format_field_instruction(field, output_format=output_format)
            field_instructions.append(field_instruction)

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

        final_instructions = (
            "Begin reconciliation now. First, review all reasoning paths and determine the most consistent, well-supported value "
            "for each required field, reasoning step-by-step within <think>...</think> tags. Then output ONLY the final structured data "
            "as a valid JSON block, starting with ```json and ending with ```."
        )

        return system_instruction, field_instructions, task_instructions, report_instructions, candidate_instructions, final_instructions, prompt_variables, output_format

    def _generate_chat_chain(self) -> None:
        """
        Generate the chat chain based on the prompt method and configuration

        Returns:
            Runnable chain for processing reports
        """
        self.logger.info(f"Stating generating prompt using {self.prompt_method} method")
        if self.prompt_method == "PromptChain":
            if self.chains is None:
                raise ValueError("PromptChain method requires chain definitions in the prompt configuration")
            
            # Generate prompts for each chain
            prompt_templates = []
            for chain in self.chains:
                system_instructions, field_instructions, task_instructions, example_instructions, report_instructions, final_instructions, prompt_variables, output_format = self._generate_prompt(chain=chain)
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
            system_instructions, field_instructions, task_instructions, example_instructions, report_instructions, final_instructions, prompt_variables, output_format = self._generate_prompt(chain=None)
            prompt = [
                SystemMessage(system_instructions),
                HumanMessage(field_instructions, name="field_instructions"),
                HumanMessagePromptTemplate.from_template(task_instructions, name="task_instructions"),
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
                    AIMessagePromptTemplate.from_template(example_instructions["assistant_output"], name="example_assistant_output")
                ])

            prompt.extend([
                HumanMessagePromptTemplate.from_template(report_instructions, name="report_instructions"),
                HumanMessage(final_instructions, name="final_instructions")
            ])
            prompt_template = ChatPromptTemplate(prompt)
            prompt_template = prompt_template.partial(**prompt_variables)

            parser = ReasoningAndDynamicJSONParser(output_format)
            self.chat_chain = prompt_template | self.llm | parser

            if self.prompt_method == "SelfConsistency":
                # For SelfConsistency, we will generate a follow-up prompt
                system_instruction, field_instructions, task_instructions, report_instructions, candidate_instructions, final_instructions, prompt_variables, output_format = self._generate_prompt_self_consistency()
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
                self.ensemble_chain = follow_up_prompt_template | self.llm
                # TODO: Implement ensemble logic for SelfConsistency method

    @staticmethod
    def _dynamic_chunks(inputs: List[Dict], batch_size: int = 32, max_tokens: int = 16000) -> Iterable[List[Dict]]:
        """
        Dynamic batching that considers both item count and approximate token size

        Args:
            inputs: List of input dictionaries to process
            batch_size: Number of items in a batch
            max_tokens: Maximum token count for the batch (approximate)
            
        Yields:
            Iterable of input lists, each representing a batch of inputs
        """
        current_batch = []
        current_batch_size = 0
        
        for item in inputs:
            # Estimate token count (adjust based on your average token/report ratio)
            approx_tokens = len(item['report']) // 4  # Rough estimate: 1 token ≈ 4 chars
            item_size = max(1, ceil(approx_tokens / 1000))  # Convert to "size units" (1 = ~1000 tokens)
            
            # Handle single items that exceed max_tokens
            if approx_tokens > max_tokens:
                logging.warning(
                    f"Single input exceeds max_tokens ({approx_tokens} > {max_tokens}). "
                    f"Processing as single-item batch. "
                    f"Patient: {item.get('patient', 'unknown')}, at index: {item.get('index', 'unknown')}"
                )
                if current_batch:
                    yield current_batch
                    current_batch = []
                    current_batch_size = 0
                yield [item]
                continue

            # Start new batch if adding this item would exceed limits
            if (current_batch and 
                (len(current_batch) >= batch_size or 
                 current_batch_size + item_size > max_tokens // 1000)):
                yield current_batch
                current_batch = []
                current_batch_size = 0
                
            current_batch.append(item)
            current_batch_size += item_size
            
        if current_batch:
            yield current_batch

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
    async def _fallback_process_items(self, items: List[Dict]) -> List[Any]:
        """
        Processes items one by one with short timeout. Returns list of results (None on failure).

        Args:
            items: List of individual inputs (e.g. from a failed chunk)

        Returns:
            List of results, one per input (or None on failure)
        """
        results = []
        for item in items:
            try:
                result = await asyncio.wait_for(self.chat_chain.abatch([item]), timeout=60)
                results.append(result[0])  # abatch still returns a list
            except asyncio.TimeoutError:
                self.logger.error(
                    f"[Fallback] Timeout on single input — Patient: {item.get('patient')}, Index: {item.get('index')}"
                )
                results.append(None)
            except Exception as e:
                self.logger.error(
                    f"[Fallback] Error on single input — Patient: {item.get('patient')}, Index: {item.get('index')}, Error: {e}"
                )
                results.append(None)
        return results

    @backoff_except_timeout(max_tries=3)
    async def _process_chunk(self, chunk_inputs: List[Dict], chunk_num: int, total_chunks: int) -> List[Any]:
        """
        Process a single chunk
        
        Args:
            chunk_inputs: List of input dictionaries to process in this chunk
            chunk_num: Current chunk number (for logging)
            total_chunks: Total number of chunks (for logging)

        Returns:
            List of processed results from the chunk
        """
        self.logger.info(
            f"Processing chunk {chunk_num}/{total_chunks} (size: {len(chunk_inputs)}), "
            f"indices: {[item['index'] for item in chunk_inputs]}, "
            f"lengths: {[len(item['report']) for item in chunk_inputs]}"
        )
        start_time = time.time()
        
        try:
            results = await asyncio.wait_for(self.chat_chain.abatch(chunk_inputs), timeout=600)
            elapsed = time.time() - start_time
            self.logger.info(f"Chunk {chunk_num} completed in {elapsed:.2f}s "
                            f"({len(chunk_inputs)/elapsed:.3f} items/sec)")
            return results
        except asyncio.TimeoutError:
            self.logger.warning(f"Chunk {chunk_num} timed out after 600s. Falling back to per-item processing.")
            return await self._fallback_process_items(chunk_inputs)
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
            return [None] * len(chunk_inputs)
        
    async def _abatch_chunked(self, inputs: List[Dict], batch_size: int = 32, max_concurrent: int = 6) -> List[Any]:
        """
        Optimized async batch processing

        Args:
            inputs: List of input dictionaries to process
            batch_size: Size of each batch for processing
            max_concurrent: Maximum number of concurrent requests to process

        Returns:
            List of processed results from all chunks
        """
        if not inputs:
            return []
            
        # Create chunks using smart batching
        chunks = list(self._dynamic_chunks(inputs, batch_size))
        total_chunks = len(chunks)
        self.logger.info(f"Processing {len(inputs)} items in {total_chunks} optimized chunks")
        
        # Process chunks with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests
        
        async def process_with_semaphore(chunk, chunk_num):
            async with semaphore:
                return await self._process_chunk(chunk, chunk_num, total_chunks)
        
        tasks = [process_with_semaphore(chunk, i+1) for i, chunk in enumerate(chunks)]
        results = []
        
        for i, future in enumerate(asyncio.as_completed(tasks), 1):
            chunk_results = await future
            results.extend(chunk_results)
            self.logger.debug(f"Completed {i}/{total_chunks} chunks")
        
        return results
    
    def run_batch(self, reports: List[str], patients: List[str], batch_size: int = 32) -> List[Any]:
        """
        Run batch processing of reports with async support

        Args:
            reports: List of report strings to process
            patients: List of patient identifiers corresponding to each report
            batch_size: Size of each batch for processing
            
        Returns:
            List of processed results from all reports
        """
        start_time = time.time()
        self.logger.info(f"Starting batch processing of {len(reports)} items")
        
        try:
            if len(reports) != len(patients):
                raise ValueError("Reports and patients lists must be of equal length")
                
            inputs = [
                {"report": r, "patient": p, "index": i}
                for i, (r, p) in enumerate(zip(reports, patients))
            ]

            # Order inputs based on length of reports for speedup batching
            inputs.sort(key=lambda x: len(x["report"]))
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            outputs = loop.run_until_complete(self._abatch_chunked(inputs, batch_size))
            
            # Map results back to original index of input
            results = [None] * len(inputs)
            for inp, result in zip(inputs, outputs):
                results[inp["index"]] = result

            elapsed = time.time() - start_time
            self.logger.info(f"Batch processing completed in {elapsed:.2f} seconds "
                           f"({len(reports)/elapsed:.1f} items/sec)")
            return results
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise
        finally:
            loop.close()

    # Optional synchronous batch processing alternative
    def run_batch_sync(self, reports: List[str], patients: List[str], batch_size: int = 32) -> List[Any]:
        """
        Synchronous version for environments where async isn't possible
        Note: This is less efficient than the async version and not recommended for large datasets
        
        Args:
            reports: List of report strings to process
            patients: List of patient identifiers corresponding to each report
            batch_size: Size of each batch for processing

        Returns:
            List of processed results from all reports
        """
        inputs = [
            {"report": r, "patient": p, "index": i}
            for i, (r, p) in enumerate(zip(reports, patients))
        ]
        outputs = [
            result for chunk in self._chunks(inputs, batch_size)
            for result in self.chat_chain.batch(chunk)
        ]
        results = [None] * len(inputs)
        for inp, result in zip(inputs, outputs):
            results[inp["index"]] = result
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
        results = self.run_batch(texts, patients)
        return adapter.format_outputs(results)