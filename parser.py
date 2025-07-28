import json
import re
from typing import TypedDict, Dict, List, Any, Optional, Union, Tuple, Iterable
import logging
import asyncio
import time
import math
import uuid
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from collections import Counter
from adapters.base_adapter import BaseAdapter
from collections import OrderedDict
from sentence_transformers import util
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from pydantic import create_model, Field
import backoff
from montoring import AsyncLLMMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
            # Fallback to remove weird characters in { ... } block
            if not fallback_blocks:
                json_candidate = re.search(r'{[\s\S]*}', text)
                if json_candidate:
                    json_text = json_candidate.group()
                    # remove lines with comments or malformed entries
                    lines = json_text.splitlines()
                    clean_lines = [line for line in lines if ':' in line and '–' not in line and '"' in line]
                    # Join and wrap in braces
                    fallback_blocks = ['{' + '\n'.join(clean_lines) + '}']

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
            reasoning = text.strip()

        return {
            "raw_output": text,
            "reasoning": reasoning,
            "extracted_data": extracted_data,
        }

class FastEnsemble(Runnable):
    def __init__(self, embedding_model: str = "all-mpnet-base-v2"):
        self.logger = logging.getLogger(__name__)

        self.logger.warning(f"Currently using embedding with API endpoints are not supported. Falling back to SentenceTransformer: {embedding_model}.")
        self.use_embeddings_API = False

        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.use_embeddings_API:
            embeddings = self.llm(texts)
        else:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return np.array(embeddings)

    def _ensemble_responses(self, candidates: List[Dict]) -> Dict:
        responses = [c["extracted_data"] for c in candidates]

        start_time = time.time()
        
        final_response = {}
        for field in responses[0].keys():  # Assuming all have same fields
            values = [r.get(field) for r in responses]
            
            # Try voting first
            if all(isinstance(v, (str, int, float)) for v in values if v is not None):
                counter = Counter([v for v in values if v is not None])
                if counter:
                    most_common = counter.most_common(1)[0]
                    if most_common[1] > 1:  # Require at least 2 votes
                        final_response[field] = most_common[0]
                        continue
            
            # Fallback to embedding similarity
            valid_values = [(i, str(v)) for i, v in enumerate(values) if v is not None]
            if not valid_values:
                final_response[field] = None
                continue
                
            indices, text_values = zip(*valid_values)
            embeddings = self._get_embeddings(text_values)
            
            # Compute similarity matrix and find most central response
            sim_matrix = util.cos_sim(embeddings, embeddings)
            centrality_scores = np.sum(sim_matrix.numpy(), axis=1)
            best_idx = indices[np.argmax(centrality_scores)]
            final_response[field] = values[best_idx]

        elapsed = time.time() - start_time
        self.logger.info(f"Item ensembling completed in {elapsed:.2f}s ")
        
        return {
            "reasoning": "Ensembled using voting + embedding similarity",
            "extracted_data": final_response,
            "candidates": candidates
        }

    def invoke(self, candidates: List[Dict], config=None) -> Dict:
        return self._ensemble_responses(candidates)

    def batch(self, inputs: List[List[Dict]], config=None) -> List[Dict]:
        return [self._ensemble_responses(c) for c in inputs]

class VLLMReportParser:
    def __init__(
        self,
        params_config: dict,
        prompt_config: dict,
        base_url: str = "http://localhost:8000/v1/",
        api_key: Optional[str] = "DummyAPIKey",
        prompt_method: str = "ZeroShot",
        batch_size: Optional[int] = None,
        timeout: int = 60,
        max_concurrent: int = 32,
        select_example: Optional[str] = None,
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
            batch_size: Batch size for processing reports. If None, internal batching not used.
            timeout: Timeout for API requests
            max_concurrent: Maximum number of concurrent requests to the API
            select_example: 1-based index of the example to use (only for example-based prompt methods).
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
            max_tokens=params_config.get('max_tokens', 2048),
            frequency_penalty=params_config.get('frequency_penalty', 0.0),
            presence_penalty=params_config.get('presence_penalty', 0.0),
            #model_kwargs={
            #    'top_k': params_config.get('top_k', 50),
            #    'repetition_penalty' : params_config.get('repetition_penalty', 1.2),
            #}
        )
        self.self_consistency_sampling = params_config.get('self_consistency_sampling', {"num_samples": 3, "temperature": [0.1, 0.3, 0.5]})
        self.batch_size = batch_size
        self.timeout = timeout
        if prompt_method == "SelfConsistency":
            self.ensemble_chain = FastEnsemble(
                embedding_model = params_config.get('embedding_model', 'all-mpnet-base-v2')
            )

        self.max_concurrent = max_concurrent
        self.select_example = select_example

        # Load prompt configuration and method
        self.prompt_config = prompt_config
        self.prompt_method = prompt_method
        self.chains = self._detect_chains() if prompt_method == "PromptChain" else None

        # Additional configurations
        self.patterns = self._load_patterns(patterns_path) if patterns_path else None
        self.save_raw_output = save_raw_output
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            from langchain_core.globals import set_verbose
            set_verbose(True)
            self.logger.setLevel(logging.DEBUG)

        if prompt_method == "PromptChain" and self.batch_size:
            self.logger.warning(
                f"Internal batch size was set to {self.batch_size}, but this is incompatible with the 'PromptChain' method. "
                "Falling back to individual patient prompting instead."
            )
            self.batch_size = None

        self.logger.info("Initializing VLLMReportParser with the following configuration:")
        self.logger.info(f"Model: {params_config.get('model')}")
        self.logger.info(f"Prompt Method: {self.prompt_method}")
        if self.select_example:
            self.logger.info(f"Selecting example: {self.select_example} at index: {self.select_example-1}")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Timeout: {self.timeout} seconds")
        self.logger.info(f"Max Concurrent Requests: {self.max_concurrent}")
        if self.chains:
            self.logger.info(f"Using the following chain sequence: {self.chains}")

        # Set up the monitor for async LLM monitoring
        self.monitor = AsyncLLMMonitor(
            base_url=base_url,
            api_key=api_key,
            update_interval=1,
            max_history=3600,
            verbose=verbose
        )

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
    
    # deprecated and not currently used
    def _adjusted_batch_size(self, batch_size: int, num_samples: int) -> int:
        """
        Adjust batch size based on the number of samples in chain.
        
        Args:
            batch_size: Original batch size
            num_samples: Number of samples for Self-Consistency or PromptChain methods
        
        Returns:
            Adjusted batch size in powers of 2, ensuring it is not less than 1.
        """
        base_size = batch_size / num_samples
        
        # Calculate the largest power of 2 less than or equal to base_size
        return int(2 ** math.floor(math.log2(base_size)) if base_size >= 1 else 1)
    
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
            return None, output_format

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
            self.logger.warning(
                    "Deprecation warning: passing system instruction in config file has been deprecated and will be unsupported in future versions. "
                    "Now using it to overwrite hardcoded system instructions, which is not advised. "
            )
            system_instructions = self.prompt_config['system_instruction'].strip()
        else:
            self.logger.debug("System instruction not found in config file. Using default system instruction, which is recommended")
            if self.prompt_method in ["ZeroShot", "OneShot", "FewShot"]:
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
            examples_instructions = []
            if 'example' in self.prompt_config:
                self.logger.warning(
                    "Deprecation warning: passing a single example is deprecated and will be unsupported in future versions. "
                    "Please reformat the yaml file with (`examples: `) instead."
                )
                examples = [self.prompt_config["example"]]
            elif 'examples' not in self.prompt_config:
                raise ValueError("Examples are required for prompt methods other than ZeroShot")
            else:
                examples = self.prompt_config["examples"]

            if self.select_example:
                if self.prompt_method == "FewShot":
                    raise ValueError("Prompt method should not be FewShot if select example was provided")
                
                if self.select_example > len(examples):
                    self.logger.warning(
                        f"Select example: {self.select_example} was out of range for number of examples in yaml file: {len(examples)}. "
                        "Switching to taking first example. "
                    )
                    self.select_example = 1
                else:
                    self.logger.info(
                        f"Selecting example {self.select_example} out of {len(examples)} examples, since OneShot was provided for prompting method"
                    )
                examples = [examples[self.select_example-1]]
            else:
                if self.prompt_method == "OneShot":
                    self.logger.warning(
                        f"Prompt method is OneShot, however no value was provided for select example. "
                        "Switching to taking first example. "
                    )
                    examples = [examples[0]]

            for idx, example in enumerate(examples):
                self._validate_example(example)
                example_instructions = {}
                example_instructions[f"user"] = "\n".join([example['input'].strip()])

                if self.prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
                    if 'reasoning' not in example:
                        raise ValueError("Examples for CoT, SelfConsistency, and PromptChain must include 'reasoning'")
                    
                    reasoning_instructions = example['reasoning'].strip()
                    if self.prompt_method == "PromptChain":
                        reasoning_instructions = reasoning_instructions.split("\n")
                        example_instructions[f"assistant_reasoning"] = "\n".join([reasoning_instructions[i] for i in chain_indexes if i < len(reasoning_instructions)])

                example_outcome = example['output'].strip()
                if self.prompt_method == "PromptChain":
                    example_outcome, example_output_variable = self._format_promptchain_instructions(example_outcome, chain_indexes, f"example_output_variable_{idx}")
                else:
                    example_outcome, example_output_variable = self._format_json_instructions(example_outcome, f"example_output_variable_{idx}")

                example_instructions[f"assistant_output"] = "\n".join([example_outcome])
                # Store example output variable in prompt variables for downstream use
                prompt_variables[f"example_output_variable_{idx}"] = example_output_variable

                examples_instructions.append(example_instructions)
        else:
            examples_instructions = None

        # Add the actual report content
        report_instructions = "\n".join([
            "[file name]: {patient}",
            "{report}",
        ])

        if self.prompt_method in ["ZeroShot", "OneShot", "FewShot"]:
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
        return system_instructions, field_instructions, task_instructions, examples_instructions, report_instructions, final_instructions, prompt_variables, output_format

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
            parsers = []
            for chain in self.chains:
                system_instructions, field_instructions, task_instructions, examples_instructions, report_instructions, final_instructions, prompt_variables, output_format = self._generate_prompt(chain=chain)
                prompt = [
                    SystemMessage(system_instructions, name="system_instructions"),
                    HumanMessage(field_instructions, name="field_instructions"),
                    HumanMessage(task_instructions, name="task_instructions"),
                ]

                if examples_instructions:
                    has_reasoning = any("assistant_reasoning" in ex for ex in examples_instructions)
                    prompt.append(
                        HumanMessage(f"Below are {len(examples_instructions)} examples of expected input{', reasoning,' if has_reasoning else ''} and output. followed by a new task.", name="example_intro")
                    )
                    for idx, example_instructions in enumerate(examples_instructions):
                        prompt.extend([
                            HumanMessage(example_instructions["user"], name=f"example_user_{idx}"),
                        ])
                        if "assistant_reasoning" in example_instructions:
                            prompt.extend([
                                AIMessage(example_instructions["assistant_reasoning"], name=f"example_assistant_reasoning_{idx}"),
                            ])
                        prompt.extend([
                            AIMessagePromptTemplate.from_template(example_instructions["assistant_output"], name=f"example_assistant_output_{idx}")
                        ])

                prompt.extend([
                    HumanMessagePromptTemplate.from_template(report_instructions, name="report_instructions"),
                    HumanMessage(final_instructions, name="final_instructions")
                ])
                prompt_template = ChatPromptTemplate(prompt)
                prompt_template = prompt_template.partial(**prompt_variables)

                prompt_templates.append(prompt_template)
                parsers.append(ReasoningAndDynamicJSONParser(output_format))
            
            self.chat_chain = self._create_prompt_chain_graph(prompt_templates, parsers)
        else:
            system_instructions, field_instructions, task_instructions, examples_instructions, report_instructions, final_instructions, prompt_variables, output_format = self._generate_prompt(chain=None)
            prompt = [
                SystemMessage(system_instructions),
                HumanMessage(field_instructions, name="field_instructions"),
                HumanMessagePromptTemplate.from_template(task_instructions, name="task_instructions"),
            ]

            if examples_instructions:
                has_reasoning = any("assistant_reasoning" in ex for ex in examples_instructions)
                prompt.append(
                    HumanMessage(f"Below are {len(examples_instructions)} examples of expected input{', reasoning,' if has_reasoning else ''} and output. followed by a new task.", name="example_intro")
                )
                for idx, example_instructions in enumerate(examples_instructions):
                    prompt.extend([
                        HumanMessage(example_instructions["user"], name=f"example_user_{idx}"),
                    ])
                    if "assistant_reasoning" in example_instructions:
                        prompt.extend([
                            AIMessage(example_instructions["assistant_reasoning"], name=f"example_assistant_reasoning_{idx}"),
                        ])
                    prompt.extend([
                        AIMessagePromptTemplate.from_template(example_instructions["assistant_output"], name=f"example_assistant_output_{idx}")
                    ])

            prompt.extend([
                HumanMessagePromptTemplate.from_template(report_instructions, name="report_instructions"),
                HumanMessage(final_instructions, name="final_instructions")
            ])
            prompt_template = ChatPromptTemplate(prompt)
            prompt_template = prompt_template.partial(**prompt_variables)

            parser = ReasoningAndDynamicJSONParser(output_format)
            base_chain = prompt_template | self.llm | parser
            if self.prompt_method == "SelfConsistency":
                # Initialize our fast ensemble strategy
                self.chat_chain = self._create_self_consistency_chain(base_chain)
            else:
                # For other methods, we can use the base chain directly
                self.chat_chain = base_chain

    def _create_prompt_chain_graph(self, prompt_templates: List[ChatPromptTemplate], parsers: List[ReasoningAndDynamicJSONParser]) -> Runnable:
        """
        Create a LangGraph with memory from the prompt templates
        for prompt chaining
        
        Args:
            prompt_templates: List of ChatPromptTemplate objects
            
        Returns:
            Runnable graph with memory
        """
        # Import langgraph here so optional dependency only required when using graphs
        from langgraph.graph import MessagesState, StateGraph, START, END
        from langgraph.checkpoint.memory import InMemorySaver

        class PromptChainState(TypedDict):
            report: str
            patient: str
            index: int
            result: Dict[str, Any]
            messages: List[BaseMessage]
            status: str

        # Initialize the graph and memory
        workflow = StateGraph(state_schema=PromptChainState)
        memory = InMemorySaver()

        for idx, (prompt_template, parser) in enumerate(zip(prompt_templates, parsers)):
            node_name = f"chain_{idx}"
            def create_node(prompt_template: ChatPromptTemplate, parser: ReasoningAndDynamicJSONParser, chain_idx: int):
                @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
                async def node(state: PromptChainState) -> Dict[str, Any]:
                    try:
                        chain = prompt_template | self.llm | parser
                        result = await chain.ainvoke({"report": state["report"], "patient": state["patient"]}, timeout=self.timeout)

                        if result.get('extracted_data', None) == None:
                            raise ValueError(f"Failed to parse JSON block or no valid JSON found for Chain {chain_idx}.")

                        return {
                            "patient": state.get("patient"),
                            "index": state.get("index"),
                            "result": {
                                "reasoning": state.get("result", {}).get("reasoning", []) + [result["reasoning"]],
                                "extracted_data": {**state.get("result", {}).get("extracted_data", {}), **result["extracted_data"]}
                            },
                            "messages": state.get("messages", []) + [AIMessage(content=result.get("raw_output"))],
                            "status": "succes"
                        }
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout for Patient: {state.get('patient')}, Index: {state.get('index')}, Chain Index: {chain_idx}")
                        raise
                    except Exception as e:
                        self.logger.error(f"Failed — Patient: {state.get('patient')}, Index: {state.get('index')}, Chain Index: {chain_idx}, Error: {e}")
                        raise

                return node

            workflow.add_node(node_name, create_node(prompt_template=prompt_template, parser=parser, chain_idx=idx))
            
            if idx == 0:
                workflow.add_edge(START, node_name)
            elif idx > 0:
                workflow.add_edge(f"chain_{idx-1}", node_name)

        workflow.add_edge(f"chain_{len(prompt_templates)-1}", END)

        graph = workflow.compile(checkpointer=memory)

        if self.verbose:
            from IPython.display import Image
            with open("graph_output.png", "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())

        return graph

    def _create_self_consistency_chain(self, base_chain: Runnable) -> Runnable:
        num_samples = self.self_consistency_sampling.get("num_samples", 3)
        temperatures = self.self_consistency_sampling.get("temperature", [0.1, 0.3, 0.5])
        
        return (
            # Generate all candidates in parallel with different temperatures
            RunnableParallel(**{
                f"candidate_{i}": base_chain.with_config(
                    run_name=f"candidate_{i}",
                    tags=[f"temp={temperatures[i]}"],
                    config={"temperature": temperatures[i]}
                )
                for i in range(num_samples)
            })
            # Prepare ensemble input while preserving raw candidates 
            | RunnableLambda(lambda x: [
                {
                    "temperature": temperatures[i],
                    "reasoning": x[f"candidate_{i}"].get("reasoning", ""),
                    "extracted_data": x[f"candidate_{i}"].get("extracted_data", {})
                }
                for i in range(num_samples)
            ])
            # Get ensemble result
            | self.ensemble_chain
        )

    def _dynamic_chunks(self, inputs: List[Dict], max_tokens: int = 16000) -> Iterable[List[Dict]]:
        """
        Dynamic batching that considers both item count and approximate token size

        Args:
            inputs: List of input dictionaries to process
            max_tokens: Maximum token count for the batch (approximate)
            
        Yields:
            Iterable of input lists, each representing a batch of inputs
        """
        current_batch = []
        current_batch_size = 0
        
        for item in inputs:
            # Estimate token count (adjust based on your average token/report ratio)
            approx_tokens = len(item['report']) // 4  # Rough estimate: 1 token ≈ 4 chars
            item_size = max(1, math.ceil(approx_tokens / 1000))  # Convert to "size units" (1 = ~1000 tokens)
            
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
                (len(current_batch) >= self.batch_size or 
                 current_batch_size + item_size > max_tokens // 1000)):
                yield current_batch
                current_batch = []
                current_batch_size = 0
                
            current_batch.append(item)
            current_batch_size += item_size
            
        if current_batch:
            yield current_batch

    async def _process_items(self, items: List[Dict]) -> List[Dict[str, Any]]:
        """
        Processes items one by one. Returns list of results (None on failure).

        Args:
            items: List of individual inputs (e.g. from a failed chunk)

        Returns:
            List of results, one per input (or None on failure)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
        async def process(item):
            async with semaphore:
                try:
                    self.logger.debug(f"Now processing patient: {item['patient']}")
                    if self.prompt_method == "PromptChain":
                        thread_id = uuid.uuid4()
                        config = {"configurable": {"thread_id": thread_id}}
                        result = await asyncio.wait_for(self.chat_chain.ainvoke(item, config={"configurable": {"thread_id": thread_id}}), timeout=self.timeout)
                        result = result["result"]
                    else:
                        result = await asyncio.wait_for(self.chat_chain.ainvoke(item), timeout=self.timeout)

                    if result.get('extracted_data', None) == None:
                        raise ValueError("Failed to parse JSON block or no valid JSON found. Returning all data as reasoning.")

                    return {
                        "patient": item["patient"],
                        "index": item["index"],
                        "result": result,
                        "status": "success"
                    }
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout for Patient: {item.get('patient')}, Index: {item.get('index')}")
                    raise
                except Exception as e:
                    self.logger.error(f"Failed — Patient: {item.get('patient')}, Index: {item.get('index')}, Error: {e}")
                    raise

        # Wrapper to catch final failure after retries:
        async def safe_process(item):
            try:
                return await process(item)
            except asyncio.TimeoutError:
                # After max retries, return timeout status instead of propagating
                return {
                    "patient": item["patient"],
                    "index": item["index"],
                    "result": {
                        "reasoning": None,
                        "extracted_data": None
                    },
                    "status": "timeout"
                }
            except Exception:
                # After max retries, return error status instead of propagating
                return {
                    "patient": item["patient"],
                    "index": item["index"],
                    "result": {
                        "reasoning": None,
                        "extracted_data": None
                    },
                    "status": "error"
                }

        return await tqdm_asyncio.gather(*[safe_process(item) for item in items])

    async def _process_chunk(self, chunk_inputs: List[Dict], chunk_num: int, total_chunks: int) -> List[Dict[str, Any]]:
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

        batch_timeout = int(self.timeout * self.batch_size / 2)
        
        @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
        async def process_chunk_inner():
            try:
                results = await asyncio.wait_for(self.chat_chain.abatch(chunk_inputs), timeout=batch_timeout)
                elapsed = time.time() - start_time
                self.logger.info(f"Chunk {chunk_num} completed in {elapsed:.2f}s "
                                f"({len(chunk_inputs)/elapsed:.3f} items/sec)")
                # Here if valid JSON is passed by status succes, and if failed downstream individually analyzed
                return [{
                    "patient": input_item["patient"],
                    "index": input_item["index"],
                    "result": llm_result,
                    "status": "success" if llm_result.get('extracted_data') is not None else "failed",
                    "report": input_item["report"]
                } for input_item, llm_result in zip(chunk_inputs, results)]
            except asyncio.TimeoutError:
                self.logger.warning(f"Chunk {chunk_num} timed out after {batch_timeout}. Falling back to per-item processing.")
                raise 
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
                raise  # propagate to retry
        
        async def safe_process_chunk():
            try:
                return await process_chunk_inner()
            except asyncio.TimeoutError:
                # Final fallback on timeout after retries
                return [{
                    "patient": item["patient"],
                    "index": item["index"],
                    "result": {
                        "reasoning": None,
                        "extracted_data": None
                    },
                    "status": "chunk_timeout",
                    "report": item["report"]
                } for item in chunk_inputs]
            except Exception as e:
                # Final fallback on other errors after retries
                return [{
                    "patient": item["patient"],
                    "index": item["index"],
                    "result": {
                        "reasoning": None,
                        "extracted_data": None
                    },
                    "status": "chunk_error",
                    "report": item["report"]
                } for item in chunk_inputs]

        return await safe_process_chunk()
        
    async def _abatch_chunked(self, inputs: List[Dict]) -> List[Dict[str, Any]]:
        """
        Optimized async batch processing

        Args:
            inputs: List of input dictionaries to process

        Returns:
            List of processed results from all chunks
        """
        if not inputs:
            return []
            
        # Create chunks using smart batching
        chunks = list(self._dynamic_chunks(inputs))
        total_chunks = len(chunks)
        self.logger.info(f"Processing {len(inputs)} items in {total_chunks} optimized chunks")
        
        # Process chunks with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)  # Limit concurrent requests
        
        async def process_with_semaphore(chunk, chunk_num):
            async with semaphore:
                return await self._process_chunk(chunk, chunk_num, total_chunks)
        
        tasks = [process_with_semaphore(chunk, i+1) for i, chunk in enumerate(chunks)]
        results = []
        fallback_items = []
        
        for i, future in enumerate(asyncio.as_completed(tasks), 1):
            chunk_results = await future

            # Check if any items need fallback processing
            needs_fallback = [r for r in chunk_results if r["status"] in ["chunk_timeout", "chunk_error", "failed"]]

            if needs_fallback:
                fallback_items.extend([{
                    "report": r["report"],
                    "patient": r["patient"],
                    "index": r["index"]
                } for r in needs_fallback])

            # Add successful results
            results.extend([r for r in chunk_results if r["status"] == "success"])
            
            self.logger.debug(f"Completed {i}/{total_chunks} chunks")
            
        # Process fallback items if any
        if fallback_items:
            self.logger.info(f"Processing {len(fallback_items)} items via fallback serially.")
            fallback_results = await self._process_items(fallback_items)
            results.extend(fallback_results)

        return results
    
    def run_batch(self, reports: List[str], patients: List[str]) -> List[Any]:
        """
        Run batch processing of reports with async support

        Args:
            reports: List of report strings to process
            patients: List of patient identifiers corresponding to each report
            
        Returns:
            List of processed results from all reports
        """
        start_time = time.time()
        self.logger.info(f"Starting batch processing of {len(reports)} items")
        
        self.monitor.start_monitoring()
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
            outputs = loop.run_until_complete(self._abatch_chunked(inputs))
            
            # Map results back to original index of input
            outputs.sort(key=lambda x: x["index"])
            results = [r["result"] for r in outputs]

            elapsed = time.time() - start_time
            self.logger.info(f"Batch processing completed in {elapsed:.2f} seconds "
                           f"({len(reports)/elapsed:.1f} items/sec)")
            return results
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise
        finally:
            self.monitor.stop_monitoring()
            loop.close()

    # Optional synchronous batch processing alternative
    def run_batch_sync(self, reports: List[str], patients: List[str]) -> List[Any]:
        """
        Synchronous version for running batches in environments where async isn't possible
        Note: This is less efficient than the async version and not recommended for large datasets
        
        Args:
            reports: List of report strings to process
            patients: List of patient identifiers corresponding to each report

        Returns:
            List of processed results from all reports
        """
        inputs = [
            {"report": r, "patient": p, "index": i}
            for i, (r, p) in enumerate(zip(reports, patients))
        ]
        outputs = [
            result for chunk in self._dynamic_chunks(inputs)
            for result in self.chat_chain.batch(chunk)
        ]
        return [r["result"] for r in outputs]

    # Optional individual item processing alternative
    async def run_patient_async(self, reports: List[str], patients: List[str]) -> List[Any]:
        """
        Asynchronous version for running individual samples environments
        Note: This is less efficient than the batch version, but sometimes recommended
        with computation constrains.
        
        Args:
            reports: List of report strings to process
            patients: List of patient identifiers corresponding to each report

        Returns:
            List of processed results from all reports
        """
        inputs = [
            {"report": r, "patient": p, "index": i}
            for i, (r, p) in enumerate(zip(reports, patients))
        ]

        results = await self._process_items(inputs)

        # Ensure results are ordered by original input order
        results_sorted = sorted(results, key=lambda x: x["index"])

        return [r["result"] for r in results_sorted]

    # Optional synchronous individual item processing alternative
    def run_patient_sync(self, reports: List[str], patients: List[str]) -> List[Any]:
        """
        Synchronous version for running individual samples environments where async isn't possible
        Note: This is less efficient than the async version and not recommended for large datasets
        
        Args:
            reports: List of report strings to process
            patients: List of patient identifiers corresponding to each report

        Returns:
            List of processed results from all reports
        """
        inputs = [
            {"report": r, "patient": p, "index": i}
            for i, (r, p) in enumerate(zip(reports, patients))
        ]
        self.monitor.start_monitoring()
        try:
            outputs = [
                self.chat_chain.invoke(patient) for patient in inputs
            ]
        finally:
            self.monitor.stop_monitoring()
            
        return [r["result"] for r in outputs]

    def process_with_adapter(self, adapter: BaseAdapter) -> Any:
        """
        Process reports using the specified adapter
        
        Args:
            adapter: Configured adapter instance
            
        Returns:
            Processed results in adapter's output format
        """
        texts, patients = adapter.prepare_inputs()
        start_time = time.time()

        if not self.batch_size:
            results = asyncio.run(self.run_patient_async(texts, patients))
        else:
            results = self.run_batch(texts, patients)

        elapsed = time.time() - start_time
        self.logger.info(f"Complete workflow took {elapsed:.2f} seconds")

        results = adapter.format_outputs(results)
        return results