import json
import re
from typing import TypedDict, Dict, List, Any, Optional, Union, Tuple, Iterable
import logging
import asyncio
import time
import math
import uuid
import random
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from collections import Counter, OrderedDict, deque, defaultdict
from adapters.base_adapter import BaseAdapter
from sentence_transformers import util
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate, BasePromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel, create_model, Field, validator
from functools import wraps
from enum import Enum
import backoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SpecialValues(str, Enum):
    MISSING = "missing"
    DEFAULT = "default"
    NOT_APPLICABLE = "n/a"
    UNKNOWN = "unknown"

class ReasoningAndDynamicJSONParser(BaseOutputParser):
    def __init__(self, output_format: Dict[str, Any], model: SentenceTransformer, dry_run: bool = False, special_values: List[str] = None):
        # Build pydantic fields dynamically from output_format
        super().__init__()
        self._model = model
        self._logger = logging.getLogger(__name__)
        self._dry_run = dry_run
        self._special_values = special_values or [v.value for v in SpecialValues]
                
        # Normalize the output_format by cleaning type names
        self._output_format = {
            field: {
                **config,
                "type": config["type"].replace("_or_missing", ""),
                "original_type": config["type"]  
            }
            for field, config in output_format.items()
        }

        # Precompute embeddings for options
        self._precompute_option_embeddings()
        
        # Create the dynamic model
        self._output_model = self._create_dynamic_model()

    def _precompute_option_embeddings(self):
        """Precompute embeddings for categorical options only."""
        self._field_embeddings = {}
        for field, config in self._output_format.items():
            if "options" in config and config["type"] not in ["int", "float", "number", "boolean", "binary"]:
                options = config["options"]
                if "default" in config and config["default"] not in options:
                    options = [config["default"]] + options
                
                self._field_embeddings[field] = {
                    "options": [str(opt) for opt in options],
                    "original_options": options,
                    "embeddings": self._model.encode([str(opt) for opt in options], convert_to_tensor=True)
                }

    def _create_dynamic_model(self):
        """Create a dynamic Pydantic model with built-in validation and transformation."""
        fields = {}
        validators = {}

        for field_name, config in self._output_format.items():
            # Determine field type
            field_type = self._get_field_type(config["type"])
            fields[field_name] = self._create_field_definition(field_type, config)
            
            # Add validator if field has options
            if field_name in self._field_embeddings:
                validators[f"validate_{field_name}"] = validator(field_name, allow_reuse=True)(
                    self._create_categorical_validator(field_name, config)
                )
            elif config["type"] in ["int", "float", "number"] and "options" in config:
                validators[f"validate_{field_name}"] = validator(field_name, allow_reuse=True)(
                    self._create_numeric_options_validator(field_name, config)
                )
            elif config["type"] in ["int", "float", "number", "boolean", "binary"]:
                validators[f"validate_{field_name}"] = validator(field_name, allow_reuse=True)(
                    self._create_special_value_validator(field_name, config)
                )

        return create_model(
            "DynamicOutputModel",
            **fields,
            __validators__=validators
        )

    def _get_field_type(self, type_str: str) -> type:
        """Map type string to Python type."""
        # There is always a string backup value, such as 'missing'
        type_map = {
            "string": str,
            "list": List[str],
            "int": Union[int, str],
            "float": Union[float, str],
            "number": Union[float, str],
            "boolean": Union[bool, int, str],
            "binary": Union[bool, int, str],
            "categorical": str
        }
        return type_map.get(type_str, Any)

    def _create_field_definition(self, field_type: type, config: Dict[str, Any]) -> tuple:
        """Create a field definition with metadata."""
        return (
            Optional[field_type],
            Field(
                default=config.get("default"),
                description=self._create_field_description(config)
            )
        )

    def _create_field_description(self, config: Dict[str, Any]) -> str:
        """Generate field description including options."""
        desc = config.get("description", "")
        if "options" in config:
            desc += f" Options: {', '.join(str(o) for o in config['options'])}"
        if config["type"] in ["int", "float", "number", "boolean", "binary"]:
            desc += f" (Special values map to default: {', '.join(self._special_values)})"
        return desc

    def _create_categorical_validator(self, field_name: str, config: Dict[str, Any]):
        """Validator for categorical fields with semantic matching."""
        def validate_categorical(cls, v):
            if v is None:
                return config.get("default")
                
            field_data = self._field_embeddings[field_name]
            v_str = str(v).strip().lower()
            
            # Check for exact match
            for opt, original in zip(field_data["options"], field_data["original_options"]):
                if v_str == opt.lower():
                    return original
            
            # Semantic matching
            v_embedding = self._model.encode(v_str, convert_to_tensor=True)
            similarities = util.cos_sim(v_embedding, field_data["embeddings"])[0].cpu()
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            
            return field_data["original_options"][best_idx] if best_score >= 0.5 else config.get("default")
            
        return validate_categorical

    def _create_numeric_options_validator(self, field_name: str, config: Dict[str, Any]):
        """Validator for numeric fields with discrete options."""
        def validate_numeric_options(cls, v):
            if v is None:
                return config.get("default")
                
            # Handle special values
            if isinstance(v, str) and v.lower() in self._special_values:
                return config.get("default")
                
            # Handle range values (e.g., "10-20")
            if isinstance(v, str) and re.match(r"^\d+\s*-\s*\d+$", v):
                try:
                    low, high = map(float, re.split(r"\s*-\s*", v))
                    midpoint = (low + high) / 2
                    return self._find_closest_numeric_option(config["options"], midpoint)
                except (ValueError, TypeError):
                    pass
                    
            # Convert to numeric value
            try:
                num_val = float(v) if config["type"] in ["float", "number"] else int(v)
                return self._find_closest_numeric_option(config["options"], num_val)
            except (ValueError, TypeError):
                return config.get("default")
                
        return validate_numeric_options

    def _find_closest_numeric_option(self, options: List[Any], value: float) -> Any:
        """Find the closest numeric option to the given value."""
        numeric_options = []
        for opt in options:
            try:
                numeric_options.append(float(opt))
            except (ValueError, TypeError):
                continue
                
        if not numeric_options:
            return options[0] if options else None
            
        closest = min(numeric_options, key=lambda x: abs(x - value))
        return closest if isinstance(closest, type(value)) else type(value)(closest)

    def _create_special_value_validator(self, field_name: str, config: Dict[str, Any]):
        """Validator for handling special values in numeric/boolean fields."""
        def validate_special_values(cls, v):
            if v is None:
                return config.get("default")
                
            # Handle special values - they always map to default
            if isinstance(v, str) and v.lower() in self._special_values:
                return config.get("default")
                
            # Convert string representations
            if isinstance(v, str):
                try:
                    if config["type"] in ["int", "float", "number"]:
                        return float(v) if config["type"] in ["float", "number"] else int(v)
                    elif config["type"] in ["boolean", "binary"]:
                        if v.lower() in ["true", "yes", "1"]:
                            return True
                        if v.lower() in ["false", "no", "0"]:
                            return False
                except (ValueError, TypeError):
                    return config.get("default")
                    
            return v
            
        return validate_special_values
    
    def parse(self, text: str) -> Dict[str, Any]:
        if self._dry_run:
            return {
                "raw_output": text,
                "reasoning": text,
                "extracted_data": self._output_model().dict()
            }
        
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
            extracted_data = {}
            print(reasoning)

        return {
            "raw_output": text,
            "reasoning": reasoning,
            "extracted_data": extracted_data,
        }

class DummyLLM(Runnable):
    """A dummy LLM that returns predictable outputs for dry run testing."""
    def __init__(self, response: str = "[DRY RUN] This is a mock response", max_content_chars=100):
        self.response = response
        self.max_content_chars = max_content_chars
        self.max_content_chars_one_side = int(max_content_chars / 2)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _return(self, message: ChatPromptValue) -> AIMessage:
        message = "\n".join(
            f"[{m.type.upper()}] - [{m.name.replace('_', ' ').upper()}]\n"
            f"{(m.content[:self.max_content_chars_one_side] + ' ... ' + m.content[-self.max_content_chars_one_side:]) if len(m.content) > self.max_content_chars else m.content}"
            for m in message.to_messages()
        )
        self.logger.info(
            "\n=== [Dry run] Prompt Messages ===" + "\n" + message + "\n" + "================================="
        )
        return AIMessage(content=self.response)

    async def ainvoke(self, message: ChatPromptValue, config: Dict[str, Any] = None) -> AIMessage:
        return self._return(message)

    def invoke(self, message: ChatPromptValue, config: Dict[str, Any] = None) -> AIMessage:
        return self._return(message)

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
        sentence_model: str = "all-mpnet-base-v2",
        save_raw_output: bool = False,
        verbose: bool = False,
        dry_run: bool = False
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
            dry_run: Enable dry run for sanity checking without llm
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
        self.merge_consecutive_messages = params_config.get('merge_consecutive_messages', False)
        self.system_instructions_allowed = params_config.get('system_instructions_allowed', True)

        # Load prompt configuration and method
        self.prompt_config = prompt_config
        self.prompt_method = prompt_method
        self.graph = self._detect_graph() if prompt_method == "PromptGraph" else None

        # Additional configurations
        self.patterns = self._load_patterns(patterns_path) if patterns_path else None
        self.save_raw_output = save_raw_output
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            from langchain_core.globals import set_verbose
            set_verbose(True)
            self.logger.setLevel(logging.DEBUG)

        if prompt_method == "PromptGraph" and self.batch_size:
            self.logger.warning(
                f"Internal batch size was set to {self.batch_size}, but this is incompatible with the 'PromptGraph' method. "
                "Falling back to individual patient prompting instead."
            )
            self.batch_size = None

        # Load sentence_model embedding model
        self.sentence_model = SentenceTransformer(sentence_model)

        self.dry_run = dry_run
        if self.dry_run:
            self.logger.info("[Dry Run] mode enabled - No actual API calls will be made")
            self.logger.setLevel(logging.INFO) if not self.verbose else self.logger.setLevel(logging.DEBUG)

        self.logger.info("Initializing VLLMReportParser with the following configuration:")
        self.logger.info(f"Model: {params_config.get('model')}")
        self.logger.info(f"Prompt Method: {self.prompt_method}")
        if self.select_example:
            self.logger.info(f"Selecting example: {self.select_example} at index: {self.select_example-1}")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Timeout: {self.timeout} seconds")
        self.logger.info(f"Max Concurrent Requests: {self.max_concurrent}")
        if self.graph:
            self.logger.info(f"Using the following graph sequence:")
            self.logger.info(self.print_graph(self.graph))

        if self.dry_run:
            self.llm = DummyLLM(max_content_chars=10**9) if self.verbose else DummyLLM()
            self.max_concurrent, self.batch_size = 1, None # Setting this for easier debugging 

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
            num_samples: Number of samples for Self-Consistency or PromptGraph methods
        
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

    def _detect_graph(self) -> Optional[List[int]]:
        """
        Detect if the prompt configuration contains a graph structure for PromptGraph method.
        Builds a Directed Acyclic Graph (DAG) representation of the prompt chains and dependencies.

        Returns:
            Dict[int, Dict] where each key is a chain_order and values contains:
                - fields: list of field names to ask at this step
                - edges: list of transitions to next steps (both unconditional and conditional)
                - dependencies: list of fields this step depends on
        """
        graph = defaultdict(lambda: {"fields": [], "edges": [], "dependencies": set()})
        field_lookup = {f["name"]: f for f in self.prompt_config["field_instructions"]}
        
        # First pass: collect all fields and their chain orders
        all_steps = set()
        for field in self.prompt_config["field_instructions"]:
            name = field["name"]
            chain_order = field.get("chain_order")
            if chain_order is not None:
                graph[chain_order]["fields"].append(name)
                all_steps.add(chain_order)

        # Validation of example if present (unchanged from original)
        if 'examples' in self.prompt_config:
            examples = self.prompt_config['examples']
            for idx, example in enumerate(examples):
                if reasoning := example.get('reasoning'):
                    reasoning = reasoning.strip().split("\n")
                    if len(field_lookup) != len(reasoning):
                        raise ValueError(f"Chain order does not match the number of reasoning steps defined in example {idx}. This will result in prompt generation errors.")
                if output := example.get('output'):
                    output = self._parse_json(output)
                    if len(field_lookup) != len(output):
                        raise ValueError(f"Chain order does not match the number of output fields defined in the example {idx}. This will result in prompt generation errors.")

        # Second pass: build dependencies and edges
        for field in self.prompt_config["field_instructions"]:
            name = field["name"]
            chain_order = field.get("chain_order")
            if chain_order is None:
                continue

            if dep := field.get("depends_on"):
                dep_field = dep.get("field")
                dep_values = dep.get("values")

                if not dep_field or dep_values is None:
                    raise ValueError(f"Invalid depends_on for '{name}'")

                if not isinstance(dep_values, list):
                    dep_values = [dep_values]

                if dep_field not in field_lookup:
                    raise ValueError(f"'{name}' depends on unknown field '{dep_field}'")
                
                target_field = field_lookup[dep_field]
                target_options = target_field.get("options")
                if target_options:
                    for v in dep_values:
                        if v not in target_options:
                            raise ValueError(f"Invalid depends_on values '{v}' for field '{dep_field}'")
                else:
                    self.logger.warning(f"Dependency for '{dep_field}' is '{target_field}', but this field is not a categorical variable.")
            
                target_order = target_field.get("chain_order")
                if target_order is None:
                    raise ValueError(f"Field '{dep_field}' has no chain_order")
                
                if chain_order <= target_order:
                    raise ValueError(
                        f"Field '{name}' (chain_order={chain_order}) must come after "
                        f"its dependency '{dep_field}' (chain_order={target_order})"
                    )
                
                # Record the dependency
                graph[chain_order]["dependencies"].add(dep_field)
                
                # Add unique edges only
                existing_conditions = {
                    (e["condition"]["field"], e["condition"]["value"])
                    for e in graph[target_order]["edges"]
                    if e["condition"] is not None
                }

                for val in dep_values:
                    if (dep_field, val) not in existing_conditions:
                        graph[target_order]["edges"].append({
                            "condition": {"field": dep_field, "value": val},
                            "next_step": chain_order
                        })
                        existing_conditions.add((dep_field, val))
            else:
                # For fields with no dependencies, we'll add default edges later
                pass

        # Third pass: add unconditional edges only where needed, and else conditional edges
        sorted_orders = sorted(graph.keys())
        for i, node in enumerate(sorted_orders):
            next_candidates = [n for n in sorted_orders[i+1:] if n > node and not graph[n]["dependencies"]]

            if next_candidates:
                if not graph[node]["edges"]:
                    next_step = min(next_candidates)
                    graph[node]["edges"].append({
                        "condition": None,  # Unconditional
                        "next_step": next_step
                    })
                else:
                    next_step = min(next_candidates)
                    unique_fields = list({edge.get('condition', {}).get("field") for edge in graph[node]['edges'] if edge.get('condition', {}).get("field") is not None})
                    if unique_fields:
                        for unique_field in unique_fields:
                            graph[node]["edges"].append({
                                "condition": {"field": unique_field, "value": "else"}, 
                                "next_step": next_step
                            })

        # Validate the DAG has no cycles
        if not self._is_dag(graph):
            raise ValueError("The dependency graph contains cycles, which is not allowed")
        
        return dict(graph) if graph else None

    def _is_dag(self, dag: Dict[int, Dict]) -> bool:
        """Helper function to check if the graph is acyclic."""
        in_degree = {node: 0 for node in dag}
        
        # Calculate in-degree for each node
        for node in dag:
            for edge in dag[node]["edges"]:
                neighbor = edge["next_step"]
                in_degree[neighbor] += 1
        
        # Kahn's algorithm for topological sorting
        queue = deque([node for node in dag if in_degree[node] == 0])
        count = 0
        
        while queue:
            node = queue.popleft()
            count += 1
            
            for edge in dag[node]["edges"]:
                neighbor = edge["next_step"]
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return count == len(dag)

    def print_graph(self, graph: Dict[int, Dict[str, Any]]):
        """
        Recursively print a graph in a readable, indented format.
        
        Args:
            graph
        """
        if not graph:
            return "No graph structure detected"
        
        sorted_nodes = sorted(graph.keys())
        lines = []
        lines.append("\nDAG Structure:")
        lines.append("=" * 40)
        
        for node in sorted_nodes:
            node_info = graph[node]
            lines.append(f"Step {node}:")
            lines.append(f"  Fields: {', '.join(node_info['fields'])}")
            
            if node_info['dependencies']:
                lines.append(f"  Depends on: {', '.join(node_info['dependencies'])}")
            
            if node_info['edges']:
                lines.append("  Edges:")
                for edge in node_info['edges']:
                    if edge['condition']:
                        if edge['condition']['value'] == "else":
                            cond = f"if {edge['condition']['field']} is anything else"
                        else:
                            cond = f"if {edge['condition']['field']} == {edge['condition']['value']}"
                    else:
                        cond = "always"
                    lines.append(f"    -> Step {edge['next_step']} ({cond})")
            else:
                lines.append("  No outgoing edges (end node)")
            
            lines.append("-" * 40)
        
        # Add topological sort information
        try:
            topo_order = self._topological_sort(graph)
            lines.append("\nTopological Order: " + " → ".join(map(str, topo_order)))
        except ValueError as e:
            lines.append(f"\nWarning: {str(e)}")
        
        return "\n".join(lines)

    def _topological_sort(self, graph: Dict[int, Dict]) -> List[int]:
        """Helper to get topological order of nodes (useful for execution order)."""
        in_degree = {node: 0 for node in graph}
        
        # Calculate in-degree for each node
        for node in graph:
            for edge in graph[node]['edges']:
                in_degree[edge['next_step']] += 1
        
        # Kahn's algorithm
        queue = deque([node for node in graph if in_degree[node] == 0])
        topo_order = []
        
        while queue:
            if len(queue) > 1:
                # This indicates multiple possible valid orders
                pass
            node = queue.popleft()
            topo_order.append(node)
            
            for edge in graph[node]['edges']:
                neighbor = edge['next_step']
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(topo_order) != len(graph):
            raise ValueError("Graph contains cycles - no valid topological order")
        
        return topo_order

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

        # Add depends on if present and chain is not defined
        if chain is not None and 'depends_on' in field and self.prompt_method != "PromptGraph":
            depends_on = field["depends_on"]
            if "field" not in depends_on:
                self.logger.warning()
                pass # Skip depends on

            field_depends_on = depends_on["field"]
            if "values" in depends_on:
                values_depends_on = depends_on["field"]
                if isinstance(values_depends_on, list):
                    if len(values_depends_on) == 1:
                        values_depends_on = depends_on["field"][0]
                        annotation_hint = "is value"
                    else:
                        values_depends_on = ", ".join(depends_on["field"][:-1]) + f", or {depends_on['field'][-1]}"
                        annotation_hint = "is one of these values"
                else:
                    values_depends_on = depends_on["field"]
                    annotation_hint = "is value"
            elif "value" in depends_on:
                values_depends_on = depends_on["field"]
                annotation_hint = "is value"
            else:
                self.logger.warning()
                pass # Skip depends on

            instruction.append(f'   - Depends on: This fields is only applicable if field: {field_depends_on}, {annotation_hint}: {values_depends_on}. Otherwise use default for this field!')

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

    def _format_promptgraph_instructions(self, instructions: str, chain_indexes: List[int], variable_name:str) -> str:
        """
        Extract and filter instructions for PromptGraph method based on chain indexes.

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
            chain: Optional chain index for PromptGraph method

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
            elif self.prompt_method in ["CoT", "SelfConsistency", "PromptGraph"]:
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
        if self.prompt_method == "PromptGraph":
            task_instructions, task_variable = self._format_promptgraph_instructions(task_instructions, chain_indexes, "task_variable")
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

                if self.prompt_method in ["CoT", "SelfConsistency", "PromptGraph"]:
                    if 'reasoning' not in example:
                        raise ValueError("Examples for CoT, SelfConsistency, and PromptGraph must include 'reasoning'")
                    
                    reasoning_instructions = example['reasoning'].strip()
                    if self.prompt_method == "PromptGraph":
                        reasoning_instructions = reasoning_instructions.split("\n")
                        example_instructions[f"assistant_reasoning"] = "\n".join([reasoning_instructions[i] for i in chain_indexes if i < len(reasoning_instructions)])

                example_outcome = example['output'].strip()
                if self.prompt_method == "PromptGraph":
                    example_outcome, example_output_variable = self._format_promptgraph_instructions(example_outcome, chain_indexes, f"example_output_variable_{idx}")
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
        elif self.prompt_method in ["CoT", "SelfConsistency", "PromptGraph"]:
            final_instructions = (
                "Begin the extraction now. First, reason step-by-step to identify and justify the value for each required field, "
                "enclosed within <think>...</think> tags. Then, output only the final structured data as a single valid JSON block, "
                "starting with ```json and ending with ```."
            )
        return system_instructions, field_instructions, task_instructions, examples_instructions, report_instructions, final_instructions, prompt_variables, output_format

    def add_names_to_messages(self, prompt_value: ChatPromptValue) -> ChatPromptValue:
        processed_messages = []
        for msg in prompt_value.messages:
            if hasattr(msg, 'additional_kwargs') and 'name' in msg.additional_kwargs:
                # Create a new message instance with the name at top level
                new_msg = msg.copy()
                new_msg.name = new_msg.additional_kwargs.pop('name')
                processed_messages.append(new_msg)
            else:
                processed_messages.append(msg)
        return ChatPromptValue(messages=processed_messages)

    def _generate_chat_chain(self) -> None:
        """
        Generate the chat chain based on the prompt method and configuration

        Returns:
            Runnable chain for processing reports
        """
        self.logger.info(f"Stating generating prompt using {self.prompt_method} method")
        if self.prompt_method == "PromptGraph":
            if self.graph is None:
                raise ValueError("PromptGraph method requires chain definitions in the prompt configuration")
            
            # Generate prompts for each chain
            prompt_templates = []
            output_formats = {}
            parsers = []
            for chain in self.graph:
                system_instructions, field_instructions, task_instructions, examples_instructions, report_instructions, final_instructions, prompt_variables, output_format = self._generate_prompt(chain=chain)
                prompt = [
                    SystemMessage(system_instructions, name="system_instructions") if self.system_instructions_allowed else HumanMessage(system_instructions, name="system_instructions"),
                    HumanMessage(field_instructions, name="field_instructions"),
                    HumanMessagePromptTemplate.from_template(task_instructions, additional_kwargs={"name": "task_instructions"}),
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
                            AIMessagePromptTemplate.from_template(example_instructions["assistant_output"], additional_kwargs={"name": f"example_assistant_output_{idx}"})
                        ])

                prompt.extend([
                    HumanMessagePromptTemplate.from_template(report_instructions, additional_kwargs={"name": "report_instructions"}),
                    HumanMessage(final_instructions, name="final_instructions")
                ])
                prompt = self._merge_consecutive_messages(prompt) if self.merge_consecutive_messages else prompt
                prompt_template = ChatPromptTemplate(prompt)
                prompt_template = prompt_template.partial(**prompt_variables)
                prompt_template = prompt_template.pipe(self.add_names_to_messages)

                prompt_templates.append(prompt_template)
                output_formats.update(output_format)
                parsers.append(ReasoningAndDynamicJSONParser(output_format, model=self.sentence_model, dry_run=self.dry_run))
            
            # Bit ugly, but saving the to the final output model manually and using after invoke to structure data
            self.final_parser = ReasoningAndDynamicJSONParser(output_formats, model=self.sentence_model, dry_run=self.dry_run)._output_model
            self.chat_chain = self._create_prompt_graph(prompt_templates, parsers)
        else:
            system_instructions, field_instructions, task_instructions, examples_instructions, report_instructions, final_instructions, prompt_variables, output_format = self._generate_prompt(chain=None)
            prompt = [
                SystemMessage(system_instructions, name="system_instructions") if self.system_instructions_allowed else HumanMessage(system_instructions, name="system_instructions"),
                HumanMessage(field_instructions, name="field_instructions"),
                HumanMessagePromptTemplate.from_template(task_instructions, additional_kwargs={"name": "task_instructions"}),
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
                        AIMessagePromptTemplate.from_template(example_instructions["assistant_output"], additional_kwargs={"name": f"example_assistant_output_{idx}"})
                    ])

            prompt.extend([
                HumanMessagePromptTemplate.from_template(report_instructions, additional_kwargs={"name": "report_instructions"}),
                HumanMessage(final_instructions, name="final_instructions")
            ])
            prompt = self._merge_consecutive_messages(prompt) if self.merge_consecutive_messages else prompt

            prompt_template = ChatPromptTemplate(prompt)
            prompt_template = prompt_template.partial(**prompt_variables)
            prompt_template = prompt_template.pipe(self.add_names_to_messages)

            parser = ReasoningAndDynamicJSONParser(output_format, model=self.sentence_model, dry_run=self.dry_run)
            base_chain = prompt_template | self.llm | parser
            if self.prompt_method == "SelfConsistency":
                # Initialize our fast ensemble strategy
                self.chat_chain = self._create_self_consistency_chain(base_chain)
            else:
                # For other methods, we can use the base chain directly
                self.chat_chain = base_chain

    def _merge_consecutive_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Merge consecutive messages of the same type (user or assistant) into a single message. 
        Required for some models that do not support multiple consecutive messages of the same type.

        Args:
            messages: List of BaseMessage objects
        Returns:
            List of BaseMessage objects with consecutive messages merged
        """
        if not messages:
            return []
        
        def to_template_message(msg: Union[HumanMessage, SystemMessage, AIMessage]) -> BasePromptTemplate:
            if isinstance(msg, HumanMessage):
                return HumanMessagePromptTemplate.from_template(msg.content, additional_kwargs={"name": getattr(msg, "name", None)})
            elif isinstance(msg, SystemMessage):
                return SystemMessagePromptTemplate.from_template(msg.content, additional_kwargs={"name": getattr(msg, "name", None)})
            elif isinstance(msg, AIMessage):
                return AIMessagePromptTemplate.from_template(msg.content, additional_kwargs={"name": getattr(msg, "name", None)})
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")
    
        def _create_message(cls: BasePromptTemplate, merged_template: str, names: List[str]) -> BasePromptTemplate:
            combined_name = " + ".join(filter(None, names)) if names else None
            prompt = PromptTemplate.from_template(merged_template)
            return cls(prompt=prompt, additional_kwargs={"name": combined_name} if combined_name else {})
        
        merged = []
        prev_cls = None
        name_buffer = []
        content_buffer = []

        for msg in messages:
            if "PromptTemplate" not in type(msg).__name__:
                msg = to_template_message(msg)

            curr_cls = type(msg)
            curr_name = getattr(msg, "name", None) or msg.additional_kwargs.get("name")

            if curr_cls == prev_cls:
                content_buffer.append(msg.prompt.template)
                if curr_name:
                    name_buffer.append(curr_name)
            else:
                if prev_cls is not None:
                    merged.append(_create_message(prev_cls, "\n\n".join(content_buffer), name_buffer))
                prev_cls = curr_cls
                content_buffer = [msg.prompt.template]
                name_buffer = [curr_name] if curr_name else []

        if content_buffer:
            merged.append(_create_message(prev_cls, "\n\n".join(content_buffer), name_buffer))

        return merged

    def _create_prompt_graph(self, prompt_templates: List[ChatPromptTemplate], parsers: List[ReasoningAndDynamicJSONParser]) -> Runnable:
        """
        Create a LangGraph from the DAG structure that can handle conditional branching for PromptGraph
        
        Args:
            prompt_templates: List of ChatPromptTemplate objects ordered by chain_order
            parsers: List of parsers corresponding to each prompt template
            
        Returns:
            Runnable graph with memory that follows the DAG structure
        """
        # Import langgraph here so optional dependency only required when using graphs
        from langgraph.graph import MessagesState, StateGraph, START, END
        from langgraph.checkpoint.memory import InMemorySaver

        class PromptGraphState(TypedDict):
            report: str
            patient: str
            index: int
            result: Dict[str, Any]
            messages: List[BaseMessage]
            status: str
            current_values: Dict[str, Any]

        # Initialize the graph and memory
        workflow = StateGraph(state_schema=PromptGraphState)
        memory = InMemorySaver()

        earliest_chain_order = min(self.graph.keys())
        last_item_used = False
        for chain_order, node_data in sorted(self.graph.items()):
            prompt_idx = chain_order - earliest_chain_order # If you start from 1, 2, etc. instead of 0
            if prompt_idx > len(prompt_templates):
                if last_item_used:
                    raise ValueError("Multiple chains out of range detected, this is not possible, please fix the order of the chains")
                else:
                    prompt_idx = len(prompt_templates) - 1
                    last_item_used = True

            prompt_template = prompt_templates[prompt_idx]
            parser = parsers[prompt_idx]
            
            def create_node(prompt_template: ChatPromptTemplate, 
                        parser: ReasoningAndDynamicJSONParser,
                        step: int,
                        fields: List[str]):
                @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
                async def node(state: PromptGraphState) -> Dict[str, Any]:
                    try:
                        # Run the chain
                        chain = prompt_template | self.llm | parser
                        result = await chain.ainvoke(
                            {"report": state["report"], "patient": state["patient"]}, 
                            timeout=self.timeout
                        )

                        if result.get('extracted_data', None) is None:
                            raise ValueError(f"Failed to parse JSON block for Step {step}")

                        # Update state with new values
                        new_values = {**state.get("current_values", {}), **result["extracted_data"]}
                        
                        return {
                            "patient": state["patient"],
                            "index": state["index"],
                            "result": {
                                "reasoning": state.get("result", {}).get("reasoning", []) + [result["reasoning"]],
                                "extracted_data": {**state.get("result", {}).get("extracted_data", {}), **result["extracted_data"]}
                            },
                            "messages": state.get("messages", []) + [AIMessage(content=result["raw_output"])],
                            "status": "success",
                            "current_values": new_values
                        }
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout for Patient: {state['patient']}, Step: {step}")
                        raise
                    except Exception as e:
                        self.logger.error(f"Failed - Patient: {state['patient']}, Step: {step}, Error: {e}")
                        raise

                return node

            node_name = f"step_{chain_order}"
            workflow.add_node(
                node_name,
                create_node(
                    prompt_template=prompt_template,
                    parser=parser,
                    step=chain_order,
                    fields=node_data["fields"]
                )
            )

        # Add conditional edges based on DAG structure
        for chain_order, node_data in self.graph.items():
            current_node = f"step_{chain_order}"
            edges = node_data["edges"]

            if not node_data["edges"]:
                # Terminal node
                workflow.add_edge(current_node, END)
                continue
        
            # Separate conditional and unconditional edges
            conditional_edges = [e for e in edges if e["condition"] is not None]
            unconditional_edges = [e for e in edges if e["condition"] is None]

            if not conditional_edges:
                # Only unconditional edges - simple case
                for edge in unconditional_edges:
                    workflow.add_edge(current_node, f"step_{edge['next_step']}")
                continue

            # Create condition function
            def create_condition_func(conditions, has_else):
                def condition(state: PromptGraphState) -> str:
                    # First check all non-else conditions
                    for edge in conditions:
                        if edge["condition"]["value"] == "else":
                            continue
                            
                        field = edge["condition"]["field"]
                        value = edge["condition"]["value"]
                        if str(state.get("current_values", {}).get(field)).strip().lower() == str(value).strip().lower():
                            return f"cond_{field}={value}"
                    
                    # Explicitly check for else condition if it exists
                    if has_else:
                        return "else_condition"
                    
                    # Fallback to unconditional edges if they exist
                    if unconditional_edges:
                        return f"unconditional"
                        
                    # Final fallback
                    return "default"
                return condition

            # Check if we have explicit else conditions
            has_else = any(e["condition"]["value"] == "else" for e in conditional_edges)
            
            # Build path map
            path_map = {}
            for edge in conditional_edges:
                if edge["condition"]["value"] == "else":
                    path_map["else_condition"] = f"step_{edge['next_step']}"
                else:
                    path_map[f"cond_{edge['condition']['field']}={edge['condition']['value']}"] = f"step_{edge['next_step']}"
            
            # Add unconditional path if exists
            if unconditional_edges:
                path_map["unconditional"] = f"step_{unconditional_edges[0]['next_step']}"
            
            # Only add default if no other options exist
            if not has_else and not unconditional_edges:
                path_map["default"] = END

            workflow.add_conditional_edges(
                current_node,
                create_condition_func(conditional_edges, has_else),
                path_map
            )

        # Set entry point
        workflow.set_entry_point(f"step_{min(self.graph.keys())}")
        
        # Compile the graph
        graph = workflow.compile(checkpointer=memory)

        if self.verbose:
            try:
                with open("dag_graph.png", "wb") as f:
                    f.write(graph.get_graph().draw_mermaid_png())
            except Exception as e:
                self.logger.warning(f"Could not generate graph visualization: {e}")

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
                    if self.prompt_method == "PromptGraph":
                        thread_id = uuid.uuid4()
                        result = await asyncio.wait_for(self.chat_chain.ainvoke(item, config={"configurable": {"thread_id": thread_id}}), timeout=self.timeout)
                        result["result"]["extracted_data"] = self.final_parser(**result["result"]["extracted_data"]).dict()
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
                    "status": "success" if llm_result.get('extracted_data') else "failed",
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

        if self.dry_run:
            self.logger.info(f"[Dry Run] Normally would process {len(inputs)} reports, now will show you one random examples")
            examples = random.sample(inputs, min(1, len(inputs)))
            await self._process_items(examples)
            return [{}] * len(inputs)

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
        outputs = [
            self.chat_chain.invoke(patient) for patient in inputs
        ]
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
