from pathlib import Path
import yaml
import json
from typing import Dict, List, Optional, Any
import sys
import re
from collections import OrderedDict

def load_yaml_config(yaml_path: str) -> Dict:
    """Load query configuration from YAML file with validation"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['field_instructions', 'task']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in YAML: {section}")

    return config

def format_field_instruction(field: Dict, index: int, chain: Optional[int]) -> str:
    """Format a single field instruction with proper indentation"""
    # Skip field instruction not in chain if chain is specified
    if chain is not None and field.get('chain_order') != chain:
        return None

    lines = [f'{index}. "{field["name"]}":']
    
    # Only add type when non-obvious
    lines.append(f'   - Type: {field["type"]}')
    
    # Add constraints if present
    if 'constraints' in field:
        constraints = field['constraints']
        if isinstance(constraints, list):
            # Handle list constraints
            lines.append(f'   - Constraints:')
            for item in constraints:
                lines.append(f'     • {item}')
        else:
            # Handle multi-line string constraints
            for line in constraints.split('\n'):
                if line.strip():
                    if line.strip().startswith('- '):
                        # Convert list items to bullet points
                        lines.append(f'     • {line.strip()[2:]}')
                    else:
                        lines.append(f'   - {line.strip()}')
    
    # Add options if present
    if 'options' in field:
        opts = ', '.join(f'"{opt}"' for opt in field['options'])
        lines.append(
            f'   - Must be EXACTLY one of the following options: [{opts}]. Please enter the value exactly as shown, without any variations.'
        )
    
    # Handle dict structures
    if field['type'] in ['dictionary', 'list of dictionaries'] and 'structure' in field:
        subfields = ', '.join(f'"{sf["key"]}"' for sf in field['structure'])
        if field['type'] == 'dictionary':
            lines.append(f'   - Expected structure keys: {{ {subfields} }}')
        elif field['type'] == 'list of dictionaries':
            lines.append(f'   - Each dictionary must contain the keys: {{ {subfields} }}')

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
            lines.append(f'     - Key "{key}": {details}')
    
    # Handle default values
    default_value = field.get('default', 'Not specified')
    if isinstance(default_value, (dict, list)):
        # Compact formatting for empty lists
        if isinstance(default_value, list) and not default_value:
            lines.append('   - Default: []')
        else:
            # Smart JSON formatting
            default_str = json.dumps(default_value, indent=2)
            if '\n' in default_str:
                # For multi-line JSON, align with field instruction
                default_str = default_str.replace('\n', '\n      ')
                lines.append(f'   - Default: {default_str}')
            else:
                # Single-line for simple values
                lines.append(f'   - Default: {default_str}')
    else:
        lines.append(f'   - Default: "{default_value}"')
    
    return '\n'.join(lines)

def validate_example(example: Dict) -> None:
    """Validate that an example has the correct structure"""
    if not isinstance(example, dict):
        raise ValueError("Example must be a dictionary")
    if 'input' not in example or 'output' not in example:
        raise ValueError("Example must contain both 'input' and 'output'")
    if not isinstance(example['input'], str) or not isinstance(example['output'], str):
        raise ValueError("Example input and output must be strings")

def generate_prompt(
    query_config: Dict, 
    sample_report: str = "[SAMPLE REPORT CONTENT]",
    prompt_method: str = "ZeroShot",
    chain: Optional[int] = None,
) -> str:
    """
    Generate a complete prompt from the YAML configuration
    
    Args:
        query_config: Loaded YAML configuration
        sample_report: Text content of the report to process
        prompt_method: Optional method type for prompt generation (e.g., ZeroShot, OneShot)
        
    Returns:
        The complete formatted prompt
    """
    # Build system instructions. Either use the config or default to a generic instruction
    if 'system_instruction' in query_config:
        print("WARNING: using system instruction from config file. This is not recommended.")
        system_instruction = query_config['system_instruction'].strip()
    else:
        print("Building system instruction based on prompt method, which is recommended")
        if prompt_method in ["ZeroShot", "OneShot"]:
            system_instruction = [
                "You are a medical data extraction system that ONLY outputs valid JSON. Maintain strict compliance with these rules:",
                "1. ALWAYS begin and end your response with ```json markers",
                "2. Use EXACT field names and structure provided",
                "3. If a value is missing or not mentioned, use the specified default for that field.",
                "4. NEVER add commentary, explanations, or deviate from the output structure"
            ]
        elif prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
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
    for idx, field in enumerate(query_config['field_instructions'], start=1):
        field_instruction = format_field_instruction(field, idx, chain)
        if field_instruction is None:
            # Skip fields not in the specified chain
            continue
        else:
            field_instructions.append(field_instruction)
            chain_indexes.append(idx-1)
    
    # Build task instruction
    task_instructions = query_config['task'].strip()
    if prompt_method == "PromptChain":
        task_instructions = extract_promptchain_instructions(task_instructions, chain_indexes)

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
    if prompt_method != 'ZeroShot':
        if 'example' not in query_config:
            raise ValueError("Examples are required for prompt methods other than ZeroShot")
        
        example = query_config["example"]
        validate_example(example)
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

        if prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
            if 'reasoning' not in example:
                raise ValueError("Examples for CoT, SelfConsistency, and PromptChain must include 'reasoning'")
            
            reasoning_instructions = example['reasoning'].strip()
            if prompt_method == "PromptChain":
                reasoning_instructions = reasoning_instructions.split("\n")
                reasoning_instructions = "\n".join([reasoning_instructions[i] for i in chain_indexes if i < len(reasoning_instructions)])

            prompt_parts.extend([
                "[EXAMPLE THINKING]",
                reasoning_instructions,
                "",
            ])

        example_instructions = example['output'].strip()
        if prompt_method == "PromptChain":
            example_instructions = extract_promptchain_instructions(example_instructions, chain_indexes)

        prompt_parts.extend([
            "[EXAMPLE EXPECTED OUTPUT]",
            example_instructions
        ])

    # Add the actual report content
    prompt_parts.extend([
        "",
        "[BEGIN FILE CONTENT]",
        f"[file name]: report_preview",
        sample_report.strip(),
        "[END FILE CONTENT]",
        "",
    ])

    if prompt_method in ["ZeroShot", "OneShot"]:
        prompt_parts.append("Begin your extraction now. Your response MUST start with: ```json")
    elif prompt_method in ["CoT", "SelfConsistency", "PromptChain"]:
        prompt_parts.append("Begin your extraction now. First, reason step-by-step to identify each required field. After reasoning, output ONLY the final structured data and ensure your response starts with: ```json")

    return "\n".join(prompt_parts)

def generate_prompt_self_consistency(
        query_config: Dict, 
        sample_report: str = "[SAMPLE REPORT CONTENT]"
    ) -> str:
    """
    Generate a prompt specifically for Self-Consistency method

    Args:
        query_config: Loaded YAML configuration
        sample_report: Text content of the report to process

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
    for idx, field in enumerate(query_config['field_instructions'], start=1):
        field_instructions.append(format_field_instruction(field, idx))

    # Construct the prompt
    prompt_parts = [
        "[SYSTEM INSTRUCTION]",
        system_instruction,
        "",
        "[FIELD INSTRUCTIONS]",
        "\n".join(field_instructions),
        "",
        "[TASK INSTRUCTION]",
        query_config['task'].strip(),
        "",
        "[BEGIN FILE CONTENT]",
        f"[file name]: report_preview",
        sample_report.strip(),
        "[END FILE CONTENT]",
        "",
        "[BEGIN SELF-CONSISTENCY CANDIDATE OUTPUTS]",
        "",
        "[Candidate Output 1]",
        "[Reasoning]",
        "..."
        "[Structured Output]",
        "..."
        "[Candidate Output 2]",
        "[Reasoning]",
        "..."
        "[Structured Output]",
        "..."
        "[Candidate Output N]",
        "[Reasoning]",
        "..."
        "[Structured Output]",
        "...",
        "[END SELF-CONSISTENCY CANDIDATE OUTPUTS]",
        "Begin reconciliation now. First, review all reasoning paths and identify the most consistent and well-supported value for each required field. After reasoning, output ONLY the final structured data and ensure your response starts with: ```json"
    ]
    return "\n".join(prompt_parts)

def detect_chains(query_config: Dict) -> Optional[List[int]]:
    """
    Detect if the query configuration contains chains for PromptChain method

    Args:
        query_config: Loaded YAML configuration

    Returns:
        List of chain indices if chains are defined, otherwise None
    """
    # Chains are defined in the query_config under 'field_instructions'
    chains = []
    for field in query_config['field_instructions']:
        if chain_order := field.get('chain_order'):
            chains.append(chain_order)

    # Now check if the length of task, example, reasoning if present and output match the chain order
    if len(chains) != len(parse_json(query_config.get('task', ''))):
        raise ValueError("Chain order does not match the number of tasks defined in the YAML configuration. This will result in prompt generation errors.")
    
    # Same validation of example if present
    if 'example' in query_config:
        example = query_config['example']
        if reasoning := example.get('reasoning'):
            # Parse reasoning as JSON if it exists
            reasoning = reasoning.strip().split("\n")
            if len(chains) != len(reasoning):
                raise ValueError("Chain order does not match the number of reasoning steps defined in the example. This will result in prompt generation errors.")
        if 'output' in example and len(parse_json(example.get('output', ''))) != len(chains):
            raise ValueError("Chain order does not match the number of output fields defined in the example. This will result in prompt generation errors.")
            
    if not chains:
        return None
    else:
        return list(set(chains))  # Return unique chain orders

def parse_json(data: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from string"""
    try:
        json_matches = list(re.finditer(r'```json\s*(?P<json>{.*?})\s*```', data, re.DOTALL))
        for match in reversed(json_matches):
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue  # Try the next candidate      
    except Exception as e:
        print(f"JSON extraction failed: {e}")
        
    return None

def extract_promptchain_instructions(instructions: str, chain_indexes: List[int]) -> str:
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
    json_part = parse_json(instructions)

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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate a prompt from YAML configuration for medical data extraction"
    )
    parser.add_argument("config_path", help="Path to the YAML config file")
    parser.add_argument("--sample", help="Optional sample report text file to include")
    parser.add_argument(
        "-p",
        "--prompt-method",
        required=True,
        choices=["ZeroShot", "OneShot", "CoT", "SelfConsistency", "PromptChain"],
        type=str,
        help="Path to YAML config for query definitions"
    )
    args = parser.parse_args()
    
    try:
        # Load and validate config
        config = load_yaml_config(args.config_path)
        
        # Load sample report if provided
        sample_text = Path(args.sample).read_text() if args.sample else "[SAMPLE REPORT CONTENT]"

        if args.prompt_method == "PromptChain":
            chains = detect_chains(config)
            for chain in chains:
                # Generate prompt for each chain
                prompt = generate_prompt(config, sample_text, args.prompt_method, chain)
                
                # Print with clear demarcation
                print("=" * 80)
                print(f"GENERATED PROMPT FOR CHAIN {chain}")
                print("=" * 80)
                print(prompt)
                print("=" * 80)
        else:
            # Generate prompt
            prompt = generate_prompt(config, sample_text, args.prompt_method)

            # Print with clear demarcation
            print("=" * 80)
            print("GENERATED PROMPT")
            print("=" * 80)
            print(prompt)
            print("=" * 80)
        
        if args.prompt_method == "SelfConsistency":
            # This is normally done on the fly with the output, but for preview purposes we generate it here
            prompt_self_consistency = generate_prompt_self_consistency(config, sample_text)
            print("=" * 80)
            print("GENERATED SELF-CONSISTENCY PROMPT")
            print("=" * 80)
            print(prompt_self_consistency)
            print("=" * 80)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()