from pathlib import Path
import yaml
import json
from typing import Dict, List, Union, Optional
import sys

def load_yaml_config(yaml_path: str) -> Dict:
    """Load query configuration from YAML file with validation"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['system_instruction', 'field_instructions', 'task']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in YAML: {section}")
    
    # Handle both single example and multiple examples
    if 'examples' not in config and 'example' not in config:
        raise ValueError("YAML config must contain either 'example' or 'examples' section")
    
    # Convert single example to list format for consistent handling
    if 'example' in config:
        config['examples'] = [config['example']]
    
    return config

def format_field_instruction(field: Dict, index: int) -> str:
    """Format a single field instruction with proper indentation"""
    lines = [f'{index}. "{field["name"]}":']
    
    # Only add type when non-obvious
    if field["type"] in ("nested", "list") or "options" in field:
        lines.append(f'   - Type: {field["type"].capitalize()}')
    
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
            f'   - Must be EXACTLY one of: [{opts}]\n'
            '      No variations allowed.'
        )
    
    # Handle nested structures
    if field['type'] == 'nested' and 'structure' in field:
        subfields = ', '.join(f'"{sf["key"]}"' for sf in field['structure'])
        lines.append(f'   - Structure: {{{subfields}}}')
        for subfield in field['structure']:
            if 'constraints' in subfield:
                lines.append(f'     - {subfield["key"]}: {subfield["constraints"]}')
    
    # Handle list of dictionaries
    elif field['type'] == 'list' and field.get('item_type') == 'dict':
        if 'required_keys' in field:
            keys = ', '.join(f'"{k}"' for k in field['required_keys'])
            lines.append(f'   - Required keys: {keys}')
    
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
    sample_report: str = "[SAMPLE REPORT CONTENT]"
) -> str:
    """
    Generate a complete prompt from the YAML configuration
    
    Args:
        query_config: Loaded YAML configuration
        sample_report: Text content of the report to process
        
    Returns:
        The complete formatted prompt
    """
    # Build field instructions
    field_instructions = []
    for idx, field in enumerate(query_config['field_instructions'], start=1):
        field_instructions.append(format_field_instruction(field, idx))
    
    # Validate all examples
    for example in query_config['examples']:
        validate_example(example)
    
    # Construct the prompt
    prompt_parts = [
        "[SYSTEM INSTRUCTION]",
        query_config['system_instruction'].strip(),
        "",
        "[FIELD INSTRUCTIONS]",
        "\n".join(field_instructions),
        "",
        "[TASK INSTRUCTION]",
        query_config['task'].strip(),
    ]

    # Add examples section
    if query_config['examples']:
        prompt_parts.extend([
            "",
            "[EXAMPLES]"
        ])
        
        for example in query_config['examples']:
            prompt_parts.extend([
                "---",
                "[EXAMPLE INPUT REPORT]",
                example['input'].strip(),
                "",
                "[EXAMPLE EXPECTED OUTPUT]",
                example['output'].strip()
            ])

    # Add the actual report content
    prompt_parts.extend([
        "",
        "[BEGIN FILE CONTENT]",
        f"[file name]: report_preview",
        sample_report.strip(),
        "[END FILE CONTENT]",
        "",
        "Begin your extraction now. Your response MUST start with: ```json"
    ])
    
    return "\n".join(prompt_parts)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate a prompt from YAML configuration for medical data extraction"
    )
    parser.add_argument("config_path", help="Path to the YAML config file")
    parser.add_argument("--sample", help="Optional sample report text file to include")
    args = parser.parse_args()
    
    try:
        # Load and validate config
        config = load_yaml_config(args.config_path)
        
        # Load sample report if provided
        sample_text = Path(args.sample).read_text() if args.sample else "[SAMPLE REPORT CONTENT]"
        
        # Generate prompt
        prompt = generate_prompt(config, sample_text)
        
        # Print with clear demarcation
        print("=" * 80)
        print("GENERATED PROMPT")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()