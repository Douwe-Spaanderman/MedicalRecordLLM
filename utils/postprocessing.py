import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
import ast
from sentence_transformers import SentenceTransformer, util

def safe_literal_eval(val: Union[str, None]) -> Union[None, List[Any]]:
    """
    Safely evaluate a string representation of a Python literal.

    Args:
        val (Union[str, None]): The string to evaluate.
    Returns:
        Union[None, List[Any]]: The evaluated list or None if the input is invalid.
    """
    if pd.isna(val) or val in ('', 'None'):
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None 

def read_prompt_config(prompt_config_path: str) -> Dict[str, Any]:
    """Read prompt configuration from a YAML file.

    Args:
        prompt_config_path (str): Path to the YAML configuration file.
    Returns:
        Dict[str, Any]: Parsed prompt configuration. 
    Raises:
        ValueError: If the configuration does not contain the 'field_instructions' key.
    """
    with open(prompt_config_path, 'r') as file:
        prompt = yaml.safe_load(file)
    
    prompt = prompt.get("field_instructions", None)
    if prompt is None:
        raise ValueError("Prompt configuration does not contain 'field_instructions' key.")
    
    prompt = {item["name"]: {key: value for key, value in item.items() if key != "name"} for item in prompt}
    return prompt

def map_option_items(LLM_output: pd.DataFrame, prompt_config: Dict[str, Any], model: SentenceTransformer) -> pd.DataFrame:
    """Map extracted data to predefined options using semantic similarity.

    Args:
        LLM_output (pd.DataFrame): DataFrame containing the LLM output with 'extracted_data' column.
        prompt_config (Dict[str, Any]): Configuration dictionary defining fields and their options.
        model (SentenceTransformer): Pretrained sentence transformer model for semantic similarity.
    Returns:
        pd.DataFrame: DataFrame with 'extracted_data' column updated to match options in prompt_config.
    """
    # Copy to avoid changing original
    df = LLM_output.copy()

    # Precompute embeddings for all options per field
    field_embeddings = {}
    for field, config in prompt_config.items():
        if "options" in config:
            options = config["options"]
            # Add default option if it exists and not already in options
            if "default" in config and config["default"] not in options:
                options = [config["default"]] + options
            option_embeddings = model.encode(options, convert_to_tensor=True)
            field_embeddings[field] = (options, option_embeddings)

    def map_dict(entry: dict, threshold: float = 0.5) -> dict:
        if not isinstance(entry, dict):
            return entry  # Skip if it's not a dictionary
        updated = entry.copy()
        for field, (options, option_embeddings) in field_embeddings.items():
            if field not in entry or entry[field] is None:
                continue
            val_str = str(entry[field]).strip()
            # Exact match
            for opt in options:
                if val_str.lower() == opt.lower():
                    updated[field] = opt
                    break
            else:
                # Semantic match
                val_embedding = model.encode(val_str, convert_to_tensor=True)
                similarities = util.cos_sim(val_embedding, option_embeddings)[0].cpu()
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])

                # Optional threshold
                updated[field] = options[best_idx] if best_score >= threshold else val_str
        return updated

    # Apply mapping to each dictionary in the column
    df['extracted_data'] = df['extracted_data'].map(map_dict)

    return df

def postprocessing(LLM_output: Union[pd.DataFrame, str], prompt_config: Union[Dict[str, Any], str], output_file: str, sentence_model: str) -> None:
    """Postprocess LLM output by mapping extracted data to predefined options.
    
    Args:
        LLM_output (Union[pd.DataFrame, str]): LLM output as a DataFrame or path to a CSV file.
        prompt_config (Union[Dict[str, Any], str]): Prompt configuration as a dictionary or path to a YAML file.
        output_file (str): Path to save the postprocessed output file.
        sentence_model (str): Sentence transformer model to use for semantic mapping.
    Raises:
        ValueError: If the LLM output does not contain 'extracted_data' column.
    """
    if isinstance(LLM_output, str):
        LLM_output = pd.read_csv(LLM_output, converters={"extracted_data": safe_literal_eval})

    if isinstance(prompt_config, str):
        prompt_config = read_prompt_config(prompt_config)

    # Postprocessing
    model = SentenceTransformer(sentence_model)
    # Map the extracted data to the prompt config for fields that have options (choices)
    LLM_output = map_option_items(LLM_output, prompt_config, model)

    # Save the postprocessed output
    LLM_output.to_csv(output_file, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Postprocess LLM output.")
    parser.add_argument(
        "-l",
        "--LLM-output",
        required=True,
        type=str, 
        help="Path to the LLM output file."
    )
    parser.add_argument(
        "-p",
        "--prompt-config",
        required=True,
        type=str,
        help="Path to YAML config for prompt definitions"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="Path to save the postprocessed output file."
    )
    parser.add_argument(
        "-m",
        "--sentence-model",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence transformer model to use for semantic mapping."
    )

    args = parser.parse_args()
    postprocessing(args.LLM_output, args.prompt_config, args.output_file, args.sentence_model)

