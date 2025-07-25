import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional, List
import ast
import yaml
from scipy.stats import t
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
    """Read prompt configuration from a YAML file."""
    with open(prompt_config_path, 'r') as file:
        prompt = yaml.safe_load(file)
    
    prompt = prompt.get("field_instructions", None)
    if prompt is None:
        raise ValueError("Prompt configuration does not contain 'field_instructions' key.")
    
    prompt = {item["name"]: {key: value for key, value in item.items() if key != "name"} for item in prompt}
    return prompt

def calculate_similarity(pred: str, gt: str, sentence_model: SentenceTransformer) -> float:
    """
    Calculate semantic similarity between two strings using a sentence transformer model.

    Args:
        pred (str): The predicted string.
        gt (str): The ground truth string.
    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    if pred == gt:
        return 1.0 # Complete match
    if not gt:
        return np.nan
    if not pred or not gt:
        return 0.0
    
    embeddings = sentence_model.encode([pred, gt], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

def calculate_list_similarity(pred_list: List[str], gt_list: List[str], sentence_model: SentenceTransformer) -> float:
    """
    Compute average maximum similarity for each ground truth item against the predicted list.
    
    Args:
        pred_list (List[str]): List of predicted strings.
        gt_list (List[str]): List of ground truth strings.
    Returns:
        float: Average maximum similarity score for ground truth items against predictions.
    """
    if not gt_list:
        return 1.0 if not pred_list else 0.0  # Handle empty GT case

    pred_list = [str(p) for p in pred_list or []]
    gt_list = [str(g) for g in gt_list or []]

    # Compute similarity of each ground truth item to the full prediction list
    scores = []
    for gt_item in gt_list:
        if not pred_list:
            scores.append(0.0)
        else:
            similarities = [
                calculate_similarity(pred_item, gt_item, sentence_model)
                for pred_item in pred_list
            ]
            scores.append(max(similarities))

    return sum(scores) / len(scores)

def calculate_results(prediction: pd.DataFrame, ground_truth: pd.DataFrame, prompt_config: Dict[str, Any], sentence_model: SentenceTransformer) -> pd.DataFrame:
    """
    Calculate performance metrics comparing prediction against ground truth.
    
    Args:
        prediction (pd.DataFrame): DataFrame containing predicted values.
        ground_truth (pd.DataFrame): DataFrame containing ground truth values.
        prompt_config (Dict[str, Any]): Configuration dictionary defining fields and their types.
        sentence_model (SentenceTransformer): Pretrained sentence transformer model for semantic similarity.
    Returns:
        pd.DataFrame: DataFrame containing calculated metrics for each field.
    """
    # Sanity check if prediction and ground_truth have the same columns and length
    if not prediction.columns.equals(ground_truth.columns):
        raise ValueError("Prediction and ground truth DataFrames must have the same columns.")
    
    if len(prediction) != len(ground_truth):
        raise ValueError("Prediction and ground truth DataFrames must have the same number of rows.")
    
    # Initialize results DataFrame
    results = []
    for field, meta in prompt_config.items():
        # Calculate metrics depending on the field type
        if meta.get("type") == "string":
            if "options" in meta:
                # Calculate the mean accuracy and standard deviation for categorical fields
                scores = (prediction[field] == ground_truth[field]).astype(int)
                metric_type = "accuracy"
            else:
                # Semantic similarity for free text
                scores = pd.Series([
                    calculate_similarity(str(p), str(g), sentence_model)
                    for p, g in zip(prediction[field], ground_truth[field])
                ])
                metric_type = "semantic_similarity"
        elif meta.get("type") == "list":
            # Ensure both prediction and ground truth are lists
            prediction[field] = prediction[field].apply(lambda x: x if isinstance(x, list) else [])
            ground_truth[field] = ground_truth[field].apply(
                lambda x: x if isinstance(x, list) else x.split(", ") if isinstance(x, str) else []
            )
            scores = pd.Series([
                calculate_list_similarity(p, g, sentence_model)
                for p, g in zip(prediction[field], ground_truth[field])
            ])
            metric_type = "list_similarity"
        elif meta.get("type") == "dict":
            raise NotImplementedError("Dictionary type fields are not supported yet.")
        else:
            raise ValueError(f"Unsupported field type for field '{field}': {meta.get('type')}")
        
        # Calculate 95% confidence interval
        n = len(scores)
        mean_score = scores.mean()
        std_err = scores.std(ddof=1) / np.sqrt(n)
        ci = t.interval(0.95, df=n-1, loc=mean_score, scale=std_err)
        
        results.append({
            "field": field,
            "metric_type": metric_type,
            "mean": scores.mean(),
            "ci_low": ci[0],
            "ci_high": ci[1],
        })

    return pd.DataFrame(results)

def calculate_macro_average(results: pd.DataFrame, prompt_config: Dict[str, Any], weights: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Calculate macro-average of all metrics in the results DataFrame.

    Args:
        results (pd.DataFrame): DataFrame containing calculated metrics for each field.
        prompt_config (Dict[str, Any]): Configuration dictionary defining fields and their types.
        weights (Optional[List[int]]): Weights for each field used for macro averaging. This overrides the weights defined in the prompt config.
    Returns:
        pd.DataFrame: DataFrame with macro averages.
    """
    result_fields = list(results["field"])
    num_fields = len(result_fields)

    # Handle weights override
    if weights is not None:
        if len(weights) != num_fields:
            raise ValueError(f"Length of weights ({len(weights)}) does not match number of result fields ({num_fields}).")
        weight_mapping = {field: weight for field, weight in zip(result_fields, weights)}
        print("Using overridden weights:")
        for field in result_fields:
            print(f"  {field}: {weight_mapping[field]}")
    else:
        # Attempt to collect weights from prompt_config
        weight_mapping = {}
        missing_weight_fields = []

        for field in result_fields:
            config = prompt_config.get(field)
            if config and "weight" in config:
                weight_mapping[field] = config["weight"]
            else:
                missing_weight_fields.append(field)

        if len(missing_weight_fields) == 0:
            print("Using weights from prompt_config:")
            for field in result_fields:
                print(f"  {field}: {weight_mapping[field]}")
        elif len(missing_weight_fields) == num_fields:
            weight_mapping = None
        else:
            raise ValueError(
                f"Incomplete weight definitions in prompt_config. "
                f"The following fields are missing weights: {missing_weight_fields}"
            )

    means = results["mean"].to_numpy()
    if weight_mapping:
        results["weight"] = results["field"].map(weight_mapping).fillna(1)
        weights_arr = results["weight"].to_numpy()
        macro_avg = np.average(means, weights=weights_arr)

        std_err = np.sqrt(np.average((means - macro_avg) ** 2, weights=weights_arr)) / np.sqrt(len(means))
    else:
        macro_avg = np.mean(means)
        std_err = np.std(means, ddof=1) / np.sqrt(len(means))

    n = len(means)
    ci = t.interval(0.95, df=n - 1, loc=macro_avg, scale=std_err)
    macro_results = {
        "field": "All Fields",
        "metric_type": "macro_avgs",
        "mean": macro_avg,
        "ci_low": ci[0],
        "ci_high": ci[1],
    }
    return pd.concat([results, pd.DataFrame([macro_results])], ignore_index=True)

    
def process_results(LLM_output: Union[pd.DataFrame, str], prompt_config: Union[Dict[str, Any], str], ground_truth: Optional[Union[pd.DataFrame, str]] = None, sentence_model: str = "all-mpnet-base-v2", scoring_weights: Optional[List[int]] = None, output_file: Optional[str] = None) -> None:
    """
    Process LLM output and compare it against ground truth.

    Args:
        LLM_output (Union[pd.DataFrame, str]): LLM output as a DataFrame or path to a CSV file.
        prompt_config (Union[Dict[str, Any], str]): Prompt configuration as a dictionary or path to a YAML file.
        ground_truth (Optional[Union[pd.DataFrame, str]]): Ground truth data as a DataFrame or path to a CSV file.
        sentence_model (str): Sentence transformer model to use for semantic mapping.
        scoring_weights (Optional[List[int]]): Weights for each field used for macro averaging. This overrides the weights defined in the prompt config.
        output_file (Optional[str]): Path to save the results output file.
    """
    if isinstance(LLM_output, str):
        LLM_output = pd.read_csv(LLM_output, converters={"extracted_data": safe_literal_eval})

    if isinstance(prompt_config, str):
        prompt_config = read_prompt_config(prompt_config)

    # Check if JSON is in LLM output by counting None values and fill those with default dictionary
    default_dict = {key: value.get("default", "Unknown") for key, value in prompt_config.items()}
    LLM_output[["extracted_data", "missing_json"]] = LLM_output["extracted_data"].apply(
        lambda x: (x, True) if isinstance(x, dict) else (default_dict, False)
    ).apply(pd.Series)

    # Remove items not in prompt config
    LLM_output["extracted_data"] = LLM_output["extracted_data"].apply(
        lambda x: {k: v for k, v in x.items() if k in prompt_config}
    )
    
    if ground_truth is not None:
        if isinstance(ground_truth, str):
            ground_truth = pd.read_csv(ground_truth)

        raise ValueError("Ground truth comparison through a different file is not implemented at this moment.")
    else:
        # Ground truth are in LLM output, so extract them from LLM_output
        ground_truth = LLM_output.copy()
        
        # Check if ground truth columns are present such as in prompt_config
        missing_fields = [field for field in prompt_config if field not in ground_truth.columns]
        if missing_fields:
            print(f"[WARNING] The following fields from prompt config are missing in ground truth so results with not be calculated for these: {missing_fields}")
        
        # Remove missing fields from prompt_config
        prompt_config = {key: value for key, value in prompt_config.items() if key not in missing_fields}
        ground_truth = ground_truth[prompt_config.keys()]

    # Extract the data from LLM output
    prediction = pd.DataFrame(LLM_output["extracted_data"].tolist())
    prediction = prediction[prompt_config.keys()]

    # Load sentence transformer model
    sentence_model = SentenceTransformer(sentence_model)

    # Calculate results
    results = calculate_results(prediction, ground_truth, prompt_config, sentence_model)

    # Calculate macro (weighted) average of all metrics
    results = calculate_macro_average(results, prompt_config, scoring_weights)

    # If output file is specified, save the results
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print("Results:\n", results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process LLM analysed results against ground truth.")
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
        "-g", 
        "--ground-truth", 
        type=str, 
        default=None, 
        help="Path to the ground truth CSV file."
    )
    parser.add_argument(
        "-m",
        "--sentence-model",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence transformer model to use for semantic mapping."
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Path to save the results output file."
    )
    parser.add_argument(
        "-w",
        "--weights",
        nargs='+',
        type=int,
        default=None,
        help="Weights for each field used for macro averaging. This overrides the weights defined in the prompt config."
    )

    args = parser.parse_args()
    process_results(args.LLM_output, args.prompt_config, args.ground_truth, args.sentence_model, args.weights, args.output_file)