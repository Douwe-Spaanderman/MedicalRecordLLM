import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional, List
import ast
import yaml
from sklearn.metrics import balanced_accuracy_score
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
    pred_list = [str(p) for p in pred_list or []]
    gt_list = [str(g) for g in gt_list or []]

    if not gt_list and not pred_list:
        return 1.0  # Both are empty = perfect match
    if not gt_list or not pred_list:
        return 0.0  # One is empty, the other isn't = total mismatch
    
    def directional_score(source_list, target_list):
        scores = []
        for source_item in source_list:
            similarities = [
                calculate_similarity(source_item, target_item, sentence_model)
                for target_item in target_list
            ]
            scores.append(max(similarities) if similarities else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    # GT -> Pred and Pred -> GT
    gt_to_pred = directional_score(gt_list, pred_list)
    pred_to_gt = directional_score(pred_list, gt_list)

    return (gt_to_pred + pred_to_gt) / 2

def bootstrap_ci(data:tuple, statistic_fn, n_resamples:int=1000, confidence_level:float=0.95, paired:bool=False):
    """
    Compute bootstrap confidence intervals.
    
    Args:
        data: Input data (tuple of arrays for paired statistics)
        statistic_fn: Function to compute the statistic
        n_resamples: Number of bootstrap samples
        confidence_level: Confidence level for the interval
        paired: Whether the data is paired (multiple arrays that should be resampled together)
    """
    np.random.seed(42)  # For reproducibility
    
    # Calculate the original statistic
    original_stat = statistic_fn(*data) if paired else statistic_fn(data)
    
    # Prepare storage for bootstrap statistics
    bootstrap_stats = np.zeros(n_resamples)
    
    n_samples = len(data[0]) if paired else len(data)
    
    for i in range(n_resamples):
        # Generate bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        if paired:
            # For paired data, resample all arrays with the same indices
            resampled_data = tuple(arr[indices] for arr in data)
            bootstrap_stats[i] = statistic_fn(*resampled_data)
        else:
            # For single array, just resample
            resampled_data = data[indices]
            bootstrap_stats[i] = statistic_fn(resampled_data)
    
    # Calculate confidence interval
    alpha = (1 - confidence_level) / 2
    ci_low = np.percentile(bootstrap_stats, 100 * alpha)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha))
    
    return original_stat, ci_low, ci_high

def bootstrap_ci_macro(results: pd.DataFrame, n_resamples: int = 1000, confidence_level: float = 0.95):
    """
    
    """
    scores = [result["scores"] for result in results]
    weights = [result["weight"] for result in results]
    metrics = [result["metric_type"] for result in results]

    np.random.seed(42)  # For reproducibility

    # Calculate the original statistic
    total = 0
    total_weight = 0
    for score, weight, metric in zip(scores, weights, metrics):
        if metric == 'balanced_accuracy':
            # For balanced accuracy, args contains resampled (y_true, y_pred) pairs
            y_true, y_pred = score[0], score[1]
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            # For other metrics, args contains resampled scores
            score = np.mean(score)
        
        total += score * weight
        total_weight += weight
            
    original_stat = total / total_weight if total_weight > 0 else 0

    # Prepare storage for bootstrap statistics
    bootstrap_stats = np.zeros(n_resamples)
    
    n_samples = len(scores[0][0]) if metrics[0] == 'balanced_accuracy' else len(scores[0])

    for i in range(n_resamples):
        # Generate bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        total = 0
        total_weight = 0
        for score, weight, metric in zip(scores, weights, metrics):
            if metric == 'balanced_accuracy':
                # For balanced accuracy, args contains resampled (y_true, y_pred) pairs
                y_true, y_pred = score[0][indices], score[1][indices]
                score_ = balanced_accuracy_score(y_true, y_pred)
            else:
                # For other metrics, args contains resampled scores
                score_ = np.mean(score[indices])
            
            total += score_ * weight
            total_weight += weight

        bootstrap_stats[i] = total / total_weight if total_weight > 0 else 0
        
    # Calculate confidence interval
    alpha = (1 - confidence_level) / 2
    ci_low = np.percentile(bootstrap_stats, 100 * alpha)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha))
    
    return original_stat, ci_low, ci_high

def calculate_results(
    prediction: pd.DataFrame, 
    ground_truth: pd.DataFrame, 
    prompt_config: Dict[str, Any], 
    sentence_model: SentenceTransformer, 
    weights: Optional[List[int]] = None,
    n_bootstrap: int = 1000
) -> pd.DataFrame:    
    """
    Calculate performance metrics comparing prediction against ground truth.
    
    Args:
        prediction (pd.DataFrame): DataFrame containing predicted values.
        ground_truth (pd.DataFrame): DataFrame containing ground truth values.
        prompt_config (Dict[str, Any]): Configuration dictionary defining fields and their types.
        sentence_model (SentenceTransformer): Pretrained sentence transformer model for semantic similarity.
        weights (Optional[List[int]]): Weights for each field used for macro averaging. This overrides the weights defined in the prompt config.
        n_bootstrap (int): Number of bootstrap samples for confidence interval calculation.
    Returns:
        pd.DataFrame: DataFrame containing calculated metrics for each field plus macro average.
    """
    # Sanity check if prediction and ground_truth have the same columns and length
    if not prediction.columns.equals(ground_truth.columns):
        raise ValueError("Prediction and ground truth DataFrames must have the same columns.")
    
    if len(prediction) != len(ground_truth):
        raise ValueError("Prediction and ground truth DataFrames must have the same number of rows.")
    
    results = []
    field_weights = []
    for field, meta in prompt_config.items():
        default_value = meta.get("default", "Unknown")
        type_value = meta.get("type", "string").replace("_or_missing", "")
        # Check for weights
        field_weight = meta.get("weight", 1) if weights is None else weights[list(prompt_config.keys()).index(field)]
        field_weights.append(field_weight)

        # Clean missing values
        for df in [ground_truth, prediction]:
            df[field] = df[field].fillna(default_value)
            df[field] = df[field].apply(
                lambda x: default_value if (
                    pd.isna(x) or (isinstance(x, str) and x.strip().lower() in {"none", "not specified", "not applicable", "missing", "unknown", None})
                ) else x
            )

        # Calculate metrics depending on the field type
        if type_value in {"string", "number", "float", "binary", "boolean", "categorical"}:
            if "options" in meta:
                # Calculate the balanced accuracy for categorical fields
                if type_value in {"number", "float", "binary", "boolean"}:
                    # Ensure both prediction and ground truth are numeric
                    prediction[field] = pd.to_numeric(prediction[field], errors='coerce')
                    ground_truth[field] = pd.to_numeric(ground_truth[field], errors='coerce')

                    # Convert all to strings and replace NaN with sentinel
                    prediction[field] = prediction[field].fillna(default_value).astype(str)
                    ground_truth[field] = ground_truth[field].fillna(default_value).astype(str)

                scores = (ground_truth[field].to_numpy(), prediction[field].to_numpy())
                score, ci_low, ci_high = bootstrap_ci(
                    scores,
                    balanced_accuracy_score,
                    n_resamples=n_bootstrap,
                    paired=True
                )
                metric_type = "balanced_accuracy"
            else:
                # Semantic similarity for free text
                scores = pd.Series([
                    calculate_similarity(str(p), str(g), sentence_model)
                    for p, g in zip(prediction[field], ground_truth[field])
                ]).to_numpy()
                score, ci_low, ci_high = bootstrap_ci(
                    scores, 
                    np.mean, 
                    n_resamples=n_bootstrap
                )
                metric_type = "semantic_similarity"
        elif type_value == "list":
            # Ensure both prediction and ground truth are lists
            prediction[field] = prediction[field].apply(lambda x: x if isinstance(x, list) else [])
            ground_truth[field] = ground_truth[field].apply(
                lambda x: x if isinstance(x, list) else x.split(", ") if isinstance(x, str) else []
            )
            scores = pd.Series([
                calculate_list_similarity(p, g, sentence_model)
                for p, g in zip(prediction[field], ground_truth[field])
            ]).to_numpy()
            score, ci_low, ci_high = bootstrap_ci(
                scores, 
                np.mean, 
                n_resamples=n_bootstrap
            )
            metric_type = "list_similarity"
        elif type_value in {"dictionary", "dict"}:
            raise NotImplementedError("Dictionary type fields are not supported yet.")
        else:
            raise ValueError(f"Unsupported field type for field '{field}': {meta.get('type')}")
        
        results.append({
            "field": field,
            "metric_type": metric_type,
            "score": score,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "weight": field_weight,
            "scores": scores,
        })

    # Calculate macro average
    if results:
        macro_mean, macro_ci_low, macro_ci_high = bootstrap_ci_macro(
            results=results,
            n_resamples=n_bootstrap
        )
        
        macro_results = {
            "field": "All Fields",
            "metric_type": "macro_avg",
            "score": macro_mean,
            "ci_low": macro_ci_low,
            "ci_high": macro_ci_high,
            "weight": None,
            "scores": None
        }
        results.append(macro_results)

    return pd.DataFrame(results)

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
    results = calculate_results(prediction, ground_truth, prompt_config, sentence_model, scoring_weights)

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