import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional, List, Tuple
import ast
import yaml
from scipy.stats import t
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")

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

def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000, average: str = 'binary') -> Dict[str, Tuple[float, float, float]]:
    """
    Calculate bootstrap confidence intervals for various metrics.
    """
    rng = np.random.default_rng()
    n_samples = len(y_true)
    
    metrics_dict = {
        'accuracy': [], 'balanced_accuracy': [], 'precision': [],
        'recall': [], 'f1': [], 'jaccard': []
    }
    
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, n_samples)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        
        try:
            metrics_dict['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            metrics_dict['balanced_accuracy'].append(balanced_accuracy_score(y_true_boot, y_pred_boot))
            metrics_dict['precision'].append(precision_score(y_true_boot, y_pred_boot, average=average, zero_division=0))
            metrics_dict['recall'].append(recall_score(y_true_boot, y_pred_boot, average=average, zero_division=0))
            metrics_dict['f1'].append(f1_score(y_true_boot, y_pred_boot, average=average, zero_division=0))
            metrics_dict['jaccard'].append(jaccard_score(y_true_boot, y_pred_boot, average=average, zero_division=0))
        except:
            pass
    
    result = {}
    for metric_name, values in metrics_dict.items():
        if values:
            result[metric_name] = (
                np.mean(values),
                np.percentile(values, 2.5),
                np.percentile(values, 97.5)
            )
        else:
            result[metric_name] = (np.nan, np.nan, np.nan)
    
    return result

def calculate_results(
    prediction: pd.DataFrame, 
    ground_truth: pd.DataFrame, 
    prompt_config: Dict[str, Any], 
    sentence_model: SentenceTransformer, 
    weights: Optional[List[int]] = None,
    n_bootstrap: int = 1000,
    use_balanced_accuracy: bool = False,
    strict_metrics: bool = False
) -> pd.DataFrame:
    """
    Calculate performance metrics comparing prediction against ground truth.
    
    Args:
        prediction (pd.DataFrame): DataFrame containing predicted values.
        ground_truth (pd.DataFrame): DataFrame containing ground truth values.
        prompt_config (Dict[str, Any]): Configuration dictionary defining fields and their types.
        sentence_model (SentenceTransformer): Pretrained sentence transformer model for semantic similarity.
        weights (Optional[List[int]]): Weights for each field used for micro averaging.
        n_bootstrap (int): Number of bootstrap samples for CI; if <=1, skip bootstrapping.
        use_balanced_accuracy (bool): If True, use balanced accuracy for categorical/binary fields instead of normal accuracy.
        strict_metrics (bool): If True, exclude entries with default values in ground truth from calculations.
    
    Returns:
        pd.DataFrame: DataFrame containing calculated metrics for each field plus micro average.
    """
    if not prediction.columns.equals(ground_truth.columns):
        raise ValueError("Prediction and ground truth DataFrames must have the same columns.")
    if len(prediction) != len(ground_truth):
        raise ValueError("Prediction and ground truth DataFrames must have the same number of rows.")
    
    results = []
    for field, meta in prompt_config.items():
        default_value = meta.get("default", "Unknown")
        type_value = meta.get("type", "string").replace("_or_missing", "")
        if meta.get("options") and not type_value in {"binary", "boolean"}:
            type_value = "categorical_number" if type_value in {"number", "float"} else "categorical"

        field_weight = meta.get("weight", 1) if weights is None else weights[list(prompt_config.keys()).index(field)]

        # Clean missing values
        for df in [ground_truth, prediction]:
            df[field] = df[field].apply(
                lambda x: default_value if (
                    (isinstance(x, list) and (len(x) == 0 or pd.isna(x).any()))
                    or (not isinstance(x, list) and pd.isna(x))
                    or (isinstance(x, str) and x.strip().lower() in {
                        "none", "not specified", "not applicable", "missing", "unknown", None
                    })
                ) else x
            )

        prediction_field = prediction[field].copy()
        ground_truth_field = ground_truth[field].copy()

        if type_value in {"number", "float"}:
            # Numeric exact match
            prediction_field = pd.to_numeric(prediction_field, errors='coerce', downcast="float")
            ground_truth_field = pd.to_numeric(ground_truth_field, errors='coerce', downcast="float")

        y_true = ground_truth_field.astype(str)
        y_pred = prediction_field.astype(str)

        # Exclude default values if specified for strict metrics
        if strict_metrics:
            mask = (y_true != str(default_value)) | (y_pred != str(default_value))
            y_pred = y_pred[mask].reset_index(drop=True)
            y_true = y_true[mask].reset_index(drop=True)

        # Initialize variables for metric_type, mean, ci_low, ci_high
        metric_type = None
        mean_score = np.nan
        ci_low = np.nan
        ci_high = np.nan

        accuracy, accuracy_ci_low, accuracy_ci_high = np.nan, np.nan, np.nan
        balanced_acc, balanced_acc_ci_low, balanced_acc_ci_high = np.nan, np.nan, np.nan
        similarity, similarity_ci_low, similarity_ci_high = np.nan, np.nan, np.nan
        precision, precision_ci_low, precision_ci_high = np.nan, np.nan, np.nan
        recall, recall_ci_low, recall_ci_high = np.nan, np.nan, np.nan
        f1, f1_ci_low, f1_ci_high = np.nan, np.nan, np.nan
        jaccard, jaccard_ci_low, jaccard_ci_high = np.nan, np.nan, np.nan

        # Calculate classification metrics for all field types except lists
        if type_value != "list":
            classes = meta.get("options", None)
            average_method = 'binary' if classes and len(classes) == 2 else 'macro'
            
            if n_bootstrap > 1:
                boot_metrics = bootstrap_metrics(y_true, y_pred, n_bootstrap, average_method)
                accuracy, accuracy_ci_low, accuracy_ci_high = boot_metrics['accuracy']
                balanced_acc, balanced_acc_ci_low, balanced_acc_ci_high = boot_metrics['balanced_accuracy']
                precision, precision_ci_low, precision_ci_high = boot_metrics['precision']
                recall, recall_ci_low, recall_ci_high = boot_metrics['recall']
                f1, f1_ci_low, f1_ci_high = boot_metrics['f1']
                jaccard, jaccard_ci_low, jaccard_ci_high = boot_metrics['jaccard']
            else:
                accuracy = accuracy_score(y_true, y_pred)
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
                recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
                f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)
                jaccard = jaccard_score(y_true, y_pred, average=average_method, zero_division=0)
                
                accuracy_ci_low, accuracy_ci_high = accuracy, accuracy
                balanced_acc_ci_low, balanced_acc_ci_high = balanced_acc, balanced_acc
                precision_ci_low, precision_ci_high = precision, precision
                recall_ci_low, recall_ci_high = recall, recall
                f1_ci_low, f1_ci_high = f1, f1
                jaccard_ci_low, jaccard_ci_high = jaccard, jaccard

            # Set metric_type, mean, ci_low, ci_high based on original logic
            if use_balanced_accuracy and type_value in {"binary", "boolean", "categorical", "categorical_number"}:
                metric_type = "balanced_accuracy"
                mean_score = balanced_acc
                ci_low = balanced_acc_ci_low
                ci_high = balanced_acc_ci_high
            else:
                metric_type = "accuracy"
                mean_score = accuracy
                ci_low = accuracy_ci_low
                ci_high = accuracy_ci_high

        if type_value == "string":
            scores = pd.Series([
                calculate_similarity(str(p), str(g), sentence_model)
                for p, g in zip(prediction_field, ground_truth_field)
            ]).to_numpy()
            
            if n_bootstrap > 1:
                rng = np.random.default_rng()
                boot_scores = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(scores), len(scores))
                    boot_scores.append(np.nanmean(scores[idx]))
                
                similarity = np.mean(boot_scores)
                similarity_ci_low, similarity_ci_high = np.percentile(boot_scores, 2.5), np.percentile(boot_scores, 97.5)
            else:
                similarity = np.nanmean(scores)
                similarity_ci_low, similarity_ci_high = similarity, similarity

            metric_type = "semantic_similarity"
            mean_score = similarity
            ci_low = similarity_ci_low
            ci_high = similarity_ci_high

        elif type_value == "list":
            # List similarity - reset classification metrics since they don't apply
            accuracy, accuracy_ci_low, accuracy_ci_high = np.nan, np.nan, np.nan
            balanced_acc, balanced_acc_ci_low, balanced_acc_ci_high = np.nan, np.nan, np.nan
            precision, precision_ci_low, precision_ci_high = np.nan, np.nan, np.nan
            recall, recall_ci_low, recall_ci_high = np.nan, np.nan, np.nan
            f1, f1_ci_low, f1_ci_high = np.nan, np.nan, np.nan
            jaccard, jaccard_ci_low, jaccard_ci_high = np.nan, np.nan, np.nan
            
            prediction_field = prediction_field.apply(
                lambda x: [s for s in x if isinstance(s, str) and s.strip()] if isinstance(x, list) else []
            )
            ground_truth_field = ground_truth_field.apply(
                lambda x: x if isinstance(x, list) else x.split(", ") if isinstance(x, str) else []
            )
            
            scores = pd.Series([
                calculate_list_similarity(p, g, sentence_model)
                for p, g in zip(prediction_field, ground_truth_field)
            ]).to_numpy()
            
            if n_bootstrap > 1:
                rng = np.random.default_rng()
                boot_scores = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(scores), len(scores))
                    boot_scores.append(np.nanmean(scores[idx]))
                
                similarity = np.mean(boot_scores)
                similarity_ci_low, similarity_ci_high = np.percentile(boot_scores, 2.5), np.percentile(boot_scores, 97.5)
            else:
                similarity = np.nanmean(scores)
                similarity_ci_low, similarity_ci_high = similarity, similarity

            metric_type = "list_similarity"
            mean_score = similarity
            ci_low = similarity_ci_low
            ci_high = similarity_ci_high

        results.append({
            "field": field,
            "field_type": type_value,
            "metric_type": metric_type,
            "mean": mean_score,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "accuracy": accuracy,
            "accuracy_CI_low": accuracy_ci_low,
            "accuracy_CI_high": accuracy_ci_high,
            "balanced_accuracy": balanced_acc,
            "balanced_accuracy_CI_low": balanced_acc_ci_low,
            "balanced_accuracy_CI_high": balanced_acc_ci_high,
            "similarity_index": similarity,
            "similarity_index_CI_low": similarity_ci_low,
            "similarity_index_CI_high": similarity_ci_high,
            "precision": precision,
            "precision_CI_low": precision_ci_low,
            "precision_CI_high": precision_ci_high,
            "recall": recall,
            "recall_CI_low": recall_ci_low,
            "recall_CI_high": recall_ci_high,
            "F1": f1,
            "F1_CI_low": f1_ci_low,
            "F1_CI_high": f1_ci_high,
            "Jaccard": jaccard,
            "Jaccard_CI_low": jaccard_ci_low,
            "Jaccard_CI_high": jaccard_ci_high,
            "weight": field_weight,
        })

    return pd.DataFrame(results)

def process_results(
    LLM_output: Union[pd.DataFrame, str], 
    prompt_config: Union[Dict[str, Any], str], 
    ground_truth: Optional[Union[pd.DataFrame, str]] = None, 
    sentence_model: str = "all-mpnet-base-v2", 
    scoring_weights: Optional[List[int]] = None, 
    output_file: Optional[str] = None,
    n_bootstrap: int = 1000,
    use_balanced_accuracy: bool = False,
    strict_metrics: bool = False,
    ) -> None:
    """
    Process LLM output and compare it against ground truth.

    Args:
        LLM_output (Union[pd.DataFrame, str]): LLM output as a DataFrame or path to a CSV file.
        prompt_config (Union[Dict[str, Any], str]): Prompt configuration as a dictionary or path to a YAML file.
        ground_truth (Optional[Union[pd.DataFrame, str]]): Ground truth data as a DataFrame or path to a CSV file.
        sentence_model (str): Sentence transformer model to use for semantic mapping.
        scoring_weights (Optional[List[int]]): Weights for each field used for micro averaging. This overrides the weights defined in the prompt config.
        output_file (Optional[str]): Path to save the results output file.
        n_bootstrap (int): Number of bootstrap samples for CI; if <=1, skip bootstrapping.
        use_balanced_accuracy (bool): If True, use balanced accuracy for categorical/binary fields instead of normal accuracy.
        strict_metrics (bool): If True, exclude entries with default values in ground truth from calculations.
    """
    if isinstance(LLM_output, str):
        LLM_output = pd.read_csv(LLM_output, converters={"extracted_data": safe_literal_eval})

    if isinstance(prompt_config, str):
        prompt_config = read_prompt_config(prompt_config)

    default_dict = {key: value.get("default", "Unknown") for key, value in prompt_config.items()}
    LLM_output[["extracted_data", "missing_json"]] = LLM_output["extracted_data"].apply(
        lambda x: (x, True) if isinstance(x, dict) else (default_dict, False)
    ).apply(pd.Series)

    LLM_output["extracted_data"] = LLM_output["extracted_data"].apply(
        lambda x: {k: v for k, v in x.items() if k in prompt_config}
    )
    
    if ground_truth is not None:
        if isinstance(ground_truth, str):
            ground_truth = pd.read_csv(ground_truth)
        raise ValueError("Ground truth comparison through a different file is not implemented at this moment.")
    else:
        ground_truth = LLM_output.copy()
        missing_fields = [field for field in prompt_config if field not in ground_truth.columns]
        if missing_fields:
            print(f"[WARNING] Missing fields in ground truth: {missing_fields}")
        
        prompt_config = {key: value for key, value in prompt_config.items() if key not in missing_fields}
        ground_truth = ground_truth[prompt_config.keys()]

    prediction = pd.DataFrame(LLM_output["extracted_data"].tolist())
    prediction = prediction[prompt_config.keys()]

    sentence_model = SentenceTransformer(sentence_model)

    results = calculate_results(
        prediction=prediction, 
        ground_truth=ground_truth, 
        prompt_config=prompt_config, 
        sentence_model=sentence_model, 
        weights=scoring_weights,
        n_bootstrap=n_bootstrap,
        use_balanced_accuracy=use_balanced_accuracy,
        strict_metrics=strict_metrics
    )

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
        "--bootstrap",
        type=int,
        default=1,
        help="Number of n_bootstrap, if <= 1 than bootstrapping is not used.",
    )
    parser.add_argument(
        "--balanced-accuracy", action="store_true", help="Do you want to calculate balanced accuracy instead of accuracy."
    )
    parser.add_argument(
        "--strict-metrics", action="store_true", help="Exclude entries with default values from calculations for strict metrics."
    )
    parser.add_argument(
        "-w",
        "--weights",
        nargs='+',
        type=int,
        default=None,
        help="Weights for each field used for micro averaging. This overrides the weights defined in the prompt config."
    )

    args = parser.parse_args()
    process_results(
        LLM_output=args.LLM_output,
        prompt_config=args.prompt_config,
        ground_truth=args.ground_truth,
        sentence_model=args.sentence_model,
        scoring_weights=args.weights,
        output_file=args.output_file,
        n_bootstrap=args.bootstrap,
        use_balanced_accuracy=args.balanced_accuracy,
        strict_metrics=args.strict_metrics
    )