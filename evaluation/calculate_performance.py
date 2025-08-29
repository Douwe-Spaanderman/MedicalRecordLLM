import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional, List
import ast
import yaml
from scipy.stats import t
from sklearn.metrics import balanced_accuracy_score
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

def calculate_results(
    prediction: pd.DataFrame, 
    ground_truth: pd.DataFrame, 
    prompt_config: Dict[str, Any], 
    sentence_model: SentenceTransformer, 
    weights: Optional[List[int]] = None,
    n_bootstrap: int = 1000,
    use_balanced_accuracy: bool = False,
    exclude_default: bool = False
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
        exclude_default (bool): If True, exclude entries with default values in ground truth from calculations.
    
    Returns:
        pd.DataFrame: DataFrame containing calculated metrics for each field plus micro average.
    """
    # Sanity check
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

        # Exclude default values if specified
        hallucinations = "N/A"
        if exclude_default:
            mask = ground_truth[field] != default_value

            # Calculate hallucinations (i.e. ground truth is default but prediction is not)
            hallucinations = ((ground_truth[field] == default_value) & (prediction[field] != default_value)).sum()

            prediction = prediction[mask].reset_index(drop=True)
            ground_truth = ground_truth[mask].reset_index(drop=True)

        scores = None
        metric_type = None

        if type_value in {"string", "binary", "boolean", "categorical", "categorical_number"}:
            if "options" in meta:
                if type_value in {"binary", "boolean", "categorical_number"}:
                    prediction[field] = pd.to_numeric(prediction[field], errors='coerce', downcast="float")
                    ground_truth[field] = pd.to_numeric(ground_truth[field], errors='coerce', downcast="float")
                    prediction[field] = prediction[field].fillna(default_value).astype(str)
                    ground_truth[field] = ground_truth[field].fillna(default_value).astype(str)

                if use_balanced_accuracy:
                    # Compute balanced accuracy per bootstrap later
                    metric_type = "balanced_accuracy"
                else:
                    scores = (prediction[field] == ground_truth[field]).astype(int).to_numpy()
                    metric_type = "accuracy"
            else:
                # Semantic similarity for free text
                scores = pd.Series([
                    calculate_similarity(str(p), str(g), sentence_model)
                    for p, g in zip(prediction[field], ground_truth[field])
                ]).to_numpy()
                metric_type = "semantic_similarity"

        elif type_value == "string_exact_match":
            scores = (prediction[field] == ground_truth[field]).astype(int).to_numpy()
            metric_type = "accuracy"

        elif type_value in {"number", "float"}:
            prediction[field] = pd.to_numeric(prediction[field], errors='coerce', downcast="float")
            ground_truth[field] = pd.to_numeric(ground_truth[field], errors='coerce', downcast="float")
            prediction[field] = prediction[field].fillna(default_value).astype(str)
            ground_truth[field] = ground_truth[field].fillna(default_value).astype(str)
            scores = (prediction[field] == ground_truth[field]).astype(int).to_numpy()
            metric_type = "accuracy"

        elif type_value == "list":
            print(field)
            prediction[field] = prediction[field].apply(
                lambda x: [s for s in x if isinstance(s, str) and s.strip()] if isinstance(x, list) else []
            )
            ground_truth[field] = ground_truth[field].apply(
                lambda x: x if isinstance(x, list) else x.split(", ") if isinstance(x, str) else []
            )
            scores = pd.Series([
                calculate_list_similarity(p, g, sentence_model)
                for p, g in zip(prediction[field], ground_truth[field])
            ]).to_numpy()
            metric_type = "list_similarity"

        else:
            raise ValueError(f"Unsupported field type for field '{field}': {meta.get('type')}")

        # Bootstrapping for CI
        if n_bootstrap > 1:
            rng = np.random.default_rng()
            boot_means = []
            if use_balanced_accuracy and metric_type == "balanced_accuracy":
                # True balanced accuracy bootstrap
                gt_arr = np.array(ground_truth[field])
                pred_arr = np.array(prediction[field])
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(gt_arr), len(gt_arr))
                    try:
                        ba = balanced_accuracy_score(gt_arr[idx], pred_arr[idx])
                        boot_means.append(ba)
                    except ValueError:
                        pass  # occurs if bootstrap sample has one class only
            else:
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(scores), len(scores))
                    boot_means.append(np.nanmean(scores[idx]))
            mean_score = np.nanmean(boot_means) if boot_means else np.nan
            ci = (np.nanpercentile(boot_means, 2.5), np.nanpercentile(boot_means, 97.5)) if boot_means else (np.nan, np.nan)
        else:
            if use_balanced_accuracy and metric_type == "balanced_accuracy":
                raise ValueError("Impossible to provide use balanced accuracy without bootstrapping.")
            else:
                n = np.isfinite(scores).sum()
                mean_score = np.nanmean(scores)
                std_err = np.nanstd(scores, ddof=1) / np.sqrt(n) if n > 1 else 0
                ci = t.interval(0.95, df=n-1, loc=mean_score, scale=std_err) if n > 1 else (mean_score, mean_score)

        results.append({
            "field": field,
            "metric_type": metric_type,
            "mean": mean_score,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "weight": field_weight,
            "scores": scores,
            "hallucinations": hallucinations if exclude_default else None
        })

    if results:
        results_df = pd.DataFrame(results)

        if use_balanced_accuracy:
            # --- Macro average ---
            if weights is None:
                macro_avg = np.nanmean(results_df["mean"])
            else:
                macro_avg = np.average(results_df["mean"], weights=results_df["weight"])

            if n_bootstrap > 1:
                rng = np.random.default_rng()
                boot_macro = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(results_df), len(results_df))
                    if weights is None:
                        boot_macro.append(np.nanmean(results_df["mean"].iloc[idx]))
                    else:
                        boot_macro.append(
                            np.average(results_df["mean"].iloc[idx], weights=results_df["weight"].iloc[idx])
                        )
                ci_macro = (np.nanpercentile(boot_macro, 2.5), np.nanpercentile(boot_macro, 97.5))
            else:
                std_err = np.nanstd(results_df["mean"], ddof=1) / np.sqrt(len(results_df))
                ci_macro = t.interval(0.95, df=len(results_df)-1, loc=macro_avg, scale=std_err)

            avg_results = {
                "field": "All Fields",
                "metric_type": "macro_avg",
                "mean": macro_avg,
                "ci_low": ci_macro[0],
                "ci_high": ci_macro[1],
                "weight": None,
                "scores": None,
                "hallucinations": None
            }
        else:
            # --- Micro average ---
            scores = np.stack(results_df["scores"].values)  # shape: (fields, samples)
            patient_means = np.nanmean(scores, axis=0)  # mean per patient
            micro_avg = np.nanmean(patient_means)

            if n_bootstrap > 1:
                rng = np.random.default_rng()
                boot_micro = [
                    np.nanmean(np.nanmean(scores[:, idx], axis=0))
                    for idx in (rng.integers(0, scores.shape[1], scores.shape[1]) for _ in range(n_bootstrap))
                ]
                ci_micro = (np.nanpercentile(boot_micro, 2.5), np.nanpercentile(boot_micro, 97.5))
            else:
                std_err = np.nanstd(patient_means, ddof=1) / np.sqrt(len(patient_means))
                ci_micro = t.interval(0.95, df=len(patient_means)-1, loc=micro_avg, scale=std_err)

            avg_results = {
                "field": "All Fields",
                "metric_type": "micro_avg",
                "mean": micro_avg,
                "ci_low": ci_micro[0],
                "ci_high": ci_micro[1],
                "weight": None,
                "scores": None,
                "hallucinations": None
            }

        return pd.concat([results_df, pd.DataFrame([avg_results])], ignore_index=True)
    return None

def process_results(
    LLM_output: Union[pd.DataFrame, str], 
    prompt_config: Union[Dict[str, Any], str], 
    ground_truth: Optional[Union[pd.DataFrame, str]] = None, 
    sentence_model: str = "all-mpnet-base-v2", 
    scoring_weights: Optional[List[int]] = None, 
    output_file: Optional[str] = None,
    n_bootstrap: int = 1000,
    use_balanced_accuracy: bool = False,
    exclude_default: bool = False
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
        exclude_default (bool): If True, exclude entries with default values in ground truth from calculations.
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
    results = calculate_results(
        prediction=prediction, 
        ground_truth=ground_truth, 
        prompt_config=prompt_config, 
        sentence_model=sentence_model, 
        weights=scoring_weights, 
        use_balanced_accuracy=use_balanced_accuracy,
        n_bootstrap=n_bootstrap
    )

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
        "--bootstrap",
        type=int,
        default=1,
        help="Number of n_bootstrap, if <= 1 than bootstrapping is not used.",
    )
    parser.add_argument(
        "--balanced-accuracy", action="store_true", help="Do you want to calculate balanced accuracy instead of accuracy."
    )
    parser.add_argument(
        "--exclude-default", action="store_true", help="Exclude entries with default values in ground truth from calculations."
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
        use_balanced_accuracy=args.balanced_accuracy
    )