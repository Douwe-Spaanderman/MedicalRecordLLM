from pathlib import Path
import pandas as pd
import yaml
import os
from tqdm import tqdm
from typing import Dict, Any, Union, List
from itertools import permutations
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")

@lru_cache(maxsize=1)
def load_sentence_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

# Global in-memory lookup for symmetric similarity cache
_similarity_cache = {}

def calculate_similarity(pred: str, gt: str, sentence_model: str) -> float:
    """
    Calculate semantic similarity between two strings using a sentence transformer model.
    Uses a global cache to avoid recomputing repeated string pairs.
    """

    # Normalize input
    pred = (pred or "").strip()
    gt = (gt or "").strip()

    # Handle trivial and missing cases
    if pred == gt and pred != "":
        return 1.0
    if not gt:
        return np.nan
    if not pred:
        return 0.0

    # Create a symmetric cache key
    key = tuple(sorted([pred, gt]))

    # Return cached result if available
    if key in _similarity_cache:
        return _similarity_cache[key]

    # Otherwise compute
    model = load_sentence_model(sentence_model)
    embeddings = model.encode([pred, gt], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    # Cache and return
    _similarity_cache[key] = similarity
    return similarity

def calculate_list_similarity(pred_list: List[str], gt_list: List[str], sentence_model: str) -> float:
    """
    Compute average maximum similarity for each ground truth item against the predicted list.
    
    Args:
        pred_list (List[str]): List of predicted strings.
        gt_list (List[str]): List of ground truth strings.
        sentence_model (str): The sentence transformer model to use.
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

def read_prompt_config(prompt_config_path: str) -> Dict[str, Any]:
    """Read prompt configuration from a YAML file."""
    with open(prompt_config_path, 'r') as file:
        prompt = yaml.safe_load(file)
    
    prompt = prompt.get("field_instructions", None)
    if prompt is None:
        raise ValueError("Prompt configuration does not contain 'field_instructions' key.")
    
    prompt = {item["name"]: {key: value for key, value in item.items() if key != "name"} for item in prompt}
    return prompt

def pairwise_metric(df_field, metric: str, sentence_model: str = None):
    """
    Compute bi-directional pairwise metric per field (across patients).

    Args:
        df_field (pd.DataFrame): rows=patients, columns=Annotator_1..N
        metric (str): "balanced_accuracy", "accuracy", "semantic_similarity", or "list_similarity"
        sentence_model (str): model name for semantic metrics
    
    Returns:
        float: Mean bi-directional pairwise agreement for the field.
    """

    pair_scores = []

    for a_name, b_name in permutations(df_field.columns, 2):
        a_vals, b_vals = df_field[a_name], df_field[b_name]
        valid_mask = ~(a_vals.isna() | b_vals.isna())
        if valid_mask.sum() == 0:
            continue

        a_vals, b_vals = a_vals[valid_mask], b_vals[valid_mask]

        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(b_vals, a_vals)
        elif metric == "accuracy":
            score = accuracy_score(b_vals, a_vals)
        elif metric == "semantic_similarity":
            sims = [
                calculate_similarity(a, b, sentence_model)
                for a, b in zip(a_vals, b_vals)
            ]
            score = np.nanmean(sims)
        elif metric == "list_similarity":
            sims = [
                calculate_list_similarity(a, b, sentence_model)
                for a, b in zip(a_vals, b_vals)
            ]
            score = np.nanmean(sims)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        pair_scores.append(score)

    return np.nanmean(pair_scores)

def bootstrap_pairwise_metric(
    df_field,
    metric: str,
    sentence_model: str = None,
    n_bootstrap: int = 1000,
    random_state: int = 42
):
    """
    Compute bootstrapped bi-directional pairwise metric for a field.
    """

    rng = np.random.default_rng(random_state)
    n_patients = len(df_field)
    bootstrap_scores = []

    for _ in tqdm(range(n_bootstrap), desc=f"Bootstrapping pairwise metric: {metric}"):
        sample_idx = rng.choice(n_patients, n_patients, replace=True)
        df_sample = df_field.iloc[sample_idx]
        score = pairwise_metric(df_sample, metric, sentence_model)
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)
    mean_score = np.nanmean(bootstrap_scores)
    ci_low, ci_high = np.nanpercentile(bootstrap_scores, [2.5, 97.5])

    return {
        "mean": mean_score,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "all_scores": bootstrap_scores,
    }

def _field_type_from_meta(meta: Dict[str, Any]) -> str:
    """Return normalized field type given YAML meta; treat categorical_number as ordinal."""
    type_value = meta.get("type", "string").replace("_or_missing", "")
    # if meta provides options and it's not binary, treat as categorical (or ordinal if numeric options)
    if meta.get("options") and type_value not in {"binary", "boolean"}:
        # if underlying type is numeric treat as ordinal
        type_value = "ordinal" if type_value in {"number", "float", "categorical_number"} else "categorical"
    # explicitly treat categorical_number as ordinal
    if type_value == "categorical_number":
        type_value = "ordinal"
    return type_value

def sanitize_rater_column(series: pd.Series, type_value: str, default_value: Any):
    """
    Clean and normalize a single rater column.
    """
    def try_float(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return x

    def clean_value(x):
        if (isinstance(x, list) and (len(x) == 0 or pd.isna(x).any())) \
           or (not isinstance(x, list) and pd.isna(x)) \
           or (isinstance(x, str) and x.strip().lower() in {"none", "not specified", "not applicable", "missing", "unknown", None}):
            return default_value
        return x

    series = series.apply(clean_value)
    if type_value in {"number", "float", "categorical_number"}:
        series = series.apply(try_float)
    
    if type_value not in "list":
        series = series.astype(str)
    return series

def _build_field_dataframe(raters: List[pd.DataFrame], field: str, type_value: str, default_value: Any) -> pd.DataFrame:
    """
    Build dataframe for a single field with one column per rater.
    Each rater column is sanitized first.
    """
    series_list = []

    for i, r in enumerate(raters):
        if "Patient-ID" not in r.columns:
            raise ValueError(f"Missing required column 'Patient-ID' in rater file index {i+1}.")
        if field not in r.columns:
            warnings.warn(f"Missing required field column '{field}' in rater file index {i+1}. Skipping this rater.")
            continue

        # Sanitize the column
        cleaned_series = sanitize_rater_column(r[field], type_value, default_value)
        # Set index
        s = cleaned_series.rename(f"Annotator_{i+1}")
        s.index = r["Patient-ID"]
        series_list.append(s)

    if not series_list:
        return None

    # Outer join to union Patient-IDs across raters
    df_field = pd.concat(series_list, axis=1, join="outer").sort_index()
    return df_field

def calculate_agreement(raters: List[pd.DataFrame], prompt_config: Dict[str, Any], sentence_model: str, n_bootstrap=1000) -> pd.DataFrame:
    """Calculate inter-rater agreement for each field based on variable type."""
    results = []
    for field, meta in prompt_config.items():
        type_value = _field_type_from_meta(meta)
        default_value = meta.get("default", "Unknown")

        # Build aligned dataframe with one column per rater
        df_field = _build_field_dataframe(raters, field, type_value, default_value)  # index = Patient-ID, columns = Annotator_1, Annotator_2, ...
        if df_field is None or df_field.empty:
            continue

        if type_value in {"binary", "categorical", "boolean", "ordinal"}:
            metric = "balanced_accuracy"
        elif type_value in {"number", "float"}:
            metric = "accuracy"
        elif type_value == "string":
            metric = "semantic_similarity"
        elif type_value == "list":
            metric = "list_similarity"
        else:
            continue

        res = bootstrap_pairwise_metric(df_field, metric, sentence_model, n_bootstrap)

        results.append({
            "field": field,
            "field_type": type_value,
            "metric_type": metric,
            "mean": res["mean"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
            "all_scores": res["all_scores"],
        })

    all_bootstrap_scores = np.stack([r["all_scores"] for r in results])
    macro_per_bootstrap = all_bootstrap_scores.mean(axis=0)
    macro_mean = macro_per_bootstrap.mean()
    ci_low, ci_high = np.percentile(macro_per_bootstrap, [2.5, 97.5])

    results.append({
        "field": "All fields",
        "field_type": "mixed",
        "metric_type": "macro_avg",
        "mean": macro_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "all_scores": macro_per_bootstrap,
    })

    return pd.DataFrame(results)

def run(rater_files: List[Union[str, os.PathLike]], prompt_config_path: Union[str, os.PathLike],
        output_file: Union[str, os.PathLike],  patient_id: str, sentence_model: str):
    """Main entry point for inter-rater agreement analysis."""
    prompt_config = read_prompt_config(prompt_config_path)

    raters = []
    for i, rater_file in enumerate(rater_files):
        df = pd.read_csv(rater_file)
        df["Annotator"] = f"Annotator_{i+1}"
        df = df.drop(columns=["Patient-ID"]) if "Patient-ID" in df.columns and patient_id != "Patient-ID" else df
        df = df.rename(columns={
            patient_id : "Patient-ID"
        })
        raters.append(df)

    results = calculate_agreement(raters, prompt_config, sentence_model)
    results.to_csv(output_file, index=False)

    print(f"Inter-rater agreement results saved to {output_file}")
    print(results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute inter-rater agreement metrics.")
    parser.add_argument(
        "--raters",
        required=True,
        nargs="+",
        type=Path,
        help="List of CSV files, one per rater."
    )
    parser.add_argument(
        "--prompt-config",
        required=True,
        type=Path,
        help="YAML prompt configuration file."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Path to output CSV file."
    )
    parser.add_argument(
        "--patient-id-col", default="Patient-ID", help="Patient ID column name."
    )
    parser.add_argument(
        "-m",
        "--sentence-model",
        type=str,
        default="embeddinggemma-300m-medical",
        help="Sentence transformer model to use for semantic mapping."
    )

    args = parser.parse_args()
    run(args.raters, args.prompt_config, args.output_file, args.patient_id_col, args.sentence_model)