from pathlib import Path
import pandas as pd
import yaml
import os
from typing import Dict, Any, Union, List
from itertools import combinations
import numpy as np
from sklearn.metrics import cohen_kappa_score
from pingouin import intraclass_corr
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
import warnings

@lru_cache(maxsize=1)
def load_sentence_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def calculate_similarity(pred: str, gt: str, sentence_model: str) -> float:
    """
    Calculate semantic similarity between two strings using a sentence transformer model.
    """
    sentence_model = load_sentence_model(sentence_model)

    if pred == gt:
        return 1.0  # Exact match
    if not gt:
        return np.nan
    if not pred or not gt:
        return 0.0

    embeddings = sentence_model.encode([pred, gt], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity


def calculate_list_similarity(pred_list: List[str], gt_list: List[str], sentence_model: str) -> float:
    """
    Compute average maximum similarity for each ground truth item against the predicted list.
    """
    pred_list = [str(p) for p in (pred_list or [])]
    gt_list = [str(g) for g in (gt_list or [])]

    if not gt_list and not pred_list:
        return 1.0
    if not gt_list or not pred_list:
        return 0.0

    def directional_score(source_list, target_list):
        scores = []
        for source_item in source_list:
            similarities = [
                calculate_similarity(source_item, target_item, sentence_model)
                for target_item in target_list
            ]
            scores.append(max(similarities) if similarities else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    gt_to_pred = directional_score(gt_list, pred_list)
    pred_to_gt = directional_score(pred_list, gt_list)
    return (gt_to_pred + pred_to_gt) / 2

def Cohen_kappa(df: pd.DataFrame) -> float:
    """Compute average pairwise Cohen's kappa."""
    raters = df.columns
    scores = []
    for r1, r2 in combinations(raters, 2):
        x, y = df[r1], df[r2]
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            continue

        kappa = cohen_kappa_score(x[mask], y[mask])
        scores.append(kappa)
    return np.nanmean(scores) if scores else np.nan

def icc_numeric(df: pd.DataFrame) -> float:
    """Compute mean pairwise ICC(2,1) between all raters."""
    raters = df.columns
    pairwise_iccs = []

    for r1, r2 in combinations(raters, 2):
        sub = df[[r1, r2]].dropna()
        if len(sub) < 3:
            continue  # not enough paired ratings for ICC

        sub = sub.copy()
        sub["Patient-ID"] = sub.index
        long_df = sub.melt(id_vars="Patient-ID", var_name="rater", value_name="score")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            res = intraclass_corr(
                data=long_df,
                targets="Patient-ID",
                raters="rater",
                ratings="score",
                nan_policy="omit"
            )

        icc_val = res.loc[res["Type"] == "ICC2", "ICC"].values
        if len(icc_val) > 0:
            pairwise_iccs.append(icc_val[0])

    return np.nanmean(pairwise_iccs) if len(pairwise_iccs) > 0 else np.nan

def semantic_similarity_matrix(df_field: pd.DataFrame, sentence_model: str) -> pd.DataFrame:
    """
    Compute pairwise semantic similarity for all annotators on a string field.
    NaN values are ignored in each pairwise comparison.
    Returns a symmetric DataFrame with annotators as both rows and columns.
    """
    annotators = df_field.columns
    n = len(annotators)
    result = pd.DataFrame(np.nan, index=annotators, columns=annotators)

    for r1, r2 in combinations(annotators, 2):
        x = df_field[r1]
        y = df_field[r2]

        # Keep only rows where both raters annotated
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            continue

        similarities = [
            calculate_similarity(str(p), str(g), sentence_model)
            for p, g in zip(x[mask], y[mask])
        ]

        mean_similarity = np.nanmean(similarities)
        result.loc[r1, r2] = mean_similarity
        result.loc[r2, r1] = mean_similarity  # symmetric

    # Fill diagonal with 1.0 (self-agreement)
    np.fill_diagonal(result.values, 1.0)
    return result

def list_similarity_matrix(df_field: pd.DataFrame, sentence_model: str) -> pd.DataFrame:
    """
    Compute pairwise similarity for list-type fields using average max similarity.
    NaNs are ignored. Returns a symmetric DataFrame.
    """
    annotators = df_field.columns
    n = len(annotators)
    result = pd.DataFrame(np.nan, index=annotators, columns=annotators)

    for r1, r2 in combinations(annotators, 2):
        x = df_field[r1]
        y = df_field[r2]

        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            continue

        similarities = [
            calculate_list_similarity(
                p if isinstance(p, list) else str(p).split(", "),
                g if isinstance(g, list) else str(g).split(", "),
                sentence_model
            )
            for p, g in zip(x[mask], y[mask])
        ]

        mean_similarity = np.nanmean(similarities)
        result.loc[r1, r2] = mean_similarity
        result.loc[r2, r1] = mean_similarity  # symmetric

    # Fill diagonal with 1.0
    np.fill_diagonal(result.values, 1.0)
    return result

def read_prompt_config(prompt_config_path: str) -> Dict[str, Any]:
    """Read prompt configuration from a YAML file."""
    with open(prompt_config_path, 'r') as file:
        prompt = yaml.safe_load(file)
    
    prompt = prompt.get("field_instructions", None)
    if prompt is None:
        raise ValueError("Prompt configuration does not contain 'field_instructions' key.")
    
    prompt = {item["name"]: {key: value for key, value in item.items() if key != "name"} for item in prompt}
    return prompt

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
            return float(-1)

    def clean_value(x):
        if (isinstance(x, list) and (len(x) == 0 or pd.isna(x).any())) \
           or (not isinstance(x, list) and pd.isna(x)) \
           or (isinstance(x, str) and x.strip().lower() in {"none", "not specified", "not applicable", "missing", "unknown", None}):
            return default_value
        return x

    series = series.apply(clean_value)
    if type_value in {"number", "float"}:
        series = series.apply(try_float)
    elif type_value not in "list":
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

def calculate_agreement(raters: List[pd.DataFrame], prompt_config: Dict[str, Any], sentence_model: str) -> pd.DataFrame:
    """Calculate inter-rater agreement for each field based on variable type."""
    results = []
    for field, meta in prompt_config.items():
        type_value = _field_type_from_meta(meta)
        default_value = meta.get("default", "Unknown")

        # Build aligned dataframe with one column per rater
        df_field = _build_field_dataframe(raters, field, type_value, default_value)  # index = Patient-ID, columns = Annotator_1, Annotator_2, ...
        if df_field is None or df_field.empty:
            continue

        if type_value in {"binary", "categorical", "boolean", "ordinal", "categorical_number", "string_exact_match"}:
            metric = Cohen_kappa(df_field)
            metric_name = "Cohen's kappa"
        elif type_value in {"number", "float"}:
            metric = icc_numeric(df_field)
            metric_name = "ICC(2,1)"
        elif type_value == "string":
            # Mean pairwise semantic similarity
            sim_matrix = semantic_similarity_matrix(df_field, sentence_model)
            # Take mean of upper triangle (pairwise mean)
            triu_idx = np.triu_indices_from(sim_matrix, k=1)
            metric = np.nanmean(sim_matrix.values[triu_idx])
            metric_name = "Semantic similarity"
        elif type_value == "list":
            # Mean pairwise list similarity
            sim_matrix = list_similarity_matrix(df_field, sentence_model)
            triu_idx = np.triu_indices_from(sim_matrix, k=1)
            metric = np.nanmean(sim_matrix.values[triu_idx])
            metric_name = "List similarity"
        else:
            metric = np.nan
            metric_name = "Unknown"

        results.append({
            "Field": field,
            "Type": type_value,
            "Metric": metric_name,
            "Score": metric.astype(float).round(6)
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