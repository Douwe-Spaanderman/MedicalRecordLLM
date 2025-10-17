from pathlib import Path
import pandas as pd
from typing import Optional, Union, List
import os
import sys

try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd().parents[2]

sys.path.insert(0, str(project_root))
from evaluation.visualize.metadata import *

def get_model_order(model_name):
    category_order = {'large': 0, 'medium': 1, 'small': 2, 'tiny': 3, 'specialized': 4}
    return (category_order[model_sizes[model_name]['category']], -model_sizes[model_name]['size'])

def categorize_and_sort(data):
    available_use_cases = [uc for uc in use_cases.keys() if uc in data["Use_Case"].unique()]

    data["Use_Case_mapped"] = data["Use_Case"].map(use_cases).fillna(data["Use_Case"])
    data["Use_Case_mapped"] = pd.Categorical(
        data["Use_Case_mapped"],
        categories=[use_cases[uc] for uc in available_use_cases],
        ordered=True
    )
    data["Use_Case"] = pd.Categorical(
        data["Use_Case"],
        categories=available_use_cases,
        ordered=True
    )
    data = data.sort_values("Use_Case")

    model_order = sorted(model_sizes.keys(), key=get_model_order)
    data["LLM"] = pd.Categorical(data["LLM"], categories=model_order, ordered=True)
    data = data.sort_values("LLM")

    # Map prompting strategy names and categories
    data["Prompting Strategy Name"] = data["Prompting Strategy"].map(
        {k: v["name"] for k, v in prompting_strategy_markers.items()}
    )
    categories_order = [v["name"] for v in prompting_strategy_markers.values()]
    data["Prompting Strategy Name"] = pd.Categorical(
        data["Prompting Strategy Name"],
        categories=categories_order,
        ordered=True
    )

    # Map LLM names and categories
    data["LLM Name"] = data["LLM"].map({k: v["name"] for k, v in model_sizes.items()})
    data["LLM Category"] = data["LLM"].map({k: v["category"] for k, v in model_sizes.items()})
    data["LLM Name"] = pd.Categorical(
        data["LLM Name"],
        categories=[v["name"] for v in model_sizes.values()],
        ordered=True
    )

    return data

def compare_strategy_preferences(data, metric_type="macro_avg"):
    """
    Compare all strategies to Zero Shot, and summarize:
    1) overall delta (like before),
    2) per-LLM strategy preference,
    3) per-LLM-category strategy preference.
    """
    df = data[data["metric_type"] == metric_type]
    
    # Pivot table for easier calculation
    df_pivot = df.pivot_table(
        index=['Use_Case_mapped', 'LLM Name', 'LLM Category'],
        columns='Prompting Strategy Name',
        values='mean'
    ).reset_index()
    
    strategies = ['One Shot', 'Few Shot', 'Chain-of-Thought', 'Self-Consistency', 'Prompt Graph']
    
    # Compute deltas w.r.t Zero Shot
    for strat in strategies:
        df_pivot[f'delta_{strat}'] = df_pivot[strat] - df_pivot['Zero Shot']
    
    delta_cols = [f'delta_{s}' for s in strategies]
    
    # Melt for general summary (as in your function)
    df_melt = df_pivot.melt(
        id_vars=['Use_Case_mapped', 'LLM Name', 'LLM Category'],
        value_vars=delta_cols,
        var_name='Strategy',
        value_name='mean'
    )
    
    # Overall summary across all LLMs & use cases
    overall_summary = df_melt.groupby(['Strategy'], observed=True).apply(compute_summary).reset_index()
    overall_summary = overall_summary.round(2)
    
    # Per-LLM summary: how each LLM responds to strategies
    llm_summary = df_melt.groupby(['LLM Name', 'Strategy'], observed=True).apply(compute_summary).reset_index()
    llm_summary = llm_summary.round(2)
    
    # Per-LLM-category summary: how categories respond to strategies
    category_summary = df_melt.groupby(['LLM Category', 'Strategy'], observed=True).apply(compute_summary).reset_index()
    category_summary = category_summary.round(2)
    
    return {
        "overall_summary": overall_summary,
        "per_llm_summary": llm_summary,
        "per_category_summary": category_summary
    }

def read_performances_and_rank(
    input_files: List[Union[str, os.PathLike]],
    rank_files: Optional[List[Union[str, os.PathLike]]] = None,
    inter_rater_agreements: Optional[List[Union[str, os.PathLike]]] = None,
    rank_method: str = "kemeny"
) -> pd.DataFrame:
    """
    Read performance data and rank files, then merge and categorize the data.

    Parameters
    ----------
    input_files : List[Union[str, os.PathLike]]
        List of performance data files.
    rank_files : Optional[List[Union[str, os.PathLike]]]
        List of rank files. If None, no rank will be added.
    inter_rater_agreement: Optional[List[Union[str, os.PathLike]]]
        List of inter-rater agreement files. If None, no inter-rater agreement will be added.
    rank_method : str, default="kemeny"
        Method used for ranking.

    Returns
    -------
    pd.DataFrame
        Merged and categorized DataFrame.
    """
    # --- Read and process performance data ---
    data = []
    for input_file in input_files:
        input_file = Path(input_file)
        performance = pd.read_csv(input_file)
        performance = performance[["field", "field_type", "metric_type", "mean", "ci_low", "ci_high"]]
        performance["Prompting Strategy"] = input_file.name.split(".")[0]
        performance["LLM"] = input_file.parent.name
        performance["Use_Case"] = input_file.parents[2].name.replace("Use_Case_", "")
        data.append(performance)
    data = pd.concat(data, ignore_index=True)

    # --- Process rank files if provided ---
    if rank_files:
        use_case_rank = []
        prompt_ranks = []
        for rank_file in rank_files:
            rank_file = Path(rank_file)
            rank = pd.read_csv(rank_file)
            LLM = rank_file.parent.name
            if LLM == "output":
                rank["Use_Case"] = rank_file.parents[1].name.replace("Use_Case_", "")
                rank[["LLM", "Prompting Strategy"]] = rank["source"].str.split(" - ", expand=True)
                rank = rank.drop(columns=["source"])
                rank = rank.rename(columns={
                    rank_method: "Use Case Rank",
                    f"{rank_method}_ci_low": "Use Case Rank CI Low",
                    f"{rank_method}_ci_high": "Use Case Rank CI High",
                })
                rank = rank[["Use_Case", "LLM", "Prompting Strategy", "Use Case Rank", "Use Case Rank CI Low", "Use Case Rank CI High"]]
                use_case_rank.append(rank)
            else:
                rank["Use_Case"] = rank_file.parents[2].name.replace("Use_Case_", "")
                rank["LLM"] = LLM
                rank = rank.rename(columns={
                    "source": "Prompting Strategy",
                    rank_method: "Prompt Rank",
                    f"{rank_method}_ci_low": "Prompt Rank CI Low",
                    f"{rank_method}_ci_high": "Prompt Rank CI High",
                })
                rank = rank[["Use_Case", "LLM", "Prompting Strategy", "Prompt Rank", "Prompt Rank CI Low", "Prompt Rank CI High"]]
                prompt_ranks.append(rank)

        # Merge rank data
        if use_case_rank:
            use_case_rank = pd.concat(use_case_rank, ignore_index=True)
            data = data.merge(
                use_case_rank,
                how="left",
                on=["Use_Case", "LLM", "Prompting Strategy"]
            )
        if prompt_ranks:
            prompt_ranks = pd.concat(prompt_ranks, ignore_index=True)
            data = data.merge(
                prompt_ranks,
                how="left",
                on=["Use_Case", "LLM", "Prompting Strategy"]
            )

    # --- Process inter-rater agreement if provided ---
    if inter_rater_agreements:
        agreement = []
        for inter_rater_agreement in inter_rater_agreements:
            inter_rater_agreement = Path(inter_rater_agreement)
            use_case = inter_rater_agreement.parents[1].name.replace("Use_Case_", "")
            inter_rater_agreement = pd.read_csv(inter_rater_agreement)
            inter_rater_agreement = inter_rater_agreement[["field", "field_type", "metric_type", "mean", "ci_low", "ci_high"]]
            inter_rater_agreement = inter_rater_agreement.rename(columns={
                "mean": "Inter-Rater Agreement Mean",
                "ci_low": "Inter-Rater Agreement CI Low",
                "ci_high": "Inter-Rater Agreement CI High",
            })
            inter_rater_agreement["Use_Case"] = use_case
            agreement.append(inter_rater_agreement)

        if agreement:
            agreement = pd.concat(agreement, ignore_index=True)
            data = data.merge(
                agreement,
                how="left",
                on=["Use_Case", "field", "field_type", "metric_type"]
            )

    # --- Categorize and sort data ---
    return categorize_and_sort(data)

def read_rank(
    rank_files: List[Union[str, os.PathLike]],
    rank_method: str = "kemeny"
) -> pd.DataFrame:
    use_case_rank = []
    prompt_ranks = []
    for rank_file in rank_files:
        rank_file = Path(rank_file)
        rank = pd.read_csv(rank_file)
        LLM = rank_file.parent.name
        if LLM == "output":
            rank["Use_Case"] = rank_file.parents[1].name.replace("Use_Case_", "")
            rank[["LLM", "Prompting Strategy"]] = rank["source"].str.split(" - ", expand=True)
            rank = rank.drop(columns=["source"])
            rank = rank.rename(columns={
                rank_method: "Use Case Rank",
                f"{rank_method}_ci_low": "Use Case Rank CI Low",
                f"{rank_method}_ci_high": "Use Case Rank CI High",
                f"{rank_method}_raw_ranks": "Use Case Rank CI Raw",
            })
            rank = rank[["Use_Case", "LLM", "Prompting Strategy", "Use Case Rank", "Use Case Rank CI Low", "Use Case Rank CI High", "Use Case Rank CI Raw"]]
            use_case_rank.append(rank)
        else:
            rank["Use_Case"] = rank_file.parents[2].name.replace("Use_Case_", "")
            rank["LLM"] = LLM
            rank = rank.rename(columns={
                "source": "Prompting Strategy",
                rank_method: "Prompt Rank",
                f"{rank_method}_ci_low": "Prompt Rank CI Low",
                f"{rank_method}_ci_high": "Prompt Rank CI High",
                f"{rank_method}_raw_ranks": "Prompt Rank CI Raw",
            })
            rank = rank[["Use_Case", "LLM", "Prompting Strategy", "Prompt Rank", "Prompt Rank CI Low", "Prompt Rank CI High", "Prompt Rank CI Raw"]]
            prompt_ranks.append(rank)

    # Concatenate all rank data
    if use_case_rank is not None and prompt_ranks:
        prompt_ranks = pd.concat(prompt_ranks, ignore_index=True)
        use_case_rank = pd.concat(use_case_rank, ignore_index=True)
        data = use_case_rank.merge(prompt_ranks, how="outer", on=["Use_Case", "LLM", "Prompting Strategy"])
    elif use_case_rank is not None:
        data = pd.concat(use_case_rank, ignore_index=True)
    elif prompt_ranks:
        data = pd.concat(prompt_ranks, ignore_index=True)
    else:
        raise ValueError("No rank data found in the provided files.")

    # --- Categorize and sort data ---
    return categorize_and_sort(data)