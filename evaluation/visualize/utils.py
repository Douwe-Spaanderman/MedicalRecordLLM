from pathlib import Path
import pandas as pd
from typing import Optional, Union, List
import os

from metadata import *

def get_model_order(model_name):
    category_order = {'large': 0, 'medium': 1, 'small': 2, 'tiny': 3, 'specialized': 4}
    return (category_order[model_sizes[model_name]['category']], -model_sizes[model_name]['size'])

from pathlib import Path
from typing import List, Union, Optional
import pandas as pd

def read_performances_and_rank(
    input_files: List[Union[str, os.PathLike]],
    rank_files: Optional[List[Union[str, os.PathLike]]] = None,
    inter_rater_agreement: Optional[Union[str, os.PathLike]] = None,
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
    inter_rater_agreement: Optional[Union[str, os.PathLike]]
        Path to inter-rater agreement file. If None, no inter-rater agreement will be added.
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

    # --- Categorize and sort data ---
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

    # --- Process rank files if provided ---
    if rank_files:
        use_case_rank = None
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
                use_case_rank = rank.copy()
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
                rank = rank.drop(columns=[f"{rank_method}_raw_ranks"])
                prompt_ranks.append(rank)

        # Merge rank data
        if use_case_rank is not None:
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
    if inter_rater_agreement:
        inter_rater_agreement = Path(inter_rater_agreement)
        inter_rater_agreement = pd.read_csv(inter_rater_agreement)
        inter_rater_agreement = inter_rater_agreement[["field", "field_type", "metric_type", "mean", "ci_low", "ci_high"]]
        inter_rater_agreement = inter_rater_agreement.rename(columns={
            "mean": "Inter-Rater Agreement Mean",
            "ci_low": "Inter-Rater Agreement CI Low",
            "ci_high": "Inter-Rater Agreement CI High",
        })
        data = data.merge(
            inter_rater_agreement,
            how="left",
            on=["field", "field_type", "metric_type"]
        )

    return data
