import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import numpy as np
import pandas as pd
import sys
import ast
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from metadata import *

try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd().parents[2]

sys.path.insert(0, str(project_root))
from evaluation.rank_aggregation import kemeny_young_aggregation

sns.set_theme(style=style, rc=custom_params, palette=sns.color_palette(palette))

def plot_rank_heatmap(data, top_X: int = None, ax=None, show_usecase_ci: bool = False):
    """
    Plot a heatmap of ranks per LLM and Use Case with optional top-X filtering,
    and show either rank range across prompting strategies or confidence intervals
    (if show_usecase_ci=True).

    Parameters:
    - data: pd.DataFrame containing rankings with columns:
        'LLM Name', 'Use_Case_mapped', 'Prompting Strategy Name', 
        'Use Case Rank', 'Final Rank', 'Final Rank CI Low', 'Final Rank CI High'
    - top_X: Optional, only show top X ranks per use case.
    - ax: Optional matplotlib axis.
    - show_usecase_ci: bool, if True shows [best–worst] rank range per LLM per use case.
    """
    data = data.copy()
    
    if top_X is not None:
        data = data[data["Use Case Rank"] <= top_X]

    # --- Aggregate rank range per LLM per use case ---
    agg_funcs = {
        "Use Case Rank": ["min", "max"]
    }
    top_ranks = (
        data.groupby(["Use_Case_mapped", "LLM Name"])
        .agg(agg_funcs)
        .reset_index()
    )
    top_ranks.columns = ["Use_Case_mapped", "LLM Name", "best_rank", "worst_rank"]

    # --- Pivot for heatmap values (best rank) ---
    heatmap_data = top_ranks.pivot(index="LLM Name", columns="Use_Case_mapped", values="best_rank")

    # --- Prepare annotation text ---
    if show_usecase_ci:
        annot_data = top_ranks.copy()
        annot_data["annot_text"] = annot_data.apply(
            lambda x: f"{int(x['best_rank'])}\n[{int(x['best_rank'])}-{int(x['worst_rank'])}]" 
            if x["best_rank"] != x["worst_rank"] else f"{int(x['best_rank'])}",
            axis=1
        )
        annot_data = annot_data.pivot(index="LLM Name", columns="Use_Case_mapped", values="annot_text")
    else:
        annot_data = heatmap_data.round(0).astype("Int64").astype(str)

    # --- Add separator and final rank + CI ---
    final_rank_idx = data.groupby("LLM Name")["Final Rank"].idxmin()
    final_rank_df = data.loc[final_rank_idx, ["LLM Name", "Final Rank", "Final Rank CI Low", "Final Rank CI High"]].reset_index(drop=True)

    heatmap_data[""] = np.nan  # optional separator
    annot_data[""] = ""  # blank separator
    heatmap_data["Overall Rank (All Use Cases)"] = final_rank_df["Final Rank"].values
    heatmap_data["Overall Rank CI Low"] = final_rank_df["Final Rank CI Low"].values
    heatmap_data["Overall Rank CI High"] = final_rank_df["Final Rank CI High"].values
    annot_data["Overall Rank (All Use Cases)"] = final_rank_df["Final Rank"].astype(int).astype(str).values
    annot_data["Overall Rank CI Low"] = final_rank_df["Final Rank CI Low"].astype(int).astype(str).values
    annot_data["Overall Rank CI High"] = final_rank_df["Final Rank CI High"].astype(int).astype(str).values

    # --- Reverse y-axis (best ranks on top) ---
    heatmap_data = heatmap_data.iloc[::-1]
    annot_data = annot_data.iloc[::-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 20)) if show_usecase_ci else plt.subplots(figsize=(14, 16)) 
    else:
        fig = ax.figure

    cmap = plt.get_cmap("RdYlGn_r")

    sns.heatmap(
        heatmap_data,
        annot=annot_data,
        fmt="",
        linewidths=0,
        square=True,
        cmap=cmap,
        cbar=False,
        ax=ax,
        annot_kws={"size": tickfontsize, "va": "center"}
    )

    # --- Draw custom borders ---
    nrows, ncols = heatmap_data.shape
    xgrid = np.arange(ncols + 1)
    ygrid = np.arange(nrows + 1)
    mask = heatmap_data.isna()

    for r in range(nrows):
        for c in range(ncols):
            val = mask.iloc[r, c]
            top_val = mask.iloc[r - 1, c] if r > 0 else None
            bottom_val = mask.iloc[r + 1, c] if r < nrows - 1 else None
            left_val = mask.iloc[r, c - 1] if c > 0 else None
            right_val = mask.iloc[r, c + 1] if c < ncols - 1 else None

            if val:
                if r > 0 and not top_val:
                    ax.hlines(y=ygrid[r], xmin=c, xmax=c + 1, colors="black", linewidth=2)
                if r < nrows - 1 and not bottom_val:
                    ax.hlines(y=ygrid[r + 1], xmin=c, xmax=c + 1, colors="black", linewidth=2)
                if c > 0 and not left_val:
                    ax.vlines(x=xgrid[c], ymin=r, ymax=r + 1, colors="black", linewidth=2)
                if c < ncols - 1 and not right_val:
                    ax.vlines(x=xgrid[c + 1], ymin=r, ymax=r + 1, colors="black", linewidth=2)
            else:
                if r == 0:
                    ax.hlines(y=ygrid[r], xmin=c, xmax=c + 1, colors="black", linewidth=4)
                elif r == nrows - 1:
                    ax.hlines(y=ygrid[r + 1], xmin=c, xmax=c + 1, colors="black", linewidth=4)
                else:
                    if mask.iloc[r + 1, c]:
                        pass
                    else:
                        ax.hlines(y=ygrid[r + 1], xmin=c, xmax=c + 1, colors="black", linewidth=0.5)

                if c == 0:
                    ax.vlines(x=xgrid[c], ymin=r, ymax=r + 1, colors="black", linewidth=4)
                elif c == ncols - 1:
                    ax.vlines(x=xgrid[c + 1], ymin=r, ymax=r + 1, colors="black", linewidth=4)
                else:
                    if mask.iloc[r, c + 1]:
                        pass
                    else:
                        ax.vlines(x=xgrid[c + 1], ymin=r, ymax=r + 1, colors="black", linewidth=0.5)

    # --- Customize ticks ---
    xticks = ["\n".join(textwrap.wrap(str(tick.get_text()), width=100)) for tick in ax.get_xticklabels()]
    ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=tickfontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tickfontsize)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="both", bottom=False, left=False)
    plt.tight_layout()

    return ax

def _run_bootstrap(args):
    b, data = args
    df_b = data.copy()
    df_b['Boot_Rank'] = df_b['Use Case Rank CI Raw'].apply(lambda x: x[b])
    votes_b = (
        df_b.sort_values(['Use_Case', 'Boot_Rank'])
            .groupby('Use_Case')['Method']
            .apply(list)
            .tolist()
    )
    consensus_b = kemeny_young_aggregation(votes_b, progress=False)
    return consensus_b

def final_rank(data, n_jobs: int = 1, n_bootstrap: bool = None):
    """
    Aggregates Use Case ranks over the combination of (LLM, Prompting Strategy) using Kemeny-Young.
    Computes bootstrap confidence intervals from 'Use Case Rank CI Raw'.
    Returns a DataFrame with Final Rank and CI.
    """

    # --- Ensure 'Use Case Rank CI Raw' is a list, not string ---
    if data['Use Case Rank CI Raw'].dtype == object:
        try:
            sample = data['Use Case Rank CI Raw'].dropna().iloc[0]
            if isinstance(sample, str):
                data['Use Case Rank CI Raw'] = data['Use Case Rank CI Raw'].apply(ast.literal_eval)
        except Exception as e:
            raise ValueError(f"Error parsing 'Use Case Rank CI Raw': {e}")

    # --- Combine LLM and Prompting Strategy as one method ---
    data['Method'] = data['LLM'].astype(str) + " | " + data['Prompting Strategy'].astype(str)

    # --- Determine number of bootstraps ---
    if n_bootstrap is None:
        n_bootstrap = len(data.iloc[0]['Use Case Rank CI Raw'])

    # --- Base consensus over Use Cases ---
    votes_base = (
        data.sort_values(['Use_Case', 'Use Case Rank'])
            .groupby('Use_Case')['Method']
            .apply(list)
            .tolist()
    )
    consensus_base = kemeny_young_aggregation(votes_base, progress=True)

    # --- Parallel bootstrap aggregation using multiprocessing ---
    n_jobs = n_jobs if n_jobs > 0 else cpu_count()
    args_list = [(b, data) for b in range(n_bootstrap)]

    all_bootstrap_consensus = []
    with Pool(n_jobs) as pool:
        for consensus_b in tqdm(pool.imap(_run_bootstrap, args_list), total=n_bootstrap, desc="Bootstrapping Ranking"):
            all_bootstrap_consensus.append(consensus_b)

    # --- Compute CI per method ---
    methods = data['Method'].unique()
    rank_distributions = {m: [] for m in methods}

    for consensus in all_bootstrap_consensus:
        for rank, method in enumerate(consensus, start=1):
            rank_distributions[method].append(rank)

    ci = {
        m: (
            np.percentile(rank_distributions[m], 2.5),
            np.percentile(rank_distributions[m], 97.5)
        )
        for m in methods
    }

    # --- Final DataFrame ---
    final_df = pd.DataFrame({
        'Method': consensus_base,
        'Final Rank': range(1, len(consensus_base) + 1),
        'Final Rank CI Low': [ci[m][0] for m in consensus_base],
        'Final Rank CI High': [ci[m][1] for m in consensus_base]
    })

    # Optionally, split Method back into LLM and Prompting Strategy
    final_df[['LLM', 'Prompting Strategy']] = final_df['Method'].str.split(" \| ", expand=True)

    return final_df[['LLM', 'Prompting Strategy', 'Final Rank', 'Final Rank CI Low', 'Final Rank CI High']]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize calculated performance results from a use case against ground truth.")
    parser.add_argument(
        "-i",
        "--input-files",
        required=True,
        nargs='+',
        help="Path(s) to the LLM output file(s)"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Path to save the results output visualization."
    )
    parser.add_argument(
        "-s",
        "--save-rank",
        type=str,
        default=None,
        help="Path to save the final rank."
    )
    parser.add_argument(
        "-l",
        "--load-rank",
        type=str,
        default=None,
        help="Path to load the final rank."
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=1,
        help="Number of concurrent jobs for bootstrapping."
    )
    args = parser.parse_args()

    from utils import read_rank
    data = read_rank(args.input_files)
    if args.load_rank and Path(args.load_rank).exists():
        final_rank_data = pd.read_csv(args.load_rank)
    else:
        final_rank_data = final_rank(data, args.n_jobs)
        if args.save_rank:
            final_rank_data.to_csv(args.save_rank, index=False)

    data = data.merge(final_rank_data, on=['LLM', 'Prompting Strategy'], how="outer")
    plot_rank_heatmap(data, show_usecase_ci=False)
    plt.savefig(args.output_file, bbox_inches='tight', pad_inches=0.5, dpi=300)
    plt.close()