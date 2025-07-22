import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional, Tuple

# --- Custom Style Parameters ---
palette = ['#66c2a5','#fc8d62','#8da0cb', '#e78ac3']
linewidth = 1
fontsize = 18
subfontsize = 16
tickfontsize = 14
edgecolor = "black"
style = "ticks"
barwidth = 0.8

custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.edgecolor": edgecolor,
    "patch.linewidth": linewidth,
    "patch.edgecolor": edgecolor,
}
sns.set_theme(style=style, rc=custom_params, palette=sns.color_palette(palette))

def plot_metric_summary(input_file: str, output_file: Optional[str] = None, weights: Dict[str, float] = None, figsize: Tuple[int]=(15, 5)) -> None:
    """
    Plots a 3-panel summary:
    - ax0: Macro-average (with optional weights)
    - ax1: Accuracy per field
    - ax2: Similarity per field
    """
    # Load results from the input file
    results = pd.read_csv(input_file)

    # Separate metric types
    acc_df = results[results["metric_type"] == "accuracy"]
    sim_df = results[results["metric_type"].str.contains("similarity")]
    num_acc = len(acc_df)
    num_sim = len(sim_df)

    # Macro-average (combined accuracy + similarity)
    if weights:
        results["weight"] = results["field"].map(weights).fillna(1)
        macro_avg = (results["mean"] * results["weight"]).sum() / results["weight"].sum()
    else:
        macro_avg = results["mean"].mean()

    # --- Dynamic subplot width allocation ---
    macro_rel = 1
    acc_rel = max(num_acc, 1)
    sim_rel = max(num_sim, 1)
    total_rel = macro_rel + acc_rel + sim_rel

    widths = [macro_rel / total_rel, acc_rel / total_rel, sim_rel / total_rel]
    
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=widths, wspace=0.1)

    # --- ax0: Macro-average ---
    ax0 = fig.add_subplot(gs[0])
    sns.barplot(x=[""], y=[macro_avg], ax=ax0, width=barwidth)
    ax0.set_ylabel("Macro-Average Score", fontsize=subfontsize, labelpad=10)
    ax0.set_xlabel("")
    ax0.set_ylim(0, 1)
    ax0.tick_params(labelsize=tickfontsize)
    ax0.set_xticks([])
    ax0.set_title("", fontsize=subfontsize)

    # --- ax1: Accuracy ---
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    if num_acc > 0:
        sns.barplot(x="field", y="mean", data=acc_df, ax=ax1, width=barwidth)
        ax1.set_ylabel("Accuracy / Precentile (%)", fontsize=subfontsize)
        ax1.set_xlabel("")
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', labelrotation=45, labelsize=tickfontsize)
        ax1.tick_params(axis='y', labelsize=tickfontsize)
        ax1.set_title("", fontsize=subfontsize)
    else:
        ax1.axis("off")

    # --- ax2: Similarity ---
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    if num_sim > 0:
        sns.barplot(x="field", y="mean", data=sim_df, ax=ax2, width=barwidth)
        ax2.set_ylabel("Mean Semantic Similarity", fontsize=subfontsize)
        ax2.set_xlabel("")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', labelrotation=45, labelsize=tickfontsize)
        ax2.tick_params(axis='y', labelsize=tickfontsize)
        ax2.set_title("", fontsize=subfontsize)
    else:
        ax2.axis("off")

    # Fix overlap
    for ax in [ax0, ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize calculated performance results from LLM output against ground truth.")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str, 
        help="Path to the LLM output file"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Path to save the results output visualization."
    )

    args = parser.parse_args()
    plot_metric_summary(args.input_file, args.output_file)