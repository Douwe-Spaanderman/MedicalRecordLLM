import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import math
import numpy as np
from metadata import *

sns.set_theme(style=style, rc=custom_params, palette=sns.color_palette(palette))

def create_heatmap(plot_data, ax=None, figsize=None):
    """
    Create a heatmap of performance scores for Prompting Strategy × LLM.
    Split rows into two subplots if too many rows.
    
    Parameters
    ----------
    plot_data : pd.DataFrame
        Filtered data for one use case.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates subplots based on row count.
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if subplots were created, None if ax was provided
    """
    # Map names and order
    # For prompting strategies
    plot_data["Prompting Strategy Name"] = plot_data["Prompting Strategy"].map(
        {k: v["name"] for k, v in prompting_strategy_markers.items()}
    )
    categories_order = [v["name"] for v in prompting_strategy_markers.values()]
    plot_data["Prompting Strategy Name"] = pd.Categorical(
        plot_data["Prompting Strategy Name"], 
        categories=categories_order, 
        ordered=True
    )

    # For LLMs
    plot_data["LLM Name"] = plot_data["LLM"].map(
        {k: v["name"] for k, v in model_sizes.items()}
    )
    plot_data["LLM Category"] = plot_data["LLM"].map(
        {k: v["category"] for k, v in model_sizes.items()}
    )
    categories_order = [v["name"] for v in model_sizes.values()]
    plot_data["LLM Name"] = pd.Categorical(
        plot_data["LLM Name"], 
        categories=categories_order, 
        ordered=True
    )

    # Sorting based on new categories
    plot_data = plot_data.sort_values(["LLM Name", "Prompting Strategy Name"])

    # Build composite row labels
    plot_data["row_label"] = plot_data["LLM Name"].astype(str) + " | " + plot_data["Prompting Strategy Name"].astype(str)
    plot_data = plot_data.reset_index(drop=True)

    heatmap_df = plot_data.pivot(
        index=["LLM Name", "Prompting Strategy Name"],
        columns="field",
        values="mean"
    )
    heatmap_df = heatmap_df.reindex(
        pd.MultiIndex.from_product(
            [
                plot_data["LLM Name"].unique(),
                plot_data["Prompting Strategy Name"].cat.categories
            ],
            names=["LLM Name", "Prompting Strategy Name"]
        )
    )

    rank_df = plot_data.pivot(
        index=["LLM Name", "Prompting Strategy Name"],
        columns="field",
        values="Use Case Rank"
    )
    if isinstance(rank_df, pd.DataFrame):
        rank_df = rank_df.iloc[:, 0]

    rank_df = rank_df.reindex(heatmap_df.index)
    cols = list(heatmap_df.columns)
    cols.append(" ")  # add spacer at the end
    heatmap_df = heatmap_df.reindex(columns=cols)
    heatmap_df[" "] = np.nan
    heatmap_df["Rank"] = rank_df

    annot_df = heatmap_df.copy()
    annot_df = annot_df.applymap(lambda x: float(f"{x:.2f}") if pd.notnull(x) else "") # two decimal

    heatmap_df["Rank"] = 1 - (heatmap_df["Rank"] - heatmap_df["Rank"].min()) / (heatmap_df["Rank"].max() - heatmap_df["Rank"].min())

    mask = heatmap_df.isna()  # True for NaNs
    # or specifically mask the " " column
    mask = heatmap_df.columns.to_series().eq(" ")  # returns a boolean Series for columns
    mask = pd.DataFrame(np.repeat(mask.values[np.newaxis, :], heatmap_df.shape[0], axis=0),
                        index=heatmap_df.index, columns=heatmap_df.columns)

    n_rows = len(heatmap_df)
    n_cols = heatmap_df.shape[1]
    n_strats = heatmap_df.index.get_level_values(1).nunique()
    llm_index_order = heatmap_df.index.get_level_values(0).unique()

    if figsize is None:
        fixed_cols = {"Instructions", " ", "Rank"}
        metric_cols = [c for c in heatmap_df.columns if c not in fixed_cols]
        n_metric_cols = len(metric_cols)
        n_fixed_cols = len(fixed_cols)
        # Fixed width for y-axis labels / LLM labels
        total_cols = n_metric_cols + n_fixed_cols
        # Width per column (smaller per-column width)
        col_width = 1.0
        yaxis_width = 4
        fig_width = yaxis_width + total_cols * col_width
        # Figure height proportional to rows
        n_rows = len(heatmap_df)
        fig_height = max(8, n_rows * 0.6)  # minimum height 8 inches
    
    def _decorate_axis(ax, sub_df):
        # set ytick labels to only the strategy part (in the sub_df order)
        ytick_locs = np.arange(len(sub_df)) + 0.5
        strategies = [str(s) for s in sub_df.index.get_level_values(1)]
        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(strategies, rotation=0, fontsize=tickfontsize)

        # add LLM names centered next to their block (computed from sub_df)
        level0 = list(sub_df.index.get_level_values(0))
        unique_llms = pd.Index(level0).unique()

        # Identify spacer columns (named " ")
        spacer_cols = [i for i, c in enumerate(sub_df.columns) if str(c) == " "]
        for i in range(len(sub_df.columns)+1):
            # draw vertical line manually
            ax.vlines(i, *ax.get_ylim(), colors="black", linewidth=0.5)

        for row in range(len(sub_df.index)+1):
            start_x = 0
            for spacer in spacer_cols + [len(sub_df.columns)]:
                ax.hlines(y=row, xmin=start_x, xmax=spacer, colors="black", linewidth=0.5)
                start_x = spacer + 1
                
        for llm in unique_llms:
            idxs = [i for i, x in enumerate(level0) if x == llm]
            start = idxs[0]
            end = idxs[-1]

            llm_text = "\n".join(textwrap.wrap(str(llm), width=15))

            mid = ytick_locs[start:end+1].mean()
            ax.text(
                -4, mid, llm_text,
                ha="right", va="center", fontsize=tickfontsize, fontweight="bold"
            )
            if not end+1 == ax.get_ylim()[0]:
                ax.hlines(y=end + 1, xmin=0, xmax=spacer_cols[0], colors="black", linewidth=3)
                ax.hlines(y=end + 1, xmin=spacer_cols[0] + 1, xmax=len(sub_df.columns)+1, colors="black", linewidth=3)

        ax.set_ylabel("")
        ax.set_xlabel("")

        # wrap x-tick labels
        xticks = ["\n".join(textwrap.wrap(str(tick.get_text()), width=30)) for tick in ax.get_xticklabels()]
        ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=tickfontsize)
        ax.tick_params(axis="both", which="both", bottom=False, left=False)

    # If ax is provided, use single plot
    if ax is not None:
        hm = sns.heatmap(
            heatmap_df,
            cmap="RdYlGn",
            cbar=False,
            annot=True,
            linewidths=0.5,
            linecolor="black",
            ax=ax,
            square=True,
            annot_kws={"size": tickfontsize}
        )
        _decorate_axis(ax, heatmap_df)
        return hm
    if n_rows > 15:
        n_models = len(llm_index_order)
        half_models = math.ceil(n_models / 2)
        left_llms = llm_index_order[:half_models]
        right_llms = llm_index_order[half_models:]

        idx = pd.IndexSlice
        heatmap_df1 = heatmap_df.loc[idx[left_llms, :], :]
        annot_df1 = annot_df.loc[idx[left_llms, :], :]
        mask1 = mask.loc[idx[left_llms, :], :]
        heatmap_df2 = heatmap_df.loc[idx[right_llms, :], :]
        annot_df2 = annot_df.loc[idx[right_llms, :], :]
        mask2 = mask.loc[idx[right_llms, :], :]

        if figsize is None:
            n_models = len(heatmap_df.index.get_level_values(0).unique())
            half_models = math.ceil(n_models / 2)
            fig_height = max(len(heatmap_df[:half_models*len(metric_cols)]), len(heatmap_df[half_models*len(metric_cols):])) * 0.4

            figsize = (fig_width, fig_height)
            
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize
        )

        sns.heatmap(
            heatmap_df1,
            cmap="RdYlGn",
            annot=annot_df1,
            cbar=False,
            ax=ax1,
            square=True,
            annot_kws={"size": 16},
            mask=mask1
        )
        sns.heatmap(
            heatmap_df2,
            cmap="RdYlGn",
            annot=annot_df2,
            cbar=False,
            ax=ax2,
            square=True,
            annot_kws={"size": 16},
            mask=mask2
        )

        _decorate_axis(ax1, heatmap_df1)
        _decorate_axis(ax2, heatmap_df2)
        
        return fig

    # small single-figure case
    fig, ax = plt.subplots(figsize=(n_cols * 1.2, n_rows * 0.8))
    hm = sns.heatmap(
        heatmap_df,
        cmap="RdYlGn",
        cbar=False,
        annot=True,
        linewidths=linewidth,
        linecolor="black",
        ax=ax,
        square=True,
        annot_kws={"size": tickfontsize},
        mask=mask
    )
    _decorate_axis(ax, heatmap_df)
    return fig

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
        "-r",
        "--ranked-results",
        nargs='+',
        help="Path(s) to the CSV file with ranked results"
    )
    args = parser.parse_args()

    from utils import read_performances_and_rank
    data = read_performances_and_rank(args.input_files, args.ranked_results)
    
    create_heatmap(data, figsize=figsizes.get(data["Use_Case"].iloc[0]))
    plt.savefig(args.output_file, bbox_inches='tight', dpi=200)
    plt.close()