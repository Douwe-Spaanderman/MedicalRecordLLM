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

    adding_rank = False
    if "Use Case Rank" in plot_data.columns:
        adding_rank = True
        ranks = {
            "Use Case Rank": "Rank",
            "Use Case Rank CI Low": "Rank CI Low",
            "Use Case Rank CI High": "Rank CI High"
        }
        rank_df = {
            new_col: plot_data.pivot(
                index=["LLM Name", "Prompting Strategy Name"],
                columns="field",
                values=old_col
            ).iloc[:, 0].reindex(heatmap_df.index)
            for old_col, new_col in ranks.items()
        }

        cols = list(heatmap_df.columns)
        cols.append(" ")  # add spacer at the end
        heatmap_df = heatmap_df.reindex(columns=cols)
        heatmap_df[" "] = np.nan
        for col in rank_df:
            heatmap_df[col] =  rank_df[col]

    # Add intra-rater annotations
    adding_agreement = False
    if "Inter-Rater Agreement Mean" in plot_data:
        adding_agreement = True
        rater = plot_data[["field", "Inter-Rater Agreement Mean", "Inter-Rater Agreement CI Low", "Inter-Rater Agreement CI High"]]
        rater = rater.drop_duplicates()
        rater = rater.drop_duplicates().set_index("field")
        inter_rater_index = pd.MultiIndex.from_tuples([("Inter-Rater Agreement", "Mean"), ("Inter-Rater Agreement", "CI Low"), ("Inter-Rater Agreement", "CI High")], names=["LLM Name", "Prompting Strategy Name"])
        inter_rater_row = pd.DataFrame(
            index=inter_rater_index,
            columns=heatmap_df.columns
        )
        for field in rater.index:
            if field in heatmap_df.columns:
                inter_rater_row.loc[("Inter-Rater Agreement", "Mean"), field] = rater.loc[field, "Inter-Rater Agreement Mean"]
                inter_rater_row.loc[("Inter-Rater Agreement", "CI Low"), field] = rater.loc[field, "Inter-Rater Agreement CI Low"]
                inter_rater_row.loc[("Inter-Rater Agreement", "CI High"), field] = rater.loc[field, "Inter-Rater Agreement CI High"]

        # Create an empty row with NaN values
        empty_index = pd.MultiIndex.from_tuples([(" ", " ")], names=["LLM Name", "Prompting Strategy Name"])
        empty_row = pd.DataFrame(
            index=empty_index,
            columns=heatmap_df.columns
        )
        empty_row.loc[(" ", " ")] = np.nan

        heatmap_df = pd.concat([inter_rater_row, empty_row, heatmap_df])

    heatmap_df = heatmap_df.astype(float)
    annot_df = heatmap_df.copy()
    annot_df = annot_df.applymap(lambda x: float(f"{x:.2f}") if pd.notnull(x) else "") # two decimal

    if adding_rank:
        rank_min = heatmap_df["Rank"].min()
        rank_max = heatmap_df["Rank"].max()
        for col in ["Rank", "Rank CI Low", "Rank CI High"]:
            heatmap_df[col] = 1 - (heatmap_df[col] - rank_min) / (rank_max - rank_min)

    mask = heatmap_df.isna()  # True for NaNs
    n_rows = len(heatmap_df)
    n_cols = heatmap_df.shape[1]
    n_strats = heatmap_df.index.get_level_values(1).nunique()
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
    
    def _decorate_axis(ax, sub_df, mask):
        """
        Decorate heatmap axis with dynamic gridlines based on mask:
        - Thin lines (0.5) between False cells
        - Thick borders (3) around False edges or True cell borders
        - No lines between adjacent True cells
        - No border for True cells at the edge
        - LLM separator lines skip rows containing True in mask
        """
        ytick_locs = np.arange(len(sub_df)) + 0.5
        strategies = [str(s) for s in sub_df.index.get_level_values(1)]
        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(strategies, rotation=0, fontsize=tickfontsize)

        level0 = list(sub_df.index.get_level_values(0))
        unique_llms = pd.Index(level0).unique()

        nrows, ncols = sub_df.shape
        xgrid = np.arange(ncols + 1)
        ygrid = np.arange(nrows + 1)

        # --- Draw cell borders based on mask ---
        for r in range(nrows):
            for c in range(ncols):
                val = mask.iloc[r, c]

                # Determine neighbor mask values
                top_val = mask.iloc[r-1, c] if r > 0 else None
                bottom_val = mask.iloc[r+1, c] if r < nrows - 1 else None
                left_val = mask.iloc[r, c-1] if c > 0 else None
                right_val = mask.iloc[r, c+1] if c < ncols - 1 else None

                # --- True cells ---
                if val:
                    # Skip borders if neighbor is also True or outside mask
                    if r > 0 and not top_val:
                        ax.hlines(y=ygrid[r], xmin=c, xmax=c+1, colors="black", linewidth=2)
                    if r < nrows - 1 and not bottom_val:
                        ax.hlines(y=ygrid[r+1], xmin=c, xmax=c+1, colors="black", linewidth=2)
                    if c > 0 and not left_val:
                        ax.vlines(x=xgrid[c], ymin=r, ymax=r+1, colors="black", linewidth=2)
                    if c < ncols - 1 and not right_val:
                        ax.vlines(x=xgrid[c+1], ymin=r, ymax=r+1, colors="black", linewidth=2)

                # --- False cells ---
                else:
                    # Horizontal lines
                    # Top edge
                    if r == 0:
                        ax.hlines(y=ygrid[r], xmin=c, xmax=c+1, colors="black", linewidth=4)
                    # Bottom edge
                    elif r == nrows - 1:
                        ax.hlines(y=ygrid[r+1], xmin=c, xmax=c+1, colors="black", linewidth=4)
                    # Internal rows
                    else:
                        if mask.iloc[r+1, c]:
                            pass  # next is True → skip thin line
                        else:
                            ax.hlines(y=ygrid[r+1], xmin=c, xmax=c+1, colors="black", linewidth=0.5)

                    # Vertical lines
                    # Left edge
                    if c == 0:
                        ax.vlines(x=xgrid[c], ymin=r, ymax=r+1, colors="black", linewidth=4)
                    # Right edge
                    elif c == ncols - 1:
                        ax.vlines(x=xgrid[c+1], ymin=r, ymax=r+1, colors="black", linewidth=4)
                    # Internal columns
                    else:
                        if mask.iloc[r, c+1]:
                            pass  # next is True → skip thin line
                        else:
                            ax.vlines(x=xgrid[c+1], ymin=r, ymax=r+1, colors="black", linewidth=0.5)

        # --- LLM group labels ---
        for llm in unique_llms:
            idxs = [i for i, x in enumerate(level0) if x == llm]
            start = idxs[0]
            end = idxs[-1]

            # Compute the mid-point for LLM label
            llm_text = "\n".join(textwrap.wrap(str(llm), width=15))
            mid = ytick_locs[start:end+1].mean()
            ax.text(
                -4, mid, llm_text,
                ha="right", va="center", fontsize=tickfontsize, fontweight="bold"
            )

            # Draw horizontal separator only if not last row
            if not end + 1 == ax.get_ylim()[0]:
                mask_row = mask.iloc[end, :]
                in_segment = False
                seg_start = 0
                for c, val in enumerate(mask_row):
                    if not val and not in_segment:
                        # Start new segment
                        seg_start = c
                        in_segment = True
                    elif val and in_segment:
                        # End segment before True
                        ax.hlines(y=ygrid[end+1], xmin=seg_start, xmax=c, colors="black", linewidth=2)
                        in_segment = False
                # Draw last segment if row ends with False
                if in_segment:
                    ax.hlines(y=ygrid[end+1], xmin=seg_start, xmax=len(mask_row), colors="black", linewidth=2)

        # --- general cleanup ---
        ax.set_ylabel("")
        ax.set_xlabel("")
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
        _decorate_axis(ax, heatmap_df, mask)
        return hm
    if n_rows > 15:
        if adding_agreement:
            n_header_rows = 4
        else:
            n_header_rows = 0

        llm_index_order = heatmap_df[n_header_rows:].index.get_level_values(0).unique()
        n_models = len(llm_index_order)
        half_models = math.ceil(n_models / 2)

        left_llms = llm_index_order[:half_models]
        right_llms = llm_index_order[half_models:]
        idx = pd.IndexSlice
        left_idx = heatmap_df.index[:n_header_rows].append(heatmap_df.loc[idx[left_llms, :], :].index)
        right_idx = heatmap_df.index[:n_header_rows].append(heatmap_df.loc[idx[right_llms, :], :].index)

        heatmap_df1 = heatmap_df.loc[left_idx, :]
        annot_df1 = annot_df.loc[left_idx, :]
        mask1 = mask.loc[left_idx, :]

        heatmap_df2 = heatmap_df.loc[right_idx, :]
        annot_df2 = annot_df.loc[right_idx, :]
        mask2 = mask.loc[right_idx, :]

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

        _decorate_axis(ax1, heatmap_df1, mask1)
        _decorate_axis(ax2, heatmap_df2, mask2)
        
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
    _decorate_axis(ax, heatmap_df, mask)
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
    parser.add_argument(
        "-in",
        "--inter-rater-agreements",
        nargs='+',
        default=[],
        help="Path(s) to the CSV file(s) with inter-rater-agreements"
    )
    args = parser.parse_args()

    from utils import read_performances_and_rank
    data = read_performances_and_rank(args.input_files, args.ranked_results, args.inter_rater_agreements)
    
    create_heatmap(data, figsize=figsizes.get(data["Use_Case"].iloc[0]))
    plt.savefig(args.output_file, bbox_inches='tight', dpi=300)
    plt.close()