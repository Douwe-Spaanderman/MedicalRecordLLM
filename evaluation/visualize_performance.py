import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional, Tuple, Union, List
import textwrap

# --- Custom Style Parameters ---
palette = ['#66c2a5','#fc8d62','#8da0cb', '#e78ac3']
linewidth = 1
fontsize = 18
subfontsize = 16
tickfontsize = 14
edgecolor = "black"
errorbar_color = "black"
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

def add_ranks(ranked_results: pd.DataFrame, data_labels: List[str]) -> List[str]:
    """
    Add rank information to the data labels based on the ranked results DataFrame.
    
    Args:
        ranked_results: DataFrame containing the ranked results.
        data_labels: List of data labels to which ranks will be added.
    
    Returns:
        List of data labels with rank information appended.
    """

    if ranked_results is not None:
        rank_icons = {
            1: "(1st)",
            2: "(2nd)",
            3: "(3rd)"
        }
        rank_map = ranked_results.set_index('source')['final_rank'].to_dict()
        rank_map = {k: rank_icons.get(v) for k, v in rank_map.items() if v in rank_icons}
        return [f"{label} {rank_map.get(label, '')}" for label in data_labels]
    return data_labels

def plot_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    ylabel: str = "",
    xlabel: str = "",
    wraptext: bool = True,
    ci_lower: Optional[str] = None,
    ci_upper: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    legend: bool = False
) -> plt.Axes:
    """
    Create a barplot with specified parameters.

    Args:
    - data: DataFrame containing the data to plot
    - x: Column name for x-axis
    - y: Column name for y-axis
    - hue: Column name for hue (optional)
    - ylabel: Label for y-axis
    - xlabel: Label for x-axis
    - wraptext: Whether to wrap text in x-axis labels (default True)
    - ax: Matplotlib Axes object to plot on (optional)
    - legend: Whether to show legend (default False)
    Returns:
    - ax: Matplotlib Axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if wraptext:
        data[x] = data[x].apply(lambda x: '\n'.join(textwrap.wrap(str(x), width=15)))

    sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, width=barwidth)
    if ci_lower and ci_upper:
        if hue:
            x_levels = data[x].unique()
            hue_levels = data[hue].unique()
            n_hue = len(hue_levels)
            for i, row in data.iterrows():
                x_idx = list(x_levels).index(row[x])
                hue_idx = list(hue_levels).index(row[hue])
                total_barwidth = barwidth / n_hue
                x_val = x_idx - barwidth / 2 + total_barwidth * (hue_idx + 0.5)
                yval = row[y]
                yerr = [[yval - row[ci_lower]], [row[ci_upper] - yval]]
                ax.errorbar(x=x_val, y=yval, yerr=yerr, fmt='none', ecolor=errorbar_color, capsize=4)
        else:
            for i, row in data.iterrows():
                yval = row[y]
                yerr = [[yval - row[ci_lower]], [row[ci_upper] - yval]]
                ax.errorbar(x=i, y=yval, yerr=yerr, fmt='none', ecolor=errorbar_color, capsize=4)
                
    ax.set_ylabel(ylabel, fontsize=subfontsize, labelpad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelrotation=45, labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    ax.set_title("", fontsize=subfontsize)
    if not legend:
        ax.legend().set_visible(False)
    
    return ax

def plot_metric_summary(
    input_data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]], 
    output_file: Optional[str] = None, 
    weights: Dict[str, float] = None, 
    figsize: Tuple[int] = (15, 5),
    data_labels: Optional[List[str]] = None,
    ranked_results: Optional[Union[str, pd.DataFrame]] = None
) -> None:
    """
    Plot metric summary for single or multiple DataFrames.
    
    Args:
        input_data: Can be a single DataFrame, a list of DataFrames, or a dictionary of DataFrames
        output_file: Path to save the figure (optional)
        weights: Dictionary of weights for fields (optional)
        figsize: Figure size
        data_labels: Labels for each DataFrame (used when input_data is a list/dict)
        ranked_results: Path to a CSV file with ranked results or a DataFrame (optional)
    """
    # Convert input to consistent format (list of DataFrames with labels)
    if isinstance(input_data, pd.DataFrame):
        dfs = [input_data]
        if data_labels is None:
            data_labels = ["Data 1"]
    elif isinstance(input_data, dict):
        dfs = list(input_data.values())
        if data_labels is None:
            data_labels = list(input_data.keys())
    else:  # list
        dfs = input_data
        if data_labels is None:
            data_labels = [f"Data {i+1}" for i in range(len(dfs))]

    if isinstance(ranked_results, str) and ranked_results != 'None':
        ranked_results = pd.read_csv(ranked_results)
    elif isinstance(ranked_results, pd.DataFrame):
        ranked_results = ranked_results
    else:
        ranked_results = None
    
    # Process each DataFrame and add a source label
    processed_dfs = []
    for df, label in zip(dfs, data_labels):
        df = df.copy()
        df['source'] = label
        processed_dfs.append(df)
    
    # Combine all DataFrames
    combined_results = pd.concat(processed_dfs)
    
    # Separate metric types
    avg_df = combined_results[combined_results["metric_type"] == "macro_avgs"]
    acc_df = combined_results[combined_results["metric_type"] == "balanced_accuracy"]
    sim_df = combined_results[combined_results["metric_type"].str.contains("similarity")]
    num_avg = len(avg_df["field"].unique())
    num_acc = len(acc_df["field"].unique())
    num_sim = len(sim_df["field"].unique())

    # --- Dynamic subplot width allocation ---
    num_avg = max(num_avg, 1)
    acc_rel = max(num_acc, 1)
    sim_rel = max(num_sim, 1)
    total_rel = num_avg + acc_rel + sim_rel
    widths = [num_avg / total_rel, acc_rel / total_rel, sim_rel / total_rel]
    
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    # Increase margins by adjusting subplot parameters
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    gs = fig.add_gridspec(1, 3, width_ratios=widths, wspace=0.15)  # Increased wspace

    # --- ax0: Macro-average ---
    ax0 = fig.add_subplot(gs[0])
    plot_barplot(data=avg_df, x="field", y="score", hue="source", ylabel="Macro-Average Score", ci_lower="ci_low", ci_upper="ci_high", ax=ax0, wraptext=True, legend=False)

    # --- ax1: Balanced accuracy ---
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    plot_barplot(data=acc_df, x="field", y="score", hue="source", ylabel="Balanced Accuracy", ci_lower="ci_low", ci_upper="ci_high", ax=ax1, wraptext=True, legend=False)

    # --- ax2: Similarity ---
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    plot_barplot(data=sim_df, x="field", y="score", hue="source", ylabel="Semantic Similarity", ci_lower="ci_low", ci_upper="ci_high", ax=ax2, wraptext=True, legend=False)

    # Shared legend for all axes, placed above the entire figure
    handles, labels = ax1.get_legend_handles_labels()

    # Add rank to labels if ranked results are provided
    labels = add_ranks(ranked_results, labels)
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.10),  # Adjust as needed
        frameon=False,
        ncol=len(data_labels) if data_labels else 1,
    )

    # Fix overlap and adjust layout
    for ax in [ax0, ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.5, dpi=300)  # Increased padding
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize calculated performance results from LLM output against ground truth.")
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
        "-l",
        "--data-labels",
        nargs='+',
        default=None,
        help="Optional labels for each input file (must match number of input files)"
    )
    parser.add_argument(
        "-r",
        "--ranked-results",
        type=str,
        default=None,
        help="Path to a CSV file with ranked results or a DataFrame (optional)"
    )
    args = parser.parse_args()

    # Load input data
    input_data = []
    for file in args.input_files:
        df = pd.read_csv(file)
        input_data.append(df)

    # If labels are provided, validate them
    if args.data_labels:
        if len(args.data_labels) != len(input_data):
            raise ValueError("Number of labels must match number of input files")
        # Convert to dictionary with labels as keys
        input_data = {label: df for label, df in zip(args.data_labels, input_data)}
    else:
        # If single file, keep as DataFrame; if multiple, keep as list
        if len(input_data) == 1:
            input_data = input_data[0]

    # Call the plotting function
    plot_metric_summary(
        input_data=input_data,
        output_file=args.output_file,
        data_labels=args.data_labels if args.data_labels else None,
        ranked_results=args.ranked_results if args.ranked_results else None
    )
