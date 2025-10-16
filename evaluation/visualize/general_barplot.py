import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import seaborn as sns
from typing import Optional
import textwrap

from metadata import *

# some global settings
sns.set_theme(style=style, rc=custom_params, palette=sns.color_palette(palette))
width_per_bar = 0.3
fig_height = 5

def get_model_color(model_name):
    """Get color for a model based on its category and size"""
    category = model_sizes[model_name]['category']
    size = model_sizes[model_name]['size']
    
    base_color = category_colors[category]
    
    # Convert hex to RGB
    rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # Calculate lightness factor based on size (larger models = darker)
    max_size = max(info['size'] for info in model_sizes.values() if info['category'] == category)
    min_size = min(info['size'] for info in model_sizes.values() if info['category'] == category)
    
    # Normalize size to 0-1 range (1 = largest)
    if max_size != min_size:
        norm_size = (size - min_size) / (max_size - min_size)
    else:
        norm_size = 0.5
    
    # Darken color based on size (larger models get darker)
    factor = 0.8 + 0.2 * (1 - norm_size)  # Range from 0.7 to 0.9
    
    # Apply factor to each RGB component
    new_rgb = tuple(int(min(255, max(0, c * factor))) for c in rgb)
    
    # Convert back to hex
    return '#%02x%02x%02x' % new_rgb

def create_custom_legend(fig, marker: bool = False, agreement: bool = False, col_width = 0.22, row_height = 0.35, start_x = -0.07, start_y = 2) -> None:
    """Create and position a custom legend with category titles and model lists"""
    # Group models by category and sort by size (descending)
    categories = {}
    for model, info in model_sizes.items():
        category = info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((model, info['size']))
    
    # Sort each category by size (descending)
    for category in categories:
        categories[category].sort(key=lambda x: -x[1])
    
    # Create a new axis for the legend
    legend_ax = fig.add_axes([0.1, 0.85, 0.8, 0.15], frameon=False)
    legend_ax.set_axis_off()
    
    # Create legend elements
    title_spacing = 0.05
    
    for col, category in enumerate(['large', 'medium', 'small', 'tiny', 'specialized']):
        if category not in categories:
            continue
        
        # Add category title (bold)
        x_pos = start_x + col * col_width
        y_pos = start_y
        legend_ax.text(
            x_pos, y_pos + title_spacing, 
            category.capitalize(), 
            fontproperties=fm.FontProperties(weight='bold'),
            transform=legend_ax.transAxes,
            ha='left', va='center', fontsize=fontsize
        )
        
        # Add models in this category
        for row, (model, size) in enumerate(categories[category]):
            y_pos = start_y - (row + 1) * row_height
            
            # Add color patch
            color = get_model_color(model)
            patch = Rectangle(
                (x_pos, y_pos - 0.03), 
                0.02, 0.06,
                facecolor=color,
                edgecolor=color,
                transform=legend_ax.transAxes,
                clip_on=False
            )
            legend_ax.add_patch(patch)
            
            # Add model label
            legend_ax.text(
                x_pos + 0.03, y_pos,
                model_sizes[model]['name'],
                ha='left', va='center',
                transform=legend_ax.transAxes,
                fontsize=fontsize
            )

    # Add legend for prompting strategy markers (2 columns, 3 rows)
    if marker:
        marker_start_x = start_x + 5 * col_width
        marker_col_width = (col_width / 2)
        
        # Add title for prompting strategies
        legend_ax.text(
            marker_start_x + (marker_col_width / 2) - 0.015, start_y + title_spacing,
            "Prompting Strategies",
            fontproperties=fm.FontProperties(weight='bold'),
            transform=legend_ax.transAxes,
            ha='left', va='center', fontsize=fontsize
        )
        
        # Create prompting strategy markers in 2 columns
        strategies = list(prompting_strategy_markers.items())
        
        for i, (strategy_key, strategy_info) in enumerate(strategies):
            # Determine column and row position
            col_idx = i // 3  # 0 for first column, 1 for second column
            row_idx = i % 3   # 0, 1, 2 for rows
            
            x_pos = marker_start_x + col_idx * marker_col_width
            y_pos = start_y - (row_idx + 1) * row_height

            if strategy_key == 'ZeroShot':
                patch = Rectangle(
                    (x_pos, y_pos - 0.03),
                    0.02, 0.06,
                    facecolor='white',
                    edgecolor='black',
                    alpha=0.3,
                    hatch='///',
                    transform=legend_ax.transAxes,
                    clip_on=False
                )
                legend_ax.add_patch(patch)
            else:
                legend_ax.text(
                    x_pos+0.01, y_pos,
                    strategy_info['symbol'],
                    ha='center', va='center',
                    fontsize=10,
                    transform=legend_ax.transAxes
                )

            # Add strategy label using the display name
            legend_ax.text(
                x_pos + 0.03, y_pos,
                strategy_info['name'],
                ha='left', va='center',
                fontsize=fontsize,
                transform=legend_ax.transAxes
            )

    # Add legend for inter-rater agreement lines
    if agreement:
        ira_start_x = start_x + 6 * col_width
        ira_col_width = col_width

        # Add title for inter-rater agreement
        legend_ax.text(
            ira_start_x, start_y + title_spacing,
            "Inter-Rater Agreement",
            fontproperties=fm.FontProperties(weight='bold'),
            transform=legend_ax.transAxes,
            ha='left', va='center', fontsize=fontsize
        )

        # Add legend entries for mean, low CI, and high CI
        ira_labels = [
            ("Mean", "-"),
            ("Confidence Interval", (0, (5, 10))),
        ]

        for i, (label, linestyle) in enumerate(ira_labels):
            y_pos = start_y - (i + 1) * row_height
            x_pos = ira_start_x

            # Draw line
            line = Line2D(
                [x_pos, x_pos + 0.01],
                [y_pos, y_pos],
                color='#dd4040',
                linestyle=linestyle,
                linewidth=1.2,
                transform=legend_ax.transAxes,
                clip_on=False
            )
            legend_ax.add_line(line)

            # Add label
            legend_ax.text(
                x_pos + 0.02, y_pos,
                label,
                ha='left', va='center',
                fontsize=fontsize,
                transform=legend_ax.transAxes
            )

def plot_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    baseline: Optional[pd.DataFrame] = None,
    ylabel: str = "",
    xlabel: str = "",
    wraptext: bool = False,
    ci_lower: Optional[str] = None,
    ci_upper: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    marker: Optional[str] = None,
    legend: bool = False,
    category_coloring: bool = False
) -> plt.Axes:
    """
    Create a barplot with specified parameters.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if wraptext:
        data[x] = data[x].apply(lambda x: '\n'.join(textwrap.wrap(str(x), width=15)))
        if baseline is not None:
            baseline[x] = baseline[x].apply(lambda x: '\n'.join(textwrap.wrap(str(x), width=15)))

    if category_coloring and hue:
        # Get custom palette based on model categories and sizes
        unique_models = data[hue].unique()
        palette = {model: get_model_color(model) for model in unique_models}
    else:
        palette = None

    sns.barplot(
        x=x, 
        y=y, 
        hue=hue, 
        data=data, 
        ax=ax, 
        width=barwidth,
        palette=palette,
        hue_order=data[hue].unique()  # Maintain the order we established
    )

    if baseline is not None:
        sns.barplot(
            x=x,
            y=y,
            hue=hue,
            data=baseline,
            ax=ax,
            width=barwidth,
            palette=palette,
            hatch='/',
            alpha=0.3,
            hue_order=baseline[hue].unique()
        )

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
                if marker:
                    strategy = row[marker]
                    ax.plot(x_val, yval, marker=prompting_strategy_markers[strategy]['marker'], color='black', markersize=8)
        else:
            for i, row in data.iterrows():
                yval = row[y]
                yerr = [[yval - row[ci_lower]], [row[ci_upper] - yval]]
                ax.errorbar(x=i, y=yval, yerr=yerr, fmt='none', ecolor=errorbar_color, capsize=4)
                if marker:
                    strategy = row[marker]
                    ax.plot(i, yval, marker=prompting_strategy_markers[strategy]['marker'], color='black', markersize=8)
                
    ax.set_ylabel(ylabel, fontsize=subfontsize, labelpad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1)
    bar_positions = [patch.get_x() for patch in ax.patches]
    bar_widths = [patch.get_width() for patch in ax.patches]
    desired_spacing = bar_widths[0]
    xmin = min(bar_positions)- desired_spacing
    xmax = max(x + w for x, w in zip(bar_positions, bar_widths)) + desired_spacing
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    # Set inter-rater-agreement if available
    for x_pos, unique_x in enumerate(data[x].unique()):
        tmp = data[data[x] == unique_x].iloc[0]  # All the same, so take the first
        if "Inter-Rater Agreement Mean" in tmp and pd.notna(tmp["Inter-Rater Agreement Mean"]):
            ira_mean = tmp["Inter-Rater Agreement Mean"]
            ira_low = tmp["Inter-Rater Agreement CI Low"]
            ira_high = tmp["Inter-Rater Agreement CI High"]

            # Get the x range for this group (all bars with the same unique_x)
            x_min = x_pos - barwidth / 2 - desired_spacing * 0.5
            x_max = x_pos + barwidth / 2 + desired_spacing * 0.5

            # Plot a horizontal line for the mean IRA
            ax.hlines(
                y=ira_mean,
                xmin=x_min,
                xmax=x_max,
                color="#dd4040",
                linestyle=(0, (5, 5)),
                linewidth=1,
                alpha=0.7,
            )

            # Plot horizontal lines for the confidence interval
            ax.hlines(
                y=ira_low,
                xmin=x_min,
                xmax=x_max,
                color="#dd4040",
                linestyle=(0, (5, 10)),
                linewidth=0.7,
                alpha=0.5,
            )
            ax.hlines(
                y=ira_high,
                xmin=x_min,
                xmax=x_max,
                color="#dd4040",
                linestyle=(0, (5, 10)),
                linewidth=0.7,
                alpha=0.5,
            )

    ax.set_title("", fontsize=subfontsize)
    if not legend:
        ax.legend().set_visible(False)
    
    return ax

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
        help="Path(s) to the CSV file(s) with ranked results"
    )
    parser.add_argument(
        "-in",
        "--inter-rater-agreements",
        nargs='+',
        default=[],
        help="Path(s) to the CSV file(s) with inter-rater-agreements"
    )
    parser.add_argument(
        "-u",
        "--use-case",
        action="store_true",
        help="Wether to create a overall or use case 'general' barplot"
    )
    args = parser.parse_args()

    from utils import read_performances_and_rank
    data = read_performances_and_rank(args.input_files, args.ranked_results, args.inter_rater_agreements)
    if args.use_case:
        num_models = data["LLM"].nunique()
        num_prompting = data["Prompting Strategy"].nunique()
        fig_width = max(6, num_prompting * num_models * width_per_bar)

        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])

        ax = plot_barplot(
            data=data[data["metric_type"] == "macro_avg"],
            x="Prompting Strategy",
            y="mean",
            hue="LLM",
            ylabel="Macro-Average Score",
            xlabel="",
            ci_lower="ci_low",
            ci_upper="ci_high",
            ax=ax,
            legend=False,
            category_coloring=True
        )

        # Create and position the custom legend
        create_custom_legend(fig, col_width = 0.20, row_height = 0.35, start_x = 0.03, start_y = 2)
        
        plt.savefig(args.output_file, bbox_inches='tight', pad_inches=0.5, dpi=300)
        plt.close()
    else:
        if not args.ranked_results:
            raise KeyError("Ranking should have been provided if giving general overview in barplot")

        data = data[data["metric_type"] == "macro_avg"]
        zero = data[data["Prompting Strategy"] == "ZeroShot"].reset_index(drop=True)
        data = data[data["Prompt Rank"] == 1].reset_index(drop=True)
        idx = data.groupby(["Use_Case_mapped", "LLM"], observed=True)["mean"].idxmax()
            
        data = data.loc[idx]
        if not data.empty:
            num_models = data["LLM"].nunique()
            num_use_cases = data["LLM"].nunique()
            fig_width = max(6, num_use_cases * num_models * width_per_bar)
            fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0])

            ax = plot_barplot(
                data=data,
                x="Use_Case_mapped",
                y="mean",
                hue="LLM",
                baseline=zero,
                ylabel="Micro-Average Score" if all(data["metric_type"] == "micro_avg") else \
                "Macro-Average Score" if all(data["metric_type"] == "macro_avg") else \
                "Micro- and Macro-Average Score",
                xlabel="",
                ci_lower="ci_low",
                ci_upper="ci_high",
                ax=ax,
                marker="Prompting Strategy",
                legend=False,
                category_coloring=True
            )

            # Create and position the custom legend
            create_custom_legend(fig, marker=True, agreement=True, col_width = 0.18, row_height = 0.35, start_x = -0.03, start_y = 2)

            plt.savefig(args.output_file, bbox_inches='tight', pad_inches=0.5, dpi=300)
            plt.close()