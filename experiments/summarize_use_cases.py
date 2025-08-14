from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import textwrap
from matplotlib.patches import Patch
from matplotlib.text import Text
import matplotlib.font_manager as fm

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

category_colors = {
    'large': '#66c2a5',
    'medium': '#fc8d62',
    'small': '#8da0cb',
    'specialized': '#e78ac3'
}


custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.edgecolor": edgecolor,
    "patch.linewidth": linewidth,
    "patch.edgecolor": edgecolor,
}
sns.set_theme(style=style, rc=custom_params, palette=sns.color_palette(palette))

model_sizes = {
    'Llama3-Med42-70B': {'category': 'specialized', 'size': 70}, 
    'Llama-3_3-Nemotron-Super-49B-v1': {'category': 'medium', 'size': 49},
    'Llama-4-Scout-17B-16E': {'category': 'medium', 'size': 109}, 
    'Qwen2.5-72B-Instruct': {'category': 'medium', 'size': 72}, 
    'Llama3-OpenBioLLM-70B': {'category': 'specialized', 'size': 70}, 
    'medgemma-27b-it': {'category': 'specialized', 'size': 27}, 
    'gemma-3-27b-it': {'category': 'small', 'size': 27},
    'Mistral-Small-3.1-24B-Instruct-2503': {'category': 'small', 'size': 24}, 
    'DeepSeek-R1-0528-Qwen3-8B': {'category': 'small', 'size': 8}
}

use_case_order = [
    "Liver",
    "Alzheimer",
    "STT_English",
    "STT_Dutch",
    "Melanoma",
    "Colorectal",
    "Unknown",
]

def get_model_color(model_name):
    """Get color for a model based on its category and size"""
    category = model_sizes[model_name]['category']
    size = model_sizes[model_name]['size']
    
    base_color = category_colors[category]
    
    # Convert hex to RGB
    rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # Calculate lightness factor based on size (larger models = darker)
    max_size = max(info['size'] for info in model_sizes.values())
    min_size = min(info['size'] for info in model_sizes.values())
    
    # Normalize size to 0-1 range (1 = largest)
    if max_size != min_size:
        norm_size = (size - min_size) / (max_size - min_size)
    else:
        norm_size = 0.5
    
    # Darken color based on size (larger models get darker)
    factor = 0.7 + 0.3 * (1 - norm_size)  # Range from 0.7 to 1.0
    
    # Apply factor to each RGB component
    new_rgb = tuple(int(min(255, max(0, c * factor))) for c in rgb)
    
    # Convert back to hex
    return '#%02x%02x%02x' % new_rgb

def create_figures(
    data: pd.DataFrame,
    root_dir: Path
) -> None:
    # Filter use_case_order to only include cases present in data
    available_use_cases = [uc for uc in use_case_order if uc in data['Use_Case'].unique()]
    
    # Order data by available use cases
    data['Use_Case'] = pd.Categorical(data['Use_Case'], categories=available_use_cases, ordered=True)
    data = data.sort_values('Use_Case')
    
    # Order models by category and size
    def get_model_order(model_name):
        category_order = {'medium': 0, 'small': 1, 'specialized': 2}
        return (category_order[model_sizes[model_name]['category']], -model_sizes[model_name]['size'])
    
    model_order = sorted(model_sizes.keys(), key=get_model_order)
    data['LLM'] = pd.Categorical(data['LLM'], categories=model_order, ordered=True)
    data = data.sort_values('LLM')

    width_per_bar = 0.6
    fig_height = 5
    
    # Individual use case plots
    for use_case in available_use_cases:
        use_case_data = data[data["Use_Case"] == use_case]
        num_models = use_case_data["LLM"].nunique()
        num_prompting = use_case_data["Prompting Strategy"].nunique()
        fig_width = max(6, num_prompting * num_models * width_per_bar)

        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])

        ax = plot_barplot(
            data=use_case_data,
            x="Prompting Strategy",
            y="Performance",
            hue="LLM",
            ylabel="Micro-Average Score" if all(use_case_data["Metric"] == "micro_avg") else \
            "Macro-Average Score" if all(use_case_data["Metric"] == "macro_avg") else \
            "Micro- and Macro-Average Score",
            xlabel="",
            ci_lower="CI Low",
            ci_upper="CI High",
            ax=ax,
            legend=False,
            category_coloring=True
        )

        # Create and position the custom legend
        create_custom_legend(fig)
        
        plt.savefig(root_dir / f"Use_Case_{use_case}" / "output" / "performance.png", bbox_inches='tight', pad_inches=0.5, dpi=300)

    # Combined plot
    data = data[data["Rank"] == 1]
    if not data.empty:
        num_models = data["LLM"].nunique()
        num_use_cases = len(available_use_cases)
        fig_width = max(6, num_use_cases * num_models * width_per_bar)
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])

        ax = plot_barplot(
            data=data,
            x="Use_Case",
            y="Performance",
            hue="LLM",
            ylabel="Micro-Average Score" if all(data["Metric"] == "micro_avg") else \
            "Macro-Average Score" if all(data["Metric"] == "macro_avg") else \
            "Micro- and Macro-Average Score",
            xlabel="",
            ci_lower="CI Low",
            ci_upper="CI High",
            ax=ax,
            legend=False,
            category_coloring=True
        )

        # Create and position the custom legend
        create_custom_legend(fig)

        plt.savefig(root_dir / "performance.png", bbox_inches='tight', pad_inches=0.5, dpi=300)

def create_custom_legend(fig):
    """Create and position a custom legend with category titles and model lists"""
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    import matplotlib.font_manager as fm
    
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
    col_width = 0.3
    row_height = 0.08
    start_x = 0.1
    start_y = 0.9
    
    for col, category in enumerate(['medium', 'small', 'specialized']):
        if category not in categories:
            continue
        
        # Add category title (bold)
        x_pos = start_x + col * col_width
        y_pos = start_y
        legend_ax.text(
            x_pos, y_pos, 
            category.capitalize(), 
            fontproperties=fm.FontProperties(weight='bold'),
            ha='left', va='center'
        )
        
        # Add models in this category
        for row, (model, size) in enumerate(categories[category]):
            y_pos = start_y - (row + 1) * row_height
            
            # Add color patch
            color = get_model_color(model)
            patch = Rectangle(
                (x_pos, y_pos - 0.015), 
                0.02, 0.02,
                facecolor=color,
                edgecolor=color,
                transform=legend_ax.transAxes
            )
            legend_ax.add_patch(patch)
            
            # Add model label
            legend_ax.text(
                x_pos + 0.03, y_pos,
                model,
                ha='left', va='center',
                fontsize=10,
                transform=legend_ax.transAxes
            )
    
    # Adjust the main plot to make room for the legend
    fig.subplots_adjust(top=0.8)

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
    bar_positions = [patch.get_x() for patch in ax.patches]
    bar_widths = [patch.get_width() for patch in ax.patches]
    xmin = min(bar_positions)-0.2
    xmax = max(x + w for x, w in zip(bar_positions, bar_widths))+0.2
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis='x', labelrotation=45, labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    ax.set_title("", fontsize=subfontsize)
    if not legend:
        ax.legend().set_visible(False)
    
    return ax

def gather_best_performances(root_dir: Path) -> pd.DataFrame:
    """
    Traverse the given root_dir and gather best performing prompting strategy
    performance per LLM per Use Case.
    """
    data_records = []

    # Traverse each use case
    for use_case_dir in root_dir.iterdir():
        output_dir = use_case_dir / "output"
        if not output_dir.is_dir():
            continue

        # Traverse each LLM folder
        for llm_dir in output_dir.iterdir():
            if not llm_dir.is_dir():
                continue

            ranked_file = llm_dir / "ranked_results.csv"
            if not ranked_file.exists():
                continue

            ranked_df = pd.read_csv(ranked_file)
            # Assuming first col = strategy name, second col = score (higher is better)
            strategy = ranked_df.sort_values(
                by="final_rank", ascending=True
            )

            for idx, rank in ranked_df.iterrows():
                perf_file = llm_dir / f"{rank['source']}.performance.csv"
                if not perf_file.exists():
                    continue

                perf_df = pd.read_csv(perf_file)
                perf_df = perf_df[perf_df["metric_type"].isin(["micro_avg", "macro_avg"])]
                if perf_df.empty or len(perf_df) > 1:
                    continue

                perf_df = perf_df.iloc[0]

                data_records.append({
                    "Use_Case": use_case_dir.name.replace("Use_Case_", ""),
                    "LLM": llm_dir.name,
                    "Prompting Strategy": rank["source"],
                    "Rank": int(rank['final_rank']),
                    "Metric": perf_df["metric_type"],
                    "Performance": perf_df["mean"],
                    "CI Low": perf_df["ci_low"],
                    "CI High": perf_df["ci_high"],
                })

    return pd.DataFrame(data_records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate best prompting strategy performance for each LLM per use case."
    )
    parser.add_argument(
        "--root_dir", 
        type=Path,
        default=Path("/home/dspaanderman/Mount/LLM/Experiments"),
        help="Path to the root directory containing use case folders."
    )
    args = parser.parse_args()

    results_df = gather_best_performances(args.root_dir)
    create_figures(results_df, args.root_dir)