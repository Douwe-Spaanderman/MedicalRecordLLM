import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import numpy as np
from metadata import *

sns.set_theme(style=style, rc=custom_params, palette=sns.color_palette(palette))

def plot_rank_heatmap(data, top_X: bool = None, ax = None):
    """
    Plot a heatmap of the (top-X) ranks per LLM and use case.

    Parameters:
    - data: pd.DataFrame containing rankings with columns for LLM, use case, and rank
    """
    # Extract LLM name from prompting strategy if needed
    data = data.copy()
    data = data[["LLM Category", "LLM Name", "Use_Case_mapped", "Prompting Strategy Name", "Prompt Rank", "Use Case Rank", "Final Rank"]]

    if top_X:
        # Filter to top X ranks
        data = data[data["Use Case Rank"] <= top_X]

    # Take best rank per LLM per use case
    top_ranks = data.groupby(["Use_Case_mapped", 'LLM Name'])["Use Case Rank"].min().reset_index()
    heatmap_data = top_ranks.pivot(index="LLM Name", columns='Use_Case_mapped', values="Use Case Rank")

    # Add Final rank
    final_rank = data.groupby("LLM Name")["Final Rank"].min().reset_index()
    heatmap_data[""] = np.nan # Add separator column
    heatmap_data = heatmap_data.merge(final_rank, on="LLM Name", how="left")
    heatmap_data = heatmap_data.set_index("LLM Name")

    # Reverse the order of the rows (LLM Names)
    heatmap_data = heatmap_data.iloc[::-1]  

    use_palette = bool(top_X)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure
        
    # Plot heatmap
    if use_palette:
        # For palette, we need to create a categorical colormap
        from matplotlib.colors import ListedColormap
        categorical_cmap = ListedColormap(palette)
        heatmap = sns.heatmap(heatmap_data, annot=True, linewidths=0,
                             square=True, fmt=".0f", cmap=categorical_cmap, ax=ax, cbar=False, annot_kws={"size": tickfontsize})
    else:
        # For continuous colormap, use cbar and position it at the top
        heatmap = sns.heatmap(heatmap_data, annot=True, linewidths=0,
                             square=True, fmt=".0f", cmap=red_to_white, ax=ax, cbar=False, annot_kws={"size": tickfontsize})
    
    # Manually add borders to all non-separator cells
    # Add left border
    ax.vlines(0, 0, len(heatmap_data), colors="black", linewidth=0.5)

    for i in range(len(heatmap_data.index) + 1):
        start, end = 0, 0
        for j in range(len(heatmap_data.columns) + 1):
            # Skip if this is the separator column
            if j < len(heatmap_data.columns) and heatmap_data.columns[j] == '':
                ax.hlines(len(heatmap_data.index), start, end, colors="black", linewidth=0.5)
                end += 1
                start = end
                continue
                
            # Add rectangle for each cell (except separator)
            rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            end += 1

        ax.hlines(len(heatmap_data.index), start, end, colors="black", linewidth=0.5)

    xticks = ["\n".join(textwrap.wrap(str(tick.get_text()), width=30)) for tick in ax.get_xticklabels()]
    ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=tickfontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tickfontsize)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="both", which="both", bottom=False, left=False)

    if use_palette:
        # Create legend for the discrete colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=palette[i], label=f'Rank {i+1}') for i in range(len(palette))
        ]
        # Add legend at the top
        ax.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, 1.15), ncol=len(palette), 
                 frameon=False)

    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return ax