import argparse
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from adjustText import adjust_text
import textwrap
from collections import defaultdict

# --- Custom Style Parameters ---
linewidth = 1
fontsize = 18
subfontsize = 18
tickfontsize = 18
edgecolor = "black"
errorbar_color = "black"
style = "ticks"
barwidth = 0.8
figsize = (8, 4)
title_size = 22
title = False 

custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.edgecolor": edgecolor,
    "patch.linewidth": linewidth,
    "patch.edgecolor": edgecolor,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": tickfontsize,
    "ytick.labelsize": tickfontsize,
    "legend.fontsize": subfontsize,
    "figure.titlesize": title_size,
}
sns.set_theme(style=style, rc=custom_params)

def tsne_text_distribution(
    texts: List[str],
    output_file: Path,
    sample_size: Optional[int] = 1000,
    perplexity: int = 10,
    random_state: int = 42,
) -> None:
    """
    Visualize distribution of free-text fields by projecting sentence embeddings into 2D using t-SNE.

    Args:
        texts (List[str]): List of text entries.
        output_file (Path): Path to save the scatter plot.
        sample_size (int, optional): Number of texts to sample for visualization (default=1000).
        perplexity (int, optional): t-SNE perplexity parameter (default=30).
        random_state (int, optional): Random seed for reproducibility.
    """
    if not texts:
        raise ValueError("Text list is empty, cannot plot distribution.")

    texts_s = pd.Series(texts)
    vc = texts_s.value_counts()

    # Optionally subsample (t-SNE can be slow)
    if sample_size and len(vc) > sample_size:
        rng = np.random.default_rng(random_state)
        texts = rng.choice(vc.index, size=sample_size, replace=False).tolist()
    else:
        texts = vc.index.tolist()

    # Encode texts with SentenceTransformer
    model = SentenceTransformer("embeddinggemma-300m-medical")
    embeddings = model.encode(texts, show_progress_bar=True)

    if perplexity > (len(embeddings) - 1) / 3:
        perplexity = max(5, (len(embeddings) - 1) // 3)
        print(f"Warning: Perplexity too high for dataset size. Setting to {perplexity}.")

    if perplexity > len(np.unique(texts)):
        perplexity = len(np.unique(texts)) - 1
        print(f"Warning: Perplexity too high for number of unique samples. Setting to {perplexity}.")

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    reduced = tsne.fit_transform(embeddings)
    spread = reduced.max(axis=0) - reduced.min(axis=0)
    spread_scale = 0.01  # Scale factor for the spread
    
    duplicated_reduced = []
    duplicated_texts = []

    np.random.seed(random_state+1)
    for i, text in enumerate(texts):
        count = vc[text]
        base_point = reduced[i]
        
        for i in range(count):
            if i == 0:
                duplicated_reduced.append(base_point)
                duplicated_texts.append(text)
            else:
                random_offset = np.random.normal(0, spread_scale, 2) * spread
                duplicated_reduced.append(base_point + random_offset)
                duplicated_texts.append(text)
    
    reduced = np.array(duplicated_reduced)
    texts = duplicated_texts

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], s=20, alpha=0.6, ax=ax)
        
    max_text_len = 15
    texts_short = [
        f"{t[:max_text_len-3]}..." if len(t) > max_text_len else t
        for t in texts
    ]

    max_labels_per_text = 3
    target_labels = 25
    expected_count = vc.clip(upper=max_labels_per_text).sum()

    texts_to_plot = []
    label_counts = defaultdict(int)
    for idx, s in enumerate(texts_short):
        if np.random.random() > target_labels/expected_count:
            continue
        label_counts[s] += 1
        if label_counts[s] > max_labels_per_text:
            continue
        texts_to_plot.append(ax.text(
            reduced[idx, 0], reduced[idx, 1], s,
            fontsize=8, alpha=0.7
        ))
    
    adjust_text(
        texts_to_plot,
        ax=ax,
        objects=ax.collections[0],
        force_text=(0.2, 0.5),
        force_static=0.2,
        force_pull=0.1,
        # force_explode=3,
    )

    sns.despine(offset=20, trim=False)
    plt.yticks([])
    plt.xticks([])
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    if title:
        plt.suptitle(output_file.with_suffix("").name)

    fig.tight_layout(rect=[0.02, 0, 1, 1])
    plt.savefig(output_file, dpi=300)
    plt.close()

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text().replace("-", " ")
        labels.append("\n".join(textwrap.wrap(text, width=width, break_long_words=break_long_words)))
    ax.set_xticklabels(labels, rotation=0, ha="center")

def histogram_plot(data: List[float], length_dataset: int, output_file: Path) -> None:
    """
    Generate a histogram plot with dynamically selected bins for numeric values.

    Args:
        data (List[float]): List of numerical values to plot.
        length_dataset (int): Total number of entries in the dataset.
        output_file (Path): Path to save the output plot image.
    """
    if not data:
        raise ValueError("Data list is empty, cannot plot distribution.")

    fig, ax = plt.subplots(figsize=figsize)

    # Ensure histogram covers full data range
    xmin, xmax = min(data), max(data)

    # Plot histogram with smart binning
    bins = np.arange(xmin, xmax + 2) - 0.5
    sns.histplot(data, bins=bins, edgecolor="black", ax=ax)
    plt.ylim(0, length_dataset)
    plt.yticks([0, length_dataset])
    plt.ylabel("")
    plt.xlim(xmin, xmax)

    fig.tight_layout()
    plt.savefig(output_file)
    plt.close()

def distribution_plot(data: List[float], length_dataset: int, output_file: Path) -> None:
    """
    Generate a distribution plot using matplotlib.

    Args:
        data (List[float]): List of numerical values to plot.
        length_dataset (int): Total number of entries in the dataset.
        output_file (Path): Path to save the output plot image.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data, ax=ax)
    plt.ylim(0, length_dataset)
    plt.yticks([0, length_dataset])
    plt.ylabel("")
    wrap_labels(ax, 7)

    fig.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def summarize_ground_truth(data_path: Path, output_dir: Path, prompt_config: Dict[str, Any]) -> None:
    """
    Analyze and summarize ground truth data for LLM experiments.

    Args:
        data_path (Path): Path to the input CSV or JSON file containing ground truth data.
        output_dir (Path): Directory to save output files.
        prompt_config (Dict[str, Any]): Configuration for prompts.
    """
    # Load data
    if data_path.suffix == '.csv':
        import pandas as pd
        data = pd.read_csv(data_path)
    elif data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    
    # Load prompt configuration
    with open(prompt_config, 'r') as f:
        prompts = yaml.safe_load(f)

    columns = [prompt['name'] for prompt in prompts['field_instructions']]
    length_dataset = len(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, column in enumerate(data.columns):
        if column not in columns:
            continue

        prompt = next((p for p in prompts['field_instructions'] if p['name'] == column), None)
        if not prompt:
            continue

        try:
            values = data[column].fillna(prompt['default']).tolist()
            if not values:
                continue
        except:
            values = data[column].dropna().tolist()

        # Generate plot based on data type
        output_file = output_dir / f"{idx+1}_{column.lower().replace(' ', '_').replace('?', '')}.png"
        type_value = prompt.get('type', 'string').replace("_or_missing", "")
        if prompt.get("options") and not type_value in {"binary", "boolean"}:
            type_value = "categorical_number" if type_value in {"number", "float"} else "categorical"

        if type_value in {"string", "binary", "boolean", "categorical", "categorical_number"}:
            if "options" in prompt:
                # Order values based on options
                options = prompt['options'] + [prompt["default"]]
                options = [str(opt) for opt in options]
                option_set = set(options)
                values = [str(v) if str(v) in option_set else prompt["default"] for v in values]
                values = sorted(values, key=lambda x: options.index(x))
                values = ['No' if v == '' else v for v in values]
                distribution_plot(values, length_dataset, output_file)
            else:
                tsne_text_distribution(values, output_file)
        elif type_value == "string_exact_match":
            tsne_text_distribution(values, output_file)
        elif type_value in {"number", "float"}:
            # Convert to float, ignoring non-convertible entries
            values = []
            for v in data[column]:
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    values.append(0)

            histogram_plot(values, length_dataset, output_file)
        elif type_value == "list":
            values = [v.split(", ") for v in values]
            values = [item for sublist in values for item in sublist]
            tsne_text_distribution(values, output_file)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all LLM experiments.")
    parser.add_argument(
        "--data-path", required=True, type=Path, help="Path to input CSV or JSON file."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to save output files."
    )
    parser.add_argument(
        "--prompt-config",
        required=True,
        type=Path,
        help="YAML file with prompt templates.",
    )

    args = parser.parse_args()
    summarize_ground_truth(args.data_path, args.output_dir, args.prompt_config)