from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from evaluation.visualize_performance import plot_barplot

def create_figures(
    data: pd.DataFrame,
    root_dir: Path
) -> None:
    use_cases = data["Use_Case"].unique()

    width_per_bar = 0.6
    fig_height = 5
    for use_case in use_cases:
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
            legend=False
        )

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.10),
            frameon=False,
            ncol=num_models if num_models else 1
        )
        plt.savefig(root_dir / f"Use_Case_{use_case}" / "output" / "performance.png", bbox_inches='tight', pad_inches=0.5, dpi=300)

    data = data[data["Rank"] == 1]
    num_models = data["LLM"].nunique()
    num_use_cases = data["Use_Case"].nunique()
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
        legend=False
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.10),
        frameon=False,
        ncol=num_models if num_models else 1
    )
    import ipdb; ipdb.set_trace()

    plt.savefig(root_dir / "performance.png", bbox_inches='tight', pad_inches=0.5, dpi=300)

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