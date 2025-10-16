from pathlib import Path
import argparse
from typing import Tuple, List
import subprocess
import shlex

try:
    project_root = Path(__file__).resolve().parents[1]
except NameError:
    project_root = Path.cwd().parent

def gather_data(root_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Traverse the given root_dir and gather performance, ranking,
    and inter-rater agreement files per Use Case.

    Args:
        root_dir (Path): The root directory to search in.

    Returns:
        Tuple[List[str], List[str], List[str]]:
            Lists of file paths (as strings) for performance, ranking,
            and inter-rater agreement CSVs.
    """
    # Find all *performance.csv files in subdirectories
    performance_files = [str(p) for p in root_dir.rglob("*performance.csv")]

    # Find all ranked_results.csv files in subdirectories
    rank_files = [str(p) for p in root_dir.rglob("ranked_results.csv")]

    # Find all inter-rater-agreement.csv files in subdirectories
    inter_rater_agreement_files = [str(p) for p in root_dir.rglob("inter-rater-agreement.csv")]

    return performance_files, rank_files, inter_rater_agreement_files

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
    parser.add_argument(
        "--ranking_method",
        type=str,
        default="kemeny",
        choices=["borda", "kemeny", "ranked_pairs", "wilcoxon_stouffer"],
        help="Method for rank aggregation."
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=1,
        help="Number of concurrent jobs for bootstrapping."
    )
    parser.add_argument(
        "--python-cmd", type=str, default="python", help="Command for python binary."
    )
    args = parser.parse_args()

    performance_files, rank_files, inter_rater_agreement_files = gather_data(args.root_dir)

    # Plot barplot
    command = [
            args.python_cmd, str(project_root / "evaluation" / "visualize" / "general_barplot.py"),
            "-i"
        ] + performance_files + [
            "-o", str(args.root_dir / "performance.png"),
            "-r"
        ] + rank_files
    
    if inter_rater_agreement_files:
        command += ["-in"] + inter_rater_agreement_files

    formatted_cmd = shlex.join([str(arg) for arg in command])
    print(f"Running performance command: {formatted_cmd}")
    subprocess.run(command, check=True)
    print(f"Figure saved: {str(args.root_dir / 'performance.png')}")

    #plot_rank_heatmap
    command = [
        args.python_cmd, str(project_root / "evaluation" / "visualize" / "rank.py"),
        "-i"
    ] + rank_files + [
        "-o", str(args.root_dir / "ranking.png"),
        "-s", str(args.root_dir / "final_rank.csv"),
        "-l", str(args.root_dir / "final_rank.csv"),
        "-j", str(args.n_jobs)
    ]

    formatted_cmd = shlex.join([str(arg) for arg in command])
    print(f"Running ranking command: {formatted_cmd}")
    subprocess.run(command, check=True)
    print(f"Figure saved: {str(args.root_dir / 'ranking.png')}")

    # Perform variance analysis
    output_dir = args.root_dir / "variance_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_cmd,
        str(project_root / "evaluation" / "variance_analysis.py"),
        "-i"
    ] + performance_files + [
        "-r"
    ] + rank_files

    if inter_rater_agreement_files:
        command += ["-in"] + inter_rater_agreement_files

    command += [
        "-o", str(output_dir)
    ]

    formatted_cmd = shlex.join([str(arg) for arg in command])
    print(f"Running prompting variance analysis command:\n{formatted_cmd}")
    subprocess.run(command, check=True)
    print(f"Variance analysis completed. Results saved in: {output_dir}")
