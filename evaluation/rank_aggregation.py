import pandas as pd
from typing import List
from pathlib import Path
import itertools
import networkx as nx
from collections import defaultdict

def pairwise_preferences(votes):
    """
    Given a list of rankings, return the pairwise preference counts.
    """
    prefs = defaultdict(lambda: 0)
    for ranking in votes:
        for i in range(len(ranking)):
            for j in range(i+1, len(ranking)):
                prefs[(ranking[i], ranking[j])] += 1
    return prefs

def ranked_pairs_aggregation(votes):
    """
    Ranked Pairs (Tideman) algorithm.
    votes: list of rankings (lists of sources)
    """
    prefs = pairwise_preferences(votes)
    items = set(itertools.chain(*votes))
    G = nx.DiGraph()

    # Sort edges by strength of victory
    edges = sorted(prefs.items(), key=lambda x: x[1], reverse=True)

    for (winner, loser), weight in edges:
        G.add_edge(winner, loser, weight=weight)
        try:
            # Check for cycles
            cycles = list(nx.find_cycle(G, orientation="original"))
            G.remove_edge(winner, loser)
        except nx.NetworkXNoCycle:
            continue

    ranking = list(nx.topological_sort(G))
    return ranking

def kemeny_young_aggregation(votes):
    """
    Kemeny-Young aggregation: brute-force version for small N.
    """
    items = set(itertools.chain(*votes))
    all_perms = list(itertools.permutations(items))
    best_score = float("inf")
    best_perm = None

    for perm in all_perms:
        score = 0
        for vote in votes:
            for i in range(len(perm)):
                for j in range(i+1, len(perm)):
                    a, b = perm[i], perm[j]
                    if vote.index(a) > vote.index(b):
                        score += 1
        if score < best_score:
            best_score = score
            best_perm = perm

    return list(best_perm)

def rank(LLM_outputs:List[Path], output_file:bool = None, method:str = "borda"):
    """
    Rank aggregation of multiple LLM performance files.
    
    Args:
        LLM_outputs (list): List of paths to the LLM performance files.
        output_file (str, optional): Path to save the aggregated results. Defaults to None.
        method (str, optional): Method for rank aggregation. Defaults to "borda".
            Options are "borda", "kemeny", or "ranked_pairs".
    """
    # Load all LLM outputs into a DataFrame
    if not LLM_outputs:
        raise ValueError("No LLM output files provided.")
    
    if not all(Path(output).exists() for output in LLM_outputs):
        raise FileNotFoundError("One or more LLM output files do not exist.")
    
    results = []
    for output in LLM_outputs:
        if not output.suffix == '.csv':
            raise ValueError(f"Invalid file format: {output}. Expected a CSV file.")
        
        # Read and concatenate all CSV files
        df = pd.read_csv(output)
        df["source"] = output.with_suffix('').stem  # Add a column to identify the source file
        df = df[df["metric_type"] != "macro_avgs"]  # Exclude the "All fields" row
        results.append(df)

    results = pd.concat(results, ignore_index=True)

    ranks = {}
    # Perform rank aggregation based on the specified method
    for field in results["field"].unique():
        field_results = results[results["field"] == field]
        if method == "borda":
            # Borda count method
            field_results['rank'] = field_results['score'].rank(ascending=False, method='min')
            ranks[field] = field_results[['source', 'rank']].set_index('source').to_dict()['rank']
        elif method == "kemeny":
            # Kemedy-Young method
            vote = list(field_results.sort_values("score", ascending=False)["source"])
            final_order = kemeny_young_aggregation([vote])
            rank_map = {source: i + 1 for i, source in enumerate(final_order)}
            field_results["rank"] = field_results["source"].map(rank_map)
            ranks[field] = field_results[["source", "rank"]].set_index('source').to_dict()['rank']
        elif method == "ranked_pairs":
            # Ranked Pairs method
            vote = list(field_results.sort_values("score", ascending=False)["source"])
            final_order = ranked_pairs_aggregation([vote])
            rank_map = {source: i + 1 for i, source in enumerate(final_order)}
            field_results["rank"] = field_results["source"].map(rank_map)
            ranks[field] = field_results[["source", "rank"]].set_index('source').to_dict()['rank']
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
    # Convert ranks to DataFrame
    rank_df = pd.DataFrame.from_dict(ranks, orient='index')
    score_df = rank_df.sum(axis=0).to_frame(name='total_rank')
    score_df['final_rank'] = score_df['total_rank'].rank(ascending=True, method='min')
    score_df = score_df.sort_values('final_rank')
    score_df.index.name = 'source'
    score_df = score_df.reset_index()

    if output_file:
        score_df.to_csv(output_file, index=False)
    else:
        print("Final aggregated scores:")
        print(score_df)
        winner = score_df.index[0]
        print(f"\n🏆 Final Winner: {winner}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rank aggregation of multiple LLM performance.")
    parser.add_argument(
        "-i",
        "--input-files",
        required=True,
        type=Path,
        nargs='+',
        help="Paths to the LLM performance files."
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=None,
        help="Path to save the results output file."
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="kemeny",
        choices=["borda", "kemeny", "ranked_pairs"],
        help="Method for rank aggregation."
    )

    args = parser.parse_args()
    rank(args.input_files, args.output_file, args.method)
