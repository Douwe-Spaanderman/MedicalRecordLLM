import pandas as pd
from typing import List
from pathlib import Path
import itertools
import networkx as nx
from collections import defaultdict
from glob import glob
import re
from tqdm.auto import tqdm
import math
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

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

def ranked_pairs_aggregation(votes, desc=None):
    """
    Ranked Pairs (Tideman) algorithm.
    votes: list of rankings (lists of sources)
    """
    prefs = pairwise_preferences(votes)
    items = set(itertools.chain(*votes))
    G = nx.DiGraph()

    # Sort edges by strength of victory
    edges = sorted(prefs.items(), key=lambda x: x[1], reverse=True)

    for (winner, loser), weight in tqdm(edges, desc=desc):
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

    n_perm_zeros = math.log10(math.factorial(len(items)))
    if n_perm_zeros > 9:
        raise ValueError(f"Too many votes: {len(items)}. Would result in > 1e{n_perm_zeros:d} permutations. Use a different method.")
    
    all_perms = list(itertools.permutations(items))
    best_score = float("inf")
    best_perm = None

    for perm in tqdm(all_perms):
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

def wilcoxon_stouffer_ranking(results_df, metric="mean", alpha=0.05):
    """
    Perform Wilcoxon signed-rank test with Stouffer's Z-score method for ranking.
    
    Args:
        results_df: DataFrame containing performance metrics for all systems
        metric: Metric to use for comparison
        alpha: Significance level for statistical tests
    
    Returns:
        Dictionary with ranking information and clusters
    """
    systems = results_df["source"].unique()
    n_systems = len(systems)
    
    # Create matrix to store pairwise comparison results
    win_matrix = np.zeros((n_systems, n_systems), dtype=int)
    p_value_matrix = np.zeros((n_systems, n_systems))
    
    # Get all fields
    fields = results_df["field"].unique()
    
    # For each pair of systems, perform Wilcoxon test on each field and combine with Stouffer's method
    for i, sys1 in enumerate(systems):
        for j, sys2 in enumerate(systems):
            if i >= j:
                continue  # Only compare each pair once
                
            p_values = []
            z_scores = []
            
            # Compare performance on each field
            for field in fields:
                sys1_scores = results_df[(results_df["source"] == sys1) & 
                                       (results_df["field"] == field)][metric].values
                sys2_scores = results_df[(results_df["source"] == sys2) & 
                                       (results_df["field"] == field)][metric].values
                
                if len(sys1_scores) > 0 and len(sys2_scores) > 0:
                    try:
                        # Perform Wilcoxon signed-rank test
                        stat, p_val = wilcoxon(sys1_scores, sys2_scores, 
                                             alternative='two-sided', zero_method='zsplit')
                        p_values.append(p_val)
                        # Convert p-value to z-score for Stouffer's method
                        z_scores.append(stats.norm.ppf(1 - p_val/2) * np.sign(sys1_scores.mean() - sys2_scores.mean()))
                    except (ValueError, ZeroDivisionError):
                        # Skip if not enough data or other issues
                        continue
            
            if p_values:
                # Combine p-values using Stouffer's Z-score method
                combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
                combined_p = 2 * (1 - stats.norm.cdf(abs(combined_z)))  # Two-tailed p-value
                
                p_value_matrix[i, j] = combined_p
                p_value_matrix[j, i] = combined_p
                
                # Determine winner based on significance and direction
                if combined_p < alpha:
                    if combined_z > 0:  # sys1 wins
                        win_matrix[i, j] = 1
                        win_matrix[j, i] = -1
                    else:  # sys2 wins
                        win_matrix[i, j] = -1
                        win_matrix[j, i] = 1
    
    # Calculate wins and losses for each system
    wins = np.sum(win_matrix == 1, axis=1)
    losses = np.sum(win_matrix == -1, axis=1)
    
    # Calculate rank ranges
    rank_ranges = []
    for i in range(n_systems):
        # Top end: l + 1, where l is number of losses
        top_rank = losses[i] + 1
        # Bottom end: n - w, where n is total systems, w is number of wins
        bottom_rank = n_systems - wins[i]
        rank_ranges.append((top_rank, bottom_rank))
    
    # Create clusters using hierarchical clustering based on win matrix
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    
    # Convert win matrix to distance matrix (systems that win against each other are closer)
    distance_matrix = 1 - (win_matrix + 1) / 2  # Convert [-1, 0, 1] to [1, 0.5, 0]
    
    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(squareform(distance_matrix), method='average')
    clusters = hierarchy.fcluster(linkage_matrix, t=0.5, criterion='distance')
    
    # Create final ranking (lower cluster number = better rank)
    cluster_ranks = {}
    for cluster_id in np.unique(clusters):
        systems_in_cluster = systems[clusters == cluster_id]
        # Within cluster, sort by mean performance
        cluster_perf = []
        for sys in systems_in_cluster:
            mean_perf = results_df[results_df["source"] == sys][metric].mean()
            cluster_perf.append((sys, mean_perf))
        
        # Sort by performance within cluster
        cluster_perf.sort(key=lambda x: x[1], reverse=True)
        
        for rank_offset, (sys, _) in enumerate(cluster_perf):
            # Assign rank based on cluster and within-cluster position
            cluster_ranks[sys] = cluster_id * 100 + rank_offset  # Scale to preserve cluster ordering
    
    # Convert to final ranks (1 = best)
    final_ranks = {}
    sorted_systems = sorted(cluster_ranks.keys(), key=lambda x: cluster_ranks[x])
    for rank, sys in enumerate(sorted_systems, 1):
        final_ranks[sys] = rank
    
    return {
        'ranks': final_ranks,
        'clusters': dict(zip(systems, clusters)),
        'rank_ranges': dict(zip(systems, rank_ranges)),
        'win_matrix': win_matrix,
        'p_value_matrix': p_value_matrix
    }

def rank(LLM_outputs:List[Path], output_file:bool = None, method:str = "borda", metric:str = "mean"):
    """
    Rank aggregation of multiple LLM performance files.
    
    Args:
        LLM_outputs (list): List of paths to the LLM performance files.
        output_file (str, optional): Path to save the aggregated results. Defaults to None.
        method (str, optional): Method for rank aggregation. Defaults to "borda".
            Options are "borda", "kemeny", or "ranked_pairs".
        metric (str, optional): Metric to use for ranking. Defaults to "mean".
            Options are "mean", "precision", "recall", "f1", etc.
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
        df = df[df["metric_type"] != "micro_avgs"]  # Exclude the "All fields" row
        results.append(df)

    results = pd.concat(results, ignore_index=True)

    ranks = {}
    # Perform rank aggregation based on the specified method
    for field in results["field"].unique():
        field_results = results[results["field"] == field].copy()
        if method == "borda":
            # Borda count method
            field_results['rank'] = field_results[metric].rank(ascending=False, method='min')
            ranks[field] = field_results[['source', 'rank']].set_index('source').to_dict()['rank']
        elif method == "kemeny":
            # Kemedy-Young method
            vote = list(field_results.sort_values(metric, ascending=False)["source"])
            final_order = kemeny_young_aggregation([vote])
            rank_map = {source: i + 1 for i, source in enumerate(final_order)}
            field_results["rank"] = field_results["source"].map(rank_map)
            ranks[field] = field_results[["source", "rank"]].set_index('source').to_dict()['rank']
        elif method == "ranked_pairs":
            # Ranked Pairs method
            vote = list(field_results.sort_values(metric, ascending=False)["source"])
            final_order = ranked_pairs_aggregation([vote], desc=f"Ranking {field:<20}")
            rank_map = {source: i + 1 for i, source in enumerate(final_order)}
            field_results["rank"] = field_results["source"].map(rank_map)
            ranks[field] = field_results[["source", "rank"]].set_index('source').to_dict()['rank']
        elif method == "wilcoxon_stouffer":
            # Wilcoxon signed-rank test with Stouffer's Z-score method
            ranking_result = wilcoxon_stouffer_ranking(field_results, metric=metric, alpha=alpha)
            field_results["rank"] = field_results["source"].map(ranking_result['ranks'])
            ranks[field] = field_results[["source", "rank"]].set_index('source').to_dict()['rank']
            
            # Also store additional information for this method
            ranks[field + '_clusters'] = ranking_result['clusters']
            ranks[field + '_rank_ranges'] = ranking_result['rank_ranges']
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
        choices=["borda", "kemeny", "ranked_pairs", "wilcoxon_stouffer"],
        help="Method for rank aggregation."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean",
        help="Primary metric for rank aggregation."
    )

    args = parser.parse_args()
    rank(args.input_files, args.output_file, args.method, args.metric)
