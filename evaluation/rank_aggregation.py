import pandas as pd
from typing import List
from pathlib import Path
import itertools
import networkx as nx
from collections import defaultdict
from tqdm.auto import tqdm
import math
import numpy as np
import random
from scipy import stats
from scipy.stats import wilcoxon
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import ast
import re
import warnings
from multiprocessing import Pool

def safe_literal_eval(x):
    try:
        if pd.isna(x) or str(x).strip() in ['[]', '', 'nan']:
            return np.array([])
        
        # Ensure x is a string
        s = str(x).strip()
        
        # Replace multiple spaces with commas (but not those after '[' or before ']')
        s = re.sub(r'\s+', ',', s.strip('[]'))
        s = f"[{s}]"
        
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return np.array(parsed)
        else:
            return np.array([])
    except (ValueError, SyntaxError, TypeError):
        return np.array([])

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

def kemeny_young_aggregation(votes, max_iter=10000, initial_temp=1000, cooling_rate=0.99):
    """
    Kemeny-Young aggregation: brute-force for small N, simulated annealing for large N.
    Stops early if the optimal solution (score = 0) is found.
    """
    items = sorted(set(itertools.chain(*votes)))
    n = len(items)
    n_perm_zeros = math.log10(math.factorial(n))

    if n_perm_zeros <= 9:
        # Use brute-force for small N
        all_perms = list(itertools.permutations(items))
        best_score = float("inf")
        best_perm = None
        for perm in tqdm(all_perms, desc="Brute-force search"):
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
    else:
        # Use simulated annealing for large N
        current_perm = items.copy()
        random.shuffle(current_perm)
        current_score = compute_kemeny_young_score(current_perm, votes)

        temp = initial_temp
        for iteration in tqdm(range(max_iter), desc="Simulated annealing"):
            # Generate a neighbor
            i, j = random.sample(range(n), 2)
            new_perm = current_perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            new_score = compute_kemeny_young_score(new_perm, votes)

            # Check if the new solution is optimal
            if new_score == 0:
                return new_perm

            # Calculate the change in score
            delta = new_score - current_score

            # Accept the new solution if it's better or with a probability if it's worse
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_perm, current_score = new_perm, new_score
                # Check if the current solution is optimal
                if current_score == 0:
                    return current_perm

            # Cool down
            temp *= cooling_rate

        return current_perm

def compute_kemeny_young_score(perm, votes):
    """Compute the Kemeny-Young score for a given permutation."""
    score = 0
    for vote in votes:
        for i in range(len(perm)):
            for j in range(i+1, len(perm)):
                a, b = perm[i], perm[j]
                if vote.index(a) > vote.index(b):
                    score += 1
    return score

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
                        stat, p_val = wilcoxon(sys1_scores[0], sys2_scores[0], 
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
                else:
                    # No significant difference (tie)
                    win_matrix[i, j] = 0
                    win_matrix[j, i] = 0
    
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
    
    # Create a proper distance matrix for clustering
    # We need to ensure symmetry - use the absolute win differences
    similarity_matrix = np.zeros((n_systems, n_systems))
    
    for i in range(n_systems):
        for j in range(n_systems):
            if i != j:
                # Convert win/loss to similarity (1 for win, 0.5 for tie, 0 for loss)
                if win_matrix[i, j] == 1:
                    similarity_matrix[i, j] = 1
                elif win_matrix[i, j] == 0:
                    similarity_matrix[i, j] = 0.5
                else:
                    similarity_matrix[i, j] = 0
    
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Ensure the matrix is symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Perform hierarchical clustering
    try:
        linkage_matrix = hierarchy.linkage(squareform(distance_matrix), method='average')
        clusters = hierarchy.fcluster(linkage_matrix, t=0.5, criterion='distance')
    except:
        # Fallback: if distance matrix issues persist, assign each system to its own cluster
        clusters = np.arange(1, n_systems + 1)
    
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
            cluster_ranks[sys] = cluster_id * 100 + rank_offset
    
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

def bootstrap_iteration(args):
    b, boot_df, method = args
    boot_df["mean"] = boot_df["all_scores"].apply(
        lambda x: x[b] if isinstance(x, np.ndarray) and len(x) > b else np.nan
    )

    # Check for NaN values and issue a warning if found
    if boot_df["mean"].isna().any():
        warnings.warn("NaN values detected in the 'mean' column after bootstrap sampling. Returning NaN.")
        return np.nan

    return aggregate_once(boot_df, method)

def aggregate_once(df, method):
    if method == "borda":
        df["rank"] = df.groupby("field")["mean"].rank(ascending=False, method="min").astype(int)
        mean_ranks = df.groupby("source")["rank"].mean().rank().astype(int)
        return mean_ranks.to_dict()
    elif method == "kemeny":
        votes = []
        for field, field_results in df.groupby("field"):
            vote = list(field_results.sort_values("mean", ascending=False)["source"])
            votes.append(vote)
        final_order = kemeny_young_aggregation(votes)
        return {src: i + 1 for i, src in enumerate(final_order)}
    elif method == "ranked_pairs":
        votes = []
        for field, field_results in df.groupby("field"):
            vote = list(field_results.sort_values("mean", ascending=False)["source"])
            votes.append(vote)
        final_order = ranked_pairs_aggregation(votes, desc="Ranked Pairs")
        return {src: i + 1 for i, src in enumerate(final_order)}
    elif method == "wilcoxon_stouffer":
        return {}
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def rank(LLM_outputs:List[Path], output_file:bool = None, methods: List[str] = ["kemeny"], n_jobs: int = 1, include_LLM: bool = False):
    """
    Rank aggregation of multiple LLM performance files.
    
    Args:
        LLM_outputs (list): List of paths to the LLM performance files.
        output_file (str, optional): Path to save the aggregated results. Defaults to None.
        methods (list[str], optional): List of methods for rank aggregation.
            Options are "borda", "kemeny", "ranked_pairs", "wilcoxon_stouffer".
        n_jobs (int): Number of concurrent jobs for bootstrap analysis
        include_LLM (bool): Bool whether to only look at prompting strategy or also LLM.
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

        df = pd.read_csv(output, converters={"all_scores": safe_literal_eval})
        if include_LLM:
            df["source"] = output.parents[0].name + " - " + output.with_suffix('').stem
        else:
            df["source"] = output.with_suffix('').stem
        df = df[~df["metric_type"].isin(["micro_avg", "macro_avg"])]  # Exclude the "All fields" row
        results.append(df)

    results = pd.concat(results, ignore_index=True)

    # Prepare rank DataFrame: index = source, columns = methods
    sources = results["source"].unique()
    rank_df = pd.DataFrame(index=sources)

    for method in methods:
        mean_df = results.copy()
        rank_map = aggregate_once(mean_df, method)
        rank_df[method] = pd.Series(rank_map)
        n_boot = min(min(len(x) for x in results["all_scores"] if isinstance(x, np.ndarray) and len(x) > 0), 100)
        if n_boot == 0:
            continue
        rank_samples = {src: [] for src in rank_df.index}
        boot_df = results.copy()
        args = [(b, boot_df, method) for b in range(n_boot)]
        with Pool(n_jobs) as pool:
            boot_rank_maps = list(tqdm(pool.imap(bootstrap_iteration, args), total=n_boot, desc=f"Bootstrapping {method}"))

        for boot_rank_map in boot_rank_maps:
            for src, r in boot_rank_map.items():
                rank_samples[src].append(r)

        ci_low = {}
        ci_high = {}
        for src, ranks in rank_samples.items():
            if ranks:
                ci_low[src] = np.percentile(ranks, 2.5)
                ci_high[src] = np.percentile(ranks, 97.5)
            else:
                ci_low[src] = np.nan
                ci_high[src] = np.nan

        if include_LLM:
            import ipdb; ipdb.set_trace()

        rank_df[f"{method}_ci_low"] = pd.Series(ci_low)
        rank_df[f"{method}_ci_high"] = pd.Series(ci_high)

    rank_df = rank_df.reset_index().rename(columns={"index": "source"})

    if output_file:
        rank_df.to_csv(output_file, index=False)
    else:
        print("Aggregated ranks by method:")
        print(rank_df)
        return rank_df
    
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
        nargs='+',
        default=["borda", "kemeny", "ranked_pairs"],
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
        "--include-LLM", 
        action="store_true", 
        help="Do you want to include the name of the LLM in the source column?"
    )

    args = parser.parse_args()
    rank(args.input_files, args.output_file, args.method, args.n_jobs, args.include_LLM)
