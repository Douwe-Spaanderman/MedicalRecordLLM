from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import sys

try:
    project_root = Path(__file__).resolve().parents[1]
except NameError:
    project_root = Path.cwd().parents[1]

sys.path.insert(0, str(project_root))
from evaluation.visualize.utils import read_performances_and_rank

def mixed_effects_variance_partition(
    df,
    strategy_col="Prompting Strategy",
    response_col="mean",
    group_col="LLM Name",
    re_formula="~C(Use_Case_mapped)"
):
    """
    Run mixed-effects models per unique strategy and decompose variance into LLM, Use Case, and residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least [strategy_col, response_col, group_col].
    strategy_col : str, default="Prompting Strategy"
        Column containing strategy labels to subset on.
    response_col : str, default="mean"
        Dependent variable.
    group_col : str, default="LLM Name"
        Column used as grouping variable (random intercept).
    re_formula : str, default="~C(Use_Case_mapped)"
        Random effects formula, e.g. "~C(Use_Case_mapped)" or "~1".

    Returns
    -------
    pd.DataFrame
        Table with variance components and contributions per strategy.
    """
    
    results = []

    for strat in df[strategy_col].unique():
        df_strat = df[df[strategy_col] == strat]

        try:
            md = smf.mixedlm(f"{response_col} ~ 1", 
                             df_strat, 
                             groups=df_strat[group_col], 
                             re_formula=re_formula)
            mdf = md.fit(reml=True, method="lbfgs", disp=False)

            # Variance components
            var_group   = float(mdf.cov_re.iloc[0,0]) if not mdf.cov_re.empty else 0
            var_other = mdf.cov_re.values.diagonal()[1] if mdf.cov_re.shape[0] > 1 else 0
            var_resid = mdf.scale

            total_var = var_group + var_other + var_resid
            contrib_group   = var_group / total_var if total_var > 0 else 0
            contrib_other = var_other / total_var if total_var > 0 else 0
            contrib_resid = var_resid / total_var if total_var > 0 else 0

            results.append({
                strategy_col: strat,
                "Var(Group)": var_group,
                "Var(Other)": var_other,
                "Var(Residual)": var_resid,
                "Group Contribution %": contrib_group * 100,
                "Other Contribution %": contrib_other * 100,
                "Residual %": contrib_resid * 100,
                "LogLik": mdf.llf,
                "Converged": mdf.converged
            })
        
        except Exception as e:
            results.append({
                strategy_col: strat,
                "Error": str(e)
            })

    df_out = pd.DataFrame(results)
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[numeric_cols] = df_out[numeric_cols].round(2)
    return df_out

def compute_summary(group):
    """Compute summary statistics for a group."""
    n = len(group)
    mean_perf = group['mean'].mean()
    std_perf = group['mean'].std(ddof=1)
    se = std_perf / np.sqrt(n)
    ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_perf, scale=se)
    return pd.Series({
        "mean_performance": mean_perf,
        "std_performance": std_perf,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_use_cases": n
    })


def get_best_strategy_stats(data, metric_type="macro_avg"):
    """Return summary statistics for the best prompting strategy per LLM."""
    df = data[data["metric_type"] == metric_type]
    df = df[df["Prompt Rank"] == 1].reset_index(drop=True)
    idx = df.groupby(["LLM Name", "Use_Case_mapped"], observed=True)["Use Case Rank"].idxmax()
    df = df.loc[idx]
    result = df.groupby("LLM Category", observed=True).apply(compute_summary).reset_index()
    result = result.round(2)
    return result


def get_strategy_counts(data, metric_type="macro_avg"):
    """Return counts of how often each strategy was best."""
    df = data[data["metric_type"] == metric_type]
    df = df[df["Prompt Rank"] == 1].reset_index(drop=True)
    result = df["Prompting Strategy Name"].value_counts().reset_index()
    result.columns = ["Prompting Strategy Name", "Count"]
    return result


def compare_to_zero_shot(data, metric_type="macro_avg"):
    """Compare all strategies to Zero Shot and return summary statistics."""
    df = data[data["metric_type"] == metric_type]
    df = df.pivot_table(
        index=['Use_Case_mapped', 'LLM Name'],
        columns='Prompting Strategy Name',
        values='mean'
    ).reset_index()
    strategies = ['One Shot', 'Few Shot', 'Chain-of-Thought', 'Self-Consistency', 'Prompt Graph']
    for strat in strategies:
        df[f'delta_{strat}'] = df[strat] - df['Zero Shot']
    df = df.melt(
        id_vars=['Use_Case_mapped', 'LLM Name'],
        value_vars=[f'delta_{s}' for s in strategies],
        var_name='Strategy',
        value_name='mean'
    )
    result = df.groupby(['Strategy'], observed=True).apply(compute_summary).reset_index()
    result = result.round(2)
    return result


def get_best_overall_settings(data):
    """Return table of best settings per use case."""
    df = data[data["Use Case Rank"] == 1].reset_index(drop=True)
    out = []
    for uc in df["Use_Case_mapped"].unique():
        tmp = df[df["Use_Case_mapped"] == uc].reset_index(drop=True)
        out.append({
            "Use_Case_mapped": uc,
            "LLM Name": tmp.loc[0, "LLM Name"],
            "Prompting Strategy Name": tmp.loc[0, "Prompting Strategy Name"],
            "Score": round(tmp.loc[0, "mean"], 2),
            "CI Low": round(tmp.loc[0, "ci_low"], 2),
            "CI High": round(tmp.loc[0, "ci_high"], 2)
        })
    return pd.DataFrame(out)


def get_stats(data, output_dir=None):
    """Compute and optionally save all statistics."""
    results = {}

    results["best_strategy_stats"] = get_best_strategy_stats(data)
    results["strategy_counts"] = get_strategy_counts(data)
    results["compare_to_zero_shot"] = compare_to_zero_shot(data)

    results["variance_partition"] = mixed_effects_variance_partition(
        data[data["metric_type"] != "macro_avg"].rename(columns={"Prompting Strategy Name": "Prompting_Strategy_Name"}),
        strategy_col="Use_Case_mapped",
        group_col="LLM Name",
        re_formula="~C(Prompting_Strategy_Name)"
    )

    results["best_overall_settings"] = get_best_overall_settings(data)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"All summaries written to {output_dir}")

    return results

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(
        description="Compute statistical summaries and mixed-effects variance partitioning for LLM prompting strategies."
    )
    parser.add_argument(
        "-i", "--input-files",
        required=True,
        nargs='+',
        help="Path(s) to the LLM output file(s)"
    )
    parser.add_argument(
        "-r", "--ranked-results",
        nargs='+',
        help="Path(s) to the CSV file(s) with ranked results"
    )
    parser.add_argument(
        "-in", "--inter-rater-agreements",
        nargs='+',
        default=[],
        help="Path(s) to the CSV file(s) with inter-rater-agreements"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save structured CSV summaries (optional)."
    )
    args = parser.parse_args()

    data = read_performances_and_rank(args.input_files, args.ranked_results, args.inter_rater_agreements)
    get_stats(data, output_dir=args.output_dir)