import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

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

    return pd.DataFrame(results)

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
    return df.groupby("LLM Category", observed=True).apply(compute_summary).reset_index()

def get_strategy_counts(data, metric_type="macro_avg"):
    """Return counts of how often each strategy was best."""
    df = data[data["metric_type"] == metric_type]
    df = df[df["Prompt Rank"] == 1].reset_index(drop=True)
    return df["Prompting Strategy Name"].value_counts()

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
    return df.groupby(['Strategy'], observed=True).apply(compute_summary).reset_index()

def get_best_overall_settings(data):
    """Print the best setting for each use case."""
    df = data[data["Use Case Rank"] == 1].reset_index(drop=True)
    for uc in df["Use_Case_mapped"].unique():
        tmp = df[df["Use_Case_mapped"] == uc].reset_index(drop=True)
        print(f"Best setting for {uc}: {tmp.loc[0]['LLM Name']} with {tmp.loc[0]['Prompting Strategy Name']} (score: {tmp.loc[0]['mean']:.3f})")
        print(tmp[["field", "field_type", "metric_type", "mean", "ci_low", "ci_high"]])

def get_stats(data):
    """Main function to compute and print all statistics."""
    # 1. Best strategy per LLM
    print("Summary statistics for best prompting strategy per LLM across use-cases (macro-avg):")
    print(get_best_strategy_stats(data))

    # 2. Strategy counts
    print("\nNumber of times prompting strategies were best (macro-avg):")
    print(get_strategy_counts(data))

    # 3. Compare strategies to Zero Shot
    print("\nSummary statistics for prompting strategies compared to Zero Shot (macro-avg):")
    print(compare_to_zero_shot(data))

    # 4. Mixed effects models
    df4 = data[data["metric_type"] != "macro_avg"].rename(columns={"Prompting Strategy Name": "Prompting_Strategy_Name"})
    print("\nMixed effects variance partition:")
    print(mixed_effects_variance_partition(
        df4,
        strategy_col="Use_Case_mapped",
        group_col="LLM Name",
        re_formula="~C(Prompting_Strategy_Name)"
    ))

    # 5. Best overall settings
    print("\nBest overall settings per use case:")
    get_best_overall_settings(data)