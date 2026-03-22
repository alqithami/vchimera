"""Comprehensive statistical analysis for the IML paper.

Implements:
    - Mann-Whitney U test (non-parametric, appropriate for small n)
    - Welch's t-test (parametric, for comparison)
    - Bootstrap confidence intervals (BCa method)
    - Cohen's d effect size with confidence intervals
    - Cliff's delta (non-parametric effect size)
    - Pairwise comparison tables for all conditions
    - LaTeX table generation for the manuscript

All tests are two-sided unless otherwise noted.
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD) for two independent samples.

    Positive d means x > y.
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if sp < 1e-12:
        return 0.0
    return float((mx - my) / sp)


def cohens_d_ci(x: np.ndarray, y: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]:
    """Cohen's d with normal-approximation CI.

    Returns (d, lo, hi).
    """
    d = cohens_d(x, y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return d, float("nan"), float("nan")
    # Hedges & Olkin (1985) variance approximation
    se_d = np.sqrt((nx + ny) / (nx * ny) + d**2 / (2 * (nx + ny)))
    z = sp_stats.norm.ppf(1 - (1 - ci) / 2)
    return d, float(d - z * se_d), float(d + z * se_d)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: non-parametric effect size.

    Returns a value in [-1, 1].  Positive means x tends to be larger than y.
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    # Count dominance pairs
    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    return float((more - less) / (nx * ny))


def cliffs_delta_interpretation(d: float) -> str:
    """Interpret Cliff's delta magnitude (Romano et al., 2006)."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def cohens_d_interpretation(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    x: np.ndarray,
    stat_func=np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval using the percentile method.

    Returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(stat_func(x))
    if n == 1:
        return point, point, point

    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = x[rng.integers(0, n, size=n)]
        boot_stats[b] = stat_func(sample)

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return point, lo, hi


def bootstrap_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_func=np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the difference stat_func(x) - stat_func(y).

    Returns (point_diff, lo, hi).
    """
    rng = np.random.default_rng(seed)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(stat_func(x) - stat_func(y))

    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        sx = x[rng.integers(0, nx, size=nx)]
        sy = y[rng.integers(0, ny, size=ny)]
        boot_diffs[b] = stat_func(sx) - stat_func(sy)

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_diffs, 100 * alpha))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Two-sided Mann-Whitney U test.

    Returns (U_statistic, p_value).
    """
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(u), float(p)


def welch_t_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Two-sided Welch's t-test (unequal variances).

    Returns (t_statistic, p_value).
    """
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, p = sp_stats.ttest_ind(x, y, equal_var=False)
    return float(t), float(p)


def kruskal_wallis(*groups: np.ndarray) -> Tuple[float, float]:
    """Kruskal-Wallis H-test for comparing multiple groups.

    Returns (H_statistic, p_value).
    """
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h, p = sp_stats.kruskal(*valid)
    return float(h), float(p)


# ---------------------------------------------------------------------------
# Pairwise comparison table
# ---------------------------------------------------------------------------

def pairwise_comparison(
    data: Dict[str, np.ndarray],
    metric_name: str = "return_mean",
) -> pd.DataFrame:
    """Compute all pairwise comparisons between conditions.

    Parameters
    ----------
    data : dict
        Mapping from condition_name -> array of per-seed metric values.
    metric_name : str
        Name of the metric being compared (for labeling).

    Returns
    -------
    pd.DataFrame
        One row per pair with test statistics and effect sizes.
    """
    conditions = sorted(data.keys())
    rows = []

    for c1, c2 in itertools.combinations(conditions, 2):
        x = np.asarray(data[c1], dtype=np.float64)
        y = np.asarray(data[c2], dtype=np.float64)

        u_stat, u_p = mann_whitney_u(x, y)
        t_stat, t_p = welch_t_test(x, y)
        d, d_lo, d_hi = cohens_d_ci(x, y)
        cd = cliffs_delta(x, y)

        diff, diff_lo, diff_hi = bootstrap_diff_ci(x, y)

        rows.append({
            "metric": metric_name,
            "condition_1": c1,
            "condition_2": c2,
            "n_1": len(x),
            "n_2": len(y),
            "mean_1": float(np.mean(x)),
            "mean_2": float(np.mean(y)),
            "std_1": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "std_2": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
            "mean_diff": diff,
            "diff_ci_lo": diff_lo,
            "diff_ci_hi": diff_hi,
            "mann_whitney_U": u_stat,
            "mann_whitney_p": u_p,
            "welch_t": t_stat,
            "welch_p": t_p,
            "cohens_d": d,
            "cohens_d_ci_lo": d_lo,
            "cohens_d_ci_hi": d_hi,
            "cohens_d_interp": cohens_d_interpretation(d) if not np.isnan(d) else "N/A",
            "cliffs_delta": cd,
            "cliffs_delta_interp": cliffs_delta_interpretation(cd) if not np.isnan(cd) else "N/A",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------

def summary_statistics(
    data: Dict[str, np.ndarray],
    metric_name: str = "return_mean",
) -> pd.DataFrame:
    """Compute summary statistics for each condition.

    Returns a DataFrame with one row per condition.
    """
    rows = []
    for cond, vals in sorted(data.items()):
        v = np.asarray(vals, dtype=np.float64)
        mean, ci_lo, ci_hi = bootstrap_ci(v)
        rows.append({
            "metric": metric_name,
            "condition": cond,
            "n": len(v),
            "mean": mean,
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)),
            "min": float(np.min(v)) if len(v) > 0 else float("nan"),
            "max": float(np.max(v)) if len(v) > 0 else float("nan"),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    """Format p-value for LaTeX."""
    if np.isnan(p):
        return "---"
    if p < 0.001:
        return "$<$0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    elif p < 0.05:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def _fmt_ci(lo: float, hi: float) -> str:
    """Format a confidence interval for LaTeX."""
    if np.isnan(lo) or np.isnan(hi):
        return "---"
    return f"[{lo:.2f}, {hi:.2f}]"


def pairwise_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert a pairwise comparison DataFrame to a LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Metric & Comparison & $\Delta\bar{x}$ & 95\% CI & "
        r"MW $p$ & Welch $p$ & Cohen's $d$ & Cliff's $\delta$ \\"
    )
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        comp = f"{row['condition_1']} vs {row['condition_2']}"
        delta = f"{row['mean_diff']:.2f}"
        ci = _fmt_ci(row["diff_ci_lo"], row["diff_ci_hi"])
        mw_p = _fmt_p(row["mann_whitney_p"])
        w_p = _fmt_p(row["welch_p"])
        cd = f"{row['cohens_d']:.2f}" if not np.isnan(row["cohens_d"]) else "---"
        cliff = f"{row['cliffs_delta']:.2f}" if not np.isnan(row["cliffs_delta"]) else "---"
        lines.append(
            f"  {row['metric']} & {comp} & {delta} & {ci} & "
            f"{mw_p} & {w_p} & {cd} & {cliff} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def summary_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert a summary statistics DataFrame to a LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Metric & Condition & $n$ & Mean $\pm$ SD & Median & 95\% CI \\"
    )
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        mean_sd = f"{row['mean']:.2f} $\\pm$ {row['std']:.2f}"
        ci = _fmt_ci(row["ci95_lo"], row["ci95_hi"])
        lines.append(
            f"  {row['metric']} & {row['condition']} & {row['n']} & "
            f"{mean_sd} & {row['median']:.2f} & {ci} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main: run full statistical analysis from aggregated results
# ---------------------------------------------------------------------------

def run_full_analysis(
    results_dir: Path,
    out_dir: Path,
) -> None:
    """Run the complete statistical analysis pipeline.

    Reads the eval_seed_sweep_agg.csv (or summary.csv) and produces:
        - summary_statistics.csv
        - pairwise_comparisons.csv
        - statistical_tables.tex (LaTeX tables for the manuscript)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to load the aggregated evaluation data
    agg_path = results_dir / "eval_seed_sweep_agg.csv"
    summary_path = results_dir / "summary.csv"

    if agg_path.exists():
        df = pd.read_csv(agg_path)
        # This has columns: run_name, env, condition, num_agents, train_seed,
        #                    return_mean_mean, return_mean_std, gini_mean, ...
        metric_col = "return_mean_mean"
        gini_col = "gini_mean"
        cond_col = "condition"
    elif summary_path.exists():
        df = pd.read_csv(summary_path)
        metric_col = "eval_return_mean"
        gini_col = "eval_gini"
        # Derive condition from iml_enabled
        if "iml_enabled" in df.columns:
            df["condition"] = df["iml_enabled"].map({True: "iml", False: "baseline"})
        cond_col = "condition"
    else:
        print(f"[WARN] No aggregated results found in {results_dir}")
        return

    all_summaries = []
    all_pairwise = []
    all_latex = []

    for env_name in sorted(df["env"].unique()):
        env_df = df[df["env"] == env_name]

        # Build per-condition arrays
        for metric, col in [("return_mean", metric_col), ("gini", gini_col)]:
            if col not in env_df.columns:
                continue

            data = {}
            for cond in sorted(env_df[cond_col].unique()):
                vals = env_df[env_df[cond_col] == cond][col].dropna().to_numpy()
                data[cond] = vals

            if len(data) < 2:
                continue

            # Summary statistics
            summ = summary_statistics(data, metric_name=f"{env_name}_{metric}")
            all_summaries.append(summ)

            # Pairwise comparisons
            pw = pairwise_comparison(data, metric_name=f"{env_name}_{metric}")
            all_pairwise.append(pw)

            # Kruskal-Wallis if > 2 conditions
            if len(data) > 2:
                groups = list(data.values())
                h, p = kruskal_wallis(*groups)
                kw_row = pd.DataFrame([{
                    "metric": f"{env_name}_{metric}",
                    "test": "Kruskal-Wallis",
                    "H": h,
                    "p": p,
                    "n_groups": len(data),
                }])
                kw_row.to_csv(out_dir / f"kruskal_wallis_{env_name}_{metric}.csv", index=False)

    # Concatenate and save
    if all_summaries:
        summ_df = pd.concat(all_summaries, ignore_index=True)
        summ_df.to_csv(out_dir / "summary_statistics.csv", index=False)
        all_latex.append(summary_to_latex(
            summ_df,
            caption="Summary statistics across conditions (bootstrap 95\\% CI).",
            label="tab:summary_stats",
        ))

    if all_pairwise:
        pw_df = pd.concat(all_pairwise, ignore_index=True)
        pw_df.to_csv(out_dir / "pairwise_comparisons.csv", index=False)
        all_latex.append(pairwise_to_latex(
            pw_df,
            caption="Pairwise statistical comparisons between conditions.",
            label="tab:pairwise",
        ))

    if all_latex:
        latex_path = out_dir / "statistical_tables.tex"
        latex_path.write_text("\n\n".join(all_latex), encoding="utf-8")
        print(f"Wrote {latex_path}")

    print(f"Statistical analysis complete. Results in {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Run statistical analysis on IML results.")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--out_dir", type=str, default="results/statistics")
    args = p.parse_args()
    run_full_analysis(Path(args.results_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
