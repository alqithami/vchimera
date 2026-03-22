#!/usr/bin/env python3
"""Lightweight statistical comparison helper (no SciPy).

Produces a small LaTeX table of paired comparisons between policies for key metrics.

Method:
  - Pair seeds within each scenario and compute differences.
  - Report mean difference, bootstrap 95% CI, and a two-sided permutation p-value.

Usage:
  python scripts/stat_tests.py --run_dir runs/<RUN_ID> --out paper/paper_assets/table_stats.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY_METRICS = [
    ("cyber_harm_auc", "Cyber harm AUC $\downarrow$"),
    ("misbelief_auc", "Misbelief AUC $\downarrow$"),
    ("trust_auc", "Trust AUC $\uparrow$"),
    ("polarization_auc", "Polarization AUC $\downarrow$"),
    ("protocol_executed", "Executed protocol violations $\downarrow$"),
]

def bootstrap_ci(x: np.ndarray, n: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if x.size == 0:
        return (0.0, 0.0)
    samples = rng.choice(x, size=(n, x.size), replace=True)
    means = samples.mean(axis=1)
    lo = float(np.quantile(means, alpha/2))
    hi = float(np.quantile(means, 1-alpha/2))
    return lo, hi

def perm_pvalue(diff: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test for paired differences."""
    rng = np.random.default_rng(seed)
    if diff.size == 0:
        return 1.0
    obs = abs(float(diff.mean()))
    # sign flips
    signs = rng.choice([-1, 1], size=(n, diff.size))
    null = abs((signs * diff).mean(axis=1))
    return float((null >= obs).mean())

def make_table(df: pd.DataFrame, scenario: str, a: str, b: str) -> str:
    sdf = df[df["scenario"] == scenario].copy()
    # pair by seed
    a_df = sdf[sdf["policy"] == a].set_index("seed")
    b_df = sdf[sdf["policy"] == b].set_index("seed")
    common = sorted(set(a_df.index).intersection(set(b_df.index)))
    if not common:
        return "% No common seeds for comparison\n"
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(f"Metric & Mean($\Delta$) & 95\\% CI & $p$ (perm.) \\\\ ")
    lines.append("\\midrule")

    for key, label in KEY_METRICS:
        da = a_df.loc[common, key].to_numpy(dtype=float)
        db = b_df.loc[common, key].to_numpy(dtype=float)
        diff = da - db  # positive means A larger than B
        m = float(diff.mean())
        lo, hi = bootstrap_ci(diff)
        p = perm_pvalue(diff)
        lines.append(f"{label} & {m:.3f} & [{lo:.3f}, {hi:.3f}] & {p:.3f} \\\\ ")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Paired comparison for scenario \\texttt{{{scenario}}}: {a} minus {b}.}} ")
    lines.append("\\label{tab:stats}" )
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy_a", default="vchimera+shield")
    ap.add_argument("--policy_b", default="pipeline+shield")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "summary_by_seed.csv")

    out = []
    for scenario in sorted(df["scenario"].unique()):
        out.append(make_table(df, scenario=scenario, a=args.policy_a, b=args.policy_b))

    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print("[OK] Wrote:", args.out)

if __name__ == "__main__":
    main()
