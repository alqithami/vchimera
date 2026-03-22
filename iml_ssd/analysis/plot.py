"""Publication-ready plotting for the IML paper.

Generates figures for:
    1. Learning curves (all conditions) — welfare + fairness
    2. Bar charts comparing final evaluation metrics
    3. Ablation study results
    4. Sanction dynamics over training
    5. Sensitivity analysis heatmaps

Style: Black & white with single accent color (#1f77b4).
All figures are saved as both PDF (for LaTeX) and PNG (for review).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- Style constants ----
ACCENT = "#1f77b4"
BLACK = "#000000"
GRAY = "#7a7a7a"
LGRAY = "#c0c0c0"

# Condition display names and styles
CONDITION_STYLES = {
    "baseline":           {"label": "Baseline (PPO)",     "color": BLACK,   "ls": "-",  "marker": "o"},
    "ia":                 {"label": "Inequity Aversion",  "color": GRAY,    "ls": "--", "marker": "s"},
    "si":                 {"label": "Social Influence",   "color": GRAY,    "ls": "-.", "marker": "^"},
    "monitor_only":       {"label": "IML (Monitor Only)", "color": LGRAY,   "ls": ":",  "marker": "d"},
    "sanction_no_review": {"label": "IML (No Review)",    "color": GRAY,    "ls": "--", "marker": "v"},
    "iml":                {"label": "IML (Full)",         "color": ACCENT,  "ls": "-",  "marker": "D"},
    "high_review":        {"label": "IML (High Review)",  "color": ACCENT,  "ls": ":",  "marker": "P"},
}

# Condition ordering for plots
CONDITION_ORDER = [
    "baseline", "ia", "si", "monitor_only", "sanction_no_review", "iml", "high_review"
]


def _setup_style():
    """Set up matplotlib style for publication."""
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _get_style(condition: str) -> dict:
    return CONDITION_STYLES.get(condition, {"label": condition, "color": BLACK, "ls": "-", "marker": "o"})


def _save_fig(fig, path_stem: Path):
    """Save figure as both PDF and PNG."""
    fig.savefig(str(path_stem) + ".pdf")
    fig.savefig(str(path_stem) + ".png")
    plt.close(fig)


# ---- Binning and aggregation helpers ----

def _bin_curves(df: pd.DataFrame, step_col: str, value_col: str, bin_size: int) -> pd.DataFrame:
    max_step = int(df[step_col].max())
    bins = np.arange(0, max_step + bin_size, bin_size)
    if len(bins) < 2:
        bins = np.array([0, max_step + 1])
    df = df.copy()
    df["step_bin"] = pd.cut(df[step_col], bins=bins, labels=bins[:-1], include_lowest=True).astype(int)
    gb = df.groupby(["env", "num_agents", "condition", "seed", "step_bin"], as_index=False)[value_col].mean()
    return gb


def _mean_ci_across_seeds(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    def agg(g):
        vals = g[value_col].to_numpy(dtype=float)
        m = float(np.mean(vals))
        if len(vals) <= 1:
            return pd.Series({"mean": m, "lo": m, "hi": m, "n": len(vals)})
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        lo = m - 1.96 * se
        hi = m + 1.96 * se
        return pd.Series({"mean": m, "lo": lo, "hi": hi, "n": len(vals)})

    out = df.groupby(["env", "num_agents", "condition", "step_bin"]).apply(agg, include_groups=False).reset_index()
    return out


# ---- Plot 1: Learning curves (all conditions) ----

def plot_learning_curves(curves: pd.DataFrame, out_dir: Path, bin_size: int = 10_000) -> None:
    """Plot learning curves for all conditions, one figure per env."""
    _ensure_dir(out_dir)
    _setup_style()

    for env_name in sorted(curves["env"].unique()):
        sub = curves[curves["env"] == env_name]

        # ---- Welfare (return_sum) ----
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

        for metric, ax, ylabel, title_suffix in [
            ("return_sum", axes[0], "Collective return (sum)", "Welfare"),
            ("return_gini", axes[1], "Gini coefficient", "Inequality"),
        ]:
            if metric not in sub.columns:
                continue
            gb = _bin_curves(sub, "env_step", metric, bin_size=bin_size)
            ci = _mean_ci_across_seeds(gb, metric)

            for cond in CONDITION_ORDER:
                s = ci[ci["condition"] == cond].sort_values("step_bin")
                if s.empty:
                    continue
                style = _get_style(cond)
                x = s["step_bin"].to_numpy()
                y = s["mean"].to_numpy()
                lo = s["lo"].to_numpy()
                hi = s["hi"].to_numpy()
                ax.plot(x, y, label=style["label"], color=style["color"],
                        linestyle=style["ls"], linewidth=1.2)
                ax.fill_between(x, lo, hi, alpha=0.12, color=style["color"])

            ax.set_xlabel("Environment steps")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{env_name.capitalize()}: {title_suffix}")

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)),
                   frameon=False, bbox_to_anchor=(0.5, 1.08))
        _save_fig(fig, out_dir / f"learning_curves_{env_name}")

        # ---- Sanctions over time (IML conditions only) ----
        if "iml_sanctions" in sub.columns:
            iml_sub = sub[sub["condition"].isin(["iml", "monitor_only", "sanction_no_review", "high_review"])]
            if not iml_sub.empty:
                fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
                gb = _bin_curves(iml_sub, "env_step", "iml_sanctions", bin_size=bin_size)
                ci = _mean_ci_across_seeds(gb, "iml_sanctions")

                for cond in CONDITION_ORDER:
                    s = ci[ci["condition"] == cond].sort_values("step_bin")
                    if s.empty:
                        continue
                    style = _get_style(cond)
                    x = s["step_bin"].to_numpy()
                    y = s["mean"].to_numpy()
                    lo = s["lo"].to_numpy()
                    hi = s["hi"].to_numpy()
                    ax.plot(x, y, label=style["label"], color=style["color"],
                            linestyle=style["ls"], linewidth=1.2)
                    ax.fill_between(x, lo, hi, alpha=0.12, color=style["color"])

                ax.set_xlabel("Environment steps")
                ax.set_ylabel("Sanctions per episode")
                ax.set_title(f"{env_name.capitalize()}: Sanction dynamics")
                ax.legend(fontsize=7, frameon=False)
                _save_fig(fig, out_dir / f"sanctions_{env_name}")


# ---- Plot 2: Evaluation bar charts ----

def plot_eval_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of evaluation metrics across conditions."""
    _ensure_dir(out_dir)
    _setup_style()

    if summary.empty or "eval_return_mean" not in summary.columns:
        return

    for env_name in sorted(summary["env"].unique()):
        sub = summary[summary["env"] == env_name]

        for metric, ylabel, title_suffix in [
            ("eval_return_mean", "Mean per-agent return", "Welfare"),
            ("eval_gini", "Gini coefficient", "Inequality"),
        ]:
            if metric not in sub.columns:
                continue

            # Group by condition
            grp = sub.groupby("condition")[metric].agg(["mean", "std", "count"]).reset_index()
            grp["se"] = grp["std"] / np.sqrt(grp["count"])

            # Order conditions
            cond_order = [c for c in CONDITION_ORDER if c in grp["condition"].values]
            grp = grp.set_index("condition").loc[cond_order].reset_index()

            fig, ax = plt.subplots(figsize=(max(4, len(cond_order) * 0.9), 3.5), constrained_layout=True)
            x = np.arange(len(grp))
            colors = [_get_style(c)["color"] for c in grp["condition"]]
            labels = [_get_style(c)["label"] for c in grp["condition"]]

            bars = ax.bar(x, grp["mean"], yerr=1.96 * grp["se"],
                          color=colors, edgecolor=BLACK, linewidth=0.5,
                          capsize=3, error_kw={"linewidth": 0.8})

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{env_name.capitalize()}: {title_suffix}")
            _save_fig(fig, out_dir / f"eval_{metric}_{env_name}")


# ---- Plot 3: Ablation study ----

def plot_ablation(summary: pd.DataFrame, out_dir: Path) -> None:
    """Plot ablation study: IML component contributions."""
    _ensure_dir(out_dir)
    _setup_style()

    if summary.empty:
        return

    ablation_conditions = ["baseline", "monitor_only", "sanction_no_review", "iml", "high_review"]

    for env_name in sorted(summary["env"].unique()):
        sub = summary[summary["env"] == env_name]
        sub = sub[sub["condition"].isin(ablation_conditions)]
        if sub.empty:
            continue

        for metric, ylabel, title_suffix in [
            ("eval_return_mean", "Mean per-agent return", "Welfare"),
            ("eval_gini", "Gini coefficient", "Inequality"),
        ]:
            if metric not in sub.columns:
                continue

            grp = sub.groupby("condition")[metric].agg(["mean", "std", "count"]).reset_index()
            grp["se"] = grp["std"] / np.sqrt(grp["count"])

            cond_order = [c for c in ablation_conditions if c in grp["condition"].values]
            grp = grp.set_index("condition").loc[cond_order].reset_index()

            fig, ax = plt.subplots(figsize=(max(4, len(cond_order) * 1.0), 3.5), constrained_layout=True)
            x = np.arange(len(grp))
            colors = [_get_style(c)["color"] for c in grp["condition"]]
            labels = [_get_style(c)["label"] for c in grp["condition"]]

            bars = ax.bar(x, grp["mean"], yerr=1.96 * grp["se"],
                          color=colors, edgecolor=BLACK, linewidth=0.5,
                          capsize=3, error_kw={"linewidth": 0.8})

            # Add value labels on bars
            for bar, val in zip(bars, grp["mean"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{env_name.capitalize()}: Ablation — {title_suffix}")
            _save_fig(fig, out_dir / f"ablation_{metric}_{env_name}")


# ---- Plot 4: Comparative radar chart ----

def plot_radar(summary: pd.DataFrame, out_dir: Path) -> None:
    """Radar chart comparing conditions on multiple dimensions."""
    _ensure_dir(out_dir)
    _setup_style()

    if summary.empty:
        return

    metrics = ["eval_return_mean", "eval_gini"]
    available = [m for m in metrics if m in summary.columns]
    if len(available) < 2:
        return

    for env_name in sorted(summary["env"].unique()):
        sub = summary[summary["env"] == env_name]
        conditions = [c for c in CONDITION_ORDER if c in sub["condition"].values]
        if len(conditions) < 2:
            continue

        # Compute per-condition means
        grp = sub.groupby("condition")[available].mean()

        # Normalize each metric to [0, 1] for radar
        normed = grp.copy()
        for col in available:
            mn, mx = grp[col].min(), grp[col].max()
            if mx - mn > 1e-12:
                normed[col] = (grp[col] - mn) / (mx - mn)
            else:
                normed[col] = 0.5

        # For Gini, lower is better, so invert
        if "eval_gini" in normed.columns:
            normed["eval_gini"] = 1.0 - normed["eval_gini"]

        labels = ["Welfare\n(return)", "Equity\n(1-Gini)"]
        n_metrics = len(available)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), constrained_layout=True)

        for cond in conditions:
            if cond not in normed.index:
                continue
            style = _get_style(cond)
            vals = normed.loc[cond, available].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, label=style["label"], color=style["color"],
                    linestyle=style["ls"], linewidth=1.3)
            ax.fill(angles, vals, alpha=0.06, color=style["color"])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{env_name.capitalize()}: Multi-dimensional comparison", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7, frameon=False)
        _save_fig(fig, out_dir / f"radar_{env_name}")


# ---- Plot 5: Paired robustness (seed-level) ----

def plot_paired_robustness(curves: pd.DataFrame, out_dir: Path) -> None:
    """Paired seed-level robustness plot (like the existing robust_eval_seed.py)."""
    _ensure_dir(out_dir)
    _setup_style()

    if curves.empty:
        return

    # We need per-run (seed) final metrics
    # Use last 10 episodes per run
    final = curves.groupby(["env", "condition", "seed"]).tail(10)
    run_means = final.groupby(["env", "condition", "seed"])[["return_mean", "return_gini"]].mean().reset_index()

    for env_name in sorted(run_means["env"].unique()):
        env_data = run_means[run_means["env"] == env_name]
        conditions = [c for c in CONDITION_ORDER if c in env_data["condition"].values]
        if len(conditions) < 2:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

        for metric, ax, ylabel in [
            ("return_mean", axes[0], "Mean per-agent return"),
            ("return_gini", axes[1], "Gini coefficient"),
        ]:
            positions = {}
            for i, cond in enumerate(conditions):
                positions[cond] = i

            for seed in sorted(env_data["seed"].unique()):
                seed_data = env_data[env_data["seed"] == seed]
                xs = []
                ys = []
                for cond in conditions:
                    row = seed_data[seed_data["condition"] == cond]
                    if not row.empty:
                        xs.append(positions[cond])
                        ys.append(float(row[metric].iloc[0]))
                if len(xs) > 1:
                    ax.plot(xs, ys, color=LGRAY, linewidth=0.7, zorder=1)

            # Overlay condition means
            for cond in conditions:
                cond_data = env_data[env_data["condition"] == cond]
                if cond_data.empty:
                    continue
                style = _get_style(cond)
                vals = cond_data[metric].to_numpy()
                m = float(np.mean(vals))
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0
                ax.errorbar(
                    [positions[cond]], [m], yerr=[1.96 * se],
                    fmt=style["marker"], color=style["color"],
                    ecolor=style["color"], elinewidth=1.0, capsize=3,
                    markersize=6, zorder=3,
                )

            ax.set_xticks(list(range(len(conditions))))
            ax.set_xticklabels([_get_style(c)["label"] for c in conditions],
                               rotation=35, ha="right", fontsize=6.5)
            ax.set_ylabel(ylabel)

        fig.suptitle(f"{env_name.capitalize()}: Seed-level robustness", fontsize=10)
        _save_fig(fig, out_dir / f"robustness_{env_name}")


# ---- Main ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--out_dir", type=str, default="figures")
    p.add_argument("--bin_size", type=int, default=10_000)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    curves_path = results_dir / "learning_curves.csv"
    summary_path = results_dir / "summary.csv"

    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        plot_learning_curves(curves, out_dir, bin_size=args.bin_size)
        plot_paired_robustness(curves, out_dir)

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        plot_eval_bars(summary, out_dir)
        plot_ablation(summary, out_dir)
        plot_radar(summary, out_dir)

    print(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
