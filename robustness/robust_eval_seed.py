"""Robustness analysis across evaluation seeds for all conditions.

Reads per-run eval_seed*.csv files and produces:
    - eval_seed_sweep.csv: one row per (run_name, eval_seed)
    - eval_seed_sweep_agg.csv: aggregated over eval seeds, per run_name
    - paired_deltas.csv: pairwise deltas between conditions
    - fig_eval_seed_robustness.pdf: paired robustness figure

Supports all conditions: baseline, ia, si, monitor_only, sanction_no_review, iml, high_review
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#1f77b4"
BLACK  = "#000000"
GRAY   = "#7a7a7a"
LGRAY  = "#c0c0c0"

CONDITION_STYLES = {
    "baseline":           {"label": "Baseline",          "color": BLACK,  "marker": "o"},
    "ia":                 {"label": "Inequity Aversion", "color": GRAY,   "marker": "s"},
    "si":                 {"label": "Social Influence",  "color": GRAY,   "marker": "^"},
    "monitor_only":       {"label": "Monitor Only",      "color": LGRAY,  "marker": "d"},
    "sanction_no_review": {"label": "No Review",         "color": GRAY,   "marker": "v"},
    "iml":                {"label": "IML (Full)",        "color": ACCENT, "marker": "D"},
    "high_review":        {"label": "High Review",       "color": ACCENT, "marker": "P"},
}

CONDITION_ORDER = ["baseline", "ia", "si", "monitor_only", "sanction_no_review", "iml", "high_review"]

RUNS_DIR = Path("runs")
OUT_DIR = Path("robustness")
RESULTS_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


def parse_run_name(run_name: str) -> dict:
    """Parse run naming convention, e.g.: cleanup_iml_agents5_seed3"""
    out = {"run_name": run_name}

    # Extract env
    for env in ("cleanup", "harvest"):
        if run_name.startswith(env):
            out["env"] = env
            break

    # Extract condition
    for cond in CONDITION_ORDER:
        if f"_{cond}_" in run_name:
            out["condition"] = cond
            break

    # Extract agents
    m = re.search(r"agents(\d+)", run_name)
    if m:
        out["num_agents"] = int(m.group(1))

    # Extract training seed
    m = re.search(r"seed(\d+)$", run_name)
    if m:
        out["train_seed"] = int(m.group(1))

    return out


def summarize_eval_file(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    out = {}
    if "return_mean" in df.columns:
        out["return_mean"] = float(df["return_mean"].mean())
    if "return_gini" in df.columns:
        out["gini"] = float(df["return_gini"].mean())
    if "iml_sanctions" in df.columns:
        out["net_sanctions"] = float(df["iml_sanctions"].mean())
    else:
        out["net_sanctions"] = 0.0
    out["n_episodes"] = int(len(df))
    return out


def main() -> None:
    eval_files = sorted(RUNS_DIR.glob("*/eval_seed*.csv"))
    print(f"Found {len(eval_files)} eval files")

    rows = []
    for f in eval_files:
        m = re.search(r"eval_seed(\d+)\.csv$", f.name)
        if not m:
            continue
        eval_seed = int(m.group(1))
        run_name = f.parent.name
        row = parse_run_name(run_name)
        row["eval_seed"] = eval_seed
        row.update(summarize_eval_file(f))
        rows.append(row)

    if not rows:
        print("No eval files found. Exiting.")
        return

    df = pd.DataFrame(rows)
    preferred = [
        "run_name", "env", "condition", "num_agents", "train_seed", "eval_seed",
        "return_mean", "gini", "net_sanctions", "n_episodes",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols].sort_values(["run_name", "eval_seed"]).reset_index(drop=True)

    out_csv = OUT_DIR / "eval_seed_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")

    # Aggregate over eval seeds per run_name
    metric_cols = [c for c in ["return_mean", "gini", "net_sanctions"] if c in df.columns]
    group_cols = [c for c in ["run_name", "env", "condition", "num_agents", "train_seed"] if c in df.columns]

    if metric_cols and group_cols:
        agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
        agg.columns = ["_".join([x for x in col if x]).rstrip("_") if isinstance(col, tuple) else col for col in agg.columns]
        out_csv_agg = OUT_DIR / "eval_seed_sweep_agg.csv"
        agg.to_csv(out_csv_agg, index=False)
        print(f"Wrote {out_csv_agg} ({len(agg)} rows)")

    # Also save to results dir for the statistics module
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "eval_seed_sweep.csv", index=False)
    if metric_cols and group_cols:
        agg.to_csv(RESULTS_DIR / "eval_seed_sweep_agg.csv", index=False)

    # ---- Figure: robustness across eval seeds ----
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
    })

    if "condition" not in df.columns or "env" not in df.columns:
        print("Cannot generate figure: missing condition/env columns")
        return

    # Per-run stats across eval seeds
    run = df.groupby(["env", "condition", "num_agents", "train_seed", "run_name"], as_index=False).agg(
        return_mean_mean=("return_mean", "mean"),
        return_mean_std=("return_mean", "std"),
        gini_mean=("gini", "mean"),
        gini_std=("gini", "std"),
    )

    envs = sorted(run["env"].unique())
    n_envs = len(envs)

    fig, axes = plt.subplots(2, n_envs, figsize=(3.4 * n_envs, 5), constrained_layout=True)
    if n_envs == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, env_name in enumerate(envs):
        env_run = run[run["env"] == env_name]
        conditions = [c for c in CONDITION_ORDER if c in env_run["condition"].values]

        for row_idx, (ycol_mean, ycol_std, ylabel) in enumerate([
            ("return_mean_mean", "return_mean_std", "Eval return (mean/agent)"),
            ("gini_mean", "gini_std", "Eval inequality (Gini)"),
        ]):
            ax = axes[row_idx, col_idx]

            for seed in sorted(env_run["train_seed"].unique()):
                xs = []
                ys = []
                for i, cond in enumerate(conditions):
                    r = env_run[(env_run["train_seed"] == seed) & (env_run["condition"] == cond)]
                    if not r.empty:
                        xs.append(i)
                        ys.append(float(r[ycol_mean].iloc[0]))
                if len(xs) > 1:
                    ax.plot(xs, ys, color=LGRAY, linewidth=0.7, zorder=1)

            for i, cond in enumerate(conditions):
                style = CONDITION_STYLES.get(cond, {"color": BLACK, "marker": "o"})
                cond_data = env_run[env_run["condition"] == cond]
                vals = cond_data[ycol_mean].to_numpy()
                stds = cond_data[ycol_std].to_numpy()
                m = float(np.mean(vals))
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0
                ax.errorbar([i], [m], yerr=[1.96 * se],
                            fmt=style["marker"], color=style["color"],
                            ecolor=style["color"], elinewidth=0.9, capsize=2,
                            markersize=5, zorder=3)

            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels([CONDITION_STYLES.get(c, {}).get("label", c)[:12] for c in conditions],
                               rotation=40, ha="right", fontsize=6)
            ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(f"{env_name.capitalize()}")

    fig.suptitle("Robustness across evaluation seeds (error bars: 95% CI over training seeds)", fontsize=9)
    fig.savefig(OUT_DIR / "fig_eval_seed_robustness.pdf")
    fig.savefig(OUT_DIR / "fig_eval_seed_robustness.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'fig_eval_seed_robustness.pdf'}")


if __name__ == "__main__":
    main()
