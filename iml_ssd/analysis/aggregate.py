"""Aggregate results across all experimental conditions.

Reads run directories and produces:
    - summary.csv: one row per run with key metrics
    - learning_curves.csv: concatenated episode-level data for plotting
    - condition_summary.csv: aggregated across seeds per env x condition

Supports conditions: baseline, iml, ia, si, monitor_only, sanction_no_review, high_review
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def collect_runs(runs_dir: Path) -> List[Path]:
    return sorted([p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists()])


def _infer_condition(cfg: Dict[str, Any], run_name: str) -> str:
    """Infer condition from config or run name."""
    # Explicit condition field
    cond = cfg.get("condition")
    if cond:
        return str(cond)

    # Try to parse from run name
    for c in ("monitor_only", "sanction_no_review", "high_review", "iml", "ia", "si", "baseline"):
        if f"_{c}_" in run_name or run_name.startswith(f"{c}_") or run_name.endswith(f"_{c}"):
            return c

    # Infer from config flags
    iml_enabled = bool(cfg.get("iml", {}).get("enabled", False))
    ia_enabled = bool(cfg.get("ia", {}).get("enabled", False))
    si_enabled = bool(cfg.get("si", {}).get("enabled", False))

    if ia_enabled:
        return "ia"
    if si_enabled:
        return "si"
    if iml_enabled:
        sanction = float(cfg.get("iml", {}).get("sanction", 0.5))
        p_review = float(cfg.get("iml", {}).get("p_review", 0.0))
        if sanction == 0.0:
            return "monitor_only"
        if p_review == 0.0:
            return "sanction_no_review"
        if p_review >= 0.7:
            return "high_review"
        return "iml"

    return "baseline"


def summarize_run(run_dir: Path, tail_episodes: int = 10) -> Dict[str, Any]:
    cfg = _load_yaml(run_dir / "config.yaml")
    env_name = str(cfg.get("env", {}).get("name", "unknown"))
    num_agents = int(cfg.get("env", {}).get("num_agents", -1))
    seed = int(cfg.get("train", {}).get("seed", -1))
    condition = _infer_condition(cfg, run_dir.name)

    # episodes.csv
    ep_path = run_dir / "episodes.csv"
    ep_df = pd.read_csv(ep_path) if ep_path.exists() else pd.DataFrame()

    # eval.csv
    eval_path = run_dir / "eval.csv"
    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()

    out: Dict[str, Any] = {
        "run_name": run_dir.name,
        "env": env_name,
        "num_agents": num_agents,
        "seed": seed,
        "condition": condition,
        "iml_enabled": condition in ("iml", "monitor_only", "sanction_no_review", "high_review"),
        "run_dir": str(run_dir),
    }

    if not ep_df.empty:
        tail = ep_df.tail(tail_episodes)
        out.update({
            "train_return_mean": float(tail["return_mean"].mean()),
            "train_return_sum": float(tail["return_sum"].mean()),
            "train_gini": float(tail["return_gini"].mean()),
            "train_iml_sanctions": float(tail.get("iml_sanctions", pd.Series([0])).mean()),
            "train_iml_false_pos": float(tail.get("iml_false_pos", pd.Series([0])).mean()),
        })
        if "env_step" in ep_df.columns:
            out["train_env_steps"] = int(ep_df["env_step"].max())
        else:
            out["train_env_steps"] = None
    else:
        out["train_env_steps"] = None

    if not eval_df.empty:
        out.update({
            "eval_return_mean": float(eval_df["return_mean"].mean()),
            "eval_return_sum": float(eval_df["return_sum"].mean()),
            "eval_gini": float(eval_df["return_gini"].mean()),
            "eval_iml_sanctions": float(eval_df.get("iml_sanctions", pd.Series([0])).mean()),
            "eval_iml_false_pos": float(eval_df.get("iml_false_pos", pd.Series([0])).mean()),
        })

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--tail_episodes", type=int, default=10)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = collect_runs(runs_dir)
    rows = []
    for rd in run_dirs:
        try:
            rows.append(summarize_run(rd, tail_episodes=args.tail_episodes))
        except Exception as e:
            print(f"[WARN] Failed to summarize {rd}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote {out_dir / 'summary.csv'} ({len(df)} runs)")

    # Condition-level summary (mean +/- std across seeds)
    if not df.empty and "condition" in df.columns:
        metric_cols = [c for c in df.columns if c.startswith("eval_") or c.startswith("train_")]
        numeric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            cond_summary = (
                df.groupby(["env", "condition"], as_index=False)[numeric_cols]
                .agg(["mean", "std", "count"])
            )
            cond_summary.columns = ["_".join(col).rstrip("_") for col in cond_summary.columns]
            cond_summary.to_csv(out_dir / "condition_summary.csv", index=False)
            print(f"Wrote {out_dir / 'condition_summary.csv'}")

    # Also concatenate learning curves if present
    curve_rows = []
    for rd in run_dirs:
        ep_path = rd / "episodes.csv"
        if not ep_path.exists():
            continue
        cfg = _load_yaml(rd / "config.yaml")
        env_name = str(cfg.get("env", {}).get("name", "unknown"))
        num_agents = int(cfg.get("env", {}).get("num_agents", -1))
        seed = int(cfg.get("train", {}).get("seed", -1))
        condition = _infer_condition(cfg, rd.name)

        ep_df = pd.read_csv(ep_path)
        ep_df["run_name"] = rd.name
        ep_df["env"] = env_name
        ep_df["num_agents"] = num_agents
        ep_df["seed"] = seed
        ep_df["condition"] = condition
        ep_df["iml_enabled"] = condition in ("iml", "monitor_only", "sanction_no_review", "high_review")
        curve_rows.append(ep_df)

    if curve_rows:
        curves = pd.concat(curve_rows, ignore_index=True)
        curves.to_csv(out_dir / "learning_curves.csv", index=False)
        print(f"Wrote {out_dir / 'learning_curves.csv'}")

    print("Aggregation complete.")


if __name__ == "__main__":
    main()
