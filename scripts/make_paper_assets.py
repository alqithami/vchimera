#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LaTeX tables and figures from a run directory.")
    p.add_argument("--run_dir", required=True, help="runs/<RUN_ID> directory containing summary_by_seed.csv")
    p.add_argument("--paper_dir", required=True, help="paper directory with paper_assets/")
    return p.parse_args()


def tex_escape(s: str) -> str:
    # minimal escaping for LaTeX tables
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def make_table_main(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("cyber_harm_auc", r"Cyber harm AUC $\downarrow$"),
        ("misbelief_auc", r"Misbelief AUC $\downarrow$"),
        ("trust_auc", r"Trust AUC $\uparrow$"),
        ("protocol_executed", r"Executed viol. $\downarrow$"),
        ("shield_interventions", r"Shield edits"),
    ]

    lines = []
    lines.append(r"\begin{tabular}{ll" + "r" * len(metrics) + "}")
    lines.append(r"\toprule")
    header = "Scenario & Policy & " + " & ".join([m[1] for m in metrics]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for (scenario, policy), sub in df.groupby(["scenario", "policy"]):
        row = [tex_escape(scenario), tex_escape(policy)]
        for key, _label in metrics:
            mu = float(sub[key].mean())
            sd = float(sub[key].std(ddof=1) if len(sub) > 1 else 0.0)
            row.append(_fmt_mean_std(mu, sd, digits=3))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_table_protocol(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("protocol_attempted", "Attempted"),
        ("protocol_executed", "Executed"),
        ("shield_interventions", "Shield edits"),
    ]
    lines = []
    lines.append(r"\begin{tabular}{ll" + "r" * len(metrics) + "}")
    lines.append(r"\toprule")
    header = "Scenario & Policy & " + " & ".join([m[1] for m in metrics]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for (scenario, policy), sub in df.groupby(["scenario", "policy"]):
        row = [tex_escape(scenario), tex_escape(policy)]
        for key, _ in metrics:
            mu = float(sub[key].mean())
            sd = float(sub[key].std(ddof=1) if len(sub) > 1 else 0.0)
            row.append(_fmt_mean_std(mu, sd, digits=2))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_pareto(df: pd.DataFrame, out_path: Path) -> None:
    # average across scenarios for each seed-policy then average
    tmp = df.groupby(["policy", "seed"]).agg({"cyber_harm_auc": "mean", "misbelief_auc": "mean"}).reset_index()
    pol = tmp.groupby("policy").agg({"cyber_harm_auc": "mean", "misbelief_auc": "mean"}).reset_index()

    plt.figure(figsize=(5.0, 3.6))
    plt.scatter(pol["cyber_harm_auc"], pol["misbelief_auc"])
    for _, r in pol.iterrows():
        plt.text(r["cyber_harm_auc"], r["misbelief_auc"], str(r["policy"]), fontsize=7)
    plt.xlabel("Cyber harm AUC (lower better)")
    plt.ylabel("Misbelief AUC (lower better)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def make_timeseries(steps: pd.DataFrame, scenario: str, metric: str, out_path: Path) -> None:
    sub = steps[steps["scenario"] == scenario].copy()
    if len(sub) == 0:
        return
    g = sub.groupby(["policy", "t"])[metric].mean().reset_index()

    plt.figure(figsize=(6.2, 3.4))
    for pol, sp in g.groupby("policy"):
        plt.plot(sp["t"], sp[metric], label=pol)
    plt.xlabel("t")
    ylabel_map = {
        "misbelief": "Misbelief",
        "trust": "Trust",
        "cyber_harm": "Cyber harm",
        "uncertainty": "Uncertainty",
        "polarization": "Polarization",
    }
    plt.ylabel(ylabel_map.get(metric, metric.replace("_", " ").title()))
    plt.legend(fontsize=7, ncols=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    paper_dir = Path(args.paper_dir)
    assets = paper_dir / "paper_assets"
    assets.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_dir / "summary_by_seed.csv")
    make_table_main(df, assets / "table_main.tex")
    make_table_protocol(df, assets / "table_protocol.tex")

    make_pareto(df, assets / "fig_pareto.pdf")

    steps_path = run_dir / "episode_steps.csv"
    if steps_path.exists():
        steps = pd.read_csv(steps_path)
        for scen in sorted(steps["scenario"].unique().tolist()):
            make_timeseries(steps, scen, "misbelief", assets / f"fig_misbelief_{scen}.pdf")
            make_timeseries(steps, scen, "trust", assets / f"fig_trust_{scen}.pdf")
            make_timeseries(steps, scen, "cyber_harm", assets / f"fig_cyberharm_{scen}.pdf")
            make_timeseries(steps, scen, "uncertainty", assets / f"fig_uncertainty_{scen}.pdf")
            make_timeseries(steps, scen, "polarization", assets / f"fig_polarization_{scen}.pdf")

    print(f"[OK] Wrote paper assets to: {assets}")


if __name__ == "__main__":
    main()
