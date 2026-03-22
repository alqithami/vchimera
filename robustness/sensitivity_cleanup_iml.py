"""Sensitivity analysis for IML hyperparameters.

Evaluates trained IML models under different (p_detect_false, p_review) settings
to understand how monitoring accuracy and contestability affect outcomes.

Also performs a sensitivity analysis on sanction magnitude.

Outputs:
    robustness/sensitivity_cleanup_iml_grid.csv
    robustness/fig_sensitivity_cleanup_iml.pdf
    robustness/sensitivity_harvest_iml_grid.csv (if harvest runs exist)
    robustness/fig_sensitivity_harvest_iml.pdf
"""
import os
import re
import glob
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#1f77b4"
BLACK  = "#000000"
GRAY   = "#7a7a7a"

P_FALSE = [1e-4, 1e-3, 1e-2, 5e-2]
P_REVIEW = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
SANCTIONS = [0.1, 0.25, 0.5, 1.0, 2.0]
EPISODES = 30
EVAL_SEED = 0

TMP_ROOT = "robustness/tmp_sens"
os.makedirs("robustness", exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)


def run_sensitivity_for_env(env_name: str):
    """Run sensitivity grid for a given environment."""
    pattern = f"runs/{env_name}_iml_agents*_seed*"
    RUNS = sorted(glob.glob(pattern))
    if not RUNS:
        print(f"[skip] No runs matched {pattern}")
        return

    out_csv = f"robustness/sensitivity_{env_name}_iml_grid.csv"
    out_fig = f"robustness/fig_sensitivity_{env_name}_iml.pdf"

    rows = []

    for run_dir in RUNS:
        run_name = os.path.basename(run_dir)
        cfg_yaml = os.path.join(run_dir, "config.yaml")
        model_pt = os.path.join(run_dir, "model.pt")

        if not (os.path.isfile(cfg_yaml) and os.path.isfile(model_pt)):
            print(f"[skip missing] {run_name}")
            continue

        with open(cfg_yaml, "r") as f:
            cfg = yaml.safe_load(f)

        cfg.setdefault("ppo", {})
        cfg["ppo"]["device"] = "cpu"
        cfg.setdefault("iml", {})
        cfg["iml"]["enabled"] = True

        # Grid: p_detect_false × p_review
        for pf in P_FALSE:
            for pr in P_REVIEW:
                tmp_dir = os.path.join(TMP_ROOT, f"{run_name}_pf{pf:g}_pr{pr:g}")
                os.makedirs(tmp_dir, exist_ok=True)

                shutil.copyfile(model_pt, os.path.join(tmp_dir, "model.pt"))

                cfg2 = dict(cfg)
                cfg2["ppo"] = dict(cfg.get("ppo", {}))
                cfg2["iml"] = dict(cfg.get("iml", {}))
                cfg2["iml"]["p_detect_false"] = float(pf)
                cfg2["iml"]["p_review"] = float(pr)

                with open(os.path.join(tmp_dir, "config.yaml"), "w") as f:
                    yaml.safe_dump(cfg2, f, sort_keys=False)

                cmd = [
                    sys.executable, "-m", "iml_ssd.experiments.evaluate",
                    "--run_dir", tmp_dir, "--episodes", str(EPISODES),
                    "--seed", str(EVAL_SEED),
                ]
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if p.returncode != 0:
                    print(f"[FAIL] {run_name} pf={pf} pr={pr}")
                    continue

                ev_path = os.path.join(tmp_dir, "eval.csv")
                if not (os.path.isfile(ev_path) and os.path.getsize(ev_path) > 0):
                    continue

                ev = pd.read_csv(ev_path)
                sanctions = float(ev["iml_sanctions"].mean()) if "iml_sanctions" in ev.columns else 0.0
                overturned = float(ev["iml_overturned"].mean()) if "iml_overturned" in ev.columns else 0.0

                rows.append({
                    "run_name": run_name,
                    "env": env_name,
                    "train_seed": int(re.search(r"_seed(\d+)$", run_name).group(1)),
                    "p_detect_false": pf,
                    "p_review": pr,
                    "sanction": float(cfg2["iml"].get("sanction", 0.5)),
                    "episodes": EPISODES,
                    "eval_seed": EVAL_SEED,
                    "return_mean": float(ev["return_mean"].mean()),
                    "gini": float(ev["return_gini"].mean()),
                    "false_pos": float(ev["iml_false_pos"].mean()) if "iml_false_pos" in ev.columns else 0.0,
                    "overturned": overturned,
                    "net_sanctions": sanctions - overturned,
                })

        # Grid: sanction magnitude (at nominal p_false and p_review)
        for s_mag in SANCTIONS:
            tmp_dir = os.path.join(TMP_ROOT, f"{run_name}_sanc{s_mag:g}")
            os.makedirs(tmp_dir, exist_ok=True)

            shutil.copyfile(model_pt, os.path.join(tmp_dir, "model.pt"))

            cfg2 = dict(cfg)
            cfg2["ppo"] = dict(cfg.get("ppo", {}))
            cfg2["iml"] = dict(cfg.get("iml", {}))
            cfg2["iml"]["sanction"] = float(s_mag)

            with open(os.path.join(tmp_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(cfg2, f, sort_keys=False)

            cmd = [
                sys.executable, "-m", "iml_ssd.experiments.evaluate",
                "--run_dir", tmp_dir, "--episodes", str(EPISODES),
                "--seed", str(EVAL_SEED),
            ]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if p.returncode != 0:
                continue

            ev_path = os.path.join(tmp_dir, "eval.csv")
            if not (os.path.isfile(ev_path) and os.path.getsize(ev_path) > 0):
                continue

            ev = pd.read_csv(ev_path)
            sanctions = float(ev["iml_sanctions"].mean()) if "iml_sanctions" in ev.columns else 0.0
            overturned = float(ev["iml_overturned"].mean()) if "iml_overturned" in ev.columns else 0.0

            rows.append({
                "run_name": run_name,
                "env": env_name,
                "train_seed": int(re.search(r"_seed(\d+)$", run_name).group(1)),
                "p_detect_false": float(cfg2["iml"].get("p_detect_false", 0.01)),
                "p_review": float(cfg2["iml"].get("p_review", 0.2)),
                "sanction": s_mag,
                "episodes": EPISODES,
                "eval_seed": EVAL_SEED,
                "return_mean": float(ev["return_mean"].mean()),
                "gini": float(ev["return_gini"].mean()),
                "false_pos": float(ev["iml_false_pos"].mean()) if "iml_false_pos" in ev.columns else 0.0,
                "overturned": overturned,
                "net_sanctions": sanctions - overturned,
            })

    if not rows:
        print(f"[WARN] No sensitivity results for {env_name}")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")

    # ---- Figures ----
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
    })

    # Figure 1: p_review × p_detect_false sensitivity
    pf_pr_df = df[df["sanction"] == 0.5]  # nominal sanction
    if not pf_pr_df.empty:
        agg = pf_pr_df.groupby(["p_detect_false", "p_review"], as_index=False).agg(
            return_mean=("return_mean", "mean"),
            net_sanctions=("net_sanctions", "mean"),
            gini=("gini", "mean"),
        )

        linestyles = {1e-4: ":", 1e-3: "--", 1e-2: "-", 5e-2: "-."}

        fig, axes_arr = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

        for pf in sorted(agg["p_detect_false"].unique()):
            sub = agg[agg["p_detect_false"] == pf].sort_values("p_review")
            x = sub["p_review"].to_numpy()
            ls = linestyles.get(pf, "-")
            axes_arr[0].plot(x, sub["return_mean"].to_numpy(), linestyle=ls, color=BLACK,
                             linewidth=1.1, label=f"p_false={pf:g}")
            axes_arr[1].plot(x, sub["net_sanctions"].to_numpy(), linestyle=ls, color=BLACK,
                             linewidth=1.1, label=f"p_false={pf:g}")
            axes_arr[2].plot(x, sub["gini"].to_numpy(), linestyle=ls, color=BLACK,
                             linewidth=1.1, label=f"p_false={pf:g}")

        # Mark nominal configuration
        nom = agg[(agg["p_detect_false"] == 1e-2) & (agg["p_review"] == 0.2)]
        if len(nom) == 1:
            axes_arr[0].plot([0.2], [float(nom["return_mean"].iloc[0])], marker="o", markersize=5, color=ACCENT)
            axes_arr[1].plot([0.2], [float(nom["net_sanctions"].iloc[0])], marker="o", markersize=5, color=ACCENT)
            axes_arr[2].plot([0.2], [float(nom["gini"].iloc[0])], marker="o", markersize=5, color=ACCENT)

        axes_arr[0].set_xlabel("p_review")
        axes_arr[0].set_ylabel("Eval return (mean/agent)")
        axes_arr[0].set_title(f"{env_name.capitalize()}: Welfare sensitivity")
        axes_arr[1].set_xlabel("p_review")
        axes_arr[1].set_ylabel("Net sanctions / episode")
        axes_arr[1].set_title(f"{env_name.capitalize()}: Sanction overhead")
        axes_arr[2].set_xlabel("p_review")
        axes_arr[2].set_ylabel("Gini coefficient")
        axes_arr[2].set_title(f"{env_name.capitalize()}: Inequality sensitivity")

        handles, labels = axes_arr[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
        fig.savefig(out_fig)
        fig.savefig(out_fig.replace(".pdf", ".png"), dpi=200)
        plt.close(fig)
        print(f"Wrote {out_fig}")

    # Figure 2: Sanction magnitude sensitivity
    sanc_df = df[df["p_review"] == 0.2]  # nominal review
    if not sanc_df.empty and len(sanc_df["sanction"].unique()) > 1:
        agg_s = sanc_df.groupby("sanction", as_index=False).agg(
            return_mean=("return_mean", "mean"),
            gini=("gini", "mean"),
            net_sanctions=("net_sanctions", "mean"),
        )

        fig, axes_arr = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
        x = agg_s["sanction"].to_numpy()
        axes_arr[0].plot(x, agg_s["return_mean"].to_numpy(), "-o", color=BLACK, linewidth=1.1, markersize=4)
        axes_arr[0].set_xlabel("Sanction magnitude")
        axes_arr[0].set_ylabel("Eval return (mean/agent)")
        axes_arr[0].set_title(f"{env_name.capitalize()}: Welfare vs sanction")

        axes_arr[1].plot(x, agg_s["gini"].to_numpy(), "-o", color=BLACK, linewidth=1.1, markersize=4)
        axes_arr[1].set_xlabel("Sanction magnitude")
        axes_arr[1].set_ylabel("Gini coefficient")
        axes_arr[1].set_title(f"{env_name.capitalize()}: Inequality vs sanction")

        # Mark nominal
        nom_s = agg_s[agg_s["sanction"] == 0.5]
        if len(nom_s) == 1:
            axes_arr[0].plot([0.5], [float(nom_s["return_mean"].iloc[0])], marker="D", markersize=6, color=ACCENT)
            axes_arr[1].plot([0.5], [float(nom_s["gini"].iloc[0])], marker="D", markersize=6, color=ACCENT)

        sanc_fig = f"robustness/fig_sanction_sensitivity_{env_name}.pdf"
        fig.savefig(sanc_fig)
        fig.savefig(sanc_fig.replace(".pdf", ".png"), dpi=200)
        plt.close(fig)
        print(f"Wrote {sanc_fig}")


def main():
    for env in ("cleanup", "harvest"):
        run_sensitivity_for_env(env)


if __name__ == "__main__":
    main()
