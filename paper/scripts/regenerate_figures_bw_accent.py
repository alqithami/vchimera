"""
Regenerate professional, print-friendly figures (grayscale + single accent color) for the
V-CHIMERA Springer "Cybersecurity" (journal 42400) submission.

This script expects:
  - episode_steps.csv from a journal_main run (contains per-step metrics for all scenarios/policies)
  - sensitivity.csv from the sensitivity_grid run (contains misbelief_auc over a (coupling, moderation) grid)
  - optional: episode_steps.csv for a CybORG transfer run (if you want the CybORG appendix figures)

Example:
  python scripts/regenerate_figures_bw_accent.py \
      --episode_steps runs/journal_main_YYYYMMDD_HHMMSS/episode_steps.csv \
      --sensitivity runs/sensitivity_grid/sensitivity.csv \
      --out_dir figures

Notes:
  - Produces PDF figures (vector) suitable for submission.
  - Uses one accent color to highlight the main system: "vchimera+shield".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ACCENT_POLICY = "vchimera-ais+shield"
ACCENT_COLOR = "#0072B2"  # Okabe–Ito blue (colorblind-safe)


def _styles():
    grays = ["#000000", "#4D4D4D", "#7A7A7A", "#A6A6A6", "#C0C0C0", "#D9D9D9"]
    styles = {
        # Shielded baselines / ablations
        "pipeline+shield": {"color": grays[0], "ls": "-", "lw": 1.4, "alpha": 0.95},
        "vchimera+shield": {"color": grays[1], "ls": "--", "lw": 1.35, "alpha": 0.95},
        "vchimera-ais+shield": {"color": ACCENT_COLOR, "ls": "-", "lw": 2.1, "alpha": 1.0},
        "vchimera-no-coupling+shield": {"color": grays[2], "ls": "-.", "lw": 1.25, "alpha": 0.95},
        "vchimera-no-targeting+shield": {"color": grays[3], "ls": ":", "lw": 1.25, "alpha": 0.95},

        # Unshielded (faint reference)
        "pipeline": {"color": grays[4], "ls": "-", "lw": 1.0, "alpha": 0.45},
        "vchimera": {"color": grays[5], "ls": "--", "lw": 1.0, "alpha": 0.45},
    }
    return styles



def _setup_rcparams():
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def make_timeseries_grid(steps: pd.DataFrame, out_path: Path) -> None:
    styles = _styles()
    policies_plot = [
    "pipeline+shield",
    "vchimera+shield",
    "vchimera-ais+shield",
    "vchimera-no-coupling+shield",
    "vchimera-no-targeting+shield",
]
    faint_policies = ["pipeline", "vchimera"]

    scenario_titles = {
        "ransomware_rumor": "Ransomware rumor",
        "outage_rumor": "Outage rumor",
        "exfiltration_scam": "Exfiltration scam",
    }
    scenarios = list(steps["scenario"].unique())

    metrics = [
        ("misbelief", "Misbelief"),
        ("trust", "Trust"),
        ("uncertainty", "Uncertainty"),
        ("cyber_harm", "Cyber harm"),
    ]

    fig, axes = plt.subplots(nrows=len(scenarios), ncols=len(metrics), figsize=(10.5, 6.5), sharex=True)

    for i, scen in enumerate(scenarios):
        sub_s = steps[steps["scenario"] == scen]
        g = sub_s.groupby(["policy", "t"], as_index=False)[[m[0] for m in metrics] + ["polarization"]].mean()

        for j, (mkey, mlabel) in enumerate(metrics):
            ax = axes[i, j]

            for pol in faint_policies:
                if pol in g["policy"].unique():
                    sp = g[g["policy"] == pol]
                    ax.plot(sp["t"], sp[mkey], **styles[pol])

            for pol in policies_plot:
                if pol in g["policy"].unique():
                    sp = g[g["policy"] == pol]
                    ax.plot(sp["t"], sp[mkey], **styles[pol])

            if i == 0:
                ax.set_title(mlabel)
            if j == 0:
                ax.set_ylabel(scenario_titles.get(scen, scen))
            if i == len(scenarios) - 1:
                ax.set_xlabel("Time step")
            ax.grid(True, linewidth=0.3, alpha=0.4)

    handles = []
    labels = []
    for pol in faint_policies + policies_plot:
        st = styles[pol]
        handles.append(Line2D([0], [0], color=st["color"], lw=st["lw"], ls=st["ls"], alpha=st["alpha"]))
        labels.append(pol)

    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_polarization_grid(steps: pd.DataFrame, out_path: Path) -> None:
    styles = _styles()
    policies_plot = [
    "pipeline+shield",
    "vchimera+shield",
    "vchimera-ais+shield",
    "vchimera-no-coupling+shield",
    "vchimera-no-targeting+shield",
]
    faint_policies = ["pipeline", "vchimera"]

    scenario_titles = {
        "ransomware_rumor": "Ransomware rumor",
        "outage_rumor": "Outage rumor",
        "exfiltration_scam": "Exfiltration scam",
    }
    scenarios = list(steps["scenario"].unique())

    fig, axes = plt.subplots(nrows=1, ncols=len(scenarios), figsize=(10.5, 2.8), sharey=True, sharex=True)

    for j, scen in enumerate(scenarios):
        ax = axes[j]
        sub_s = steps[steps["scenario"] == scen]
        g = sub_s.groupby(["policy", "t"], as_index=False)["polarization"].mean()

        for pol in faint_policies:
            if pol in g["policy"].unique():
                sp = g[g["policy"] == pol]
                ax.plot(sp["t"], sp["polarization"], **styles[pol])

        for pol in policies_plot:
            if pol in g["policy"].unique():
                sp = g[g["policy"] == pol]
                ax.plot(sp["t"], sp["polarization"], **styles[pol])

        ax.set_title(scenario_titles.get(scen, scen))
        ax.set_xlabel("Time step")
        ax.grid(True, linewidth=0.3, alpha=0.4)

    axes[0].set_ylabel("Polarization")

    handles = []
    labels = []
    for pol in faint_policies + policies_plot:
        st = styles[pol]
        handles.append(Line2D([0], [0], color=st["color"], lw=st["lw"], ls=st["ls"], alpha=st["alpha"]))
        labels.append(pol)

    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_tradeoff(summary_by_seed: pd.DataFrame, out_path: Path) -> None:
    styles = _styles()
    grays = ["#000000", "#4D4D4D", "#7A7A7A", "#A6A6A6", "#C0C0C0", "#D9D9D9"]

    tmp = summary_by_seed.groupby(["policy", "seed"], as_index=False).agg(
        cyber_harm_auc=("cyber_harm_auc", "mean"),
        misbelief_auc=("misbelief_auc", "mean"),
        trust_auc=("trust_auc", "mean"),
    )
    pol = tmp.groupby("policy", as_index=False).agg(
        cyber_harm_auc=("cyber_harm_auc", "mean"),
        misbelief_auc=("misbelief_auc", "mean"),
        trust_auc=("trust_auc", "mean"),
    )

    markers = {
        "pipeline": "o",
        "pipeline+shield": "s",
        "vchimera": "^",
        "vchimera+shield": "D",
        "vchimera-ais+shield": "*",
        "vchimera-no-coupling+shield": "v",
        "vchimera-no-targeting+shield": "P",
    }

    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    for _, r in pol.iterrows():
        p = r["policy"]
        m = markers.get(p, "o")
        if p == ACCENT_POLICY:
            ax.scatter(
                r["cyber_harm_auc"],
                r["misbelief_auc"],
                s=70,
                marker=m,
                color=ACCENT_COLOR,
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
        else:
            col = grays[1] if "+shield" in p else grays[4]
            ax.scatter(
                r["cyber_harm_auc"],
                r["misbelief_auc"],
                s=55,
                marker=m,
                color=col,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
            )
        ax.text(
            r["cyber_harm_auc"],
            r["misbelief_auc"],
            " " + p.replace("vchimera", "V").replace("pipeline", "P"),
            fontsize=7,
            va="center",
        )

    ax.set_xlabel("Cyber harm AUC (↓)")
    ax.set_ylabel("Misbelief AUC (↓)")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_title("Average trade-off across scenarios")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_sensitivity_heatmap(sens: pd.DataFrame, out_path: Path) -> None:
    pivot = sens.pivot(index="coupling_scale", columns="moderation_scale", values="misbelief_auc").sort_index(
        ascending=True
    )
    couplings = pivot.index.values
    mods = pivot.columns.values
    data = pivot.values

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    im = ax.imshow(data, cmap="Greys", aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(mods)))
    ax.set_xticklabels([f"{m:.1f}" for m in mods])
    ax.set_yticks(np.arange(len(couplings)))
    ax.set_yticklabels([f"{c:.1f}" for c in couplings])
    ax.set_xlabel("Moderation scale")
    ax.set_ylabel("Coupling scale")
    ax.set_title("Sensitivity: Misbelief AUC (ransomware rumor)")

    for i in range(len(couplings)):
        for j in range(len(mods)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7, color="black")

    if 1.0 in couplings and 1.0 in mods:
        i = np.where(couplings == 1.0)[0][0]
        j = np.where(mods == 1.0)[0][0]
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ACCENT_COLOR, linewidth=2.0))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Misbelief AUC (↓)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)




def make_sensitivity_compare_heatmap(sens_a: pd.DataFrame, sens_b: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side sensitivity comparison (same grayscale scale).

    Intended for comparing naive coupling vs. immune-gated coupling (IGC).

    Fixes common Matplotlib layout issues where the colorbar overlaps the right panel by
    reserving explicit space for the colorbar and placing it in its own axis.
    """
    piv_a = (
        sens_a.pivot(index="coupling_scale", columns="moderation_scale", values="misbelief_auc")
        .sort_index(ascending=True)
    )
    piv_b = (
        sens_b.pivot(index="coupling_scale", columns="moderation_scale", values="misbelief_auc")
        .sort_index(ascending=True)
    )

    couplings = piv_a.index.values
    mods = piv_a.columns.values

    A = piv_a.values
    B = piv_b.values

    vmin = float(np.nanmin([np.nanmin(A), np.nanmin(B)]))
    vmax = float(np.nanmax([np.nanmax(A), np.nanmax(B)]))

    def _cell_text_color(v: float) -> str:
        # Use white text on dark cells for readability.
        if not np.isfinite(v) or vmax <= vmin:
            return "black"
        z = (v - vmin) / (vmax - vmin + 1e-12)
        return "white" if z >= 0.62 else "black"

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), sharey=True)

    # Reserve space on the right for the colorbar.
    fig.subplots_adjust(right=0.86, wspace=0.22, bottom=0.18, top=0.88)

    im = None
    for ax, data, title in [
        (axes[0], A, "Naive coupling"),
        (axes[1], B, "Immune-gated coupling (IGC)"),
    ]:
        im = ax.imshow(data, cmap="Greys", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(mods)))
        ax.set_xticklabels([f"{m:.1f}" for m in mods])
        ax.set_yticks(np.arange(len(couplings)))
        ax.set_yticklabels([f"{c:.1f}" for c in couplings])
        ax.set_xlabel("Moderation scale")
        ax.set_title(title)

        # Cell annotations
        for i in range(len(couplings)):
            for j in range(len(mods)):
                v = data[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=_cell_text_color(v),
                )

        # Highlight the default setting (1.0, 1.0)
        if 1.0 in couplings and 1.0 in mods:
            i0 = np.where(couplings == 1.0)[0][0]
            j0 = np.where(mods == 1.0)[0][0]
            ax.add_patch(
                Rectangle(
                    (j0 - 0.5, i0 - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=ACCENT_COLOR,
                    linewidth=2.0,
                )
            )

    axes[0].set_ylabel("Coupling scale")

    # Dedicated axis for the colorbar (prevents overlap with the right panel).
    cax = fig.add_axes([0.885, 0.20, 0.018, 0.64])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Misbelief AUC (↓)", labelpad=8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_steps", type=str, required=True, help="Path to episode_steps.csv for journal_main.")
    ap.add_argument("--summary_by_seed", type=str, default=None, help="Path to summary_by_seed.csv for trade-off plot.")
    ap.add_argument("--sensitivity", type=str, required=True, help="Path to sensitivity.csv for the sensitivity plot.")
    ap.add_argument("--sensitivity2", type=str, default=None, help="Optional second sensitivity.csv for a side-by-side comparison figure.")
    ap.add_argument("--out_dir", type=str, default="figures", help="Output directory for PDFs.")
    args = ap.parse_args()

    _setup_rcparams()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = pd.read_csv(args.episode_steps)
    make_timeseries_grid(steps, out_dir / "fig_timeseries_grid_gray.pdf")
    make_polarization_grid(steps, out_dir / "fig_polarization_grid_gray.pdf")

    if args.summary_by_seed:
        summ = pd.read_csv(args.summary_by_seed)
        make_tradeoff(summ, out_dir / "fig_tradeoff_bw.pdf")

    sens = pd.read_csv(args.sensitivity)
    make_sensitivity_heatmap(sens, out_dir / "fig_sensitivity_misbelief_gray.pdf")

    if args.sensitivity2:
        sens2 = pd.read_csv(args.sensitivity2)
        make_sensitivity_compare_heatmap(sens, sens2, out_dir / "fig_sensitivity_compare_gray.pdf")

    print("[OK] Wrote figures to:", out_dir)


if __name__ == "__main__":
    main()
