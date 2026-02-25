#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from vchimera.factory import build_env, deep_update, load_yaml
from vchimera.metrics import summarize_episode, summary_to_row
from vchimera.policies import make_policy
from vchimera.protocol import CommsAction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robustness/sensitivity sweep.")
    p.add_argument("--config", required=True, help="Sensitivity config YAML")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    scenario_cfg = load_yaml(cfg["scenario"]["config"])
    scenario_name = cfg["scenario"].get("name", "scenario")

    policy_name = cfg["policy"]["name"]
    use_shield = bool(cfg["policy"].get("shield", True))
    label = cfg["policy"].get("label", policy_name + ("+shield" if use_shield else ""))

    seeds = cfg.get("seeds", [0, 1, 2, 3])

    coupling_grid = cfg["grid"]["coupling_scale"]
    moderation_grid = cfg["grid"]["moderation_scale"]

    out_dir = Path(cfg.get("output_dir", "runs")) / (cfg.get("name", "sensitivity") + "_grid")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for c_scale in tqdm(coupling_grid, desc="coupling"):
        for m_scale in moderation_grid:
            override = {
                "coupling": {"cyber_to_social_scale": float(c_scale), "social_to_cyber_scale": float(c_scale)},
                "social": {
                    "platforms": [
                        # scale moderation knobs uniformly across platforms
                        {"name": "microblog", "mod_remove_prob": float(0.10 * m_scale), "mod_label_prob": float(0.15 * m_scale)},
                        {"name": "messaging", "mod_remove_prob": float(0.05 * m_scale), "mod_label_prob": float(0.08 * m_scale)},
                        {"name": "video", "mod_remove_prob": float(0.12 * m_scale), "mod_label_prob": float(0.20 * m_scale)},
                    ]
                },
            }
            scen = deep_update(scenario_cfg, override)
            env, evaluator, _ = build_env(scen)

            policy = make_policy(policy_name)

            # run seeds
            per_seed = []
            for seed in seeds:
                obs = env.reset(seed=seed)
                policy.reset()
                last_msg_t = -10_000
                steps = []
                horizon = int(scen.get("env", {}).get("horizon", 60))
                for t in range(horizon):
                    cyber_action, proposed = policy.act(obs, t)
                    attempted = evaluator.count_violations(proposed, obs, t, last_msg_t)
                    if use_shield:
                        executed, interventions = evaluator.shield(proposed, obs, t, last_msg_t)
                    else:
                        executed, interventions = proposed, 0
                    executed_v = evaluator.count_violations(executed, obs, t, last_msg_t)
                    if executed.type != "silence":
                        last_msg_t = t
                    obs, log, done = env.step(cyber_action, executed, attempted, executed_v, interventions)
                    steps.append(log)
                    if done:
                        break
                summ = summarize_episode(steps)
                per_seed.append(summary_to_row(summ))

            # average over seeds
            df = pd.DataFrame(per_seed)
            row = {
                "scenario": scenario_name,
                "policy": label,
                "coupling_scale": float(c_scale),
                "moderation_scale": float(m_scale),
                "cyber_harm_auc": float(df["cyber_harm_auc"].mean()),
                "misbelief_auc": float(df["misbelief_auc"].mean()),
                "trust_auc": float(df["trust_auc"].mean()),
                "protocol_executed": float(df["protocol_executed"].mean()),
            }
            rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "sensitivity.csv", index=False)

    # heatmap figure: misbelief_auc as function of coupling x moderation
    # pivot
    piv = res.pivot_table(index="coupling_scale", columns="moderation_scale", values="misbelief_auc", aggfunc="mean")
    plt.figure(figsize=(6.0, 3.8))
    plt.imshow(piv.values, aspect="auto", origin="lower")
    plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns], rotation=45, ha="right")
    plt.yticks(range(len(piv.index)), [str(r) for r in piv.index])
    plt.xlabel("Moderation scale")
    plt.ylabel("Coupling scale")
    plt.title("Misbelief AUC (lower better)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_sensitivity_misbelief.pdf", bbox_inches="tight")
    plt.close()

    print("[OK] Wrote:", out_dir)


if __name__ == "__main__":
    main()
