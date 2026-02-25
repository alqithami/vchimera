#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from tqdm import tqdm

from vchimera.factory import build_env, deep_update, load_yaml
from vchimera.metrics import summarize_episode, summary_to_row
from vchimera.policies import make_policy
from vchimera.protocol import CommsAction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V-CHIMERA experiments from a YAML config.")
    p.add_argument("--config", required=True, help="Path to experiment config YAML.")
    p.add_argument("--override", default=None, help="Optional YAML override applied to all scenarios (deep merge).")
    p.add_argument("--write_steps", action="store_true", help="Write per-step logs to CSV (can be large).")
    return p.parse_args()


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    args = parse_args()
    exp_cfg = load_yaml(args.config)

    exp_name = exp_cfg.get("name", Path(args.config).stem)
    output_root = Path(exp_cfg.get("output_root", "runs"))
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{exp_name}_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    override = {}
    if args.override:
        override = load_yaml(args.override)

    if exp_cfg.get("override"):
        override = deep_update(override, load_yaml(exp_cfg["override"]))

    seeds = exp_cfg.get("seeds", list(range(8)))

    scenario_specs = exp_cfg.get("scenarios", [])
    policy_specs = exp_cfg.get("policies", [])

    rows = []
    step_rows = []

    for scen in scenario_specs:
        scen_name = scen["name"]
        scen_cfg_path = scen["config"]
        base_scen_cfg = load_yaml(scen_cfg_path)
        scen_cfg = deep_update(base_scen_cfg, override)

        env, evaluator, _ = build_env(scen_cfg)

        for pol in policy_specs:
            pol_name = pol["name"]
            label = pol.get("label", pol_name + ("+shield" if pol.get("shield", False) else ""))
            use_shield = bool(pol.get("shield", False))

            policy = make_policy(pol_name)

            for seed in tqdm(seeds, desc=f"{scen_name} | {label}", leave=False):
                obs = env.reset(seed=seed)
                policy.reset()
                last_msg_t = -10_000
                steps = []

                for t in range(int(scen_cfg.get("env", {}).get("horizon", 60))):
                    cyber_action, proposed = policy.act(obs, t)
                    if not isinstance(proposed, CommsAction):
                        proposed = CommsAction(**proposed) if isinstance(proposed, dict) else CommsAction(type=str(proposed))

                    attempted = evaluator.count_violations(proposed, obs, t, last_msg_t)
                    if use_shield:
                        executed_action, interventions = evaluator.shield(proposed, obs, t, last_msg_t)
                    else:
                        executed_action, interventions = proposed, 0

                    executed = evaluator.count_violations(executed_action, obs, t, last_msg_t)

                    if executed_action.type != "silence":
                        last_msg_t = t

                    obs, log, done = env.step(
                        cyber_action=cyber_action,
                        comms_action=executed_action,
                        protocol_attempted=attempted,
                        protocol_executed=executed,
                        shield_interventions=interventions,
                    )
                    steps.append(log)

                    if args.write_steps or exp_cfg.get("write_steps", False):
                        step_rows.append({
                            "scenario": scen_name,
                            "policy": label,
                            "seed": seed,
                            **log.__dict__,
                        })

                    if done:
                        break

                summ = summarize_episode(steps)
                row = {
                    "scenario": scen_name,
                    "policy": label,
                    "seed": seed,
                    **summary_to_row(summ),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "summary_by_seed.csv", index=False)

    # aggregate by scenario+policy
    agg = df.groupby(["scenario", "policy"]).agg(["mean", "std", "count"])
    # flatten columns
    agg.columns = ["_".join([c[0], c[1]]) for c in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    agg.to_csv(run_dir / "summary.csv", index=False)

    if len(step_rows) > 0:
        pd.DataFrame(step_rows).to_csv(run_dir / "episode_steps.csv", index=False)

    # write run manifest
    manifest = {
        "experiment": exp_name,
        "config": str(args.config),
        "override": args.override,
        "seeds": seeds,
        "scenarios": scenario_specs,
        "policies": policy_specs,
    }
    with open(run_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f)

    print(f"[OK] Wrote results to: {run_dir}")


if __name__ == "__main__":
    main()