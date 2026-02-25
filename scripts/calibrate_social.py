#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from vchimera.backends.social_abm import SocialABMBackend, SocialABMConfig, PlatformConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Moment-matching calibration for the social ABM (crisis-regime).")
    p.add_argument("--targets", required=True, help="YAML file with target ranges.")
    p.add_argument("--out", required=True, help="Output YAML override with calibrated social parameters.")
    p.add_argument("--n_iter", type=int, default=None, help="Override search.n_iter in targets YAML.")
    p.add_argument("--seed", type=int, default=None, help="Override search.seed in targets YAML.")
    return p.parse_args()


def mid(rng: Tuple[float, float]) -> float:
    return 0.5 * (float(rng[0]) + float(rng[1]))


def within(x: float, rng: Tuple[float, float]) -> float:
    lo, hi = float(rng[0]), float(rng[1])
    if x < lo:
        return (lo - x) / max(1e-9, (hi - lo))
    if x > hi:
        return (x - hi) / max(1e-9, (hi - lo))
    return 0.0


def score(stats: Dict[str, float], targets: Dict[str, Tuple[float, float]]) -> float:
    """
    Penalty for being outside target ranges, plus weak pull toward midpoints.
    Targets are intended to represent *crisis-regime averages* over a fixed horizon.
    """
    s = 0.0
    for k, rng in targets.items():
        s += within(float(stats[k]), rng) ** 2
        # encourage closeness to midpoint (weakly)
        s += 0.15 * (float(stats[k]) - mid(rng)) ** 2
    return float(s)


def run_social(cfg: SocialABMConfig, seed: int, horizon: int) -> Dict[str, float]:
    """
    Simulate a *crisis-regime* trajectory:
    - initial shock
    - persistent low-intensity narrative pressure (outage/exfiltration rumor background)
    We return time-averaged metrics over the horizon to support calibration.
    """
    backend = SocialABMBackend(cfg)
    rng = np.random.default_rng(seed)

    obs = backend.reset(seed=seed, initial_events={"outage": 0.35, "exfiltration": 0.25})

    acc = {"misbelief": 0.0, "trust": 0.0, "uncertainty": 0.0, "polarization": 0.0,
           "compliance": 0.0, "reporting": 0.0}

    for t in range(horizon):
        # persistent narrative pressure with mild stochasticity
        events = {
            "outage": float(np.clip(0.04 + 0.03 * rng.standard_normal(), 0.0, 0.20)),
            "exfiltration": float(np.clip(0.03 + 0.02 * rng.standard_normal(), 0.0, 0.15)),
            "ransomware": 0.0,
            "verified_update": 0.0,
        }
        obs, metrics = backend.step({"type": "silence"}, events)
        for k in acc:
            acc[k] += float(metrics[k])

    for k in acc:
        acc[k] /= float(horizon)
    return acc


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.targets).read_text(encoding="utf-8"))

    targets = cfg["targets"]
    search = cfg.get("search", {})
    n_iter = int(args.n_iter if args.n_iter is not None else search.get("n_iter", 120))
    horizon = int(search.get("horizon", 40))
    seed0 = int(args.seed if args.seed is not None else search.get("seed", 0))

    rng = np.random.default_rng(seed0)

    best_s = 1e9
    best_params = None
    best_stats = None

    # base platform defaults (can be changed in scenario YAMLs; calibration focuses on ABM core)
    platforms = [
        PlatformConfig(name="microblog", mean_degree=14, homophily=0.62, mod_remove_prob=0.10, mod_label_prob=0.15, amplification=1.15),
        PlatformConfig(name="messaging", mean_degree=10, homophily=0.72, mod_remove_prob=0.05, mod_label_prob=0.08, amplification=0.95),
        PlatformConfig(name="video", mean_degree=8, homophily=0.58, mod_remove_prob=0.12, mod_label_prob=0.20, amplification=1.05),
    ]

    for i in range(n_iter):
        params = dict(
            bots_frac=float(rng.uniform(0.02, 0.10)),
            base_post_misinfo=float(rng.uniform(0.025, 0.090)),
            base_post_correction=float(rng.uniform(0.010, 0.060)),
            susceptibility=float(rng.uniform(0.10, 0.30)),
            correction_efficacy=float(rng.uniform(0.10, 0.30)),
            official_efficacy=float(rng.uniform(0.15, 0.40)),
            uncertainty_decay=float(rng.uniform(0.005, 0.040)),
            uncertainty_from_labeled_misinfo=float(rng.uniform(0.02, 0.10)),
            shock_uncertainty=float(rng.uniform(0.15, 0.45)),
            shock_trust_drop=float(rng.uniform(0.10, 0.35)),
            shock_misbelief_inject=float(rng.uniform(0.02, 0.12)),
        )

        social_cfg = SocialABMConfig(
            n_agents=900,
            n_communities=9,
            platforms=platforms,
            **params,
        )
        stats = run_social(social_cfg, seed=seed0 + i, horizon=horizon)

        s = score(stats, targets)
        if s < best_s:
            best_s = s
            best_params = params
            best_stats = stats

    out = {
        "social": best_params,
        "calibration_report": {
            "score": float(best_s),
            "final_metrics": best_stats,
            "targets": targets,
            "notes": "Time-averaged metrics over a crisis-regime horizon under persistent narrative pressure.",
        },
    }
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print("[OK] Wrote calibrated override:", args.out)
    print("Best score:", best_s)
    print("Best stats:", best_stats)


if __name__ == "__main__":
    main()
