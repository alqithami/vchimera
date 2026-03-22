"""Unified evaluation script supporting all experimental conditions.

Loads a trained model from a run directory and evaluates it for a given number
of episodes.  Supports baseline, IML (all ablations), IA, and SI conditions.

Usage:
    python -m iml_ssd.experiments.evaluate --run_dir runs/cleanup_iml_agents5_seed0 --episodes 50
    python -m iml_ssd.experiments.evaluate --run_dir runs/cleanup_ia_agents5_seed0 --episodes 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from iml_ssd.config import load_yaml
from iml_ssd.envs.ssd_env import get_action_space_n, get_agent_ids, make_ssd_env, preprocess_obs
from iml_ssd.institution.iml_wrapper import IMLConfig, IMLWrapper
from iml_ssd.institution.rules import HighWasteNoCleanRule, LowAppleDensityHarvestRule, NoPunishmentBeamRule
from iml_ssd.baselines.inequity_aversion import IAConfig, IAWrapper
from iml_ssd.baselines.social_influence import SIConfig, SIWrapper
from iml_ssd.rl.networks import SharedCNNActorCritic
from iml_ssd.utils.logging import CSVLogger
from iml_ssd.utils.metrics import gini


def _build_iml_config(cfg: Dict[str, Any]) -> IMLConfig:
    iml_cfg = cfg.get("iml", {}) or {}
    enabled = bool(iml_cfg.get("enabled", False))

    rules: List[Any] = []
    rule_names = iml_cfg.get("rules", ["no_punishment_beam"])
    if isinstance(rule_names, str):
        rule_names = [rule_names]
    for r in rule_names:
        r = str(r)
        if r == "no_punishment_beam":
            rules.append(NoPunishmentBeamRule())
        elif r == "low_density_harvest":
            params = iml_cfg.get("low_density_harvest", {}) or {}
            rules.append(LowAppleDensityHarvestRule(**params))
        elif r == "high_waste_no_clean":
            params = iml_cfg.get("high_waste_no_clean", {}) or {}
            rules.append(HighWasteNoCleanRule(**params))
        else:
            raise ValueError(f"Unknown rule '{r}'.")

    return IMLConfig(
        enabled=enabled,
        p_detect_true=float(iml_cfg.get("p_detect_true", 0.9)),
        p_detect_false=float(iml_cfg.get("p_detect_false", 0.01)),
        sanction=float(iml_cfg.get("sanction", 0.5)),
        p_review=float(iml_cfg.get("p_review", 0.0)),
        write_ledger=False,
        ledger_path=None,
        rules=rules,
    )


def _build_ia_config(cfg: Dict[str, Any]) -> IAConfig:
    ia_cfg = cfg.get("ia", {}) or {}
    return IAConfig(
        enabled=bool(ia_cfg.get("enabled", False)),
        alpha=float(ia_cfg.get("alpha", 5.0)),
        beta=float(ia_cfg.get("beta", 0.05)),
    )


def _build_si_config(cfg: Dict[str, Any]) -> SIConfig:
    si_cfg = cfg.get("si", {}) or {}
    return SIConfig(
        enabled=bool(si_cfg.get("enabled", False)),
        influence_weight=float(si_cfg.get("influence_weight", 1.0)),
        mode=str(si_cfg.get("mode", "reward_deviation")),
    )


def _determine_condition(cfg: Dict[str, Any]) -> str:
    """Determine the experimental condition from the config."""
    condition = cfg.get("condition", None)
    if condition:
        return str(condition)

    iml_enabled = bool(cfg.get("iml", {}).get("enabled", False))
    ia_enabled = bool(cfg.get("ia", {}).get("enabled", False))
    si_enabled = bool(cfg.get("si", {}).get("enabled", False))

    if ia_enabled:
        return "ia"
    if si_enabled:
        return "si"
    if iml_enabled:
        iml_cfg = cfg.get("iml", {})
        sanction = float(iml_cfg.get("sanction", 0.5))
        p_review = float(iml_cfg.get("p_review", 0.0))
        if sanction == 0.0:
            return "monitor_only"
        if p_review == 0.0:
            return "sanction_no_review"
        if p_review >= 0.7:
            return "high_review"
        return "iml"
    return "baseline"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_suffix", type=str, default="",
                        help="Suffix for output CSV, e.g., '_seed42' -> eval_seed42.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    cfg = load_yaml(cfg_path)

    seed = int(args.seed) if args.seed is not None else int(cfg.get("train", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load model
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    obs_shape = tuple(ckpt["obs_shape"])
    n_actions = int(ckpt["n_actions"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharedCNNActorCritic(obs_shape=obs_shape, n_actions=n_actions).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    env_name = str(cfg.get("env", {}).get("name", "cleanup")).lower()
    num_agents = int(cfg.get("env", {}).get("num_agents", 5))
    env_kwargs = cfg.get("env", {}).get("kwargs", {}) or {}
    env = make_ssd_env(env_name, num_agents=num_agents, seed=seed, **env_kwargs)

    condition = _determine_condition(cfg)

    # Apply the appropriate wrapper
    if condition in ("iml", "monitor_only", "sanction_no_review", "high_review"):
        iml_cfg = _build_iml_config(cfg)
        if iml_cfg.enabled:
            env = IMLWrapper(env, iml_cfg, run_dir=None, seed=seed)
    elif condition == "ia":
        ia_cfg = _build_ia_config(cfg)
        env = IAWrapper(env, ia_cfg, seed=seed)
    elif condition == "si":
        si_cfg = _build_si_config(cfg)
        si_wrapper = SIWrapper(env, si_cfg, seed=seed)
        if si_wrapper.cfg.mode == "policy_kl":
            si_wrapper.set_model(model)
        env = si_wrapper

    # Output CSV
    suffix = args.out_suffix
    out_name = f"eval{suffix}.csv"
    out_csv = CSVLogger(run_dir / out_name)

    for ep in range(int(args.episodes)):
        obs = env.reset()
        agent_ids = get_agent_ids(obs)
        ep_returns = {aid: 0.0 for aid in agent_ids}
        ep_returns_original = {aid: 0.0 for aid in agent_ids}  # pre-wrapper rewards
        ep_len = 0
        iml_counts = dict(truth=0, detected=0, sanctions=0, false_pos=0, overturned=0)
        ia_penalty_total = 0.0
        si_bonus_total = 0.0

        while True:
            obs_batch = np.stack([preprocess_obs(obs[aid]) for aid in agent_ids], axis=0)
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, values = model.forward(obs_t)
                actions = torch.argmax(logits, dim=-1)

            action_dict = {aid: int(actions[i].item()) for i, aid in enumerate(agent_ids)}
            next_obs, rewards, dones, infos = env.step(action_dict)

            ep_len += 1
            for aid in agent_ids:
                ep_returns[aid] += float(rewards.get(aid, 0.0))
                if isinstance(infos, dict) and aid in infos and isinstance(infos[aid], dict):
                    # IML counters
                    iml = infos[aid].get("iml")
                    if isinstance(iml, dict):
                        iml_counts["truth"] += int(bool(iml.get("truth_any", False)))
                        iml_counts["detected"] += int(bool(iml.get("detected_any", False)))
                        iml_counts["sanctions"] += int(iml.get("sanctions", 0) or 0)
                        iml_counts["false_pos"] += int(bool(iml.get("false_positive", False)))
                        iml_counts["overturned"] += int(bool(iml.get("overturned", False)))
                    # IA counters
                    ia = infos[aid].get("ia")
                    if isinstance(ia, dict):
                        ep_returns_original[aid] += float(ia.get("r_original", 0.0))
                        ia_penalty_total += float(ia.get("disadv_penalty", 0.0))
                        ia_penalty_total += float(ia.get("adv_penalty", 0.0))
                    else:
                        ep_returns_original[aid] += float(rewards.get(aid, 0.0))
                    # SI counters
                    si = infos[aid].get("si")
                    if isinstance(si, dict):
                        ep_returns_original[aid] += float(si.get("r_original", 0.0))
                        si_bonus_total += float(si.get("influence_bonus", 0.0))
                    elif ia is None:
                        ep_returns_original[aid] += float(rewards.get(aid, 0.0))

            obs = next_obs

            episode_done = False
            if isinstance(dones, dict):
                episode_done = bool(dones.get("__all__", False))
                if not episode_done:
                    agent_keys = [k for k in dones.keys() if k != "__all__"]
                    if agent_keys:
                        episode_done = all(bool(dones[k]) for k in agent_keys)

            if episode_done:
                returns_list = [ep_returns[aid] for aid in agent_ids]
                returns_orig_list = [ep_returns_original[aid] for aid in agent_ids]
                out_csv.write({
                    "episode": ep,
                    "episode_len": ep_len,
                    "return_mean": float(np.mean(returns_list)),
                    "return_sum": float(np.sum(returns_list)),
                    "return_gini": gini(returns_list),
                    "return_original_mean": float(np.mean(returns_orig_list)),
                    "return_original_sum": float(np.sum(returns_orig_list)),
                    "iml_truth": iml_counts["truth"],
                    "iml_detected": iml_counts["detected"],
                    "iml_sanctions": iml_counts["sanctions"],
                    "iml_false_pos": iml_counts["false_pos"],
                    "iml_overturned": iml_counts["overturned"],
                    "ia_penalty_total": ia_penalty_total,
                    "si_bonus_total": si_bonus_total,
                    "condition": condition,
                })
                out_csv.flush()
                break

    out_csv.close()
    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass

    print(f"Evaluation saved to {run_dir / out_name}")


if __name__ == "__main__":
    main()
