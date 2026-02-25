#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from vchimera.factory import load_yaml, build_env
from vchimera.policies import make_policy
from vchimera.metrics import summarize_episode


def main() -> None:
    scen = load_yaml("configs/scenarios/ransomware_rumor.yaml")
    env, evaluator, _ = build_env(scen)

    policy = make_policy("vchimera")
    obs = env.reset(seed=0)

    last_msg_t = -10_000
    steps = []
    for t in range(20):
        cyber_action, proposed = policy.act(obs, t)
        attempted = evaluator.count_violations(proposed, obs, t, last_msg_t)
        executed, interventions = evaluator.shield(proposed, obs, t, last_msg_t)
        exec_v = evaluator.count_violations(executed, obs, t, last_msg_t)
        if executed.type != "silence":
            last_msg_t = t
        obs, log, done = env.step(cyber_action, executed, attempted, exec_v, interventions)
        steps.append(log)
        if done:
            break

    summ = summarize_episode(steps)
    print("[SMOKE TEST] Episode summary (20 steps):")
    print(summ)


if __name__ == "__main__":
    main()
