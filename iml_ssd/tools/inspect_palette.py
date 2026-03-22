from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from iml_ssd.envs.ssd_env import get_agent_ids, make_ssd_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="cleanup", choices=["cleanup", "harvest"])
    p.add_argument("--num_agents", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=20)
    args = p.parse_args()

    env = make_ssd_env(args.env, num_agents=args.num_agents, seed=args.seed)
    obs = env.reset()
    agent_ids = get_agent_ids(obs)

    counter = Counter()

    for t in range(args.steps):
        for aid in agent_ids:
            o = obs[aid]
            if o.dtype != np.uint8:
                o = (np.clip(o, 0, 1) * 255).astype(np.uint8)
            flat = o.reshape(-1, 3)
            # Count colors
            for rgb in map(tuple, flat):
                counter[rgb] += 1

        # random step to change view
        action_dict = {aid: 0 for aid in agent_ids}  # 0 often = NOOP or move; harmless
        obs, rewards, dones, infos = env.step(action_dict)
        if isinstance(dones, dict) and dones.get("__all__", False):
            obs = env.reset()

    print(f"Unique RGB colors observed: {len(counter)}")
    for rgb, cnt in counter.most_common(args.topk):
        print(f"{rgb}: {cnt}")

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
