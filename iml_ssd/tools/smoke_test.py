from __future__ import annotations

import argparse
import random

import numpy as np

from iml_ssd.envs.ssd_env import get_action_space_n, get_agent_ids, make_ssd_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="cleanup", choices=["cleanup", "harvest"])
    p.add_argument("--num_agents", type=int, default=5)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = make_ssd_env(args.env, num_agents=args.num_agents, seed=args.seed)
    obs = env.reset()
    agent_ids = get_agent_ids(obs)
    n_actions = get_action_space_n(env)

    print(f"Env={args.env}, agents={agent_ids}, n_actions={n_actions}, obs_shape={obs[agent_ids[0]].shape}")

    for t in range(args.steps):
        action_dict = {aid: int(np.random.randint(0, n_actions)) for aid in agent_ids}
        obs, rewards, dones, infos = env.step(action_dict)
        if isinstance(dones, dict) and dones.get("__all__", False):
            print(f"Episode ended at step {t}. Resetting.")
            obs = env.reset()

    if hasattr(env, "close"):
        env.close()
    print("Smoke test OK.")


if __name__ == "__main__":
    main()
