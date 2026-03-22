from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SSDEnvError(RuntimeError):
    pass


@dataclass
class EnvSpec:
    name: str
    module: str
    cls: str


SSD_SPECS = {
    "cleanup": EnvSpec(name="cleanup", module="social_dilemmas.envs.cleanup", cls="CleanupEnv"),
    "harvest": EnvSpec(name="harvest", module="social_dilemmas.envs.harvest", cls="HarvestEnv"),
}


def _try_instantiate(env_cls, *, num_agents: int, seed: Optional[int], env_kwargs: Dict[str, Any]):
    """Best-effort constructor calling across possible signatures."""
    # Common constructor kwargs seen across multi-agent envs
    candidates = [
        dict(num_agents=num_agents, **env_kwargs),
        dict(n_agents=num_agents, **env_kwargs),
        dict(num_agents=num_agents),
        dict(n_agents=num_agents),
        dict(**env_kwargs),
        dict(),
    ]

    last_exc = None
    for kwargs in candidates:
        try:
            env = env_cls(**kwargs)  # type: ignore[arg-type]
            # Seed if supported
            if seed is not None:
                if hasattr(env, "seed") and callable(getattr(env, "seed")):
                    try:
                        env.seed(seed)
                    except Exception:
                        pass
                np.random.seed(seed)
            return env
        except Exception as e:
            last_exc = e
            continue

    raise SSDEnvError(
        f"Failed to instantiate {env_cls} with tried kwargs variants. Last error: {last_exc}"
    )


def make_ssd_env(env_name: str, num_agents: int, seed: Optional[int] = None, **env_kwargs):
    env_name = env_name.lower().strip()
    if env_name not in SSD_SPECS:
        raise SSDEnvError(f"Unknown SSD env '{env_name}'. Choose from {list(SSD_SPECS.keys())}")
    spec = SSD_SPECS[env_name]

    try:
        mod = importlib.import_module(spec.module)
    except Exception as e:
        raise SSDEnvError(
            f"Could not import '{spec.module}'. Make sure SSD is installed (social_dilemmas). Error: {e}"
        )
    if not hasattr(mod, spec.cls):
        raise SSDEnvError(f"Module '{spec.module}' has no attribute '{spec.cls}'.")

    env_cls = getattr(mod, spec.cls)
    env = _try_instantiate(env_cls, num_agents=num_agents, seed=seed, env_kwargs=env_kwargs)
    return env


def get_agent_ids(obs_dict: Dict[Any, Any]) -> List[str]:
    """Return agent ids as strings, stable order."""
    # obs_dict keys can be strings already, or ints.
    keys = list(obs_dict.keys())
    # Filter out RLlib's special __all__ if present
    keys = [k for k in keys if k != "__all__"]
    return sorted([str(k) for k in keys])


def get_action_space_n(env) -> int:
    """Best effort to find a discrete action space size."""
    # SSD envs often expose env.action_space as gym.spaces.Discrete or dict-like
    asp = getattr(env, "action_space", None)
    if asp is None:
        raise SSDEnvError("Environment has no action_space.")
    if hasattr(asp, "n"):
        return int(asp.n)
    # Dict of spaces
    if isinstance(asp, dict):
        first = next(iter(asp.values()))
        if hasattr(first, "n"):
            return int(first.n)
    # gym.spaces.Dict
    if hasattr(asp, "spaces") and isinstance(asp.spaces, dict):
        first = next(iter(asp.spaces.values()))
        if hasattr(first, "n"):
            return int(first.n)
        # Nested Dict: e.g. Dict({"curr_obs": Box, ...}) per agent
        # Try to find the action space from the env directly
    # PettingZoo style
    if hasattr(env, "action_spaces"):
        spaces = getattr(env, "action_spaces")
        if isinstance(spaces, dict):
            first = next(iter(spaces.values()))
            if hasattr(first, "n"):
                return int(first.n)
    # SSD-specific: check env.agents dict for action_space
    if hasattr(env, "agents") and isinstance(env.agents, dict):
        for agent in env.agents.values():
            if hasattr(agent, "action_space") and hasattr(agent.action_space, "n"):
                return int(agent.action_space.n)
    # Fallback: SSD Cleanup/Harvest have 8 actions (7 movement + 1 fire/clean)
    # or 9 actions depending on version. Try to infer from _MAP_ENV_ACTIONS + extra.
    if hasattr(env, "all_actions"):
        return len(env.all_actions)
    # Hard fallback for known SSD envs
    return 8


def extract_obs(agent_obs: Any) -> np.ndarray:
    """Extract the raw observation array from an agent's observation.

    SSD environments return observations in one of two formats:
      1. A raw numpy array (older versions or simple wrappers)
      2. A dict like {"curr_obs": np.ndarray, ...} (standard SSD MapEnv)

    This function handles both cases and returns the numpy array.
    """
    if isinstance(agent_obs, np.ndarray):
        return agent_obs
    if isinstance(agent_obs, dict):
        # Standard SSD format: {"curr_obs": rgb_array, ...}
        if "curr_obs" in agent_obs:
            return agent_obs["curr_obs"]
        # Try common alternative keys
        for key in ("obs", "observation", "rgb"):
            if key in agent_obs:
                return agent_obs[key]
        # Last resort: return the first numpy array found
        for v in agent_obs.values():
            if isinstance(v, np.ndarray):
                return v
        raise SSDEnvError(
            f"Could not extract observation array from dict with keys: {list(agent_obs.keys())}"
        )
    raise SSDEnvError(f"Unexpected observation type: {type(agent_obs)}")


def preprocess_obs(obs: Any) -> np.ndarray:
    """Extract and convert observation to float32 in [0,1].

    Handles both raw numpy arrays and SSD's nested dict format
    ({"curr_obs": np.ndarray, ...}).
    """
    arr = extract_obs(obs)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    # Many SSD observations are 0..255
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr
