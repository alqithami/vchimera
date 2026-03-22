from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping/dict, got {type(data)}")
    return data


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dict, returning a new dict."""
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_kv_overrides(items: Iterable[str]) -> Dict[str, Any]:
    """Parse CLI overrides of the form:

    - key=value
    - nested.key=value

    Values are parsed as JSON when possible; fallback to string.
    """
    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Override must be key=value, got: {it}")
        key, val = it.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            parsed = json.loads(val)
        except Exception:
            parsed = val

        # set nested
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = parsed
    return out


def add_base_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config.")
    parser.add_argument(
        "--set",
        type=str,
        nargs="*",
        default=[],
        help="Override config keys, e.g., --set train.total_steps=2000000 iml.enabled=true",
    )
    return parser


def load_config_with_overrides(args: argparse.Namespace) -> Tuple[Dict[str, Any], Path]:
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    overrides = parse_kv_overrides(args.set)
    cfg = deep_update(cfg, overrides)
    if args.seed is not None:
        cfg.setdefault("train", {})
        cfg["train"]["seed"] = int(args.seed)
    return cfg, cfg_path
