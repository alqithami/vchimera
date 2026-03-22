"""IML Ablation configurations.

Provides named ablation presets that selectively enable/disable IML components:
    - monitor_only:  detection runs but no sanctions are applied
    - sanction_only: detection + sanctions but no review/appeal channel
    - review_only:   detection + sanctions + review (full IML minus ledger logging)
    - full:          all components enabled (equivalent to standard IML)

These ablations allow us to measure the marginal contribution of each
institutional component to welfare, inequality, and compliance outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .iml_wrapper import IMLConfig


# Named ablation presets
ABLATION_PRESETS = {
    "monitor_only": {
        "description": "Detection runs, violations are logged, but NO sanctions applied.",
        "enabled": True,
        "sanction": 0.0,       # zero sanction magnitude
        "p_review": 0.0,       # no review needed since no sanctions
        "write_ledger": True,
    },
    "sanction_no_review": {
        "description": "Detection + sanctions, but NO review/appeal channel.",
        "enabled": True,
        "p_review": 0.0,       # no review
        "write_ledger": True,
    },
    "full_iml": {
        "description": "Full IML: detection + sanctions + review + ledger (default).",
        "enabled": True,
        "p_review": 0.2,       # standard review probability
        "write_ledger": True,
    },
    "high_review": {
        "description": "Full IML with high review probability (stress-test contestability).",
        "enabled": True,
        "p_review": 0.8,
        "write_ledger": True,
    },
}


def get_ablation_config(
    ablation_name: str,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return an IML config dict for the named ablation.

    Parameters
    ----------
    ablation_name : str
        One of the keys in ABLATION_PRESETS.
    base_config : dict, optional
        Base IML config dict to override.  If None, uses sensible defaults.

    Returns
    -------
    dict
        IML config section suitable for merging into a full experiment config.
    """
    if ablation_name not in ABLATION_PRESETS:
        raise ValueError(
            f"Unknown ablation '{ablation_name}'. "
            f"Choose from: {list(ABLATION_PRESETS.keys())}"
        )

    preset = ABLATION_PRESETS[ablation_name]

    # Start from base or defaults
    if base_config is not None:
        cfg = dict(base_config)
    else:
        cfg = {
            "enabled": True,
            "p_detect_true": 0.9,
            "p_detect_false": 0.01,
            "sanction": 0.5,
            "p_review": 0.2,
            "write_ledger": True,
            "rules": ["no_punishment_beam"],
        }

    # Apply preset overrides
    for k, v in preset.items():
        if k != "description":
            cfg[k] = v

    return cfg


def list_ablations() -> Dict[str, str]:
    """Return a dict of ablation_name -> description."""
    return {k: v["description"] for k, v in ABLATION_PRESETS.items()}
