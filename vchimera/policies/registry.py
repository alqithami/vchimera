from __future__ import annotations

from typing import Dict

from .pipeline import PipelinePolicy
from .vchimera_policy import VChimeraPolicy, VChimeraConfig


def make_policy(name: str):
    n = name.strip().lower()
    if n == "pipeline":
        return PipelinePolicy()
    if n == "vchimera":
        return VChimeraPolicy(VChimeraConfig(use_coupling=True, use_targeting=True))
    if n in ("vchimera-no-coupling", "vchimera_no_coupling"):
        return VChimeraPolicy(VChimeraConfig(use_coupling=False, use_targeting=True))
    if n in ("vchimera-no-targeting", "vchimera_no_targeting"):
        return VChimeraPolicy(VChimeraConfig(use_coupling=True, use_targeting=False))
    if n in ("vchimera-no-coupling-no-targeting",):
        return VChimeraPolicy(VChimeraConfig(use_coupling=False, use_targeting=False))
    raise ValueError(f"Unknown policy: {name}")
