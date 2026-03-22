from __future__ import annotations

"""Immune-inspired variant of the V-CHIMERA policy.

This policy is deliberately lightweight (no deep learning dependencies) while
implementing canonical Artificial Immune System (AIS) motifs:

* **Antigen load:** community-level risk score derived from misbelief and low trust.
* **Danger signal:** cyber-severity and uncertainty cues that up-regulate response.
* **Immune memory:** exponential moving average of antigen load (persistence).
* **Clonal expansion:** response intensity scales with antigen load.
* **Tolerance / suppression:** avoid over-reacting when antigen is below threshold.

The protocol shield is applied externally by the experiment runner and provides
the hard safety guarantee (zero executed protocol violations).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .vchimera_policy import VChimeraConfig, VChimeraPolicy
from ..protocol import CommsAction


@dataclass
class ImmuneVChimeraConfig(VChimeraConfig):
    """AIS-style control knobs on top of :class:`VChimeraConfig`."""

    # Immune memory (EMA) over community antigen load
    memory_decay: float = 0.85

    # Antigen composition weights (community-level)
    w_antigen_misbelief: float = 1.00
    w_antigen_low_trust: float = 0.80
    w_antigen_uncertainty: float = 0.30

    # Danger signal weights (episode-level)
    w_danger_severity: float = 0.70
    w_danger_detection_gap: float = 0.30
    # Note: used both for comms urgency and for danger-gated coupling.
    danger_threshold: float = 0.65

    # When the immune response should activate (tolerance threshold)
    antigen_threshold: float = 0.35

    # Clonal expansion: intensity increases with antigen load
    base_intensity: float = 0.35
    clonal_gain: float = 0.45

    # Local refractory period (avoid repeatedly targeting the same community)
    target_cooldown: int = 0


class ImmuneVChimeraPolicy(VChimeraPolicy):
    """Immune-inspired coupling-aware policy (pre-shield)."""

    name = "vchimera-ais"

    def __init__(self, cfg: Optional[ImmuneVChimeraConfig] = None):
        super().__init__(cfg=cfg or ImmuneVChimeraConfig())
        self.cfg: ImmuneVChimeraConfig
        self._mem: Optional[np.ndarray] = None
        self._last_target: Optional[int] = None
        self._last_target_t: int = -10_000
        self._t: int = 0

    def reset(self) -> None:
        self._mem = None
        self._last_target = None
        self._last_target_t = -10_000
        self._t = 0

    # --- helper utilities ---
    def _community_arrays(self, obs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        comm_b = obs.get("social_comm_misbelief") or obs.get("comm_misbelief")
        comm_t = obs.get("social_comm_trust") or obs.get("comm_trust")
        if comm_b is None or comm_t is None:
            return np.array([], dtype=float), np.array([], dtype=float)
        b = np.array(comm_b, dtype=float)
        t = np.array(comm_t, dtype=float)
        n = min(b.size, t.size)
        return b[:n], t[:n]

    def _antigen_load(self, obs: Dict) -> np.ndarray:
        cfg = self.cfg
        b, t = self._community_arrays(obs)
        if b.size == 0:
            return np.array([], dtype=float)
        u = float(obs.get("social_uncertainty", obs.get("uncertainty", 0.0)))
        antigen = (
            cfg.w_antigen_misbelief * b
            + cfg.w_antigen_low_trust * (1.0 - t)
            + cfg.w_antigen_uncertainty * u
        )
        antigen = np.nan_to_num(
            antigen,
            nan=float(np.nanmean(antigen)) if np.any(~np.isnan(antigen)) else 0.0,
        )
        return np.clip(antigen, 0.0, 2.0)

    def _danger_signal(self, obs: Dict) -> float:
        cfg = self.cfg
        detection = float(obs.get("cyber_detection_conf", obs.get("detection_conf", 0.0)))
        severity = float(obs.get("cyber_severity", obs.get("severity", 0.0)))
        det_gap = max(0.0, cfg.detection_good - detection)
        danger = cfg.w_danger_severity * severity + cfg.w_danger_detection_gap * det_gap
        return float(np.clip(danger, 0.0, 1.0))

    # --- target selection (immune memory + tolerance + local cooldown) ---
    def _pick_target(self, obs: Dict) -> str:
        cfg = self.cfg
        if not cfg.use_targeting:
            return "all"

        antigen = self._antigen_load(obs)
        if antigen.size == 0:
            return "all"

        if self._mem is None or self._mem.size != antigen.size:
            self._mem = antigen.copy()
        else:
            self._mem = cfg.memory_decay * self._mem + (1.0 - cfg.memory_decay) * antigen

        k = int(np.argmax(self._mem))
        if float(self._mem[k]) < cfg.antigen_threshold:
            return "all"

        # avoid repeated targeting (local refractory period)
        if (
            self._last_target is not None
            and k == self._last_target
            and (self._t - self._last_target_t) <= cfg.target_cooldown
        ):
            order = np.argsort(-self._mem)
            for kk in order:
                kk = int(kk)
                if kk != self._last_target and float(self._mem[kk]) >= cfg.antigen_threshold:
                    k = kk
                    break

        self._last_target = k
        self._last_target_t = self._t
        return f"comm:{k}"

    # --- main policy ---
    def act(self, obs: Dict, t: int):
        self._t = int(t)
        cfg = self.cfg

        detection = float(obs.get("cyber_detection_conf", obs.get("detection_conf", 0.0)))
        severity = float(obs.get("cyber_severity", obs.get("severity", 0.0)))
        compromised = float(obs.get("cyber_compromised_frac", obs.get("compromised_frac", 0.0)))
        services_down = float(obs.get("cyber_services_down", obs.get("services_down", 0.0)))
        evidence_avail = bool(obs.get("cyber_evidence_available", obs.get("evidence_available", False)))

        misbelief = float(obs.get("social_misbelief", obs.get("misbelief", 0.0)))
        trust = float(obs.get("social_trust", obs.get("trust", 0.0)))
        uncertainty = float(obs.get("social_uncertainty", obs.get("uncertainty", 0.0)))
        compliance = float(obs.get("social_compliance", obs.get("compliance", 0.6)))
        reporting = float(obs.get("social_reporting", obs.get("reporting", 0.6)))

        danger = self._danger_signal(obs)
        # immune-style regulation: only couple when the "danger" signal is high and
        # social measurements are not dominated by uncertainty.
        use_coupling_now = bool(cfg.use_coupling and (danger >= cfg.danger_threshold) and (uncertainty < cfg.uncertainty_high))

        # cyber action mirrors the transparent baseline, but with danger-gated coupling
        if use_coupling_now:
            if severity >= cfg.severity_high and (not evidence_avail) and detection < cfg.detection_good:
                cyber_action = "monitor" if reporting >= 0.45 else "hunt"
            elif compromised > 0.30:
                cyber_action = "hunt"
            elif services_down > 0.35:
                cyber_action = "restore"
            else:
                cyber_action = "monitor" if compliance < 0.45 else "patch"
        else:
            if compromised > 0.30:
                cyber_action = "hunt"
            elif services_down > 0.35:
                cyber_action = "restore"
            elif detection < 0.40:
                cyber_action = "monitor"
            else:
                cyber_action = "patch"

        target = self._pick_target(obs)

        # broadcast is often safer under fast cross-community diffusion
        broadcast = (misbelief >= cfg.misbelief_mid) or (trust < cfg.trust_low)
        target_eff = "all" if broadcast else target

        # intensity via clonal expansion (based on immune memory)
        intensity = cfg.base_intensity
        if target.startswith("comm:") and self._mem is not None and self._mem.size > 0:
            try:
                k = int(target.split(":")[1])
                intensity = cfg.base_intensity + cfg.clonal_gain * float(self._mem[k])
            except Exception:
                pass
        # reduce backfire risk in low-trust regimes
        intensity *= float(np.clip(0.40 + 0.60 * trust, 0.40, 1.00))
        intensity = float(np.clip(intensity, 0.15, 0.90))

        # immune-inspired comms logic
        # 1) Danger high + evidence weak => transparency with uncertainty label.
        if (danger >= cfg.danger_threshold) and (not evidence_avail):
            comms = CommsAction(
                type="transparency_update",
                target=target_eff,
                evidence=False,
                uncertainty_label=True,
                intensity=intensity,
            )

        # 2) Evidence available + high misbelief => debunk.
        elif evidence_avail and (misbelief >= cfg.misbelief_high) and (detection >= cfg.detection_good):
            comms = CommsAction(
                type="debunk",
                target=target_eff,
                evidence=True,
                uncertainty_label=False,
                intensity=intensity,
            )

        # 3) Early prebunking to build resilience.
        elif (t <= 6) and (misbelief >= cfg.misbelief_mid) and (trust >= cfg.trust_low):
            comms = CommsAction(
                type="prebunk",
                target="all",
                evidence=False,
                uncertainty_label=(uncertainty >= cfg.uncertainty_high),
                intensity=float(np.clip(0.25 + 0.45 * misbelief, 0.20, 0.70)),
            )

        # 4) Reporting weak => solicit reports.
        elif reporting < 0.45:
            comms = CommsAction(
                type="request_reports",
                target="all" if target == "all" else target,
                evidence=False,
                uncertainty_label=(uncertainty >= cfg.uncertainty_high),
                intensity=float(np.clip(0.40 + 0.30 * (0.45 - reporting), 0.30, 0.70)),
            )

        # 5) Default tolerance.
        else:
            comms = CommsAction(type="silence", target="all", evidence=False, uncertainty_label=False, intensity=0.0)

        return cyber_action, comms
