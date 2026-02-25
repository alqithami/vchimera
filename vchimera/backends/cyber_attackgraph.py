from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from .base import CyberBackend


@dataclass
class AttackGraphConfig:
    n_hosts: int = 30
    edge_prob: float = 0.08
    critical_frac: float = 0.15
    base_exploit_prob: float = 0.25
    attacker_budget: int = 6
    patch_reduction: float = 0.75
    isolate_reduction: float = 0.85

    monitor_gain: float = 0.12
    hunt_removal_rate: float = 0.25
    hunt_requires_detection: float = 0.5
    patch_rate: float = 0.18
    isolate_rate: float = 0.12
    restore_rate: float = 0.22

    exfiltrate_when_critical_compromised: bool = True
    exfil_gain: float = 0.05

    ransomware_trigger_compromised: float = 0.35
    ransomware_prob: float = 0.04
    ransomware_services_down: float = 0.55

    downtime_from_isolation: float = 0.25  # additional downtime when isolating compromised hosts
    downtime_decay: float = 0.10

    evidence_progress_gain_monitor: float = 0.07
    evidence_progress_gain_hunt: float = 0.12
    evidence_progress_threshold: float = 0.55

    harm_w_compromise: float = 1.0
    harm_w_critical: float = 2.0
    harm_w_downtime: float = 1.8
    harm_w_exfil: float = 2.2
    harm_w_ransom: float = 1.5
    harm_norm: float = 6.0  # normalization constant for severity in [0,1]


class AttackGraphCyberBackend(CyberBackend):
    """
    Abstract cyber backend: a stochastic attack-graph incident simulator.

    Safety: this is NOT an exploit simulator; edges represent abstract attack pathways
    and only use probabilities and difficulty weights.
    """

    def __init__(self, cfg: AttackGraphConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)
        self.G: nx.DiGraph = nx.DiGraph()
        self.critical: set[int] = set()

        # state
        self.compromised: set[int] = set()
        self.patched: set[int] = set()
        self.isolated: set[int] = set()
        self.detection_conf: float = 0.15
        self.services_down: float = 0.0
        self.exfil_risk: float = 0.0
        self.ransomware: bool = False
        self.evidence_progress: float = 0.0
        self.t: int = 0
        self.max_steps: int = 60

    def _build_graph(self) -> None:
        cfg = self.cfg
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(cfg.n_hosts))

        # Random directed edges with difficulty weights
        for u in range(cfg.n_hosts):
            for v in range(cfg.n_hosts):
                if u == v:
                    continue
                if self.rng.random() < cfg.edge_prob:
                    difficulty = float(self.rng.uniform(0.0, 2.0))  # higher = harder
                    self.G.add_edge(u, v, difficulty=difficulty)

        # Ensure weak connectivity by adding some edges if needed
        if self.G.number_of_edges() == 0:
            for u in range(cfg.n_hosts - 1):
                self.G.add_edge(u, u + 1, difficulty=float(self.rng.uniform(0.0, 1.5)))

        # Critical hosts
        n_crit = max(1, int(cfg.critical_frac * cfg.n_hosts))
        self.critical = set(self.rng.choice(cfg.n_hosts, size=n_crit, replace=False).tolist())

    def reset(self, seed: int) -> Dict[str, Any]:
        self.rng = np.random.default_rng(seed)
        self._build_graph()

        self.compromised = set()
        self.patched = set()
        self.isolated = set()
        self.detection_conf = float(self.rng.uniform(0.10, 0.20))
        self.services_down = 0.0
        self.exfil_risk = 0.0
        self.ransomware = False
        self.evidence_progress = 0.0
        self.t = 0

        # initial foothold: 1–2 hosts
        footholds = self.rng.choice(self.cfg.n_hosts, size=int(self.rng.integers(1, 3)), replace=False).tolist()
        self.compromised.update(footholds)

        return self._obs_metrics()[0]

    def _attacker_step(self) -> None:
        cfg = self.cfg
        if len(self.compromised) == 0:
            return

        attempts = 0
        compromised_list = list(self.compromised)
        self.rng.shuffle(compromised_list)

        for u in compromised_list:
            if attempts >= cfg.attacker_budget:
                break
            for v in self.G.successors(u):
                if attempts >= cfg.attacker_budget:
                    break
                if v in self.compromised:
                    continue
                if v in self.isolated:
                    # isolation reduces reachability dramatically
                    if self.rng.random() < cfg.isolate_reduction:
                        continue

                difficulty = float(self.G[u][v].get("difficulty", 1.0))
                p = cfg.base_exploit_prob * np.exp(-difficulty)

                # patched hosts are harder
                if v in self.patched:
                    p *= (1.0 - cfg.patch_reduction)

                # defender detection reduces attacker success a bit
                p *= (1.0 - 0.35 * self.detection_conf)

                if self.rng.random() < p:
                    self.compromised.add(v)
                attempts += 1

        # exfil risk dynamics
        if cfg.exfiltrate_when_critical_compromised and (len(self.compromised.intersection(self.critical)) > 0):
            self.exfil_risk = min(1.0, self.exfil_risk + cfg.exfil_gain * (0.5 + 0.5 * self.rng.random()))

        # ransomware trigger
        compromised_frac = len(self.compromised) / max(1, cfg.n_hosts)
        if (not self.ransomware) and compromised_frac >= cfg.ransomware_trigger_compromised:
            if self.rng.random() < cfg.ransomware_prob:
                self.ransomware = True

    def _defender_step(self, action: str, modifiers: Dict[str, float]) -> None:
        cfg = self.cfg
        compliance = float(modifiers.get("compliance", 0.6))
        reporting = float(modifiers.get("reporting", 0.6))

        action = action.lower().strip()

        if action == "monitor":
            # improve detection confidence, boosted by reporting
            gain = cfg.monitor_gain * (0.6 + 0.8 * reporting)
            self.detection_conf = float(min(1.0, self.detection_conf + gain * (1.0 - self.detection_conf)))
            self.evidence_progress = float(min(1.0, self.evidence_progress + cfg.evidence_progress_gain_monitor * (0.6 + reporting)))

        elif action == "hunt":
            # remove some compromised hosts if detection is good enough
            self.detection_conf = float(min(1.0, self.detection_conf + 0.03 * (0.5 + reporting)))
            self.evidence_progress = float(min(1.0, self.evidence_progress + cfg.evidence_progress_gain_hunt * (0.5 + reporting)))
            if self.detection_conf >= cfg.hunt_requires_detection and len(self.compromised) > 0:
                n_remove = int(max(0, round(cfg.hunt_removal_rate * len(self.compromised) * (0.7 + 0.6 * reporting))))
                n_remove = min(n_remove, len(self.compromised))
                if n_remove > 0:
                    to_remove = self.rng.choice(list(self.compromised), size=n_remove, replace=False).tolist()
                    for h in to_remove:
                        self.compromised.discard(h)
                        # hunting can also lead to hardening (patch)
                        if self.rng.random() < 0.25:
                            self.patched.add(h)

        elif action == "patch":
            # patch fraction of hosts, effectiveness boosted by compliance (public compliance / internal user compliance)
            n_patch = int(max(0, round(cfg.patch_rate * cfg.n_hosts * (0.5 + compliance))))
            candidates = [h for h in range(cfg.n_hosts) if h not in self.patched]
            if len(candidates) > 0 and n_patch > 0:
                n_patch = min(n_patch, len(candidates))
                patched_hosts = self.rng.choice(candidates, size=n_patch, replace=False).tolist()
                self.patched.update(patched_hosts)
            # patching may cause transient downtime
            self.services_down = float(min(1.0, self.services_down + 0.04 * (1.0 - compliance)))

        elif action == "isolate":
            # isolate some compromised hosts (or random if none)
            n_iso = int(max(0, round(cfg.isolate_rate * cfg.n_hosts)))
            candidates = list(self.compromised) if len(self.compromised) > 0 else list(range(cfg.n_hosts))
            if len(candidates) > 0 and n_iso > 0:
                n_iso = min(n_iso, len(candidates))
                iso_hosts = self.rng.choice(candidates, size=n_iso, replace=False).tolist()
                self.isolated.update(iso_hosts)
            # isolation causes downtime
            self.services_down = float(min(1.0, self.services_down + cfg.downtime_from_isolation))

        elif action == "restore":
            # reduce downtime and ransomware impact over time
            self.services_down = float(max(0.0, self.services_down - cfg.restore_rate))
            if self.ransomware and self.services_down < 0.15 and self.rng.random() < 0.25:
                self.ransomware = False

        elif action == "noop":
            pass
        else:
            # unknown action => noop
            pass

        # natural downtime decay
        self.services_down = float(max(0.0, self.services_down - cfg.downtime_decay))

        # ransomware increases downtime floor
        if self.ransomware:
            self.services_down = float(max(self.services_down, cfg.ransomware_services_down))

        # if exfil risk high, detection tends to increase slowly (more signals)
        if self.exfil_risk > 0.5:
            self.detection_conf = float(min(1.0, self.detection_conf + 0.01 * (self.exfil_risk - 0.5)))

    def _obs_metrics(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        cfg = self.cfg
        compromised_frac = len(self.compromised) / max(1, cfg.n_hosts)
        critical_comp = len(self.compromised.intersection(self.critical)) / max(1, len(self.critical))

        harm = (
            cfg.harm_w_compromise * compromised_frac
            + cfg.harm_w_critical * critical_comp
            + cfg.harm_w_downtime * self.services_down
            + cfg.harm_w_exfil * self.exfil_risk
            + cfg.harm_w_ransom * (1.0 if self.ransomware else 0.0)
        )
        severity = float(min(1.0, harm / max(1e-9, cfg.harm_norm)))

        evidence_available = bool(self.evidence_progress >= cfg.evidence_progress_threshold and self.detection_conf >= cfg.hunt_requires_detection)

        obs = {
            "t": self.t,
            "detection_conf": float(self.detection_conf),
            "severity": severity,
            "compromised_frac": float(compromised_frac),
            "services_down": float(self.services_down),
            "exfil_risk": float(self.exfil_risk),
            "ransomware": float(1.0 if self.ransomware else 0.0),
            "evidence_available": evidence_available,
        }
        metrics = {
            "cyber_harm": float(harm),
            "detection_conf": float(self.detection_conf),
            "severity": severity,
            "compromised_frac": float(compromised_frac),
            "services_down": float(self.services_down),
            "exfil_risk": float(self.exfil_risk),
            "ransomware": float(1.0 if self.ransomware else 0.0),
            "evidence_available": float(1.0 if evidence_available else 0.0),
        }
        return obs, metrics

    def step(self, action: str, modifiers: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
        self._defender_step(action, modifiers)
        self._attacker_step()
        self.t += 1
        obs, metrics = self._obs_metrics()
        done = self.t >= self.max_steps
        return obs, metrics, done
