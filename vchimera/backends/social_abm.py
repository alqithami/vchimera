from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import SocialBackend
from ..protocol import CommsAction


@dataclass
class PlatformConfig:
    name: str
    mean_degree: int = 12
    homophily: float = 0.65  # higher => more within-community edges
    mod_remove_prob: float = 0.10
    mod_label_prob: float = 0.15
    amplification: float = 1.0  # exposure multiplier


@dataclass
class SocialABMConfig:
    n_agents: int = 800
    n_communities: int = 8
    platforms: List[PlatformConfig] = field(default_factory=list)

    bots_frac: float = 0.04
    bot_misinfo_multiplier: float = 3.0

    base_post_misinfo: float = 0.040
    base_post_correction: float = 0.020

    susceptibility: float = 0.14  # misinfo influence strength
    correction_efficacy: float = 0.18
    official_efficacy: float = 0.26

    trust_drift: float = 0.004  # natural drift toward baseline
    trust_baseline: float = 0.55

    uncertainty_decay: float = 0.03
    uncertainty_from_labeled_misinfo: float = 0.04

    inoculation_gain: float = 0.10
    inoculation_decay: float = 0.01

    fatigue_gain: float = 0.15     # per official message
    fatigue_decay: float = 0.08
    fatigue_trust_penalty: float = 0.08

    # Narrative shock parameters (scaled by coupling bus)
    shock_uncertainty: float = 0.22
    shock_trust_drop: float = 0.18
    shock_misbelief_inject: float = 0.06


class SocialABMBackend(SocialBackend):
    """
    Cross-platform agent-based model for misinformation/trust dynamics.
    Designed to be nontrivial (agent-level, networked, multi-platform) while remaining lightweight.
    """

    def __init__(self, cfg: SocialABMConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)

        # agent attributes/state
        self.community: np.ndarray = np.zeros(cfg.n_agents, dtype=int)
        self.is_bot: np.ndarray = np.zeros(cfg.n_agents, dtype=bool)

        self.b: np.ndarray = np.zeros(cfg.n_agents, dtype=float)  # misbelief
        self.u: np.ndarray = np.zeros(cfg.n_agents, dtype=float)  # uncertainty
        self.t: np.ndarray = np.zeros(cfg.n_agents, dtype=float)  # trust
        self.r: np.ndarray = np.zeros(cfg.n_agents, dtype=float)  # inoculation/resilience

        self.fatigue: float = 0.0

        # adjacency lists per platform: list of list[int]
        self.adj: Dict[str, List[np.ndarray]] = {}

    def _build_platform_graph(self, platform: PlatformConfig) -> List[np.ndarray]:
        N = self.cfg.n_agents
        K = self.cfg.n_communities
        mean_deg = max(1, int(platform.mean_degree))
        hom = float(np.clip(platform.homophily, 0.0, 1.0))

        # Precompute community member indices for sampling
        members = [np.where(self.community == k)[0] for k in range(K)]

        adj = []
        for i in range(N):
            deg = int(self.rng.poisson(lam=mean_deg))
            deg = max(1, min(deg, N - 1))

            ci = int(self.community[i])
            same_pool = members[ci]
            diff_pool = np.where(self.community != ci)[0]

            # mixture sampling for homophily
            n_same = int(round(hom * deg))
            n_diff = deg - n_same

            # avoid self loops
            same_choices = same_pool[same_pool != i] if len(same_pool) > 1 else diff_pool

            neigh = []
            if len(same_choices) > 0 and n_same > 0:
                n_same = min(n_same, len(same_choices))
                neigh.append(self.rng.choice(same_choices, size=n_same, replace=False))
            if len(diff_pool) > 0 and n_diff > 0:
                n_diff = min(n_diff, len(diff_pool))
                neigh.append(self.rng.choice(diff_pool, size=n_diff, replace=False))
            if len(neigh) == 0:
                # fallback
                pool = np.array([j for j in range(N) if j != i], dtype=int)
                neigh = [self.rng.choice(pool, size=deg, replace=False)]
            neigh = np.unique(np.concatenate(neigh)).astype(int)
            adj.append(neigh)
        return adj

    def reset(self, seed: int, initial_events: Dict[str, float]) -> Dict[str, Any]:
        self.rng = np.random.default_rng(seed)
        cfg = self.cfg

        # communities: balanced
        self.community = np.repeat(np.arange(cfg.n_communities), repeats=int(np.ceil(cfg.n_agents / cfg.n_communities)))[: cfg.n_agents]
        self.rng.shuffle(self.community)

        # bots
        n_bots = int(round(cfg.bots_frac * cfg.n_agents))
        self.is_bot[:] = False
        if n_bots > 0:
            bot_idx = self.rng.choice(cfg.n_agents, size=n_bots, replace=False)
            self.is_bot[bot_idx] = True

        # initialize latent states with heterogeneity
        self.b = np.clip(self.rng.beta(2.0, 6.0, size=cfg.n_agents), 0.0, 1.0)  # mean ~0.25
        self.u = np.clip(self.rng.beta(2.2, 3.5, size=cfg.n_agents), 0.0, 1.0)  # mean ~0.39
        self.t = np.clip(self.rng.beta(3.4, 2.8, size=cfg.n_agents), 0.0, 1.0)  # mean ~0.55
        self.r = np.zeros(cfg.n_agents, dtype=float)

        self.fatigue = 0.0

        # build platform graphs
        if not cfg.platforms:
            cfg.platforms = [
                PlatformConfig(name="microblog", mean_degree=14, homophily=0.62, mod_remove_prob=0.10, mod_label_prob=0.15, amplification=1.15),
                PlatformConfig(name="messaging", mean_degree=10, homophily=0.72, mod_remove_prob=0.05, mod_label_prob=0.08, amplification=0.95),
                PlatformConfig(name="video", mean_degree=8, homophily=0.58, mod_remove_prob=0.12, mod_label_prob=0.20, amplification=1.05),
            ]

        self.adj = {}
        for p in cfg.platforms:
            self.adj[p.name] = self._build_platform_graph(p)

        # apply initial narrative shocks (e.g., outage at t=0)
        self._apply_narrative_shocks(initial_events)

        obs, _ = self._obs_metrics()
        return obs

    def _apply_narrative_shocks(self, narrative_events: Dict[str, float]) -> None:
        cfg = self.cfg
        outage = float(narrative_events.get("outage", 0.0))
        ransom = float(narrative_events.get("ransomware", 0.0))
        exfil = float(narrative_events.get("exfiltration", 0.0))
        verified = float(narrative_events.get("verified_update", 0.0))

        shock = outage + 1.3 * ransom + 0.9 * exfil
        if shock > 0:
            self.u = np.clip(self.u + cfg.shock_uncertainty * shock, 0.0, 1.0)
            self.t = np.clip(self.t - cfg.shock_trust_drop * shock, 0.0, 1.0)
            # misbelief injection is stronger for bots and high-uncertainty agents
            inject = cfg.shock_misbelief_inject * shock * (0.4 + 0.6 * self.u)
            self.b = np.clip(self.b + inject, 0.0, 1.0)

        if verified > 0:
            self.t = np.clip(self.t + 0.08 * verified, 0.0, 1.0)
            self.u = np.clip(self.u - 0.05 * verified, 0.0, 1.0)

    def _peer_posting(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        # probabilities per agent
        p_mis = cfg.base_post_misinfo * (0.35 + 0.65 * self.b) * (0.35 + 0.65 * self.u)
        p_cor = cfg.base_post_correction * (0.35 + 0.65 * (1.0 - self.b)) * (0.35 + 0.65 * self.t)

        # bots amplify misinfo posting; reduce correction
        p_mis = p_mis * np.where(self.is_bot, cfg.bot_misinfo_multiplier, 1.0)
        p_cor = p_cor * np.where(self.is_bot, 0.25, 1.0)

        p_mis = np.clip(p_mis, 0.0, 0.6)
        p_cor = np.clip(p_cor, 0.0, 0.3)

        mis_post = self.rng.random(self.cfg.n_agents) < p_mis
        cor_post = self.rng.random(self.cfg.n_agents) < p_cor
        return mis_post, cor_post

    def _diffuse(self, mis_post: np.ndarray, cor_post: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns per-agent exposure counts:
          - misinfo exposures
          - labeled misinfo exposures
          - correction exposures
        """
        N = self.cfg.n_agents
        mis_exp = np.zeros(N, dtype=float)
        mis_lab = np.zeros(N, dtype=float)
        cor_exp = np.zeros(N, dtype=float)

        # per platform diffusion + moderation
        for p in self.cfg.platforms:
            adj = self.adj[p.name]
            amp = float(p.amplification)

            posters_mis = np.where(mis_post)[0]
            posters_cor = np.where(cor_post)[0]

            # misinfo propagation
            for i in posters_mis:
                neigh = adj[i]
                if len(neigh) == 0:
                    continue
                # algorithmic amplification: sample additional recipients proportional to amp
                k = int(max(1, round(amp * len(neigh) * 0.45)))
                k = min(k, len(neigh))
                rec = self.rng.choice(neigh, size=k, replace=False)
                # moderation: remove or label some exposures
                r = self.rng.random(k)
                removed = r < p.mod_remove_prob
                labeled = (~removed) & (r < (p.mod_remove_prob + p.mod_label_prob))
                clean = (~removed) & (~labeled)
                if np.any(clean):
                    mis_exp[rec[clean]] += 1.0
                if np.any(labeled):
                    mis_lab[rec[labeled]] += 1.0

            # correction propagation
            for i in posters_cor:
                neigh = adj[i]
                if len(neigh) == 0:
                    continue
                k = int(max(1, round(amp * len(neigh) * 0.35)))
                k = min(k, len(neigh))
                rec = self.rng.choice(neigh, size=k, replace=False)
                cor_exp[rec] += 1.0

        return mis_exp, mis_lab, cor_exp

    def _apply_official_action(self, a: CommsAction) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns deltas (misbelief_delta, trust_delta, uncertainty_delta, inoculation_delta_scalar)
        computed at agent-level.
        """
        cfg = self.cfg
        N = cfg.n_agents

        if a.type == "silence":
            return np.zeros(N), np.zeros(N), np.zeros(N), 0.0

        # message fatigue accumulates with any non-silence message
        self.fatigue = min(3.0, self.fatigue + cfg.fatigue_gain)
        fatigue_scale = float(np.exp(-cfg.fatigue_trust_penalty * self.fatigue))

        # targeting mask
        if a.target == "all":
            mask = np.ones(N, dtype=bool)
        elif a.target.startswith("comm:"):
            try:
                k = int(a.target.split(":")[1])
            except Exception:
                k = 0
            mask = self.community == k
        else:
            mask = np.ones(N, dtype=bool)

        intensity = float(np.clip(a.intensity, 0.0, 1.0))
        evidence_bonus = 1.0 + (0.35 if a.evidence else 0.0)
        unc_label_bonus = 1.0 + (0.10 if a.uncertainty_label else 0.0)

        db = np.zeros(N, dtype=float)
        dt = np.zeros(N, dtype=float)
        du = np.zeros(N, dtype=float)
        inoc_delta = 0.0

        if a.type == "transparency_update":
            # Transparency reduces uncertainty and improves trust; evidence makes it stronger.
            du[mask] -= (0.10 * intensity) * unc_label_bonus
            dt[mask] += (0.08 * intensity) * evidence_bonus * fatigue_scale

        elif a.type == "debunk":
            # Debunk reduces misbelief; trust mediates acceptance.
            accept = (0.35 + 0.65 * self.t) * evidence_bonus
            db[mask] -= (cfg.official_efficacy * intensity) * accept[mask]
            dt[mask] += (0.03 * intensity) * evidence_bonus * fatigue_scale
            # if debunk without evidence (shouldn't happen under shield), penalize trust
            if not a.evidence:
                dt[mask] -= (0.06 * intensity)

        elif a.type == "prebunk":
            inoc_delta = (cfg.inoculation_gain * intensity)
            dt[mask] += (0.02 * intensity) * fatigue_scale
            du[mask] -= (0.04 * intensity)

        elif a.type == "request_reports":
            # Slight increase in uncertainty but also trust from participatory framing
            du[mask] += (0.02 * intensity)
            dt[mask] += (0.015 * intensity) * fatigue_scale

        return db, dt, du, inoc_delta

    def step(self, comms_action: Any, narrative_events: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        cfg = self.cfg

        # narrative shocks from cyber layer
        self._apply_narrative_shocks(narrative_events)

        # peer diffusion
        mis_post, cor_post = self._peer_posting()
        mis_exp, mis_lab, cor_exp = self._diffuse(mis_post, cor_post)

        # apply effects
        # Labeled misinfo increases uncertainty (and weakly increases misbelief due to exposure)
        self.u = np.clip(self.u + cfg.uncertainty_from_labeled_misinfo * np.tanh(mis_lab / 3.0), 0.0, 1.0)
        self.b = np.clip(self.b + 0.02 * cfg.susceptibility * np.tanh(mis_lab / 3.0), 0.0, 1.0)

        # Plain misinfo affects misbelief, reduced by inoculation and trust
        mis_effect = cfg.susceptibility * np.tanh(mis_exp / 3.0) * (1.0 - 0.55 * self.t) * (1.0 - 0.70 * self.r)
        # bots are less persuadable but maintain high posting; keep them stable
        mis_effect = np.where(self.is_bot, 0.4 * mis_effect, mis_effect)
        self.b = np.clip(self.b + mis_effect, 0.0, 1.0)

        # corrections decrease misbelief, mediated by trust
        cor_effect = cfg.correction_efficacy * np.tanh(cor_exp / 3.0) * (0.25 + 0.75 * self.t)
        self.b = np.clip(self.b - cor_effect, 0.0, 1.0)

        # official communications
        if isinstance(comms_action, dict):
            a = CommsAction(**comms_action)
        else:
            a = comms_action if isinstance(comms_action, CommsAction) else CommsAction(type=str(comms_action))

        db, dt, du, inoc_delta = self._apply_official_action(a)
        self.b = np.clip(self.b + db, 0.0, 1.0)
        self.t = np.clip(self.t + dt, 0.0, 1.0)
        self.u = np.clip(self.u + du, 0.0, 1.0)
        self.r = np.clip(self.r + inoc_delta, 0.0, 1.0)

        # fatigue decays
        self.fatigue = max(0.0, self.fatigue - cfg.fatigue_decay)

        # uncertainty decays naturally, faster when trust is higher
        self.u = np.clip(self.u - cfg.uncertainty_decay * (0.4 + 0.6 * self.t), 0.0, 1.0)

        # trust drifts toward baseline (mean reversion)
        self.t = np.clip(self.t + cfg.trust_drift * (cfg.trust_baseline - self.t), 0.0, 1.0)

        # inoculation decays slowly
        self.r = np.clip(self.r - cfg.inoculation_decay, 0.0, 1.0)

        obs, metrics = self._obs_metrics()
        return obs, metrics

    def _obs_metrics(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        cfg = self.cfg
        misbelief = float(np.mean(self.b))
        trust = float(np.mean(self.t))
        uncertainty = float(np.mean(self.u))

        # polarization: std of community mean misbelief
        comm_means = []
        for k in range(cfg.n_communities):
            idx = np.where(self.community == k)[0]
            if len(idx) == 0:
                continue
            comm_means.append(float(np.mean(self.b[idx])))
        polarization = float(np.std(np.array(comm_means))) if len(comm_means) > 1 else 0.0

        # compliance and reporting proxies (these feed back into cyber backend)
        compliance = float(np.clip(np.mean(self.t * (1.0 - self.b)), 0.0, 1.0))
        reporting = float(np.clip(np.mean(self.t * (1.0 - self.u)), 0.0, 1.0))

        comm_misbelief = []
        comm_trust = []
        for k in range(cfg.n_communities):
            idx = np.where(self.community == k)[0]
            if len(idx) == 0:
                comm_misbelief.append(float("nan"))
                comm_trust.append(float("nan"))
            else:
                comm_misbelief.append(float(np.mean(self.b[idx])))
                comm_trust.append(float(np.mean(self.t[idx])))

        obs = {
            "misbelief": misbelief,
            "trust": trust,
            "uncertainty": uncertainty,
            "polarization": polarization,
            "compliance": compliance,
            "reporting": reporting,
            # for targeting (used by V-CHIMERA policy)
            "comm_misbelief": comm_misbelief,
            "comm_trust": comm_trust,
        }
        metrics = {
            "misbelief": misbelief,
            "trust": trust,
            "uncertainty": uncertainty,
            "polarization": polarization,
            "compliance": compliance,
            "reporting": reporting,
        }
        return obs, metrics
