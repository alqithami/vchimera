from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed python + numpy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def auc(xs: Sequence[float]) -> float:
    """Discrete AUC proxy (mean over time). Assumes uniform step size.

    Note: Many papers report AUC as the integral (sum) over time. We report the
    *time-normalized* AUC (mean), which makes metrics comparable across horizons.
    """
    if len(xs) == 0:
        return float("nan")
    return float(np.mean(xs))


def mean_ci95(values: Sequence[float]) -> Tuple[float, float]:
    """
    Mean and 95% CI (normal approx). Intended for quick reporting in tables.
    For robustness, prefer bootstrap_ci95.
    """
    arr = np.array(values, dtype=float)
    m = float(np.mean(arr))
    if len(arr) <= 1:
        return m, 0.0
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return m, 1.96 * se


def bootstrap_ci95(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """
    Bootstrap mean CI.
    Returns (mean, low, high).
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    means = np.array(means, dtype=float)
    return float(np.mean(arr)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@dataclass
class StepLog:
    t: int
    cyber_harm: float
    detection_conf: float
    severity: float
    compromised_frac: float
    services_down: float
    exfil_risk: float
    ransomware: float

    misbelief: float
    trust: float
    uncertainty: float
    polarization: float

    protocol_attempted: int
    protocol_executed: int
    shield_interventions: int
