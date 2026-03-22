from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


def gini(x: Iterable[float]) -> float:
    """Gini coefficient for non-negative values.

    Returns 0 for all-equal and approaches 1 for maximal inequality.
    """
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if np.allclose(arr, 0):
        return 0.0
    # Shift to non-negative
    if arr.min() < 0:
        arr = arr - arr.min()
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    # Gini: 1 - 2 * (sum_{i} (n+1-i)*x_i) / (n*sum x)
    # Equivalent: (2 * sum(i*x_i) / (n*sum x)) - (n+1)/n
    idx = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * np.sum(idx * arr) / (n * np.sum(arr))) - ((n + 1.0) / n)
    return float(g)


def mean_ci(values: List[float], ci: float = 0.95) -> Tuple[float, float, float]:
    """Mean and normal-approx CI (good enough for >= 5 seeds).

    Returns (mean, lo, hi).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if v.size == 1:
        return m, m, m
    se = float(v.std(ddof=1) / np.sqrt(v.size))
    # 1.96 for 95%. If ci differs, approximate with normal.
    z = 1.96 if abs(ci - 0.95) < 1e-6 else float(np.sqrt(2) * np.math.erfcinv(2 * (1 - ci)))
    return m, m - z * se, m + z * se


def rolling_mean(x: List[float], window: int) -> List[float]:
    if window <= 1:
        return x
    out = []
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(x[lo : i + 1])))
    return out
