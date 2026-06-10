"""Statistical comparison helpers for paired experiments.

All take per-query metric vectors (one float per test record) so two systems
that ran on the same test set can be compared. Wilcoxon for two systems,
Friedman for >2 systems, bootstrap for confidence intervals on any single
metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TestResult:
    statistic: float
    p_value: float
    method: str


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Two paired vectors. Tests H0: median(a - b) == 0."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Wilcoxon requires equal-length paired samples")
    res = wilcoxon(a, b, zero_method="wilcox", correction=False)
    return TestResult(float(res.statistic), float(res.pvalue), "wilcoxon_signed_rank")


def friedman_test(*samples: Sequence[float]) -> TestResult:
    """Three or more paired vectors. Tests H0: all distributions are identical."""
    from scipy.stats import friedmanchisquare
    if len(samples) < 3:
        raise ValueError("Friedman requires >= 3 paired samples")
    arrs = [np.asarray(s, dtype=np.float64) for s in samples]
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError("Friedman requires equal-length paired samples")
    res = friedmanchisquare(*arrs)
    return TestResult(float(res.statistic), float(res.pvalue), "friedman")


def bootstrap_ci(values: Sequence[float], n_resamples: int = 5000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap on the *mean*. Returns (mean, low, high)."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    n = a.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = a[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(a.mean()), lo, hi
