"""Latency timer — perf_counter based, list of millisecond samples."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import List

import numpy as np


class LatencyTimer:
    def __init__(self):
        self.samples_ms: List[float] = []

    @contextmanager
    def measure(self):
        t0 = time.perf_counter()
        yield
        self.samples_ms.append((time.perf_counter() - t0) * 1000.0)

    def stats(self) -> dict:
        if not self.samples_ms:
            return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        a = np.asarray(self.samples_ms, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
        }
