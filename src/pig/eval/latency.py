from __future__ import annotations

import time

import numpy as np

from pig.models.base import Detector


def measure_latency(detector: Detector, texts: list[str], n: int = 200, warmup: int = 5, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    sample = [texts[i] for i in rng.integers(0, len(texts), n)]
    for t in sample[:warmup]:
        detector.predict_proba([t])
    times = []
    for t in sample:
        start = time.perf_counter()
        detector.predict_proba([t])
        times.append((time.perf_counter() - start) * 1000)
    arr = np.array(times)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)), "mean_ms": float(arr.mean())}
