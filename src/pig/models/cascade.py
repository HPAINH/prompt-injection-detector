from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from pig.models.base import Detector


class CascadeDetector(Detector):
    name = "M4_cascade"

    def __init__(self, fast: Detector, slow: Detector, low: float = 0.2, high: float = 0.8):
        super().__init__()
        self.fast = fast
        self.slow = slow
        self.low = low
        self.high = high
        self.last_slow_share = 0.0

    def combine(self, fast_scores: np.ndarray, slow_fn) -> np.ndarray:
        scores = fast_scores.astype(np.float32).copy()
        unsure = (scores > self.low) & (scores < self.high)
        self.last_slow_share = float(unsure.mean()) if len(scores) else 0.0
        if unsure.any():
            scores[unsure] = slow_fn(np.where(unsure)[0])
        return scores

    def predict_proba(self, texts):
        texts = list(texts)
        fast = self.fast.predict_proba(texts)
        return self.combine(fast, lambda idx: self.slow.predict_proba([texts[i] for i in idx]))

    def predict_type(self, texts):
        return self.fast.predict_type(texts)

    def fit_band(self, fast_val: np.ndarray, slow_val: np.ndarray, y_val, max_share: float = 0.3,
                 max_fpr: float = 0.02) -> dict:
        from pig.eval.thresholds import threshold_at_fpr

        y = np.asarray(y_val)
        best = None
        grid = np.linspace(0.0, 1.0, 41)
        for lo in grid:
            for hi in grid:
                if hi <= lo:
                    continue
                unsure = (fast_val > lo) & (fast_val < hi)
                share = unsure.mean()
                if share > max_share:
                    continue
                s = np.where(unsure, slow_val, fast_val)
                t = threshold_at_fpr(y, s, max_fpr)
                f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
                key = (f1, -share)
                if best is None or key > best[0]:
                    best = (key, lo, hi, t, share)
        _, self.low, self.high, self.threshold, share = best
        return {"low": float(self.low), "high": float(self.high), "threshold": float(self.threshold),
                "val_f1": float(best[0][0]), "val_slow_share": float(share)}
