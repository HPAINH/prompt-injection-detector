from __future__ import annotations

import numpy as np


def threshold_at_fpr(y_true, scores, max_fpr: float = 0.02) -> float:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    benign = np.sort(s[y == 0])[::-1]
    if len(benign) == 0:
        return 0.5
    k = int(np.floor(max_fpr * len(benign)))
    if k >= len(benign):
        return float(benign[-1])
    return float(np.nextafter(benign[k], np.inf))
