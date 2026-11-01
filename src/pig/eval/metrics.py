from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

from pig.config import ATTACK_TYPES


def _safe(fn, *args, **kw) -> float:
    try:
        return float(fn(*args, **kw))
    except ValueError:
        return float("nan")


def _fpr(y: np.ndarray, pred: np.ndarray) -> float:
    neg = y == 0
    return float(pred[neg].mean()) if neg.any() else float("nan")


def bootstrap_ci(y, pred, metric=f1_score, n: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(metric(y[idx], pred[idx], zero_division=0))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def evaluate(name: str, df: pd.DataFrame, scores: np.ndarray, threshold: float,
             latency: dict | None = None, ci: bool = True) -> dict:
    y = df["label"].to_numpy()
    s = np.asarray(scores)
    pred = (s >= threshold).astype(int)
    hard = df["is_hard_negative"].to_numpy(bool)
    res = {
        "model": name,
        "n": int(len(y)),
        "threshold": float(threshold),
        "precision": _safe(precision_score, y, pred, zero_division=0),
        "recall": _safe(recall_score, y, pred, zero_division=0),
        "f1": _safe(f1_score, y, pred, zero_division=0),
        "roc_auc": _safe(roc_auc_score, y, s),
        "pr_auc": _safe(average_precision_score, y, s),
        "fpr_benign": _fpr(y, pred),
        "fpr_hard_neg": float(pred[hard].mean()) if hard.any() else float("nan"),
    }
    if ci and len(y) > 10:
        res["f1_ci_low"], res["f1_ci_high"] = bootstrap_ci(y, pred)
    for t in ATTACK_TYPES:
        m = (df["attack_type"].to_numpy() == t) & (y == 1)
        res[f"recall_{t}"] = float(pred[m].mean()) if m.any() else float("nan")
    for lang in ["ru", "en"]:
        m = df["lang"].to_numpy() == lang
        res[f"f1_{lang}"] = _safe(f1_score, y[m], pred[m], zero_division=0) if m.any() else float("nan")
    if latency:
        res.update({f"latency_{k}": v for k, v in latency.items()})
    return res


def error_table(df: pd.DataFrame, scores: np.ndarray, threshold: float, k: int = 30, seed: int = 42) -> pd.DataFrame:
    d = df.assign(score=np.asarray(scores), pred=(np.asarray(scores) >= threshold).astype(int))
    fp = d[(d["label"] == 0) & (d["pred"] == 1)].assign(error="FP")
    fn = d[(d["label"] == 1) & (d["pred"] == 0)].assign(error="FN")
    half = k // 2
    fp = fp.sort_values("score", ascending=False).head(half)
    fn = fn.sort_values("score").head(k - len(fp))
    cols = ["error", "score", "text", "attack_type", "lang", "source", "is_hard_negative"]
    return pd.concat([fp, fn])[cols].assign(reason="").reset_index(drop=True)


def summary_table(results: list[dict]) -> pd.DataFrame:
    cols = ["model", "f1", "f1_ci_low", "f1_ci_high", "precision", "recall", "roc_auc", "pr_auc",
            "fpr_benign", "fpr_hard_neg", "latency_p50_ms", "latency_p95_ms"]
    df = pd.DataFrame(results)
    return df[[c for c in cols if c in df.columns]].round(4)
