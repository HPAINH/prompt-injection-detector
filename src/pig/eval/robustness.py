from __future__ import annotations

import random

import numpy as np
import pandas as pd

from pig.data.generators import OBFUSCATORS, TRAIN_OBFUSCATORS
from pig.models.base import Detector


def obfuscated_sets(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    attacks = df[(df["label"] == 1) & (df["attack_type"] != "obfuscation")]
    rng = random.Random(seed)
    out = {"original": attacks}
    for name, fn in OBFUSCATORS.items():
        if name == "wrap":
            texts = [fn(t, l, rng) for t, l in zip(attacks["text"], attacks["lang"])]
        else:
            texts = [fn(t, l) for t, l in zip(attacks["text"], attacks["lang"])]
        out[name] = attacks.assign(text=texts)
    return out


def robustness_report(detector: Detector, df: pd.DataFrame, threshold: float | None = None, seed: int = 42) -> pd.DataFrame:
    t = detector.threshold if threshold is None else threshold
    rows = []
    for name, part in obfuscated_sets(df, seed).items():
        if part.empty:
            continue
        s = detector.predict_proba(part["text"].tolist())
        rows.append({
            "model": detector.name,
            "obfuscation": name,
            "seen_in_train": name in TRAIN_OBFUSCATORS,
            "n": len(part),
            "recall": float((np.asarray(s) >= t).mean()),
        })
    return pd.DataFrame(rows)
