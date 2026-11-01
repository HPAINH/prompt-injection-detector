from __future__ import annotations

import numpy as np
import pandas as pd

from pig.text import normalize, text_id

COLUMNS = ["id", "text", "label", "attack_type", "lang", "source", "template_id", "is_hard_negative"]


def clean(df: pd.DataFrame, min_len: int = 3, max_len: int = 2000) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).map(lambda t: normalize(t, max_len))
    df = df[df["text"].str.len() >= min_len]
    df["id"] = df["text"].map(text_id)
    df["template_id"] = df["template_id"].fillna("").astype(str)
    df["is_hard_negative"] = df["is_hard_negative"].fillna(False).astype(bool)
    df["_prio"] = (df["source"] == "manual").astype(int)
    df = df.sort_values("_prio", ascending=False, kind="stable")
    conflicts = df.groupby("id")["label"].nunique()
    conflict_ids = set(conflicts[conflicts > 1].index)
    has_manual = set(df.loc[df["_prio"] == 1, "id"])
    drop = df["id"].isin(conflict_ids) & ~(df["id"].isin(has_manual) & (df["_prio"] == 1))
    df = df[~drop]
    df = df.drop_duplicates("id", keep="first").sort_index()
    return df[COLUMNS].reset_index(drop=True)


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_duplicate_groups(emb: np.ndarray, threshold: float = 0.95, block: int = 1024) -> np.ndarray:
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    n = len(emb)
    dsu = _DSU(n)
    for start in range(0, n, block):
        sims = emb[start : start + block] @ emb.T
        rows, cols = np.where(sims > threshold)
        for r, c in zip(rows, cols):
            i = start + r
            if i < c:
                dsu.union(i, c)
    return np.array([dsu.find(i) for i in range(n)])


def drop_label_conflicting_groups(df: pd.DataFrame, groups: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    g = pd.Series(groups, index=df.index)
    mixed = df.groupby(g)["label"].nunique()
    bad = g.isin(mixed[mixed > 1].index)
    keep = ~bad | (df["source"] == "manual")
    return df[keep.values].reset_index(drop=True), groups[keep.values]
