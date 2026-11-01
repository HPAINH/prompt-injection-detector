from __future__ import annotations

import numpy as np

from pig.config import ATTACK_TYPES
from pig.features.embedder import Embedder
from pig.text import fragments

FEATURE_NAMES = (
    ["d_task"]
    + [f"s_att_{t}" for t in ATTACK_TYPES]
    + ["s_att", "s_ben", "delta", "shift", "drift", "n_frag"]
)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)


class DeviationFeaturizer:
    def __init__(self, embedder: Embedder, task_descriptions: list[str], window: int = 12, stride: int = 6):
        self.embedder = embedder
        self.task_descriptions = task_descriptions
        self.window = window
        self.stride = stride
        self.task_vecs: np.ndarray | None = None
        self.att_centroids: np.ndarray | None = None
        self.ben_centroid: np.ndarray | None = None

    def fit(self, emb: np.ndarray, labels, attack_types) -> "DeviationFeaturizer":
        labels = np.asarray(labels)
        types = np.asarray(attack_types)
        self.task_vecs = _l2(self.embedder.encode(self.task_descriptions))
        cents = []
        att_mean = _l2(emb[labels == 1].mean(0, keepdims=True))[0]
        for t in ATTACK_TYPES:
            mask = (labels == 1) & (types == t)
            cents.append(_l2(emb[mask].mean(0, keepdims=True))[0] if mask.any() else att_mean)
        self.att_centroids = np.stack(cents)
        self.ben_centroid = _l2(emb[labels == 0].mean(0, keepdims=True))[0]
        return self

    def _fragment_stats(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frags, owner = [], []
        for i, t in enumerate(texts):
            fs = fragments(t, self.window, self.stride)
            frags.extend(fs)
            owner.extend([i] * len(fs))
        owner = np.array(owner)
        fe = _l2(self.embedder.encode(frags))
        task_sim = (fe @ self.task_vecs.T).max(1)
        shift = np.zeros(len(texts), np.float32)
        drift = np.zeros(len(texts), np.float32)
        n_frag = np.zeros(len(texts), np.float32)
        for i in range(len(texts)):
            idx = np.where(owner == i)[0]
            n_frag[i] = len(idx)
            drift[i] = float((1 - task_sim[idx]).max())
            if len(idx) > 1:
                v = fe[idx]
                shift[i] = float((1 - (v[:-1] * v[1:]).sum(1)).max())
        return shift, drift, n_frag

    def transform(self, texts: list[str], emb: np.ndarray) -> np.ndarray:
        if self.att_centroids is None:
            raise RuntimeError("featurizer is not fitted")
        emb = _l2(emb)
        d_task = 1 - (emb @ self.task_vecs.T).max(1)
        s_types = emb @ self.att_centroids.T
        s_att = s_types.max(1)
        s_ben = emb @ self.ben_centroid
        shift, drift, n_frag = self._fragment_stats(texts)
        feats = np.column_stack([d_task, s_types, s_att, s_ben, s_att - s_ben, shift, drift, np.log1p(n_frag)])
        return feats.astype(np.float32)

    def state(self) -> dict:
        return {
            "task_descriptions": self.task_descriptions,
            "window": self.window,
            "stride": self.stride,
            "task_vecs": self.task_vecs,
            "att_centroids": self.att_centroids,
            "ben_centroid": self.ben_centroid,
        }

    @classmethod
    def from_state(cls, state: dict, embedder: Embedder) -> "DeviationFeaturizer":
        obj = cls(embedder, state["task_descriptions"], state["window"], state["stride"])
        obj.task_vecs = state["task_vecs"]
        obj.att_centroids = state["att_centroids"]
        obj.ben_centroid = state["ben_centroid"]
        return obj
