from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pig.config import ATTACK_TYPES
from pig.features.deviation import FEATURE_NAMES, DeviationFeaturizer
from pig.features.embedder import Embedder
from pig.models.base import Detector

MODES = {"emb": "M2a_embedding", "dev": "M2b_deviation", "full": "M2_semantic"}


class SemanticDetector(Detector):
    def __init__(self, embedder: Embedder, task_descriptions: list[str], mode: str = "full",
                 C: float = 2.0, window: int = 12, stride: int = 6, seed: int = 42):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        self.mode = mode
        self.name = MODES[mode]
        self.embedder = embedder
        self.featurizer = DeviationFeaturizer(embedder, task_descriptions, window, stride)
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed)
        self.typer = LogisticRegression(C=C, max_iter=3000, random_state=seed)
        self.type_scaler = StandardScaler()
        self.seed = seed

    def _features(self, texts: list[str], emb: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        emb = self.embedder.encode(texts) if emb is None else emb
        dev = self.featurizer.transform(texts, emb)
        if self.mode == "emb":
            x = emb
        elif self.mode == "dev":
            x = dev
        else:
            x = np.hstack([emb, dev])
        return x, np.hstack([emb, dev])

    def fit(self, texts, labels, attack_types=None):
        texts = list(texts)
        labels = np.asarray(labels)
        types = np.asarray(attack_types if attack_types is not None else ["none"] * len(texts))
        emb = self.embedder.encode(texts)
        self.featurizer.fit(emb, labels, types)
        x, full = self._features(texts, emb)
        self.clf.fit(self.scaler.fit_transform(x), labels)
        mask = (labels == 1) & np.isin(types, ATTACK_TYPES)
        if mask.sum() > 0 and len(set(types[mask])) > 1:
            self.typer.fit(self.type_scaler.fit_transform(full[mask]), types[mask])
            self._has_typer = True
        else:
            self._has_typer = False
        return self

    def predict_proba(self, texts):
        x, _ = self._features(list(texts))
        return self.clf.predict_proba(self.scaler.transform(x))[:, 1].astype(np.float32)

    def predict_with_type(self, texts: list[str]) -> tuple[np.ndarray, list[str]]:
        texts = list(texts)
        x, full = self._features(texts)
        proba = self.clf.predict_proba(self.scaler.transform(x))[:, 1].astype(np.float32)
        if getattr(self, "_has_typer", False):
            types = self.typer.predict(self.type_scaler.transform(full)).tolist()
        else:
            types = super().predict_type(texts)
        return proba, types

    def predict_type(self, texts):
        return self.predict_with_type(texts)[1]

    def deviation_frame(self, texts: list[str]):
        import pandas as pd

        emb = self.embedder.encode(list(texts))
        return pd.DataFrame(self.featurizer.transform(list(texts), emb), columns=FEATURE_NAMES)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "mode": self.mode, "threshold": self.threshold, "seed": self.seed,
            "featurizer": self.featurizer.state(), "scaler": self.scaler, "clf": self.clf,
            "typer": self.typer, "type_scaler": self.type_scaler,
            "has_typer": getattr(self, "_has_typer", False), "embedder": self.embedder.name,
        }, path)

    @classmethod
    def load(cls, path, embedder: Embedder) -> "SemanticDetector":
        s = joblib.load(path)
        if s["embedder"] != embedder.name:
            raise ValueError(f"model trained with {s['embedder']}, got {embedder.name}")
        obj = cls(embedder, s["featurizer"]["task_descriptions"], s["mode"], seed=s["seed"])
        obj.featurizer = DeviationFeaturizer.from_state(s["featurizer"], embedder)
        obj.scaler, obj.clf = s["scaler"], s["clf"]
        obj.typer, obj.type_scaler, obj._has_typer = s["typer"], s["type_scaler"], s["has_typer"]
        obj.threshold = s["threshold"]
        return obj
