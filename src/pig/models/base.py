from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np

from pig.data.labeling import infer_attack_type


class Detector(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.threshold: float = 0.5

    def fit(self, texts: list[str], labels, attack_types=None) -> "Detector":
        return self

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    def predict(self, texts: list[str]) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def predict_type(self, texts: list[str]) -> list[str]:
        return [infer_attack_type(t) for t in texts]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Detector":
        return joblib.load(path)
