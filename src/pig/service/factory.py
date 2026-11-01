from __future__ import annotations

import os
import time

from pig.config import Config
from pig.models.base import Detector
from pig.models.rules import RuleDetector

NAMES = {"rules": "M0_rules", "tfidf": "M1_tfidf", "semantic": "M2_semantic", "cascade": "M4_cascade"}


class Guard:
    def __init__(self, detector: Detector):
        self.detector = detector

    def check(self, text: str) -> dict:
        start = time.perf_counter()
        if hasattr(self.detector, "predict_with_type"):
            proba, types = self.detector.predict_with_type([text])
        else:
            proba = self.detector.predict_proba([text])
            types = None
        score = float(proba[0])
        attack = score >= self.detector.threshold
        attack_type = "none"
        if attack:
            attack_type = types[0] if types else self.detector.predict_type([text])[0]
        return {
            "verdict": "attack" if attack else "benign",
            "score": round(score, 4),
            "attack_type": attack_type,
            "model": self.detector.name,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }


def make_guard(cfg: Config, kind: str | None = None) -> Guard:
    kind = kind or os.environ.get("PIG_DETECTOR", cfg["service"]["detector"])
    if kind == "rules":
        det = RuleDetector()
        det.threshold = 0.5
        return Guard(det)
    from pig.pipeline import load_models

    embedder = os.environ.get("PIG_EMBEDDER", "sentence")
    models = load_models(cfg, embedder, with_transformer=(kind == "cascade"))
    name = NAMES[kind]
    if name not in models:
        raise RuntimeError(f"model {name} not found in {cfg.path('models')}")
    return Guard(models[name])
