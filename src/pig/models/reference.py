from __future__ import annotations

import numpy as np

from pig.models.base import Detector


class ReferenceDetector(Detector):
    name = "REF_deberta"

    def __init__(self, model_name: str = "protectai/deberta-v3-base-prompt-injection-v2", batch_size: int = 32, device: int | None = None):
        super().__init__()
        import torch
        from transformers import pipeline

        device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self.pipe = pipeline("text-classification", model=model_name, device=device)
        self.batch_size = batch_size

    def predict_proba(self, texts):
        res = self.pipe(list(texts), batch_size=self.batch_size, top_k=None, truncation=True, max_length=512)
        out = []
        for r in res:
            scores = {d["label"].upper(): d["score"] for d in r}
            out.append(scores.get("INJECTION", 1 - scores.get("SAFE", 0.0)))
        return np.array(out, dtype=np.float32)
