from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from pig.models.base import Detector


class TfidfDetector(Detector):
    name = "M1_tfidf"

    def __init__(self, word_ngrams=(1, 2), char_ngrams=(2, 5), max_features: int = 50000, C: float = 4.0, seed: int = 42):
        super().__init__()
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=tuple(word_ngrams), max_features=max_features, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(char_ngrams), max_features=max_features, sublinear_tf=True, lowercase=True)),
        ])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
        self.pipe = Pipeline([("features", features), ("clf", clf)])

    def fit(self, texts, labels, attack_types=None):
        self.pipe.fit(list(texts), np.asarray(labels))
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(list(texts))[:, 1].astype(np.float32)

    def top_features(self, k: int = 25) -> dict[str, list[str]]:
        names = self.pipe.named_steps["features"].get_feature_names_out()
        coef = self.pipe.named_steps["clf"].coef_[0]
        order = np.argsort(coef)
        return {"attack": names[order[-k:]][::-1].tolist(), "benign": names[order[:k]].tolist()}
