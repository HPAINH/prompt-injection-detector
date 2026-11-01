from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from pig.text import text_id


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"hashing-{dim}"
        self.vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=dim, alternate_sign=False, norm=None)

    def encode(self, texts: list[str]) -> np.ndarray:
        x = self.vec.transform([t.lower() for t in texts]).toarray().astype(np.float32)
        return _l2(np.log1p(x))


class SentenceEmbedder:
    def __init__(self, model_name: str, prefix: str = "", batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.prefix = prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        x = self.model.encode(
            [self.prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 500,
            convert_to_numpy=True,
        )
        return x.astype(np.float32)


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache_dir: str | Path | None = None):
        self.inner = inner
        self.name = inner.name
        self.mem: dict[str, np.ndarray] = {}
        self.path = None
        if cache_dir:
            safe = inner.name.replace("/", "__")
            self.path = Path(cache_dir) / f"emb_{safe}.npz"
            if self.path.exists():
                data = np.load(self.path)
                self.mem = dict(zip(data["keys"].tolist(), data["vecs"]))

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [text_id(t) for t in texts]
        missing = {k: t for k, t in zip(keys, texts) if k not in self.mem}
        if missing:
            vecs = self.inner.encode(list(missing.values()))
            self.mem.update(zip(missing.keys(), vecs))
            self.save()
        return np.stack([self.mem[k] for k in keys]) if keys else np.zeros((0, 1), np.float32)

    def save(self) -> None:
        if self.path and self.mem:
            keys = np.array(list(self.mem.keys()))
            vecs = np.stack(list(self.mem.values()))
            np.savez(self.path, keys=keys, vecs=vecs)


def make_embedder(cfg: dict | None = None, kind: str = "sentence", cache_dir: str | Path | None = None) -> Embedder:
    if kind == "hashing":
        inner = HashingEmbedder()
    else:
        sc = (cfg or {}).get("models", {}).get("semantic", {})
        inner = SentenceEmbedder(
            sc.get("encoder", "intfloat/multilingual-e5-small"),
            prefix=sc.get("prefix", "query: "),
            batch_size=sc.get("batch_size", 64),
        )
    return CachedEmbedder(inner, cache_dir)
