from __future__ import annotations

import pandas as pd

from pig.text import text_id

MODEL = "Helsinki-NLP/opus-mt-en-ru"


def translate_attacks(df: pd.DataFrame, n: int = 300, seed: int = 42, batch_size: int = 16, device: int = -1) -> pd.DataFrame:
    from transformers import pipeline

    src = df[(df["label"] == 1) & (df["lang"] == "en") & (df["template_id"] == "")]
    src = src[src["text"].str.len() < 600]
    src = src.sample(min(n, len(src)), random_state=seed)
    tr = pipeline("translation", model=MODEL, device=device)
    out = tr(src["text"].tolist(), batch_size=batch_size, max_length=512, truncation=True)
    res = src.copy()
    res["text"] = [o["translation_text"] for o in out]
    res["lang"] = "ru"
    res["source"] = "translated:" + res["source"]
    res["reviewed"] = False
    return res


def review_sheet(df: pd.DataFrame, share: float = 0.1, seed: int = 42) -> pd.DataFrame:
    sample = df.sample(max(1, int(len(df) * share)), random_state=seed).copy()
    sample["id"] = sample["text"].map(text_id)
    return sample.assign(ok="", comment="")[["id", "text", "attack_type", "ok", "comment"]]
