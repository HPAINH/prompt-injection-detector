from __future__ import annotations

import pandas as pd

from pig.text import text_id

MODEL = "Helsinki-NLP/opus-mt-en-ru"


def translate_texts(texts: list[str], batch_size: int = 16, device: str | None = None, max_length: int = 512) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            gen = model.generate(**batch, max_new_tokens=max_length, num_beams=2)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def translate_attacks(df: pd.DataFrame, n: int = 300, seed: int = 42, batch_size: int = 16, device: str | None = None) -> pd.DataFrame:
    src = df[(df["label"] == 1) & (df["lang"] == "en") & (df["template_id"] == "")]
    src = src[src["text"].str.len() < 600]
    src = src.sample(min(n, len(src)), random_state=seed)
    res = src.copy()
    res["original"] = src["text"]
    res["text"] = translate_texts(src["text"].tolist(), batch_size, device)
    res["lang"] = "ru"
    res["source"] = "translated:" + res["source"]
    return res


def review_sheet(df: pd.DataFrame, share: float = 0.1, seed: int = 42) -> pd.DataFrame:
    sample = df.sample(max(1, int(len(df) * share)), random_state=seed).copy()
    sample["id"] = sample["text"].map(text_id)
    cols = ["id", "original", "text", "attack_type", "ok", "comment"] if "original" in sample else ["id", "text", "attack_type", "ok", "comment"]
    return sample.assign(ok="", comment="")[cols]
