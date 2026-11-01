from __future__ import annotations

import logging
import re

import pandas as pd

from pig.data.labeling import infer_attack_type
from pig.text import detect_lang

log = logging.getLogger(__name__)

_GERMAN = re.compile(
    r"\b(der|die|das|und|ist|nicht|ich|sie|wie|was|ein|eine|mit|für|auf|bitte|vergiss|alle)\b",
    re.IGNORECASE,
)


def _lang(text: str) -> str:
    lang = detect_lang(text)
    if lang == "en" and len(_GERMAN.findall(text)) >= 2:
        return "other"
    return lang


def _rows(texts, labels, source: str) -> pd.DataFrame:
    df = pd.DataFrame({"text": list(texts), "label": [int(x) for x in labels]})
    df["lang"] = df["text"].map(_lang)
    df["attack_type"] = [
        infer_attack_type(t) if y == 1 else "none" for t, y in zip(df["text"], df["label"])
    ]
    df["source"] = source
    df["template_id"] = ""
    df["is_hard_negative"] = False
    return df[df["lang"].isin(["ru", "en"])]


def _load(name: str):
    from datasets import load_dataset

    ds = load_dataset(name)
    return pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)


def load_deepset() -> pd.DataFrame:
    df = _load("deepset/prompt-injections")
    return _rows(df["text"], df["label"], "deepset/prompt-injections")


def load_jackhhao() -> pd.DataFrame:
    df = _load("jackhhao/jailbreak-classification")
    labels = (df["type"].str.lower() == "jailbreak").astype(int)
    return _rows(df["prompt"], labels, "jackhhao/jailbreak-classification")


def load_gandalf() -> pd.DataFrame:
    df = _load("Lakera/gandalf_ignore_instructions")
    return _rows(df["text"], [1] * len(df), "Lakera/gandalf_ignore_instructions")


LOADERS = {
    "deepset/prompt-injections": load_deepset,
    "jackhhao/jailbreak-classification": load_jackhhao,
    "Lakera/gandalf_ignore_instructions": load_gandalf,
}


def load_hf_sources(max_per_source: int = 1500, seed: int = 42) -> pd.DataFrame:
    frames = []
    for name, loader in LOADERS.items():
        try:
            df = loader()
        except Exception as exc:
            log.warning("source %s skipped: %s", name, exc)
            continue
        if len(df) > max_per_source:
            df = df.sample(max_per_source, random_state=seed)
        log.info("source %s: %d rows", name, len(df))
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["text", "label", "lang", "attack_type", "source", "template_id", "is_hard_negative"]
        )
    return pd.concat(frames, ignore_index=True)
