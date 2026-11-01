from __future__ import annotations

import logging
import re

import pandas as pd

from pig.data.labeling import infer_attack_type
from pig.text import detect_lang

log = logging.getLogger(__name__)

COLUMNS = ["text", "label", "lang", "attack_type", "source", "template_id", "is_hard_negative"]

_GERMAN = re.compile(
    r"\b(der|die|das|und|ist|nicht|ich|sie|wie|was|ein|eine|mit|für|auf|bitte|vergiss|alle)\b",
    re.IGNORECASE,
)

CLASS_MAP = [
    (("suffix", "encod", "obfusc", "token", "base64", "typo", "cipher", "smuggl", "unicode"), "obfuscation"),
    (("virtual", "role", "persona", "jailbreak", "dan", "fiction", "hypothet", "story", "game"), "roleplay"),
    (("leak", "extract", "reveal", "system_prompt", "exfil"), "leak"),
    (("ignore", "override", "instruction", "context", "hijack", "payload", "switch", "direct"), "override"),
]


def _lang(text: str) -> str:
    lang = detect_lang(text)
    if lang == "en" and len(_GERMAN.findall(text)) >= 2:
        return "other"
    return lang


def _pick(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"none of {candidates} in columns {list(df.columns)}")


def _rows(texts, labels, source, attack_types=None) -> pd.DataFrame:
    df = pd.DataFrame({"text": [str(t) for t in texts], "label": [int(x) for x in labels]})
    df["lang"] = df["text"].map(_lang)
    if attack_types is None:
        attack_types = [None] * len(df)
    df["attack_type"] = [
        "none" if y == 0 else (a or infer_attack_type(t))
        for t, y, a in zip(df["text"], df["label"], attack_types)
    ]
    df["source"] = source if isinstance(source, str) else list(source)
    df["template_id"] = ""
    df["is_hard_negative"] = False
    return df[df["lang"].isin(["ru", "en"])][COLUMNS]


def _load(name: str, revision: str | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(name, revision=revision) if revision else load_dataset(name)
    return pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)


def filter_origins(df: pd.DataFrame, origins: list[str] | None, cap: int | None, seed: int = 42) -> pd.DataFrame:
    origin = df["source"].str.split("|").str[-1]
    if origins:
        keep = origin.str.lower().apply(lambda o: any(k.lower() in o for k in origins))
        df, origin = df[keep], origin[keep]
    if cap and len(df):
        df = df.sample(frac=1.0, random_state=seed)
        origin = origin.loc[df.index]
        df = df[(origin.groupby(origin).cumcount() < cap).values]
    return df.reset_index(drop=True)


def map_class(value) -> str | None:
    v = str(value or "").lower()
    for keys, target in CLASS_MAP:
        if any(k in v for k in keys):
            return target
    return None


def load_deepset() -> pd.DataFrame:
    df = _load("deepset/prompt-injections")
    return _rows(df[_pick(df, "text", "prompt")], df["label"], "deepset/prompt-injections")


def load_jackhhao() -> pd.DataFrame:
    df = _load("jackhhao/jailbreak-classification")
    labels = (df[_pick(df, "type", "label")].astype(str).str.lower() == "jailbreak").astype(int)
    return _rows(df[_pick(df, "prompt", "text")], labels, "jackhhao/jailbreak-classification")


def load_gandalf() -> pd.DataFrame:
    df = _load("Lakera/gandalf_ignore_instructions")
    return _rows(df[_pick(df, "text", "prompt")], [1] * len(df), "Lakera/gandalf_ignore_instructions")


def load_ru_injections(origins: list[str] | None = None, origin_cap: int | None = None, seed: int = 42) -> pd.DataFrame:
    name = "dmtrdr/russian_prompt_injections"
    df = _load(name)
    text = df[_pick(df, "prompt_ru", "text", "prompt")]
    cls_col = next((c for c in ("class", "category", "type") if c in df.columns), None)
    cls = df[cls_col] if cls_col else pd.Series([None] * len(df))
    orig = df["source"].astype(str) if "source" in df.columns else pd.Series([""] * len(df))
    out = _rows(text, [1] * len(df), [f"{name}|{o}" for o in orig], [map_class(c) for c in cls])
    out = out[out["lang"] == "ru"]
    return filter_origins(out, origins, origin_cap, seed)


RU_BENIGN_CANDIDATES = [
    ("IlyaGusev/ru_turbo_alpaca", "refs/convert/parquet"),
    ("d0rj/alpaca-cleaned-ru", None),
]


def load_ru_alpaca() -> pd.DataFrame:
    errors = []
    for name, revision in RU_BENIGN_CANDIDATES:
        try:
            df = _load(name, revision)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        ins = df[_pick(df, "instruction", "prompt", "question")].fillna("").astype(str)
        if "input" in df.columns:
            inp = df["input"].fillna("").astype(str)
            text = ins + ("\n" + inp).where(inp.str.len() > 0, "")
        else:
            text = ins
        text = text[text.str.len().between(10, 600)]
        out = _rows(text, [0] * len(text), name)
        return out[out["lang"] == "ru"]
    raise RuntimeError("; ".join(errors))


LOADERS = {
    "deepset/prompt-injections": load_deepset,
    "jackhhao/jailbreak-classification": load_jackhhao,
    "Lakera/gandalf_ignore_instructions": load_gandalf,
    "dmtrdr/russian_prompt_injections": load_ru_injections,
    "IlyaGusev/ru_turbo_alpaca": load_ru_alpaca,
}


def load_hf_sources(max_per_source: int = 1500, seed: int = 42, overrides: dict | None = None,
                    only: list[str] | None = None, options: dict | None = None) -> pd.DataFrame:
    overrides = overrides or {}
    options = options or {}
    frames = []
    for name, loader in LOADERS.items():
        if only and name not in only:
            continue
        try:
            df = loader(**options.get(name, {}))
        except Exception as exc:
            log.warning("source %s skipped: %s", name, exc)
            continue
        cap = overrides.get(name, max_per_source)
        if len(df) > cap:
            df = df.sample(cap, random_state=seed)
        log.info("source %s: %d rows", name, len(df))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_manual(path) -> pd.DataFrame:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(p, encoding="utf-8-sig", sep=None, engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding="cp1251", sep=None, engine="python")
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    types = df["attack_type"] if "attack_type" in df.columns else pd.Series([None] * len(df))
    types = [None if pd.isna(t) or str(t).strip() == "" else str(t).strip() for t in types]
    out = _rows(df["text"], df["label"].astype(int), "manual", types)
    if "is_hard_negative" in df.columns:
        flag = df["is_hard_negative"].astype(str).str.strip().str.lower().isin({"1", "true", "да", "yes", "y"})
        out["is_hard_negative"] = flag.loc[out.index].values
    return out
