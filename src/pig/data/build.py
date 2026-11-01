from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pig.config import DEFAULT_CONFIG, Config, set_seed
from pig.data.clean import clean, drop_label_conflicting_groups, near_duplicate_groups
from pig.data.generators import generate_all
from pig.data.sources import load_hf_sources, load_manual
from pig.data.split import assign_splits, check_no_leakage

log = logging.getLogger(__name__)


def dataset_stats(df: pd.DataFrame) -> dict:
    return {
        "total": int(len(df)),
        "attack_share": round(float(df["label"].mean()), 4),
        "by_split": df["split"].value_counts().to_dict(),
        "by_split_label": {f"{s}_{l}": int(v) for (s, l), v in df.groupby(["split", "label"]).size().items()},
        "by_lang": df["lang"].value_counts().to_dict(),
        "by_type": df["attack_type"].value_counts().to_dict(),
        "by_source": df["source"].value_counts().to_dict(),
        "hard_negatives": int(df["is_hard_negative"].sum()),
        "hard_negatives_by_split": df[df["is_hard_negative"]]["split"].value_counts().to_dict(),
    }


def manual_path(cfg: Config) -> Path:
    p = Path(cfg["data"]["manual_file"])
    return p if p.is_absolute() else DEFAULT_CONFIG.parents[1] / p


def build_dataset(cfg: Config, use_hf: bool = True, embedder=None, translated: pd.DataFrame | None = None,
                  out_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    set_seed(cfg["seed"])
    d = cfg["data"]
    manual = load_manual(manual_path(cfg)) if d.get("manual_file") else None
    parts = [manual] if manual is not None and len(manual) else []
    parts.append(generate_all(cfg, cfg["seed"]))
    if use_hf:
        parts.append(load_hf_sources(d["max_per_hf_source"], cfg["seed"], d.get("max_per_source_overrides"),
                                     options=d.get("source_options")))
    if translated is not None and len(translated):
        parts.append(translated)
    raw = pd.concat(parts, ignore_index=True)
    df = clean(raw, d["min_len"], d["max_len"])
    log.info("after exact dedup: %d", len(df))

    if embedder is not None:
        emb = embedder.encode(df["text"].tolist())
        groups = near_duplicate_groups(emb, d["near_dup_threshold"])
        df, groups = drop_label_conflicting_groups(df, groups)
    else:
        groups = np.arange(len(df))

    df = assign_splits(df, groups, tuple(d["split"]), d["heldout_sources"], d["heldout_template_share"], cfg["seed"])
    leakage = check_no_leakage(df)
    stats = dataset_stats(df)
    stats["leakage"] = leakage
    stats["near_dup_groups_merged"] = int(len(df) - df["group"].nunique())

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "dataset.parquet", index=False)
        (out / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, stats


def load_splits(path: str | Path) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(Path(path) / "dataset.parquet") if Path(path).is_dir() else pd.read_parquet(path)
    return {s: g.reset_index(drop=True) for s, g in df.groupby("split")}
