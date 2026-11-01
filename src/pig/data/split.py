from __future__ import annotations

import random

import numpy as np
import pandas as pd


def base_template(tid: str) -> str:
    return tid.split("+")[0] if tid else ""


def pick_heldout_templates(df: pd.DataFrame, share: float, seed: int) -> set[str]:
    rng = random.Random(seed)
    gen = df[df["template_id"] != ""].copy()
    gen["base"] = gen["template_id"].map(base_template)
    held = set()
    for _, grp in gen.groupby(["lang", "attack_type", "is_hard_negative"]):
        bases = sorted(grp["base"].unique())
        k = int(round(len(bases) * share))
        k = min(k, len(bases) - 1)
        if k > 0:
            held.update(rng.sample(bases, k))
    return held


def assign_splits(
    df: pd.DataFrame,
    groups: np.ndarray | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    heldout_sources: list[str] | str | None = None,
    heldout_template_share: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    groups = np.arange(len(df)) if groups is None else np.asarray(groups)
    df["_group"] = groups

    held_templates = pick_heldout_templates(df, heldout_template_share, seed)
    is_held = df["template_id"].map(base_template).isin(held_templates)
    if isinstance(heldout_sources, str):
        heldout_sources = [heldout_sources]
    src = df["source"].str.lower()
    for pattern in heldout_sources or []:
        is_held |= src.str.contains(pattern.lower(), regex=False)
    manual_groups = set(df.loc[df["source"] == "manual", "_group"])
    held_groups = set(df.loc[is_held, "_group"]) - manual_groups
    df["split"] = np.where(df["_group"].isin(manual_groups), "manual",
                           np.where(df["_group"].isin(held_groups), "heldout", ""))

    rng = np.random.default_rng(seed)
    rest = df[df["split"] == ""]
    first = rest.groupby("_group").first()
    first["stratum"] = first["label"].astype(str) + "_" + first["lang"] + "_" + first["is_hard_negative"].astype(str)
    sizes = rest.groupby("_group").size()
    mapping = {}
    for _, grp in first.groupby("stratum"):
        gids = grp.index.to_numpy().copy()
        rng.shuffle(gids)
        total = sizes.loc[gids].sum()
        cum = 0
        for gid in gids:
            frac = cum / total if total else 0
            if frac < ratios[0]:
                mapping[gid] = "train"
            elif frac < ratios[0] + ratios[1]:
                mapping[gid] = "val"
            else:
                mapping[gid] = "test"
            cum += sizes.loc[gid]
    mask = df["split"] == ""
    df.loc[mask, "split"] = df.loc[mask, "_group"].map(mapping)
    df = df[~((df["split"] == "manual") & (df["source"] != "manual"))]
    return df.rename(columns={"_group": "group"}).reset_index(drop=True)


def check_no_leakage(df: pd.DataFrame, groups: np.ndarray | None = None) -> dict:
    report = {}
    by_split = {s: set(g["id"]) for s, g in df.groupby("split")}
    names = sorted(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            report[f"id:{a}&{b}"] = len(by_split[a] & by_split[b])
    if groups is None and "group" in df.columns:
        groups = df["group"].to_numpy()
    if groups is not None:
        g = pd.Series(groups, index=df.index)
        per_group = df.groupby(g)["split"].nunique()
        report["groups_in_multiple_splits"] = int((per_group > 1).sum())
    return report
