from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pig.config import Config, set_seed
from pig.data.build import load_splits
from pig.eval.latency import measure_latency
from pig.eval.metrics import error_table, evaluate, summary_table
from pig.eval.plots import plot_bars, plot_confusion, plot_curves
from pig.eval.robustness import robustness_report
from pig.eval.thresholds import threshold_at_fpr
from pig.features.embedder import make_embedder
from pig.models.base import Detector
from pig.models.cascade import CascadeDetector
from pig.models.rules import RuleDetector
from pig.models.semantic import SemanticDetector
from pig.models.tfidf import TfidfDetector

log = logging.getLogger(__name__)


def task_descriptions(cfg: Config) -> list[str]:
    return list(cfg["task_description"].values())


def calibrate(det: Detector, val: pd.DataFrame, max_fpr: float) -> float:
    det.threshold = threshold_at_fpr(val["label"], det.predict_proba(val["text"].tolist()), max_fpr)
    return det.threshold


def train_classic(cfg: Config, splits: dict[str, pd.DataFrame], embedder_kind: str = "sentence",
                  modes: tuple[str, ...] = ("emb", "dev", "full")) -> dict[str, Detector]:
    set_seed(cfg["seed"])
    train, val = splits["train"], splits["val"]
    max_fpr = cfg["threshold"]["max_fpr"]
    models_dir = cfg.path("models")
    out: dict[str, Detector] = {}

    m0 = RuleDetector()
    calibrate(m0, val, max_fpr)
    m0.save(models_dir / f"{m0.name}.joblib")
    out[m0.name] = m0

    tc = cfg["models"]["tfidf"]
    m1 = TfidfDetector(tc["word_ngrams"], tc["char_ngrams"], tc["max_features"], tc["C"], cfg["seed"])
    m1.fit(train["text"], train["label"])
    calibrate(m1, val, max_fpr)
    m1.save(models_dir / f"{m1.name}.joblib")
    out[m1.name] = m1

    emb = make_embedder(cfg, embedder_kind, cfg.path("cache"))
    sc = cfg["models"]["semantic"]
    for mode in modes:
        m = SemanticDetector(emb, task_descriptions(cfg), mode, sc["C"], sc["window"], sc["stride"], cfg["seed"])
        m.fit(train["text"].tolist(), train["label"], train["attack_type"])
        calibrate(m, val, max_fpr)
        m.save(models_dir / f"{m.name}.joblib")
        out[m.name] = m
        log.info("%s threshold=%.4f", m.name, m.threshold)
    return out


def train_transformer(cfg: Config, splits: dict[str, pd.DataFrame]):
    from pig.models.transformer import TransformerDetector

    tc = cfg["models"]["transformer"]
    out_dir = cfg.path("models") / "M3_transformer"
    history = TransformerDetector.train(
        splits["train"]["text"], splits["train"]["label"], splits["val"]["text"], splits["val"]["label"],
        tc["base"], out_dir, tc["epochs"], tc["lr"], tc["batch_size"], tc["max_length"], tc["weight_decay"], cfg["seed"],
    )
    det = TransformerDetector(out_dir, tc["max_length"])
    calibrate(det, splits["val"], cfg["threshold"]["max_fpr"])
    det.save_threshold(out_dir)
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    return det, history


def build_cascade(cfg: Config, fast: Detector, slow: Detector, val: pd.DataFrame) -> tuple[CascadeDetector, dict]:
    cas = CascadeDetector(fast, slow)
    texts = val["text"].tolist()
    info = cas.fit_band(fast.predict_proba(texts), slow.predict_proba(texts), val["label"],
                        cfg["cascade"]["max_m3_share"], cfg["threshold"]["max_fpr"])
    (cfg.path("models") / "M4_cascade.json").write_text(json.dumps(info, indent=2))
    return cas, info


def load_models(cfg: Config, embedder_kind: str = "sentence", with_transformer: bool = True,
                with_reference: bool = False) -> dict[str, Detector]:
    md = cfg.path("models")
    out: dict[str, Detector] = {}
    for name in ["M0_rules", "M1_tfidf"]:
        p = md / f"{name}.joblib"
        if p.exists():
            out[name] = Detector.load(p)
    emb = make_embedder(cfg, embedder_kind, cfg.path("cache"))
    for name in ["M2a_embedding", "M2b_deviation", "M2_semantic"]:
        p = md / f"{name}.joblib"
        if p.exists():
            out[name] = SemanticDetector.load(p, emb)
    if with_transformer and (md / "M3_transformer" / "config.json").exists():
        from pig.models.transformer import TransformerDetector

        out["M3_transformer"] = TransformerDetector(md / "M3_transformer")
        cj = md / "M4_cascade.json"
        if cj.exists() and "M2_semantic" in out:
            info = json.loads(cj.read_text())
            cas = CascadeDetector(out["M2_semantic"], out["M3_transformer"], info["low"], info["high"])
            cas.threshold = info["threshold"]
            out[cas.name] = cas
    if with_reference:
        from pig.models.reference import ReferenceDetector

        ref = ReferenceDetector(cfg["models"]["reference"]["name"])
        out[ref.name] = ref
    return out


def evaluate_all(cfg: Config, models: dict[str, Detector], splits: dict[str, pd.DataFrame],
                 latency_n: int = 200, val: pd.DataFrame | None = None) -> dict:
    reports = cfg.path("reports")
    max_fpr = cfg["threshold"]["max_fpr"]
    results: dict[str, list[dict]] = {"test": [], "heldout": []}
    curves: dict[str, dict] = {"test": {}, "heldout": {}}
    robust, errors = [], {}
    for name, det in models.items():
        if name.startswith("REF") and val is not None:
            calibrate(det, val, max_fpr)
        lat = measure_latency(det, splits["test"]["text"].tolist(), n=latency_n)
        for split in ["test", "heldout"]:
            part = splits.get(split)
            if part is None or part.empty:
                continue
            s = det.predict_proba(part["text"].tolist())
            r = evaluate(name, part, s, det.threshold, lat if split == "test" else None)
            if isinstance(det, CascadeDetector):
                r["slow_share"] = det.last_slow_share
            results[split].append(r)
            curves[split][name] = (part["label"].to_numpy(), s)
            if split == "test":
                plot_confusion(part["label"], (s >= det.threshold).astype(int), name, reports / "confusion")
                errors[name] = error_table(part, s, det.threshold)
        robust.append(robustness_report(det, splits["test"]))
        log.info("evaluated %s", name)

    for split in ["test", "heldout"]:
        if results[split]:
            summary_table(results[split]).to_csv(reports / f"summary_{split}.csv", index=False)
            plot_curves(curves[split], reports, split)
    rob = pd.concat(robust, ignore_index=True) if robust else pd.DataFrame()
    if not rob.empty:
        rob.to_csv(reports / "robustness.csv", index=False)
        plot_bars(rob, "obfuscation", "recall", "model", "Recall на обфусцированных атаках (test)", reports / "robustness.png", "Recall")
    (reports / "errors").mkdir(exist_ok=True)
    for name, et in errors.items():
        et.to_csv(reports / "errors" / f"{name}.csv", index=False)

    by_type = []
    for r in results["test"]:
        for t in ["override", "leak", "roleplay", "obfuscation"]:
            by_type.append({"model": r["model"], "attack_type": t, "recall": r.get(f"recall_{t}", np.nan)})
    if by_type:
        plot_bars(pd.DataFrame(by_type), "attack_type", "recall", "model", "Recall по типам атак (test)", reports / "recall_by_type.png", "Recall")

    payload = {"test": results["test"], "heldout": results["heldout"]}
    (reports / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return payload


def ablation_table(results: dict) -> pd.DataFrame:
    rows = []
    for split in ["test", "heldout"]:
        for r in results.get(split, []):
            if r["model"] in {"M2a_embedding", "M2b_deviation", "M2_semantic"}:
                rows.append({"split": split, "model": r["model"], "f1": r["f1"], "f1_ci_low": r.get("f1_ci_low"),
                             "f1_ci_high": r.get("f1_ci_high"), "fpr_hard_neg": r["fpr_hard_neg"]})
    return pd.DataFrame(rows)


def load_data(cfg: Config) -> dict[str, pd.DataFrame]:
    return load_splits(cfg.path("processed"))
