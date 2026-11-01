from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

PALETTE = ["#2a6fdb", "#e0703a", "#2f9e6e", "#8b5cf6", "#c2410c", "#64748b", "#0e7490"]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_eda(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pd.crosstab(df["split"], df["label"]).plot.bar(ax=axes[0], color=PALETTE[:2], rot=0)
    axes[0].set_title("Классы по выборкам")
    axes[0].legend(["норма", "атака"])
    pd.crosstab(df["lang"], df["label"]).plot.bar(ax=axes[1], color=PALETTE[:2], rot=0)
    axes[1].set_title("Классы по языкам")
    axes[1].legend(["норма", "атака"])
    df[df["label"] == 1]["attack_type"].value_counts().plot.barh(ax=axes[2], color=PALETTE[0])
    axes[2].set_title("Типы атак")
    paths.append(_save(fig, out / "eda_classes.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    lens = df["text"].str.len().clip(upper=1000)
    ax.hist([lens[df["label"] == 0], lens[df["label"] == 1]], bins=40, label=["норма", "атака"], color=PALETTE[:2])
    ax.set_xlabel("Длина запроса, символов (обрезано на 1000)")
    ax.set_ylabel("Число запросов")
    ax.legend()
    paths.append(_save(fig, out / "eda_lengths.png"))
    return paths


def plot_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: str | Path, suffix: str = "test") -> list[Path]:
    out = Path(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (name, (y, s)) in enumerate(curves.items()):
        c = PALETTE[i % len(PALETTE)]
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, label=name, color=c)
        p, r, _ = precision_recall_curve(y, s)
        axes[1].plot(r, p, label=name, color=c)
    axes[0].plot([0, 1], [0, 1], "--", color="#bbb")
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xlabel("FPR (лог. шкала)")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    return [_save(fig, out / f"curves_{suffix}.png")]


def plot_confusion(y, pred, name: str, out_dir: str | Path) -> Path:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["норма", "атака"])
    ax.set_yticks([0, 1], ["норма", "атака"])
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    ax.set_title(name)
    return _save(fig, Path(out_dir) / f"confusion_{name}.png")


def plot_bars(df: pd.DataFrame, x: str, y: str, hue: str, title: str, path: str | Path, ylabel: str = "") -> Path:
    pivot = df.pivot_table(index=x, columns=hue, values=y)
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot.bar(ax=ax, color=PALETTE[: pivot.shape[1]], rot=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel or y)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, Path(path))
