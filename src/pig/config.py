from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ATTACK_TYPES = ["override", "leak", "roleplay", "obfuscation"]
ALL_TYPES = ATTACK_TYPES + ["none"]
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class Config(dict):
    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __deepcopy__(self, memo) -> "Config":
        import copy

        return Config(copy.deepcopy(dict(self), memo))

    def path(self, key: str) -> Path:
        root = Path(os.environ.get("PIG_ROOT", self["paths"]["root"]))
        if not root.is_absolute():
            root = DEFAULT_CONFIG.parents[1] / root
        p = root / self["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("PIG_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
