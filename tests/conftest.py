import copy

import pytest

from pig.config import load_config
from pig.data.build import build_dataset
from pig.features.embedder import make_embedder


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    import os

    root = tmp_path_factory.mktemp("pig")
    os.environ["PIG_ROOT"] = str(root)
    c = copy.deepcopy(load_config())
    g = c["data"]["generator"]
    g.update({"ru_attacks": 300, "en_attacks": 100, "ru_benign": 400, "en_benign": 100,
              "hard_negatives_ru": 120, "hard_negatives_en": 40})
    return c


@pytest.fixture(scope="session")
def embedder(cfg):
    return make_embedder(cfg, "hashing", cfg.path("cache"))


@pytest.fixture(scope="session")
def dataset(cfg, embedder):
    df, stats = build_dataset(cfg, use_hf=False, embedder=embedder, out_dir=cfg.path("processed"))
    return df, stats


@pytest.fixture(scope="session")
def splits(dataset):
    df, _ = dataset
    return {s: g.reset_index(drop=True) for s, g in df.groupby("split")}
