import numpy as np
import pandas as pd

from pig.eval.metrics import error_table, evaluate
from pig.eval.robustness import obfuscated_sets, robustness_report
from pig.models.rules import RuleDetector


def _df():
    return pd.DataFrame({
        "text": ["a", "b", "c", "d"],
        "label": [1, 1, 0, 0],
        "attack_type": ["override", "leak", "none", "none"],
        "lang": ["ru", "en", "ru", "en"],
        "source": ["t"] * 4,
        "is_hard_negative": [False, False, True, False],
    })


def test_evaluate_perfect():
    r = evaluate("x", _df(), np.array([0.9, 0.8, 0.1, 0.2]), 0.5, ci=False)
    assert r["f1"] == 1.0 and r["fpr_benign"] == 0.0 and r["recall_override"] == 1.0


def test_evaluate_hard_negative_fp():
    r = evaluate("x", _df(), np.array([0.9, 0.8, 0.7, 0.2]), 0.5, ci=False)
    assert r["fpr_hard_neg"] == 1.0 and r["fpr_benign"] == 0.5


def test_error_table():
    et = error_table(_df(), np.array([0.1, 0.8, 0.7, 0.2]), 0.5)
    assert set(et["error"]) == {"FP", "FN"}


def test_robustness(splits):
    sets = obfuscated_sets(splits["test"])
    assert {"original", "base64", "translit", "wrap"} <= set(sets)
    rep = robustness_report(RuleDetector(), splits["test"])
    assert rep["recall"].between(0, 1).all()
