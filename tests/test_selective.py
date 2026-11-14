import numpy as np
import pandas as pd

from freqsub.detect.selective import fit_and_evaluate, risk_coverage


def test_risk_coverage_perfect_ranking_has_low_aurc():
    correct = np.array([1, 1, 1, 0, 0])
    _, _, good = risk_coverage(correct, np.array([5, 4, 3, 2, 1.0]))
    _, _, bad = risk_coverage(correct, np.array([1, 2, 3, 4, 5.0]))
    assert good < bad


def test_counts_signal_is_detected():
    rng = np.random.default_rng(0)
    n = 600
    cooc = rng.normal(size=n)
    correct = (cooc + rng.normal(scale=0.7, size=n) > 0).astype(int)  # correctness driven by exposure
    feats = pd.DataFrame({"split": np.where(np.arange(n) < 300, "dev", "test"), "correct": correct,
                          "max_p": rng.uniform(size=n), "margin": rng.uniform(size=n), "entropy": rng.uniform(size=n),
                          "chosen_log_cooc": cooc, "chosen_cooc_pct": rng.uniform(size=n),
                          "log_subj_count": rng.normal(size=n), "log_s_pop": rng.normal(size=n)})
    res = fit_and_evaluate(feats, "test")
    assert res["counts"]["auroc_error"] > 0.75
    assert abs(res["confidence"]["auroc_error"] - 0.5) < 0.1


def test_h5_bootstrap_detects_added_value_of_counts():
    rng = np.random.default_rng(1)
    n = 800
    cooc, conf = rng.normal(size=n), rng.normal(size=n)
    correct = (cooc + conf + rng.normal(scale=0.7, size=n) > 0).astype(int)
    feats = pd.DataFrame({"split": np.where(np.arange(n) < 400, "dev", "test"), "correct": correct,
                          "max_p": conf, "margin": rng.uniform(size=n), "entropy": rng.uniform(size=n),
                          "chosen_log_cooc": cooc, "chosen_cooc_pct": rng.uniform(size=n),
                          "log_subj_count": rng.normal(size=n), "log_s_pop": rng.normal(size=n)})
    res = fit_and_evaluate(feats, "test")
    assert res["h5_auroc_diff"]["verdict"] == "supported"
