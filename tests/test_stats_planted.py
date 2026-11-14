"""Planted-effect tests: the H1 pipeline must recover a known effect and report ~0 when there is none."""
import numpy as np
import pandas as pd
import pytest

from freqsub.analysis.choice_model import fit_conditional_logit, substitution_rank_test


def synth(beta_cooc: float, n_q: int = 600, n_c: int = 40, seed: int = 0) -> pd.DataFrame:
    """Each question: n_c candidates, one gold. Utility = 2*is_gold + 0.5*log_freq + beta*log_cooc + Gumbel.
    log_cooc is correlated with log_freq (as in real corpora) so the control matters."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        log_freq = rng.normal(0, 1, n_c)
        log_cooc = 0.6 * log_freq + rng.normal(0, 0.8, n_c)
        log_opop = 0.5 * log_freq + rng.normal(0, 1, n_c)
        n_tokens = rng.integers(1, 5, n_c)
        is_gold = np.zeros(n_c, int)
        is_gold[rng.integers(n_c)] = 1
        u = 2.0 * is_gold + 0.5 * log_freq + beta_cooc * log_cooc + rng.gumbel(size=n_c)
        chosen = np.zeros(n_c, int)
        chosen[np.argmax(u)] = 1
        for i in range(n_c):
            rows.append({"qid": q, "subj_id": q, "chosen": chosen[i], "is_gold": is_gold[i],
                         "log_cooc": log_cooc[i], "log_freq": log_freq[i], "log_opop": log_opop[i],
                         "n_tokens": n_tokens[i], "logprob": u[i]})
    return pd.DataFrame(rows)


def test_recovers_planted_effect():
    """Across independent datasets the cooc CI covers the truth and the estimates average near it.
    (Single datasets can be ~2 SE off on the correlated log_freq term: that is sampling noise.)"""
    cooc, freq = [], []
    for seed in (10, 11, 12):
        table, info = fit_conditional_logit(synth(beta_cooc=0.8, seed=seed), standardize=False)
        assert info["converged"]
        b = table.loc["log_cooc"]
        assert b.ci_low < 0.8 < b.ci_high, b
        cooc.append(b.coef)
        freq.append(table.loc["log_freq", "coef"])
    assert abs(sum(cooc) / 3 - 0.8) < 0.08
    assert abs(sum(freq) / 3 - 0.5) < 0.12


def test_null_effect_not_detected():
    df = synth(beta_cooc=0.0, seed=1)
    table, _ = fit_conditional_logit(df, standardize=False)
    b = table.loc["log_cooc"]
    assert b.ci_low < 0 < b.ci_high, b


def test_rank_test_detects_and_nulls():
    pos = substitution_rank_test(synth(0.8, seed=2), n_perm=2000)
    assert pos.mean_rank > 0.6 and pos.p_value < 0.01
    # with beta=0, the chosen wrong answer still tracks log_freq, which correlates with log_cooc,
    # so the *rank test alone* is confounded: this is exactly why the conditional logit is primary.
    null = substitution_rank_test(synth(0.0, seed=3), n_perm=2000, feature="n_tokens")
    assert abs(null.mean_rank - 0.5) < 0.05 and null.p_value > 0.01


@pytest.mark.parametrize("seed", [4, 5])
def test_coverage_is_reasonable(seed):
    """Two independent null datasets: CIs should usually contain 0 (sanity, not a full coverage study)."""
    table, _ = fit_conditional_logit(synth(0.0, seed=seed, n_q=400), standardize=False)
    assert abs(table.loc["log_cooc", "z"]) < 3.5
