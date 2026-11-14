import numpy as np
import pandas as pd

from freqsub.calibration.priors import apply_correction, paired_bootstrap, tune_alpha


def toy(n_q=300, n_c=10, seed=0):
    """Scores = truth signal + a frequency bias. Subtracting the frequency prior should help."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        freq = rng.normal(0, 1, n_c)
        gold = rng.integers(n_c)
        for c in range(n_c):
            s = 1.0 * (c == gold) + 1.5 * freq[c] + rng.normal(0, 0.5)
            rows.append({"qid": q, "is_gold": int(c == gold), "logprob": s, "prior": freq[c]})
    return pd.DataFrame(rows)


def test_one_choice_per_question():
    df = toy()
    ch = apply_correction(df, "prior", 0.5)
    assert (ch.groupby(df.qid).sum() == 1).all()


def test_alpha_tuning_finds_bias_correction():
    df = toy()
    alpha, curve = tune_alpha(df, "prior")
    acc0 = dict(curve)[0.0]
    assert 1.0 <= alpha <= 2.0 and dict(curve)[alpha] > acc0 + 0.1


def test_noise_prior_gives_negligible_gain():
    """Grid search can pick a nonzero alpha by chance (that is why alpha is tuned on dev and the
    test split is evaluated once); but a pure-noise prior must not buy meaningful accuracy."""
    df = toy()
    df["prior"] = np.random.default_rng(1).normal(size=len(df))
    alpha, curve = tune_alpha(df, "prior")
    assert dict(curve)[alpha] - dict(curve)[0.0] < 0.04


def test_paired_bootstrap_direction():
    df = toy()
    res = paired_bootstrap(df, apply_correction(df, "prior", 0.0), apply_correction(df, "prior", 1.5))
    assert res["diff"] > 0 and res["ci"][0] > 0
