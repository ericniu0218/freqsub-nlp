"""H1 statistics.

Input: a long table with one row per (question, candidate) and columns
    qid, subj_id, chosen (0/1: model's argmax), is_gold (0/1), log_cooc, log_freq, log_opop, n_tokens

1. ``substitution_rank_test``: among WRONG predictions, the percentile rank of the chosen candidate's
   co-occurrence among the non-gold candidates. Null (choice unrelated to co-occurrence) => mean 0.5.
   Permutation test: replace each chosen wrong candidate by a uniformly random non-gold candidate.
2. ``fit_conditional_logit``: P(choose c | q) ∝ exp(x_c · beta), grouped by question (McFadden).
   Primary estimand: beta[log_cooc], controlling for log_freq, log_opop, n_tokens, is_gold.
3. ``within_question_ols`` (secondary): regress each candidate's log-prob on the same features with
   question fixed effects (demeaning) and subject-clustered standard errors. Uses the full score
   distribution, not just the argmax.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

FEATURES = ["log_cooc", "log_freq", "log_opop", "n_tokens", "is_gold"]


@dataclass
class RankTestResult:
    n_wrong: int
    mean_rank: float
    null_mean: float
    p_value: float
    ci_low: float
    ci_high: float


def substitution_rank_test(df: pd.DataFrame, n_perm: int = 10_000, seed: int = 0,
                           feature: str = "log_cooc") -> RankTestResult:
    rng = np.random.default_rng(seed)
    obs, groups = [], []
    for _, g in df.groupby("qid", sort=False):
        chosen = g[g["chosen"] == 1]
        if len(chosen) != 1 or int(chosen["is_gold"].iloc[0]) == 1:
            continue  # only wrong predictions
        ng = g[g["is_gold"] == 0]
        if len(ng) < 2:
            continue
        pct = (rankdata(ng[feature].to_numpy(), method="average") - 1) / (len(ng) - 1)  # 0..1
        chosen_pos = np.flatnonzero(ng["chosen"].to_numpy() == 1)[0]
        obs.append(pct[chosen_pos])
        groups.append(pct)
    obs_arr = np.asarray(obs)
    if len(obs_arr) == 0:
        return RankTestResult(0, float("nan"), 0.5, float("nan"), float("nan"), float("nan"))
    # permutation null: random non-gold pick per question (vectorised by drawing indices)
    null_means = np.empty(n_perm)
    sizes = np.array([len(p) for p in groups])
    flat = np.concatenate(groups)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    for b in range(n_perm):
        picks = offsets + (rng.random(len(sizes)) * sizes).astype(int)
        null_means[b] = flat[picks].mean()
    m = obs_arr.mean()
    p = (1 + np.sum(np.abs(null_means - 0.5) >= abs(m - 0.5))) / (n_perm + 1)  # two-sided
    boots = [rng.choice(obs_arr, size=len(obs_arr), replace=True).mean() for _ in range(2000)]
    return RankTestResult(len(obs_arr), float(m), float(null_means.mean()), float(p),
                          float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def _standardize(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Z-score continuous features (not is_gold) so coefficients are per-SD and optimisation is stable."""
    out, scale = df.copy(), {}
    for c in cols:
        if c == "is_gold":
            continue
        mu, sd = out[c].mean(), out[c].std()
        sd = sd if sd > 0 else 1.0
        out[c] = (out[c] - mu) / sd
        scale[c] = (float(mu), float(sd))
    return out, scale


def fit_conditional_logit(df: pd.DataFrame, features: list[str] | None = None, standardize: bool = True):
    from statsmodels.discrete.conditional_models import ConditionalLogit

    features = features or FEATURES
    d = df.dropna(subset=features + ["chosen"]).copy()
    # drop groups without exactly one chosen row (defensive)
    ok = d.groupby("qid")["chosen"].transform("sum") == 1
    d = d[ok]
    d, scale = _standardize(d, features) if standardize else (d, {})
    model = ConditionalLogit(d["chosen"].astype(int), d[features].astype(float), groups=d["qid"])
    res = model.fit(disp=False, method="bfgs", maxiter=500)
    table = pd.DataFrame({"coef": res.params, "se": res.bse, "z": res.tvalues, "p": res.pvalues,
                          "ci_low": res.conf_int()[0], "ci_high": res.conf_int()[1]})
    return table, {"n_groups": int(d["qid"].nunique()), "n_rows": len(d), "scale": scale,
                   "converged": bool(getattr(getattr(res, "mle_retvals", None) or {}, "get", lambda k, d=True: d)("converged", True))}


def within_question_ols(df: pd.DataFrame, y: str = "logprob", features: list[str] | None = None):
    import statsmodels.api as sm

    features = features or FEATURES
    d = df.dropna(subset=features + [y]).copy()
    d, _ = _standardize(d, features)
    cols = features + [y]
    d[cols] = d[cols] - d.groupby("qid")[cols].transform("mean")  # question fixed effects
    res = sm.OLS(d[y], d[features]).fit(cov_type="cluster", cov_kwds={"groups": d["subj_id"]})
    return pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues,
                         "ci_low": res.conf_int()[0], "ci_high": res.conf_int()[1]})
