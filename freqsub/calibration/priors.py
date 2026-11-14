"""Correction methods for H4: subtract a prior from candidate scores.

    score'(c) = score(c) - alpha * prior(c)

Priors (all in log space):
  corpus_marginal : log(1 + corpus frequency of c)                       [new, corpus-derived]
  corpus_cooc     : log(1 + estimated co-occurrence of subject and c)    [new, corpus-derived; can
                    also penalise true associations, so alpha is tuned on dev and may end up ~0]
  contextual_cal  : log p_model(c | demos + content-free question), averaged over content-free
                    inputs ["N/A", "", "[MASK]"]  (Zhao et al. 2021, "Calibrate Before Use")
  pmi_dc          : log p_model(c | domain premise)  (Holtzman et al. 2021, domain-conditional PMI)
alpha is tuned on the DEV split by grid search; the TEST split is evaluated once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTENT_FREE = ["N/A", "", "[MASK]"]
DOMAIN_PREMISE = {"country": "The name of a country:"}
ALPHA_GRID = np.round(np.arange(0.0, 2.01, 0.05), 2)


def model_priors(lm, bank, template: str, relation: str, cand_names: list[str]) -> dict[str, np.ndarray]:
    """Question-independent model priors for every candidate (computed once per template)."""
    from ..prompts.builder import build_prompt
    from ..scoring.hf_scorer import score_cached

    demos = bank.primary
    cf = []
    for x in CONTENT_FREE:
        # same demos and template as real questions, with the question replaced by a content-free one
        prompt = build_prompt(template, demos, x).rstrip()
        s, _ = score_cached(lm, prompt, cand_names)
        cf.append(np.asarray(s))
    # average probabilities (not log-probs) across content-free inputs, as in CBU
    cf_logp = np.log(np.mean(np.exp(np.vstack(cf) - np.max(np.vstack(cf))), axis=0)) + np.max(np.vstack(cf))
    pmi, _ = score_cached(lm, DOMAIN_PREMISE[relation], cand_names)
    return {"contextual_cal": cf_logp, "pmi_dc": np.asarray(pmi)}


def apply_correction(df: pd.DataFrame, prior_col: str, alpha: float, score_col: str = "logprob") -> pd.Series:
    """Return a 0/1 'chosen' column after subtracting alpha * prior."""
    adj = df[score_col] - alpha * df[prior_col]
    best = adj.groupby(df["qid"]).transform("max")
    chosen = (adj == best).astype(int)
    return chosen * (chosen.groupby(df["qid"]).cumsum() == 1)


def accuracy(df: pd.DataFrame, chosen: pd.Series) -> float:
    return float(df.loc[chosen == 1, "is_gold"].mean())


def tune_alpha(dev: pd.DataFrame, prior_col: str, grid=ALPHA_GRID) -> tuple[float, list[tuple[float, float]]]:
    curve = [(float(a), accuracy(dev, apply_correction(dev, prior_col, a))) for a in grid]
    best = max(curve, key=lambda x: (x[1], -x[0]))  # ties -> smaller alpha (less intervention)
    return best[0], curve


def paired_bootstrap(df: pd.DataFrame, chosen_a: pd.Series, chosen_b: pd.Series, n: int = 2000,
                     seed: int = 0) -> dict:
    """Accuracy difference (b - a) with a question-level paired bootstrap CI."""
    ca = df.loc[chosen_a == 1].set_index("qid")["is_gold"]
    cb = df.loc[chosen_b == 1].set_index("qid")["is_gold"]
    qids = ca.index.intersection(cb.index)
    a, b = ca.loc[qids].to_numpy(), cb.loc[qids].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), size=(n, len(qids)))
    diffs = b[idx].mean(1) - a[idx].mean(1)
    return {"acc_a": float(a.mean()), "acc_b": float(b.mean()), "diff": float(b.mean() - a.mean()),
            "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "n": int(len(qids))}
