"""H5: predict when the model is wrong, from model confidence and/or training-corpus counts.

Per question features (computed from the question x candidate table):
  confidence : max_p (softmax over candidates), margin (top1 - top2 prob), entropy
  counts     : log_cooc of the chosen candidate, its percentile within the candidate set,
               log subject count, log subject pageviews
Models are fit on DEV and scored on the evaluation split. Metrics: AUROC for detecting errors,
area under the risk-coverage curve (AURC; lower is better), accuracy at fixed coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONF = ["max_p", "margin", "entropy"]
COUNTS = ["chosen_log_cooc", "chosen_cooc_pct", "log_subj_count", "log_s_pop"]


def question_features(df: pd.DataFrame, score_col: str = "logprob") -> pd.DataFrame:
    rows = []
    for qid, g in df.groupby("qid", sort=False):
        s = g[score_col].to_numpy()
        p = np.exp(s - s.max())
        p /= p.sum()
        order = np.argsort(-p)
        top = g.iloc[order[0]]
        cooc = g["log_cooc"].to_numpy()
        rows.append({"qid": qid, "split": top["split"], "correct": int(top["is_gold"]),
                     "max_p": p[order[0]], "margin": p[order[0]] - p[order[1]],
                     "entropy": float(-(p * np.log(p + 1e-12)).sum()),
                     "chosen_log_cooc": top["log_cooc"],
                     "chosen_cooc_pct": float((cooc < top["log_cooc"]).mean()),
                     "log_subj_count": float(np.log1p(top["subj_count"])),
                     "log_s_pop": float(np.log1p(top["s_pop"] if pd.notna(top["s_pop"]) else 0))})
    return pd.DataFrame(rows)


def risk_coverage(correct: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-confidence)
    c = correct[order]
    n = np.arange(1, len(c) + 1)
    risk = 1 - np.cumsum(c) / n
    coverage = n / len(c)
    return coverage, risk, float(np.trapezoid(risk, coverage)) if len(c) > 1 else float("nan")


def fit_and_evaluate(feats: pd.DataFrame, eval_split: str, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, ev = feats[feats["split"] == "dev"], feats[feats["split"] == eval_split].reset_index(drop=True)
    out = {"n_train": len(tr), "n_eval": len(ev), "eval_split": eval_split}
    if tr["correct"].nunique() < 2 or ev["correct"].nunique() < 2:
        out["note"] = "need both correct and wrong examples"
        return out
    preds = {}
    for name, cols in [("confidence", CONF), ("counts", COUNTS), ("combined", CONF + COUNTS)]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        clf.fit(tr[cols], tr["correct"])
        p_correct = clf.predict_proba(ev[cols])[:, 1]
        preds[name] = p_correct
        _, _, aurc = risk_coverage(ev["correct"].to_numpy(), p_correct)
        out[name] = {"auroc_error": float(roc_auc_score(1 - ev["correct"], 1 - p_correct)), "aurc": aurc}
        for cov in (0.5, 0.8):
            k = max(1, int(round(cov * len(ev))))
            keep = np.argsort(-p_correct)[:k]
            out[name][f"acc_at_{int(cov*100)}"] = float(ev["correct"].to_numpy()[keep].mean())
    out["base_accuracy"] = float(ev["correct"].mean())
    # pre-registered H5 test: bootstrap CI of AUROC(combined) - AUROC(confidence) over eval questions
    err = (1 - ev["correct"]).to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(err), len(err))
        if err[idx].min() == err[idx].max():
            continue
        diffs.append(roc_auc_score(err[idx], 1 - preds["combined"][idx]) -
                     roc_auc_score(err[idx], 1 - preds["confidence"][idx]))
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    out["h5_auroc_diff"] = {"diff": out["combined"]["auroc_error"] - out["confidence"]["auroc_error"],
                            "ci": [lo, hi], "n_boot": len(diffs),
                            "verdict": "supported" if lo > 0 else ("contradicted" if hi < 0 else "inconclusive")}
    return out
