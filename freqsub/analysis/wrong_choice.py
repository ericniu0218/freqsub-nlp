"""Predict the chosen wrong answer, conditional on a model error.

This module never determines whether a question is an error. It takes complete
question-by-candidate scores, selects error questions, removes every valid row,
and compares a candidate-identity softmax with identity plus co-occurrence.
The historical country test must not be presented as fresh confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from freqsub.data.aliases import normalize

REQUIRED = {"qid", "subj_id", "subj", "cand_obj_id", "is_gold", "is_valid", "chosen",
            "log_cooc", "split", "relation"}
LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0)


def validate_grid(frame: pd.DataFrame, *, relation: str, split: str) -> pd.DataFrame:
    """Reject incomplete candidate grids, malformed choices, and mixed cohorts."""
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    d = frame.copy()
    if d.empty or set(d["relation"]) != {relation} or set(d["split"]) != {split}:
        raise ValueError("empty or mixed relation/split")
    if d[["qid", "subj_id", "subj", "cand_obj_id"]].isna().any().any():
        raise ValueError("missing question, subject, or candidate identity")
    if d.duplicated(["qid", "cand_obj_id"]).any():
        raise ValueError("duplicate question/candidate row")
    if any(not d[col].isin([0, 1]).all() for col in ("is_gold", "is_valid", "chosen")):
        raise ValueError("gold/valid/chosen indicators must be binary")
    if ((d["is_gold"] == 1) & (d["is_valid"] != 1)).any():
        raise ValueError("the designated gold answer must be valid")
    if not np.isfinite(d["log_cooc"].to_numpy(dtype=float)).all():
        raise ValueError("co-occurrence contains missing or nonfinite values")
    if (d["log_cooc"] < 0).any():
        raise ValueError("log1p co-occurrence cannot be negative")
    pool = set(d["cand_obj_id"])
    if len(pool) < 3:
        raise ValueError("candidate pool must contain at least three distinct answers")
    for qid, g in d.groupby("qid", sort=False):
        if (set(g["cand_obj_id"]) != pool or int(g["is_gold"].sum()) != 1
                or int(g["is_valid"].sum()) < 1 or int(g["is_valid"].sum()) >= len(pool)
                or int(g["chosen"].sum()) != 1):
            raise ValueError(f"incomplete candidate grid or malformed labels for {qid}")
        if g["subj_id"].nunique() != 1 or g["subj"].nunique() != 1:
            raise ValueError(f"mixed subject in {qid}")
    d["subject_key"] = d["subj"].map(normalize)
    if (d["subject_key"] == "").any():
        raise ValueError("empty normalized subject")
    return d


@dataclass
class ChoiceData:
    groups: list[np.ndarray]
    x: list[np.ndarray]
    winners: list[int]
    qids: list[str]
    subject_keys: list[str]
    candidate_ids: list[list[str]]
    x_sd: float
    n_all_questions: int


def error_choices(frame: pd.DataFrame, candidate_order: list[str], *, x_sd: float | None = None) -> ChoiceData:
    """Create grouped wrong-choice observations without consulting test outcomes for scaling."""
    lookup = {c: i for i, c in enumerate(candidate_order)}
    groups, xvals, winners, qids, subjects, ids = [], [], [], [], [], []
    for qid, g in frame.groupby("qid", sort=False):
        if int(g.loc[g["chosen"] == 1, "is_valid"].iloc[0]) == 1:
            continue
        wrong = g[g["is_valid"] == 0].sort_values("cand_obj_id", key=lambda s: s.astype(str))
        candidate_ids = [str(c) for c in wrong["cand_obj_id"]]
        if any(c not in lookup for c in candidate_ids):
            raise ValueError("evaluation contains a candidate absent from training")
        selected = np.flatnonzero(wrong["chosen"].to_numpy() == 1)
        if len(selected) != 1:
            raise ValueError(f"error question {qid} lacks one chosen wrong candidate")
        raw = wrong["log_cooc"].to_numpy(dtype=float)
        groups.append(np.asarray([lookup[c] for c in candidate_ids], dtype=int))
        xvals.append(raw - raw.mean())
        winners.append(int(selected[0]))
        qids.append(str(qid))
        subjects.append(str(wrong["subject_key"].iloc[0]))
        ids.append(candidate_ids)
    if not groups:
        raise ValueError("no wrong questions")
    sd = float(np.std(np.concatenate(xvals))) if x_sd is None else float(x_sd)
    if not np.isfinite(sd) or sd < 0:
        raise ValueError("invalid co-occurrence scale")
    if sd == 0:
        sd = 1.0  # negative control: the added feature cannot change any choice
    return ChoiceData(groups, [x / sd for x in xvals], winners, qids, subjects, ids, sd,
                      frame["qid"].nunique())


@dataclass
class Fit:
    offsets: np.ndarray
    beta: float
    penalty: float
    with_cooc: bool


def fit(data: ChoiceData, n_candidates: int, penalty: float, *, with_cooc: bool) -> Fit:
    """Fit a ridge-regularized grouped softmax with an analytic gradient."""
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    n = len(data.groups)
    size = n_candidates + int(with_cooc)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        offsets = params[:n_candidates]
        beta = params[-1] if with_cooc else 0.0
        grad = np.zeros_like(params)
        loss = 0.0
        for ix, x, winner in zip(data.groups, data.x, data.winners):
            scores = offsets[ix] + beta * x
            p = np.exp(scores - logsumexp(scores))
            loss += float(logsumexp(scores) - scores[winner])
            p[winner] -= 1.0
            np.add.at(grad, ix, p)
            if with_cooc:
                grad[-1] += float(p @ x)
        loss = loss / n + penalty * float(params @ params) / 2
        grad = grad / n + penalty * params
        return loss, grad

    result = minimize(objective, np.zeros(size), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"grouped softmax did not converge: {result.message}")
    return Fit(result.x[:n_candidates], float(result.x[-1]) if with_cooc else 0.0,
               penalty, with_cooc)


def predict_rows(model: Fit, data: ChoiceData) -> pd.DataFrame:
    """Return per-question proper loss, winner rank, and top-1 correctness."""
    rows = []
    for qid, subject, ids, ix, x, winner in zip(data.qids, data.subject_keys, data.candidate_ids,
                                                 data.groups, data.x, data.winners):
        scores = model.offsets[ix] + model.beta * x
        logp = scores - logsumexp(scores)
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == winner)[0]) + 1
        rows.append({"qid": qid, "subject_key": subject, "chosen_wrong_id": ids[winner],
                     "predicted_wrong_id": ids[int(order[0])], "n_candidates": len(ids),
                     "loss": float(-logp[winner]), "top1": int(rank == 1), "rank": rank,
                     "winner_probability": float(np.exp(logp[winner]))})
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(base: pd.DataFrame, augmented: pd.DataFrame, *, seed: int = 0,
                             draws: int = 2000) -> dict:
    joined = base[["qid", "subject_key", "loss"]].merge(
        augmented[["qid", "subject_key", "loss"]], on=["qid", "subject_key"],
        suffixes=("_base", "_aug"), validate="one_to_one")
    if len(joined) != len(base) or len(joined) != len(augmented):
        raise ValueError("baseline and augmented evaluation cohorts differ")
    delta = joined["loss_base"].to_numpy() - joined["loss_aug"].to_numpy()
    clusters = [g.index.to_numpy() for _, g in joined.groupby("subject_key", sort=True)]
    if len(clusters) < 2:
        raise ValueError("need at least two distinct evaluation subjects")
    rng = np.random.default_rng(seed)
    draws_out = []
    for _ in range(draws):
        selected = rng.integers(0, len(clusters), len(clusters))
        row_ids = np.concatenate([clusters[i] for i in selected])
        draws_out.append(float(delta[row_ids].mean()))
    return {"improvement": float(delta.mean()), "ci95": [float(np.percentile(draws_out, 2.5)),
                                                         float(np.percentile(draws_out, 97.5))],
            "n_questions": len(joined), "n_subjects": len(clusters), "draws": draws}


def select_penalty(data: ChoiceData, n_candidates: int, *, with_cooc: bool) -> tuple[float, dict]:
    """Three-fold development-only grouped CV; never inspect evaluation questions."""
    losses = {}
    for lam in LAMBDAS:
        rows = cross_validated_rows(data, n_candidates, lam, with_cooc=with_cooc)
        losses[lam] = float(rows["loss"].mean())
    selected = min(LAMBDAS, key=lambda lam: (losses[lam], -lam))
    return selected, {str(lam): losses[lam] for lam in LAMBDAS}


def cross_validated_rows(data: ChoiceData, n_candidates: int, penalty: float,
                         *, with_cooc: bool) -> pd.DataFrame:
    """Return one out-of-fold prediction per development error question."""
    subjects = sorted(set(data.subject_keys))
    if len(subjects) < 15 or len(data.groups) < 30:
        raise ValueError("at least 30 development errors across 15 subjects required for CV")
    folds = {s: i % 3 for i, s in enumerate(subjects)}

    def subset(indices: list[int], scale: float) -> ChoiceData:
        return ChoiceData([data.groups[i] for i in indices], [data.x[i] / scale for i in indices],
                          [data.winners[i] for i in indices], [data.qids[i] for i in indices],
                          [data.subject_keys[i] for i in indices], [data.candidate_ids[i] for i in indices],
                          data.x_sd * scale, len(indices))

    parts = []
    for fold in range(3):
        tr = [i for i, s in enumerate(data.subject_keys) if folds[s] != fold]
        va = [i for i, s in enumerate(data.subject_keys) if folds[s] == fold]
        train_scale = float(np.std(np.concatenate([data.x[i] for i in tr])))
        if train_scale == 0:
            train_scale = 1.0
        model = fit(subset(tr, train_scale), n_candidates, penalty, with_cooc=with_cooc)
        pred = predict_rows(model, subset(va, train_scale))
        pred["fold"] = fold
        parts.append(pred)
    out = pd.concat(parts, ignore_index=True)
    if len(out) != len(data.qids) or out["qid"].duplicated().any():
        raise ValueError("cross-validation did not produce one prediction per question")
    return out


def evaluate(train: pd.DataFrame, holdout: pd.DataFrame, *, relation: str,
             train_split: str = "dev", eval_split: str = "test", seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """Fit on dev errors, evaluate once on fresh error questions."""
    if relation == "country":
        raise ValueError("the historical country test is closed")
    if train_split != "dev" or eval_split != "test":
        raise ValueError("only a fresh dev/test study may use this confirmatory evaluator")
    train = validate_grid(train, relation=relation, split=train_split)
    holdout = validate_grid(holdout, relation=relation, split=eval_split)
    if set(train["qid"]) & set(holdout["qid"]):
        raise ValueError("question IDs overlap")
    if set(train["subject_key"]) & set(holdout["subject_key"]):
        raise ValueError("normalized subject names overlap")
    candidates = sorted({str(c) for c in train["cand_obj_id"]})
    if set(candidates) != {str(c) for c in holdout["cand_obj_id"]}:
        raise ValueError("candidate pools differ")
    dev = error_choices(train, candidates)
    test = error_choices(holdout, candidates, x_sd=dev.x_sd)
    base_lam, base_cv = select_penalty(dev, len(candidates), with_cooc=False)
    aug_lam, aug_cv = select_penalty(dev, len(candidates), with_cooc=True)
    base_fit = fit(dev, len(candidates), base_lam, with_cooc=False)
    aug_fit = fit(dev, len(candidates), aug_lam, with_cooc=True)
    base = predict_rows(base_fit, test)
    aug = predict_rows(aug_fit, test)
    comparison = paired_cluster_bootstrap(base, aug, seed=seed)
    rows = base.add_prefix("baseline_").merge(aug.add_prefix("augmented_"),
                                              left_on="baseline_qid", right_on="augmented_qid",
                                              validate="one_to_one")
    result = {"relation": relation, "n_dev_questions": dev.n_all_questions,
              "n_dev_errors": len(dev.qids), "n_test_questions": test.n_all_questions,
              "n_test_errors": len(test.qids), "n_candidates": len(candidates),
              "dev_within_question_cooc_sd": dev.x_sd,
              "baseline": {"penalty": base_lam, "cv_loss": base_cv,
                           "test_log_loss": float(base["loss"].mean()),
                           "test_top1": float(base["top1"].mean()),
                           "test_mean_rank": float(base["rank"].mean())},
              "augmented": {"penalty": aug_lam, "cv_loss": aug_cv,
                            "cooc_beta_per_dev_sd": aug_fit.beta,
                            "test_log_loss": float(aug["loss"].mean()),
                            "test_top1": float(aug["top1"].mean()),
                            "test_mean_rank": float(aug["rank"].mean())},
              "paired_log_loss": comparison}
    return result, rows
