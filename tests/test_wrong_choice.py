"""Meaningful checks for the new wrong-answer target and held-out evaluator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from freqsub.analysis.wrong_choice import (cross_validated_rows, error_choices, evaluate, fit, predict_rows,
                                           validate_grid)


def fixture(n_dev: int = 60, n_test: int = 30, *, signal: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(9)
    rows = []
    for i in range(n_dev + n_test):
        x = rng.normal(size=8) + 4 if signal else np.full(8, float(i))
        gold = int(rng.integers(8))
        wrong = np.array([j for j in range(8) if j != gold])
        chosen = int(wrong[np.argmax(x[wrong])]) if signal else int(wrong[i % len(wrong)])
        for c in range(8):
            rows.append({"qid": str(i), "subj_id": i, "subj": f"Person {i}", "cand_obj_id": str(c),
                         "is_gold": int(c == gold), "is_valid": int(c == gold),
                         "chosen": int(c == chosen),
                         "log_cooc": float(x[c]), "split": "dev" if i < n_dev else "test",
                         "relation": "occupation"})
    frame = pd.DataFrame(rows)
    return frame[frame["split"] == "dev"].copy(), frame[frame["split"] == "test"].copy()


def test_planted_wrong_answer_signal_beats_identity() -> None:
    dev, holdout = fixture()
    result, rows = evaluate(dev, holdout, relation="occupation")
    assert result["n_dev_errors"] == 60 and result["n_test_errors"] == 30
    assert result["paired_log_loss"]["improvement"] > 0.5
    assert result["paired_log_loss"]["ci95"][0] > 0
    assert len(rows) == 30 and rows["augmented_top1"].mean() > rows["baseline_top1"].mean()


def test_constant_within_question_cooccurrence_has_no_effect() -> None:
    dev, holdout = fixture(signal=False)
    order = sorted(dev["cand_obj_id"].unique())
    data = error_choices(validate_grid(dev, relation="occupation", split="dev"), order)
    base = predict_rows(fit(data, len(order), 0.1, with_cooc=False), data)
    added = predict_rows(fit(data, len(order), 0.1, with_cooc=True), data)
    assert np.allclose(base["loss"], added["loss"], atol=1e-9)


def test_rejects_leakage_and_incomplete_grid() -> None:
    dev, holdout = fixture()
    holdout.loc[holdout["qid"] == "60", "subj"] = "PERSON 0"
    with pytest.raises(ValueError, match="normalized subject names overlap"):
        evaluate(dev, holdout, relation="occupation")
    dev, holdout = fixture()
    with pytest.raises(ValueError, match="closed"):
        evaluate(dev, holdout, relation="country")
    with pytest.raises(ValueError, match="incomplete candidate grid"):
        validate_grid(dev.iloc[1:], relation="occupation", split="dev")


def test_correct_questions_do_not_enter_wrong_choice_target() -> None:
    dev, _ = fixture()
    first = dev["qid"] == "0"
    dev.loc[first, "chosen"] = dev.loc[first, "is_gold"]
    data = error_choices(validate_grid(dev, relation="occupation", split="dev"),
                         sorted(dev["cand_obj_id"].unique()))
    assert len(data.qids) == 59 and "0" not in data.qids


def test_alternative_accepted_answer_is_not_an_error() -> None:
    dev, _ = fixture()
    chosen = dev[(dev["qid"] == "0") & (dev["chosen"] == 1)]["cand_obj_id"].iloc[0]
    dev.loc[(dev["qid"] == "0") & (dev["cand_obj_id"] == chosen), "is_valid"] = 1
    data = error_choices(validate_grid(dev, relation="occupation", split="dev"),
                         sorted(dev["cand_obj_id"].unique()))
    assert "0" not in data.qids


def test_tracked_dev_scores_have_complete_grid_and_80_dataset_errors() -> None:
    path = Path(__file__).resolve().parents[1] / "results/occupation_dev_scores/candidate_rows.parquet"
    d = pd.read_parquet(path)
    d["log_cooc"] = 0.0  # only validating score cohort and accepted-answer labels
    checked = validate_grid(d, relation="occupation", split="dev")
    data = error_choices(checked, sorted({str(v) for v in checked["cand_obj_id"]}))
    assert data.n_all_questions == 111
    assert len(data.qids) == 80
    assert all(len(g) >= 74 for g in data.groups)


def test_cross_validated_predictions_cover_each_error_once() -> None:
    dev, _ = fixture(signal=False)
    order = sorted(dev["cand_obj_id"].unique())
    data = error_choices(validate_grid(dev, relation="occupation", split="dev"), order)
    rows = cross_validated_rows(data, len(order), 0.01, with_cooc=False)
    assert len(rows) == 60
    assert rows["qid"].is_unique
    assert set(rows["fold"]) == {0, 1, 2}
    assert set(rows["qid"]) == set(data.qids)
