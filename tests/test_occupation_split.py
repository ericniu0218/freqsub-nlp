"""Guard the outcome-blind occupation split against silent changes."""
from __future__ import annotations

import json
import csv
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_occupation_inventory() -> None:
    manifest = json.loads((ROOT / "configs/occupation_split.json").read_text())
    assert manifest["relation"] == "occupation"
    assert manifest["dataset_sha256"] == "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b"
    assert len(manifest["dev_question_ids"]) == 111
    assert len(manifest["test_question_ids"]) == 421
    assert not set(manifest["dev_question_ids"]) & set(manifest["test_question_ids"])
    assert len(manifest["candidate_ids_names"]) == 84
    assert not set(manifest["demos_primary_subject_ids"]) & set(manifest["demos_backup_subject_ids"])


def test_split_replays_when_public_tsv_available() -> None:
    path = ROOT / "data_raw/popqa_test.tsv"
    if not path.exists():
        pytest.skip("pinned public TSV not downloaded")
    import importlib.util
    spec = importlib.util.spec_from_file_location("prep_occupation", ROOT / "scripts/11_prepare_occupation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frozen = json.loads((ROOT / "configs/occupation_split.json").read_text())
    assert mod.prepare(path) == frozen
    from freqsub.data.aliases import normalize
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {int(r["id"]): r for r in csv.DictReader(handle, delimiter="\t") if r["prop"] == "occupation"}
    dev_names = {normalize(rows[qid]["subj"]) for qid in frozen["dev_question_ids"]}
    test_names = {normalize(rows[qid]["subj"]) for qid in frozen["test_question_ids"]}
    assert not dev_names & test_names


def test_blinded_test_review_queue_matches_frozen_ids() -> None:
    frozen = json.loads((ROOT / "configs/occupation_split.json").read_text())
    with (ROOT / "configs/occupation_test_review_queue.tsv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert "model_choice" not in reader.fieldnames and "log_cooc" not in reader.fieldnames
        rows = list(reader)
    assert len(rows) == 421
    assert {int(r["question_id"]) for r in rows} == set(frozen["test_question_ids"])
    assert all(r["reviewer_1_other_valid_candidate_ids"] == "UNREVIEWED" and
               r["reviewer_2_other_valid_candidate_ids"] == "UNREVIEWED" for r in rows)


def test_unreviewed_test_queue_cannot_unlock_evaluation() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_occupation", ROOT / "scripts/11_wrong_choice.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frozen = json.loads((ROOT / "configs/occupation_split.json").read_text())
    with pytest.raises(ValueError, match="unreviewed"):
        mod.adjudicated_ids(ROOT / "configs/occupation_test_review_queue.tsv", frozen)
    with pytest.raises(ValueError, match="unreviewed"):
        mod.adjudicated_ids(ROOT / "configs/occupation_dev_full_review_queue.tsv", frozen, split="dev")


def test_blind_adjudication_excludes_invalid_gold(tmp_path: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_occupation", ROOT / "scripts/11_wrong_choice.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frozen = {"candidate_ids_names": [[1, "actor"], [2, "writer"]], "test_question_ids": [10, 11]}
    path = tmp_path / "review.tsv"
    fields = ["question_id", "reviewer_1_gold_valid", "reviewer_2_gold_valid",
              "adjudicated_gold_valid", "reviewer_1_other_valid_candidate_ids",
              "reviewer_2_other_valid_candidate_ids", "adjudicated_other_valid_candidate_ids",
              "reviewer_1_evidence", "reviewer_2_evidence", "adjudication_note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({"question_id": 10, "reviewer_1_gold_valid": "yes",
                         "reviewer_2_gold_valid": "yes", "adjudicated_gold_valid": "yes",
                         "reviewer_1_other_valid_candidate_ids": "2",
                         "reviewer_2_other_valid_candidate_ids": "none",
                         "adjudicated_other_valid_candidate_ids": "2",
                         "reviewer_1_evidence": "source for writer", "reviewer_2_evidence": "none",
                         "adjudication_note": "none"})
        writer.writerow({"question_id": 11, "reviewer_1_gold_valid": "no",
                         "reviewer_2_gold_valid": "no", "adjudicated_gold_valid": "no",
                         "reviewer_1_other_valid_candidate_ids": "none",
                         "reviewer_2_other_valid_candidate_ids": "none",
                         "adjudicated_other_valid_candidate_ids": "none",
                         "reviewer_1_evidence": "source contradicts gold",
                         "reviewer_2_evidence": "independent source contradicts gold",
                         "adjudication_note": "exclude under blind rule"})
    extras, excluded = mod.adjudicated_ids(path, frozen)
    assert extras == {10: {2}, 11: set()}
    assert excluded == {11}
