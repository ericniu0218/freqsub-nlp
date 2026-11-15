"""Evaluate which wrong occupation the model chooses, once fresh inputs exist.

Both inputs must be complete question x candidate Parquet tables with the columns
specified by freqsub.analysis.wrong_choice.REQUIRED. The old country test is refused.
Do not run on held-out occupation rows until the dated protocol and semantic audit
in WRONG_ANSWER_PLAN.md have been completed and committed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from freqsub.analysis.wrong_choice import evaluate  # noqa: E402
from freqsub.data.aliases import normalize, parse_alias_field  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def adjudicated_ids(path: Path, frozen: dict, split: str = "test") -> tuple[dict[int, set[int]], set[int]]:
    """Require two completed blind reviews and a resolved candidate set per question."""
    if split not in {"dev", "test"}:
        raise ValueError("review split must be dev or test")
    allowed = {int(item[0]) for item in frozen["candidate_ids_names"]}
    expected = set(frozen[f"{split}_question_ids"])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != len(expected):
        raise ValueError("test adjudication has incomplete question coverage")

    def parse_ids(raw: str) -> set[int]:
        if not raw or not raw.strip() or raw.strip().lower() == "unreviewed":
            raise ValueError("unreviewed occupation candidate set; enter 'none' explicitly")
        if raw.strip().lower() == "none":
            return set()
        ids = {int(part.strip()) for part in raw.split(",")}
        if not ids.issubset(allowed):
            raise ValueError("adjudication contains candidate ID outside frozen pool")
        return ids

    result, excluded = {}, set()
    for row in rows:
        qid = int(row["question_id"])
        if qid not in expected or qid in result:
            raise ValueError("unknown or repeated question in adjudication")
        first = parse_ids(row["reviewer_1_other_valid_candidate_ids"])
        second = parse_ids(row["reviewer_2_other_valid_candidate_ids"])
        final = parse_ids(row["adjudicated_other_valid_candidate_ids"])
        gold_flags = [row[key].strip().lower() for key in
                      ("reviewer_1_gold_valid", "reviewer_2_gold_valid", "adjudicated_gold_valid")]
        if any(flag not in {"yes", "no"} for flag in gold_flags):
            raise ValueError("unreviewed gold validity; enter yes or no")
        if ((gold_flags[0] != gold_flags[1] or gold_flags[2] == "no")
                and row["adjudication_note"].strip().lower() in {"", "unreviewed"}):
            raise ValueError("gold validity disagreement or exclusion needs an adjudication note")
        if ((gold_flags[0] == "no" and row["reviewer_1_evidence"].strip().lower() in {"", "unreviewed"})
                or (gold_flags[1] == "no" and row["reviewer_2_evidence"].strip().lower() in
                    {"", "unreviewed"})):
            raise ValueError("rejected gold answer needs source evidence")
        if gold_flags[2] == "no":
            excluded.add(qid)
        if (first and row["reviewer_1_evidence"].strip().lower() in {"", "unreviewed"}) or (
                second and row["reviewer_2_evidence"].strip().lower() in {"", "unreviewed"}):
            raise ValueError("reviewer listed an additional valid answer without evidence")
        result[qid] = final
    if set(result) != expected:
        raise ValueError("test adjudication has missing IDs")
    return result, excluded


def apply_validity(frame: pd.DataFrame, ids_by_question: dict[int, set[int]], frozen: dict) -> pd.DataFrame:
    """Recompute test validity from pinned public aliases plus blind adjudication."""
    names = {int(cid): str(name) for cid, name in frozen["candidate_ids_names"]}
    source_path = ROOT / "data_raw/popqa_test.tsv"
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != frozen["dataset_sha256"]:
        raise ValueError("test validity source differs from pinned PopQA TSV")
    with source_path.open(newline="", encoding="utf-8") as handle:
        source = {int(r["id"]): r for r in csv.DictReader(handle, delimiter="\t") if r["prop"] == "occupation"}
    out = frame.copy()
    accepted = {qid: {normalize(a) for a in parse_alias_field(source[qid]["possible_answers"])}
                for qid in ids_by_question}
    out["is_valid"] = [int(bool(row.is_gold) or
                           normalize(names[int(row.cand_obj_id)]) in accepted[int(row.qid)] or
                           int(row.cand_obj_id) in ids_by_question[int(row.qid)])
                       for row in out.itertuples()]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True, help="complete development candidate table")
    parser.add_argument("--test", type=Path, required=True, help="complete fresh held-out candidate table")
    parser.add_argument("--relation", default="occupation")
    parser.add_argument("--adjudication", type=Path, required=True,
                        help="completed, two-reader blind test review queue")
    parser.add_argument("--dev-adjudication", type=Path, required=True,
                        help="completed, two-reader blind development review queue")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.relation != "occupation":
        parser.error("this frozen fresh protocol supports occupation only; country is closed")
    if args.output.exists():
        parser.error("output already exists; evaluation is one-shot and cannot be silently overwritten")
    frozen = json.loads((ROOT / "configs/occupation_split.json").read_text())
    reviewed, test_excluded = adjudicated_ids(args.adjudication, frozen, split="test")
    dev_reviewed, dev_excluded = adjudicated_ids(args.dev_adjudication, frozen, split="dev")
    dev = pd.read_parquet(args.dev)
    test = pd.read_parquet(args.test)
    if set(dev["qid"].astype(int)) != set(frozen["dev_question_ids"]) or (
            set(test["qid"].astype(int)) != set(frozen["test_question_ids"])):
        parser.error("input question IDs do not match the frozen full dev/test split")
    dev = apply_validity(dev[~dev["qid"].astype(int).isin(dev_excluded)], dev_reviewed, frozen)
    test = apply_validity(test[~test["qid"].astype(int).isin(test_excluded)], reviewed, frozen)
    result, rows = evaluate(dev, test, relation=args.relation)
    result["blind_gold_invalid_exclusions"] = {"dev_question_ids": sorted(dev_excluded),
                                                "test_question_ids": sorted(test_excluded)}
    candidate_names = {str(cid): str(name) for cid, name in frozen["candidate_ids_names"]}
    for prefix in ("baseline", "augmented"):
        rows[f"{prefix}_chosen_wrong_name"] = rows[f"{prefix}_chosen_wrong_id"].map(candidate_names)
        rows[f"{prefix}_predicted_wrong_name"] = rows[f"{prefix}_predicted_wrong_id"].map(candidate_names)
    result["inputs_sha256"] = {"dev": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
                               "test": hashlib.sha256(args.test.read_bytes()).hexdigest(),
                               "dev_adjudication": hashlib.sha256(args.dev_adjudication.read_bytes()).hexdigest(),
                               "test_adjudication": hashlib.sha256(args.adjudication.read_bytes()).hexdigest()}
    args.output.mkdir(parents=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    rows.to_parquet(args.output / "question_predictions.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
