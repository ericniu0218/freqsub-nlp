"""Freeze outcome-blind PopQA occupation split and candidate list."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from freqsub.data.aliases import normalize  # noqa: E402
from freqsub.prompts.builder import PromptBank  # noqa: E402

DATA_SHA = "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b"


def prepare(path: Path, seed: int = 0) -> dict:
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("PopQA TSV differs from pinned revision")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["prop"] == "occupation"]
    if len(rows) != 532:
        raise ValueError("unexpected occupation question count")
    groups = defaultdict(list)
    for row in rows:
        groups[normalize(row["subj"])].append(row)
    if "" in groups:
        raise ValueError("empty normalized subject")
    names = sorted(groups, key=lambda k: (max(int(r["s_pop"]) for r in groups[k]), k))
    deciles = defaultdict(list)
    for i, name in enumerate(names):
        deciles[min(9, i * 10 // len(names))].append(name)
    rng = random.Random(seed)
    dev_names = set()
    for decile in sorted(deciles):
        bucket = deciles[decile].copy()
        rng.shuffle(bucket)
        dev_names.update(bucket[:round(len(bucket) * 0.2)])
    dev = [r for r in rows if normalize(r["subj"]) in dev_names]
    test = [r for r in rows if normalize(r["subj"]) not in dev_names]
    counts = Counter(int(r["obj_id"]) for r in rows)
    dev_df = pd.DataFrame({"subj_id": [int(r["subj_id"]) for r in dev],
                           "obj_id": [int(r["obj_id"]) for r in dev],
                           "obj": [r["obj"] for r in dev],
                           "question": [r["question"] for r in dev], "split": "dev"})
    bank = PromptBank(dev_df, counts, seed=seed)
    candidates = sorted({(int(r["obj_id"]), r["obj"]) for r in rows})
    if len({c[0] for c in candidates}) != len(candidates):
        raise ValueError("one candidate ID has conflicting names")
    return {"dataset_revision": "098765c79ea10a2cb19c828324e33281b8336ec0",
            "dataset_sha256": DATA_SHA, "relation": "occupation", "seed": seed,
            "split_unit": "normalized subject string", "n_questions": len(rows),
            "n_subject_names": len(groups), "n_candidates": len(candidates),
            "dev_question_ids": sorted(int(r["id"]) for r in dev),
            "test_question_ids": sorted(int(r["id"]) for r in test),
            "candidate_ids_names": [list(candidate) for candidate in candidates],
            "demos_primary_subject_ids": [int(v) for v in bank.primary["subj_id"]],
            "demos_backup_subject_ids": [int(v) for v in bank.backup["subj_id"]],
            "no_model_scores_inspected": True}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--popqa", type=Path, default=Path("data_raw/popqa_test.tsv"))
    parser.add_argument("--output", type=Path, default=Path("configs/occupation_split.json"))
    parser.add_argument("--review-queue", type=Path, default=Path("configs/occupation_review_queue.tsv"))
    parser.add_argument("--test-review-queue", type=Path,
                        default=Path("configs/occupation_test_review_queue.tsv"))
    parser.add_argument("--dev-full-review-queue", type=Path,
                        default=Path("configs/occupation_dev_full_review_queue.tsv"))
    args = parser.parse_args()
    manifest = prepare(args.popqa)
    content = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text() != content:
        raise ValueError("frozen split differs; refusing overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    with args.popqa.open(newline="", encoding="utf-8") as handle:
        source = {int(r["id"]): r for r in csv.DictReader(handle, delimiter="\t")}
    selected = sorted(manifest["dev_question_ids"],
                      key=lambda qid: hashlib.sha256(f"{manifest['seed']}:{qid}".encode()).hexdigest())[:40]
    fields = ["question_id", "subject", "question", "recorded_gold", "dataset_possible_answers",
              "reviewer_1_gold_valid", "reviewer_2_gold_valid", "reviewer_1_other_valid",
              "reviewer_2_other_valid", "reviewer_1_evidence", "reviewer_2_evidence"]
    review = []
    for qid in selected:
        row = source[qid]
        review.append({"question_id": qid, "subject": row["subj"], "question": row["question"],
                       "recorded_gold": row["obj"], "dataset_possible_answers": row["possible_answers"],
                       **{key: "UNREVIEWED" for key in fields[5:]}})
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(review)
    queue_content = buf.getvalue()
    if args.review_queue.exists() and args.review_queue.read_bytes() != queue_content.encode():
        raise ValueError("existing review queue differs; refusing overwrite")
    args.review_queue.parent.mkdir(parents=True, exist_ok=True)
    args.review_queue.write_bytes(queue_content.encode())
    test_fields = ["question_id", "subject", "question", "recorded_gold", "dataset_possible_answers",
                   "reviewer_1_gold_valid", "reviewer_2_gold_valid", "adjudicated_gold_valid",
                   "reviewer_1_other_valid_candidate_ids", "reviewer_2_other_valid_candidate_ids",
                   "reviewer_1_evidence", "reviewer_2_evidence", "adjudicated_other_valid_candidate_ids",
                   "adjudication_note"]
    test_buf = io.StringIO()
    test_writer = csv.DictWriter(test_buf, fieldnames=test_fields, delimiter="\t", lineterminator="\n")
    test_writer.writeheader()
    for qid in manifest["test_question_ids"]:
        row = source[qid]
        test_writer.writerow({"question_id": qid, "subject": row["subj"],
                              "question": row["question"], "recorded_gold": row["obj"],
                              "dataset_possible_answers": row["possible_answers"],
                              **{key: "UNREVIEWED" for key in test_fields[5:]}})
    test_content = test_buf.getvalue()
    if args.test_review_queue.exists() and args.test_review_queue.read_bytes() != test_content.encode():
        raise ValueError("existing test review queue differs; refusing overwrite")
    args.test_review_queue.parent.mkdir(parents=True, exist_ok=True)
    args.test_review_queue.write_bytes(test_content.encode())
    dev_buf = io.StringIO()
    dev_writer = csv.DictWriter(dev_buf, fieldnames=test_fields, delimiter="\t", lineterminator="\n")
    dev_writer.writeheader()
    for qid in manifest["dev_question_ids"]:
        row = source[qid]
        dev_writer.writerow({"question_id": qid, "subject": row["subj"],
                             "question": row["question"], "recorded_gold": row["obj"],
                             "dataset_possible_answers": row["possible_answers"],
                             **{key: "UNREVIEWED" for key in test_fields[5:]}})
    dev_content = dev_buf.getvalue()
    if args.dev_full_review_queue.exists() and args.dev_full_review_queue.read_bytes() != dev_content.encode():
        raise ValueError("existing full dev review queue differs; refusing overwrite")
    args.dev_full_review_queue.parent.mkdir(parents=True, exist_ok=True)
    args.dev_full_review_queue.write_bytes(dev_content.encode())
    print(f"{len(manifest['dev_question_ids'])} dev and {len(manifest['test_question_ids'])} test questions; "
          f"{manifest['n_candidates']} candidates")


if __name__ == "__main__":
    main()
