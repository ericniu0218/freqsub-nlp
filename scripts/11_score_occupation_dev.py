"""Score all 111 frozen occupation DEV questions; never load test prompts into OLMo."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from freqsub.data.aliases import normalize, parse_alias_field  # noqa: E402
from freqsub.prompts.builder import PromptBank  # noqa: E402
from freqsub.scoring.hf_scorer import load_model, score_cached, score_naive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "allenai/OLMo-2-0425-1B"
REVISION = "9d3e43659f00c17e6da23cf32333afd1fc39fa1a"


def main() -> None:
    frozen_path = ROOT / "configs/occupation_split.json"
    frozen = json.loads(frozen_path.read_text())
    source_path = ROOT / "data_raw/popqa_test.tsv"
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != frozen["dataset_sha256"]:
        raise ValueError("public PopQA file differs from frozen dataset")
    with source_path.open(newline="", encoding="utf-8") as handle:
        source = {int(r["id"]): r for r in csv.DictReader(handle, delimiter="\t") if r["prop"] == "occupation"}
    dev_ids = set(frozen["dev_question_ids"])
    test_ids = set(frozen["test_question_ids"])
    if dev_ids & test_ids or dev_ids | test_ids != set(source):
        raise ValueError("frozen question IDs do not partition the pinned dataset")
    q = pd.DataFrame([{"id": qid, "subj_id": int(row["subj_id"]), "subj": row["subj"],
                       "obj_id": int(row["obj_id"]), "obj": row["obj"],
                       "question": row["question"], "relation": "occupation",
                       "split": "dev" if qid in dev_ids else "test"} for qid, row in source.items()])
    bank = PromptBank.from_ids(q, "occupation", frozen["demos_primary_subject_ids"],
                               frozen["demos_backup_subject_ids"])
    candidates = frozen["candidate_ids_names"]
    ids, names = [int(x[0]) for x in candidates], [str(x[1]) for x in candidates]
    outdir = ROOT / "results/occupation_dev_scores"
    if outdir.exists():
        raise ValueError("development output exists; refusing silent overwrite")
    torch.set_num_threads(2)
    lm = load_model(MODEL, REVISION, "float32", device="mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for ix, qid in enumerate(sorted(dev_ids)):
        item = q[q["id"] == qid].iloc[0]
        prompt, demo_set, _ = bank.prompt("qa", item["subj_id"], item["question"])
        scores, lengths = score_cached(lm, prompt, names, batch_size=16)
        if ix == 0:
            top = np.argsort(scores)[-2:]
            ref = score_naive(lm, prompt, [names[int(i)] for i in top])
            if max(abs(ref[j] - scores[int(i)]) for j, i in enumerate(top)) > 1e-4:
                raise ValueError("trained-model scorer parity failed")
        accepted = {normalize(a) for a in parse_alias_field(source[qid]["possible_answers"])}
        for cid, name, score, length in zip(ids, names, scores, lengths):
            is_gold = int(cid == int(item["obj_id"]))
            rows.append({"qid": qid, "subj_id": int(item["subj_id"]), "subj": item["subj"],
                         "question": item["question"], "cand_obj_id": cid, "candidate": name,
                         "is_gold": is_gold, "is_valid": int(bool(is_gold) or normalize(name) in accepted),
                         "logprob": score, "n_tokens": length, "demo_set": demo_set,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "split": "dev", "relation": "occupation"})
        if (ix + 1) % 10 == 0 or ix + 1 == len(dev_ids):
            print(f"scored {ix+1}/{len(dev_ids)} DEV questions", flush=True)
    data = pd.DataFrame(rows)
    data["chosen"] = (data["logprob"] == data.groupby("qid")["logprob"].transform("max")).astype(int)
    if (data.groupby("qid")["chosen"].sum() != 1).any():
        raise ValueError("score tie or missing winner")
    chosen = data[data["chosen"] == 1]
    summary = {"status": "development outcomes only; no corpus proximity or held-out test",
               "model": MODEL, "revision": REVISION,
               "split_manifest_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
               "n_questions": len(dev_ids), "n_candidates": len(ids),
               "n_dataset_gold_correct": int(chosen["is_gold"].sum()),
               "n_possible_answers_correct": int(chosen["is_valid"].sum()),
               "n_possible_answers_errors": int((chosen["is_valid"] == 0).sum()),
               "chosen_label_counts": chosen["candidate"].value_counts().to_dict()}
    outdir.mkdir(parents=True)
    data.to_parquet(outdir / "candidate_rows.parquet", index=False)
    (outdir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_questions", "n_dataset_gold_correct",
                                               "n_possible_answers_correct")}, indent=2))


if __name__ == "__main__":
    main()
