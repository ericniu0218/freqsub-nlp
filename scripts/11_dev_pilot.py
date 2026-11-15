"""Bounded, descriptive occupation DEV probe. Never reads or scores test IDs.

Scores 12 outcome-blind development questions against all 84 primary-name
occupations, then samples at most 10 indexed subject occurrences each. At most
132 serial API requests are attempted; throttling stops the run. Raw snippets
stay in the ignored local SQLite cache and are not committed.
"""
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

from freqsub.counts.client import InfiniGramClient  # noqa: E402
from freqsub.counts.contexts import fetch_subject_contexts, estimate_cooc  # noqa: E402
from freqsub.data.aliases import normalize, parse_alias_field  # noqa: E402
from freqsub.prompts.builder import PromptBank  # noqa: E402
from freqsub.scoring.hf_scorer import load_model, score_cached, score_naive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "allenai/OLMo-2-0425-1B"
REVISION = "9d3e43659f00c17e6da23cf32333afd1fc39fa1a"
INDEX = "v4_olmo-mix-1124_llama"
K = 10


def main() -> None:
    manifest = json.loads((ROOT / "configs/occupation_split.json").read_text())
    source_path = ROOT / "data_raw/popqa_test.tsv"
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("public PopQA file differs from frozen dataset")
    with source_path.open(newline="", encoding="utf-8") as handle:
        source = {int(r["id"]): r for r in csv.DictReader(handle, delimiter="\t") if r["prop"] == "occupation"}
    pilot_ids = sorted(manifest["dev_question_ids"],
                       key=lambda qid: hashlib.sha256(f"pilot:{qid}".encode()).hexdigest())[:12]
    if set(pilot_ids) & set(manifest["test_question_ids"]):
        raise AssertionError("pilot selected a test question")
    pool = manifest["candidate_ids_names"]
    ids = [int(c[0]) for c in pool]
    names = [str(c[1]) for c in pool]
    q = pd.DataFrame([{"id": qid, "subj_id": int(row["subj_id"]), "subj": row["subj"],
                       "obj_id": int(row["obj_id"]), "obj": row["obj"],
                       "question": row["question"],
                       "split": "dev" if qid in manifest["dev_question_ids"] else "test",
                       "relation": "occupation"} for qid, row in source.items()])
    bank = PromptBank.from_ids(q, "occupation",
                               manifest["demos_primary_subject_ids"], manifest["demos_backup_subject_ids"])
    cache = ROOT / "artifacts/cache/occupation_dev_pilot"
    cache.mkdir(parents=True, exist_ok=True)
    score_path = cache / "scores.parquet"
    if score_path.exists():
        scores = pd.read_parquet(score_path)
        if set(scores["qid"]) != set(pilot_ids) or len(scores) != len(pilot_ids) * len(ids):
            raise ValueError("cached pilot scores have wrong cohort or candidate grid")
    else:
        torch.set_num_threads(2)
        lm = load_model(MODEL, REVISION, "float32", device="mps" if torch.backends.mps.is_available() else "cpu")
        rows = []
        for ix, qid in enumerate(pilot_ids):
            item = q[q["id"] == qid].iloc[0]
            prompt, demo_set, _ = bank.prompt("qa", item["subj_id"], item["question"])
            values, lengths = score_cached(lm, prompt, names, batch_size=16)
            if ix == 0:
                top = np.argsort(values)[-2:]
                reference = score_naive(lm, prompt, [names[int(i)] for i in top])
                if max(abs(reference[j] - values[int(i)]) for j, i in enumerate(top)) > 1e-4:
                    raise ValueError("trained-model cached scorer differs from naive reference")
            for cid, name, score, length in zip(ids, names, values, lengths):
                rows.append({"qid": qid, "subj_id": int(item["subj_id"]), "subj": item["subj"],
                             "question": item["question"], "cand_obj_id": cid, "candidate": name,
                             "gold_obj_id": int(item["obj_id"]), "logprob": score,
                             "n_tokens": length, "demo_set": demo_set,
                             "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
            print(f"scored DEV {ix+1}/{len(pilot_ids)}: {qid}", flush=True)
        scores = pd.DataFrame(rows)
        scores.to_parquet(score_path, index=False)
    client = InfiniGramClient(INDEX, cache / "api.sqlite", rate_per_s=0.45, max_retries=1)
    output = []
    for ix, qid in enumerate(pilot_ids):
        block = scores[scores["qid"] == qid].copy()
        subject = str(block["subj"].iloc[0])
        context = fetch_subject_contexts(client, subject, K, 200, seed=0, workers=1)
        if context.n_blocked or (context.n_occurrences and not context.k):
            raise ValueError(f"blocked or empty corpus windows for DEV question {qid}")
        block["subject_occurrences"] = context.n_occurrences
        block["sampled_windows"] = context.k
        block["hits"] = [estimate_cooc(context, [name]).hits for name in block["candidate"]]
        block["log_cooc"] = np.log1p(block["hits"] * context.n_occurrences / max(context.k, 1))
        output.append(block)
        print(f"sampled DEV {ix+1}/{len(pilot_ids)}: {qid}, windows={context.k}", flush=True)
    rows = pd.concat(output, ignore_index=True)
    rows["is_gold"] = (rows["cand_obj_id"] == rows["gold_obj_id"]).astype(int)
    accepted = {qid: {normalize(a) for a in parse_alias_field(source[qid]["possible_answers"])}
                for qid in pilot_ids}
    rows["is_valid"] = [int(bool(row.is_gold) or normalize(row.candidate) in accepted[int(row.qid)])
                        for row in rows.itertuples()]
    rows["chosen"] = (rows["logprob"] == rows.groupby("qid")["logprob"].transform("max")).astype(int)
    if (rows.groupby("qid")["chosen"].sum() != 1).any():
        raise ValueError("score tie or missing winner")
    chosen = rows[rows["chosen"] == 1]
    selected = chosen[["qid", "subj", "question", "candidate", "is_gold", "is_valid",
                       "hits", "sampled_windows"]]
    results = {"status": "development diagnostic only; not a hypothesis test",
               "model": MODEL, "revision": REVISION, "index": INDEX,
               "dataset_sha256": manifest["dataset_sha256"], "split_manifest_sha256":
               hashlib.sha256((ROOT / "configs/occupation_split.json").read_bytes()).hexdigest(),
               "pilot_question_ids": pilot_ids, "n_questions": len(pilot_ids),
               "n_wrong_by_single_popqa_gold": int((selected["is_gold"] == 0).sum()),
               "n_wrong_by_popqa_possible_answers": int((selected["is_valid"] == 0).sum()),
               "n_accepted_alternative_chosen": int(((selected["is_gold"] == 0) &
                                                     (selected["is_valid"] == 1)).sum()),
               "n_zero_occurrence_subjects": int((rows.groupby("qid")["subject_occurrences"].first() == 0).sum()),
               "n_candidate_window_hits": int(rows["hits"].sum()),
               "api_calls_this_invocation": client.stats["api_calls"],
               "cache_hits_this_invocation": client.stats["cache_hits"],
               "chosen": selected.to_dict(orient="records")}
    outdir = ROOT / "results/occupation_dev_pilot"
    if outdir.exists():
        raise ValueError("published pilot directory exists; refusing overwrite")
    outdir.mkdir(parents=True)
    rows.to_parquet(outdir / "candidate_rows.parquet", index=False)
    (outdir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"n_questions": len(pilot_ids),
                      "n_wrong_by_possible_answers": results["n_wrong_by_popqa_possible_answers"],
                      "api_calls": client.stats["api_calls"]}, indent=2))


if __name__ == "__main__":
    main()
