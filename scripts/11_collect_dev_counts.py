"""Bounded 50-window corpus probe on 30 preselected occupation DEV errors."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freqsub.analysis.wrong_choice import error_choices, select_penalty, validate_grid  # noqa: E402
from freqsub.counts.client import InfiniGramClient  # noqa: E402
from freqsub.counts.contexts import estimate_cooc, fetch_subject_contexts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INDEX = "v4_olmo-mix-1124_llama"
K = 50
N_ERROR_QUESTIONS = 30


def main() -> None:
    score_path = ROOT / "results/occupation_dev_scores/candidate_rows.parquet"
    scores = pd.read_parquet(score_path)
    frozen = json.loads((ROOT / "configs/occupation_split.json").read_text())
    score_meta = json.loads((ROOT / "results/occupation_dev_scores/results.json").read_text())
    frozen_sha = hashlib.sha256((ROOT / "configs/occupation_split.json").read_bytes()).hexdigest()
    if (score_meta["revision"] != "9d3e43659f00c17e6da23cf32333afd1fc39fa1a"
            or score_meta["split_manifest_sha256"] != frozen_sha
            or set(scores["qid"].astype(int)) != set(frozen["dev_question_ids"])
            or len(scores) != len(frozen["dev_question_ids"]) * frozen["n_candidates"]):
        raise ValueError("development score provenance or candidate grid differs from frozen run")
    chosen = scores[scores["chosen"] == 1]
    errors = chosen[chosen["is_valid"] == 0]
    qids = sorted((int(v) for v in errors["qid"]),
                  key=lambda qid: hashlib.sha256(f"count-probe:{qid}".encode()).hexdigest())[:N_ERROR_QUESTIONS]
    if len(qids) != N_ERROR_QUESTIONS or not set(qids).issubset(set(scores["qid"])):
        raise ValueError("development error cohort is incomplete")
    if set(qids) & set(frozen["test_question_ids"]):
        raise ValueError("test question reached corpus probe")
    scratch = ROOT / "artifacts/cache/occupation_dev_count_probe"
    scratch.mkdir(parents=True, exist_ok=True)
    client = InfiniGramClient(INDEX, scratch / "api.sqlite", rate_per_s=0.45, max_retries=1)
    chunks = []
    for ix, qid in enumerate(qids):
        path = scratch / f"{qid}.parquet"
        if path.exists():
            block = pd.read_parquet(path)
            if set(block["cand_obj_id"]) != set(scores["cand_obj_id"]):
                raise ValueError(f"cached candidate grid differs for {qid}")
        else:
            block = scores[scores["qid"] == qid].copy()
            if len(block) != frozen["n_candidates"]:
                raise ValueError(f"incomplete score grid for {qid}")
            context = fetch_subject_contexts(client, str(block["subj"].iloc[0]), K, 200,
                                             seed=0, workers=1)
            if context.n_blocked or (context.n_occurrences and not context.k):
                raise ValueError(f"blocked or empty corpus windows for {qid}")
            block["subject_occurrences"] = context.n_occurrences
            block["sampled_windows"] = context.k
            block["hits"] = [estimate_cooc(context, [str(name)]).hits for name in block["candidate"]]
            block["log_cooc"] = np.log1p(block["hits"] * context.n_occurrences / max(context.k, 1))
            tmp = path.with_suffix(".parquet.tmp")
            block.to_parquet(tmp, index=False)
            tmp.replace(path)
        chunks.append(block)
        print(f"DEV error context {ix+1}/{len(qids)}: {qid}, windows={block['sampled_windows'].iloc[0]}",
              flush=True)
    all_rows = pd.concat(chunks, ignore_index=True)
    usable_qids = set(all_rows.loc[all_rows["sampled_windows"] > 0, "qid"])
    used = all_rows[all_rows["qid"].isin(usable_qids)].copy()
    result = {"status": "exploratory development diagnostic; test not touched",
              "index": INDEX, "k_max": K, "selected_dev_error_question_ids": qids,
              "n_selected": len(qids), "n_usable": len(usable_qids),
              "n_zero_window": len(qids) - len(usable_qids),
              "n_sampled_windows": int(all_rows.groupby("qid")["sampled_windows"].first().sum()),
              "n_candidate_window_hits": int(all_rows["hits"].sum()),
              "n_chosen_wrong_positive_hits": int(((all_rows["chosen"] == 1) &
                                                  (all_rows["hits"] > 0)).sum()),
              "api_calls_this_invocation": client.stats["api_calls"],
              "cache_hits_this_invocation": client.stats["cache_hits"]}
    if len(usable_qids) >= N_ERROR_QUESTIONS:
        checked = validate_grid(used, relation="occupation", split="dev")
        candidates = sorted({str(v) for v in checked["cand_obj_id"]})
        data = error_choices(checked, candidates)
        base, base_curve = select_penalty(data, len(candidates), with_cooc=False)
        aug, aug_curve = select_penalty(data, len(candidates), with_cooc=True)
        result["development_cv"] = {"baseline_penalty": base, "augmented_penalty": aug,
                                    "baseline_loss_by_penalty": base_curve,
                                    "augmented_loss_by_penalty": aug_curve,
                                    "warning": "same dev cases select penalty and report CV; exploratory"}
    else:
        result["development_cv"] = {"status": "not run: fewer than 30 usable error questions"}
    out = ROOT / "results/occupation_dev_count_probe"
    if out.exists():
        raise ValueError("published context probe exists; refusing overwrite")
    out.mkdir(parents=True)
    all_rows.to_parquet(out / "candidate_rows.parquet", index=False)
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("n_selected", "n_usable", "n_sampled_windows",
                                                "n_chosen_wrong_positive_hits")}, indent=2))


if __name__ == "__main__":
    main()
