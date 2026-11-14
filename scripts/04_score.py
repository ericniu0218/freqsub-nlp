"""Stage 04: score every candidate entity for every question with an exact LM log-probability.

For small typed pools (country: 107 entities) every pool entity is a candidate: no candidate
sampling, so no candidate-construction confound. Writes one shard per 50 questions; finished shards
are skipped on restart (resumable on Kaggle / after interruptions).

Output rows: id, subj_id, obj_id_gold, cand_obj_id, logprob, n_tokens, template, demo_set,
             gold_in_demos, revision
Usage: python scripts/04_score.py [templates=[qa]] [model.revision=stage1-...] [limit=40]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from freqsub.prompts.builder import PromptBank  # noqa: E402
from freqsub.scoring.hf_scorer import load_model, score_cached  # noqa: E402
from freqsub.utils.config import load_config  # noqa: E402
from freqsub.utils.io import stage_dir, write_manifest, write_parquet  # noqa: E402
from freqsub.utils.stages import dir_01, dir_02a, frozen_demos  # noqa: E402

SHARD = 50



def main() -> None:
    cfg = load_config()
    templates = cfg.get("templates") or ["qa"]
    limit = cfg.get("limit")
    data_dir = dir_01(cfg)
    q = pd.read_parquet(data_dir / "questions.parquet")
    ents = pd.read_parquet(dir_02a(cfg) / "entities.parquet")
    # demonstrations are frozen in configs/demos.yaml so every run uses identical prompts

    scfg = {"data": cfg["data"], "seed": cfg["seed"], "model": cfg["model"], "templates": templates,
            "limit": limit}
    q_all = q  # demos are DEV questions: always look them up in the unfiltered table
    subset = cfg.get("question_subset")  # e.g. "primary_test" for H2 checkpoint runs
    if subset:
        scfg["question_subset"] = subset  # only added when set, so earlier runs keep their directories
        if subset != "primary_test":
            raise ValueError(f"unknown question_subset {subset!r}")
        subj = pd.read_parquet(dir_02a(cfg) / "subjects.parquet")
        keep_subj = set(subj.loc[~subj["ambiguous"], "subj_id"])
        q = q[(q["split"] == "test") & (q["subj_id"].isin(keep_subj))].reset_index(drop=True)
    out = stage_dir("04_scores", scfg)
    torch.set_num_threads(int(cfg.get("threads", 2)))
    # skip the (6 GB) model download/load entirely if every shard of this run already exists
    expected = []
    for rel in cfg["data"]["relations"]:
        n_q = int((q["relation"] == rel).sum()) if not limit else min(int(limit), int((q["relation"] == rel).sum()))
        expected += [out / f"{rel.replace(' ', '_')}__{t}__{s0:05d}.parquet"
                     for t in templates for s0 in range(0, n_q, SHARD)]
    if all(p.exists() for p in expected):
        print(f"all {len(expected)} shards already exist for {cfg['model']['revision']} {templates}; nothing to do")
        return
    lm = load_model(cfg["model"]["name"], cfg["model"]["revision"], cfg["model"]["dtype"])
    t0, done = time.time(), 0

    for rel in cfg["data"]["relations"]:
        rq = q[q["relation"] == rel].reset_index(drop=True)
        if limit:
            rq = rq.head(int(limit))
        pool = ents[ents["relation"] == rel].reset_index(drop=True)
        cand_names, cand_ids = list(pool["obj"]), list(pool["obj_id"])
        bank = PromptBank.from_ids(q_all, rel, **frozen_demos(rel))
        for template in templates:
            for s0 in range(0, len(rq), SHARD):
                path = out / f"{rel.replace(' ', '_')}__{template}__{s0:05d}.parquet"
                if path.exists():
                    continue
                rows = []
                for _, r in rq.iloc[s0:s0 + SHARD].iterrows():
                    prompt, which, demo_objs = bank.prompt(template, r["subj_id"], r["question"])
                    scores, lens = score_cached(lm, prompt, cand_names, batch_size=cfg.get("batch_size", 64))
                    for cid, sc, n in zip(cand_ids, scores, lens):
                        rows.append({"id": r["id"], "subj_id": r["subj_id"], "relation": rel,
                                     "obj_id_gold": r["obj_id"], "cand_obj_id": cid, "logprob": sc,
                                     "n_tokens": n, "template": template, "demo_set": which,
                                     "gold_in_demos": r["obj_id"] in demo_objs,
                                     "revision": cfg["model"]["revision"]})
                    done += 1
                write_parquet(pd.DataFrame(rows), path)
                el = time.time() - t0
                print(f"{rel}/{template} shard {s0}: {done} questions, {el/60:.1f} min, "
                      f"{el/max(done,1):.1f}s/question", flush=True)

    # record the exact demonstrations used, for the write-up and for audits
    demo_rows = []
    for rel in cfg["data"]["relations"]:
        bank = PromptBank.from_ids(q_all, rel, **frozen_demos(rel))
        for name, d in [("primary", bank.primary), ("backup", bank.backup)]:
            demo_rows += [{"relation": rel, "set": name, "subj_id": x["subj_id"], "question": x["question"],
                           "obj": x["obj"]} for _, x in d.iterrows()]
    write_parquet(pd.DataFrame(demo_rows), out / "demos.parquet")
    write_manifest(out, "04_scores", scfg, {"questions_scored": done})
    print(out)


if __name__ == "__main__":
    main()
