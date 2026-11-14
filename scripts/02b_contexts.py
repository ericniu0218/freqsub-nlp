"""Stage 02b: sampled training-corpus contexts per subject -> co-occurrence estimates for every
candidate entity in the subject's relation pool (see freqsub/counts/contexts.py for the estimator).

Order: non-ambiguous subjects first (primary analysis), then ambiguous ones (robustness).
Resumable at two levels: every API response is cached, and results are written in chunks of 25
subjects that are skipped on restart.

Outputs (artifacts/02b_contexts/<hash>/): chunk_*.parquet with
  cooc rows:    subj_id, obj_id, hits, k, n_subject, exhaustive, p_hat, cooc_hat, se_p
  windows_*.parquet: the raw windows (audit trail)
Usage: python scripts/02b_contexts.py [counts.contexts_per_subject=50] [only_primary=true]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from freqsub.counts.client import InfiniGramClient  # noqa: E402
from freqsub.counts.contexts import estimate_cooc, fetch_subject_contexts  # noqa: E402
from freqsub.utils.config import load_config, resolve  # noqa: E402
from freqsub.utils.io import stage_dir, write_manifest, write_parquet  # noqa: E402
from freqsub.utils.stages import cfg_02b, dir_02a  # noqa: E402

CHUNK = 25



def main() -> None:
    cfg = load_config()
    c = cfg["counts"]
    src = dir_02a(cfg)
    ents = pd.read_parquet(src / "entities.parquet")
    subj = pd.read_parquet(src / "subjects.parquet")
    subj = subj.sort_values(["ambiguous", "subj_id"]).reset_index(drop=True)  # primary first
    if cfg.get("only_primary"):
        subj = subj[~subj["ambiguous"]].reset_index(drop=True)

    scfg = cfg_02b(cfg)
    assert scfg["source"] == src.name
    out = stage_dir("02b_contexts", scfg)
    client = InfiniGramClient(c["index"], resolve(c["cache_path"]), rate_per_s=c["rate_per_s"],
                              cooldown_start_s=c.get("cooldown_start_s", 60.0))
    pools = {r: g.reset_index(drop=True) for r, g in ents.groupby("relation")}

    t0 = time.time()
    for s0 in range(0, len(subj), CHUNK):
        path = out / f"chunk_{s0:05d}.parquet"
        if path.exists():
            continue
        cooc_rows, win_rows = [], []
        for j, (_, s) in enumerate(subj.iloc[s0:s0 + CHUNK].iterrows()):
            if j % 5 == 0:
                print(f"  subject {s0 + j + 1}/{len(subj)} ({(time.time()-t0)/60:.1f} min) | {client.stats}", flush=True)
            ctx = fetch_subject_contexts(client, s["subj"], k=c["contexts_per_subject"],
                                         max_disp_len=c["max_disp_len"], seed=cfg["seed"], workers=c["workers"])
            win_rows += [{"subj_id": s["subj_id"], "window_ix": j, "text": w} for j, w in enumerate(ctx.windows)]
            for _, e in pools[s["relation"]].iterrows():
                est = estimate_cooc(ctx, list(e["search_aliases"]))
                cooc_rows.append({"subj_id": s["subj_id"], "obj_id": e["obj_id"], "hits": est.hits, "k": est.k,
                                  "n_subject": est.n_subject, "exhaustive": est.exhaustive,
                                  "n_blocked": ctx.n_blocked, "p_hat": est.p_hat, "cooc_hat": est.cooc_hat,
                                  "se_p": est.se_p})
        write_parquet(pd.DataFrame(win_rows), out / f"windows_{s0:05d}.parquet")
        write_parquet(pd.DataFrame(cooc_rows), path)
        el = time.time() - t0
        print(f"subjects {min(s0+CHUNK, len(subj))}/{len(subj)} | {el/60:.1f} min | {client.stats}", flush=True)

    write_manifest(out, "02b_contexts", scfg, {"client_stats": client.stats, "n_subjects": len(subj)})
    print(out)


if __name__ == "__main__":
    main()
