"""Stage 01: PopQA -> cleaned question table with subject-level dev/test splits.

Usage: python scripts/01_build_data.py [data.relations=[country]] [seed=0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freqsub.data.popqa import download_popqa, load_popqa, object_pool  # noqa: E402
from freqsub.data.splits import assign_splits  # noqa: E402
from freqsub.utils.config import load_config, resolve  # noqa: E402
from freqsub.utils.io import stage_dir, write_manifest, write_parquet  # noqa: E402


def main() -> None:
    cfg = load_config()
    dcfg = {"data": cfg["data"], "seed": cfg["seed"]}
    path = download_popqa(resolve(cfg["data"]["popqa_path"]))
    df = load_popqa(path, cfg["data"]["relations"], cfg["data"]["max_aliases"])
    df = assign_splits(df, dev_frac=cfg["data"]["dev_frac"], seed=cfg["seed"])

    out = stage_dir("01_data", dcfg)
    write_parquet(df, out / "questions.parquet")
    pools = {r: object_pool(df, r) for r in cfg["data"]["relations"]}
    for r, pool in pools.items():
        write_parquet(pool, out / f"pool_{r.replace(' ', '_')}.parquet")

    summary = {r: {"questions": int((df.relation == r).sum()),
                   "subjects": int(df[df.relation == r].subj_id.nunique()),
                   "pool_size": len(pools[r]),
                   "dev_questions": int(((df.relation == r) & (df.split == "dev")).sum())}
               for r in cfg["data"]["relations"]}
    write_manifest(out, "01_data", dcfg, {"summary": summary})
    print(out)
    for r, s in summary.items():
        print(f"  {r}: {s}")


if __name__ == "__main__":
    main()
