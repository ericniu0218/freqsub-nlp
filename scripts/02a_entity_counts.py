"""Stage 02a: exact corpus counts for candidate entities (all aliases) and subjects; alias filtering;
subject ambiguity flags.  Cheap: one count/find query per string, all cached.

Outputs (artifacts/02a_counts/<hash>/):
  entities.parquet   one row per (relation, obj_id): search aliases kept, freq_max, freq_sum
  alias_log.parquet  every alias decision (keep/drop + reason) -- review ratio_flag rows by hand
  subjects.parquet   subj_id, corpus count, token ids, ambiguity score, flag, threshold
Usage: python scripts/02a_entity_counts.py [counts.rate_per_s=1]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from freqsub.counts.client import InfiniGramClient  # noqa: E402
from freqsub.data.aliases import filter_search_aliases  # noqa: E402
from freqsub.data.ambiguity import ambiguity_score, flag_ambiguous  # noqa: E402
from freqsub.utils.config import load_config, resolve  # noqa: E402
from freqsub.utils.io import ARTIFACTS, config_hash, stage_dir, write_manifest, write_parquet  # noqa: E402


def main() -> None:
    cfg = load_config()
    data_dir = ARTIFACTS / "01_data" / config_hash({"data": cfg["data"], "seed": cfg["seed"]})
    q = pd.read_parquet(data_dir / "questions.parquet")
    overrides = OmegaConf.to_container(OmegaConf.load(resolve("configs/alias_overrides.yaml")))
    scfg = {"data": cfg["data"], "seed": cfg["seed"], "index": cfg["counts"]["index"],
            "overrides": overrides}
    out = stage_dir("02a_counts", scfg)
    c = cfg["counts"]
    client = InfiniGramClient(c["index"], resolve(c["cache_path"]), rate_per_s=c["rate_per_s"])

    ent_rows, log_rows = [], []
    for rel in cfg["data"]["relations"]:
        pool = pd.read_parquet(data_dir / f"pool_{rel.replace(' ', '_')}.parquet")
        rel_over = (overrides or {}).get(rel) or {}
        for _, e in pool.iterrows():
            counts = {a: int(client.count(a)["count"]) for a in e["o_alias_list"]}
            kept, log = filter_search_aliases(e["obj"], counts, rel_over.get(e["obj"]))
            log_rows += [{"relation": rel, "obj_id": e["obj_id"], **r} for r in log]
            kc = [counts[a] for a in kept]
            ent_rows.append({"relation": rel, "obj_id": e["obj_id"], "obj": e["obj"], "o_pop": e["o_pop"],
                             "n_gold": e["n_gold"], "search_aliases": kept,
                             "freq_max": max(kc) if kc else 0, "freq_sum": sum(kc)})
    ents, alog = pd.DataFrame(ent_rows), pd.DataFrame(log_rows)
    write_parquet(ents, out / "entities.parquet")
    write_parquet(alog, out / "alias_log.parquet")

    subj = q.drop_duplicates("subj_id")[["subj_id", "subj", "relation", "s_pop", "split"]].reset_index(drop=True)
    found = []
    for i, s in subj.iterrows():
        d = client.find(s["subj"])
        found.append({"subj_count": int(d["cnt"]), "n_shard_segments": len(d["segment_by_shard"])})
        if (i + 1) % 100 == 0:
            print(f"subjects {i+1}/{len(subj)} | {client.stats}", flush=True)
    subj = pd.concat([subj, pd.DataFrame(found)], axis=1)
    subj["amb"] = ambiguity_score(subj["subj_count"], subj["s_pop"].fillna(0))
    subj["ambiguous"], thr = flag_ambiguous(subj)
    write_parquet(subj, out / "subjects.parquet")

    n_flag = int((alog.reason == "ratio_flag").sum())
    write_manifest(out, "02a_counts", scfg, {"client_stats": client.stats, "ambiguity_threshold": thr,
                                             "n_ratio_flags": n_flag})
    print(out)
    print(f"aliases: {len(alog)} total, {int((alog.decision=='drop').sum())} dropped "
          f"({alog.reason.value_counts().to_dict()})")
    print(f"subjects: {len(subj)}, ambiguous: {int(subj.ambiguous.sum())} (threshold amb>{thr:.2f})")
    print(f"zero-count subjects: {int((subj.subj_count==0).sum())}")


if __name__ == "__main__":
    main()
