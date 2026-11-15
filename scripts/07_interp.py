"""Stage 07 (H3): which components push the model toward the wrong (substituted) answer?

Cases: wrong predictions (template qa, primary checkpoint) whose gold and chosen answers differ in
their first token. Per case: logit lens and exact direct logit attribution (DLA) of
logit[gold first token] - logit[chosen first token] at the answer position.

Discovery on DEV: rank components by mean push toward the wrong answer (most negative DLA).
Evaluation on EVAL split (default test; only after H3 is pre-registered): mean-ablate the top-k
components (dev-mean contribution vectors) and measure
  (a) flip rate: share of wrong cases where gold-first-token now beats chosen-first-token,
  (b) collateral damage: share of correct cases where the gold first token stops being the argmax
      among all candidates' first tokens,
against 20 random sets of k components (same k).
Usage: python scripts/07_interp.py [eval_split=dev] [k=3] [max_cases=300]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from freqsub.interp.components import (ablation_at_last_position, component_modules,  # noqa: E402
                                       direct_logit_attribution, logit_lens, run_with_cache)
from freqsub.prompts.builder import PromptBank  # noqa: E402
from freqsub.scoring.hf_scorer import load_model, prompt_token_ids, score_cached  # noqa: E402
from freqsub.utils.config import load_config  # noqa: E402
from freqsub.utils.io import ARTIFACTS, stage_dir, write_manifest, write_parquet  # noqa: E402
from freqsub.utils.stages import dir_01, dir_02a, frozen_demos  # noqa: E402



def first_tok(tok, name: str) -> int:
    return tok(" " + name, add_special_tokens=False)["input_ids"][0]


def main() -> None:
    cfg = load_config()
    torch.set_num_threads(int(cfg.get("threads", 2)))
    eval_split, k, max_cases = cfg.get("eval_split", "dev"), int(cfg.get("k", 3)), int(cfg.get("max_cases", 300))
    rev = cfg["model"]["revision"]
    data_dir = dir_01(cfg)
    q = pd.read_parquet(data_dir / "questions.parquet")
    ents = pd.read_parquet(dir_02a(cfg) / "entities.parquet")
    subj = pd.read_parquet(dir_02a(cfg) / "subjects.parquet")
    names = dict(zip(ents["obj_id"], ents["obj"]))
    ambiguous = set(subj.loc[subj["ambiguous"], "subj_id"])

    shards = [pd.read_parquet(f) for f in (ARTIFACTS / "04_scores").glob("*/country__qa__*.parquet")]
    s = pd.concat(shards, ignore_index=True)  # shard indices overlap; idxmax below needs unique labels
    s = s[s["revision"] == rev].drop_duplicates(["id", "cand_obj_id"])
    top = s.loc[s.groupby("id")["logprob"].idxmax(), ["id", "subj_id", "obj_id_gold", "cand_obj_id"]]
    assert top["id"].is_unique and len(top) == s["id"].nunique(), "top-candidate table must have one row per question"
    second = (s.sort_values("logprob", ascending=False).groupby("id").nth(1)[["id", "cand_obj_id"]]
              .rename(columns={"cand_obj_id": "runner_up"}))
    top = top.merge(q[["id", "question", "split"]], on="id").merge(second, on="id", how="left")
    top = top[~top["subj_id"].isin(ambiguous)]
    # label policy (configs/label_policy.yaml): drop label-ambiguous golds; merged duplicates count as correct
    from omegaconf import OmegaConf
    from freqsub.utils.config import resolve
    pol = OmegaConf.to_container(OmegaConf.load(resolve("configs/label_policy.yaml")))["country"]
    name_to_id = {v: k for k, v in names.items()}
    top = top[~top["obj_id_gold"].isin({name_to_id[n] for n in pol["label_ambiguous_gold"]})]
    for keep, drop in pol["merge"]:
        top.loc[top["obj_id_gold"] == name_to_id[drop], "obj_id_gold"] = name_to_id[keep]
        top.loc[top["cand_obj_id"] == name_to_id[drop], "cand_obj_id"] = name_to_id[keep]
    top["correct"] = top["obj_id_gold"] == top["cand_obj_id"]

    lm = load_model(cfg["model"]["name"], rev, cfg["model"]["dtype"])
    tok = lm.tokenizer
    bank = PromptBank.from_ids(q, "country", **frozen_demos("country"))  # exactly the scored prompts

    def ids_for(r):
        prompt, _, _ = bank.prompt("qa", r["subj_id"], r["question"])
        return prompt_token_ids(tok, prompt)

    # Resumable: every phase is persisted in the stage directory (container restarts are frequent).
    out = stage_dir("07_interp", {"rev": rev, "k": k, "eval_split": eval_split, "max_cases": max_cases})
    import sqlite3
    pc = sqlite3.connect(out / "pair_cache.sqlite")
    pc.execute("CREATE TABLE IF NOT EXISTS s (key TEXT PRIMARY KEY, a REAL, b REAL)")

    # ---- per-case DLA + logit lens on wrong predictions -----------------------------------------
    rows, lens_rows = [], []
    have_dla = (out / "dla.parquet").exists() and (out / "logit_lens.parquet").exists()
    for split in ([] if have_dla else sorted({"dev", eval_split})):
        wrong = top[(top["split"] == split) & (~top["correct"])].head(max_cases)
        for _, r in wrong.iterrows():
            t, f = first_tok(tok, names[r["obj_id_gold"]]), first_tok(tok, names[r["cand_obj_id"]])
            if t == f:
                continue
            cache = run_with_cache(lm, ids_for(r))
            if cache["logits"][f] <= cache["logits"][t]:
                continue  # wrong by full-answer score but gold already wins on the first token: not a first-token case
            a = direct_logit_attribution(lm, cache, t, f)
            rows.append({"id": r["id"], "split": split, **a})
            lens_rows += [{"id": r["id"], "split": split, "layer": i, "logit_diff": v}
                          for i, v in enumerate(logit_lens(lm, cache, t, f))]
    if have_dla:
        dla, lens = pd.read_parquet(out / "dla.parquet"), pd.read_parquet(out / "logit_lens.parquet")
    else:
        dla, lens = pd.DataFrame(rows), pd.DataFrame(lens_rows)
        write_parquet(dla, out / "dla.parquet")
        write_parquet(lens, out / "logit_lens.parquet")
    comp_names = list(component_modules(lm.model))

    # ---- discovery on dev: components pushing toward the wrong answer -------------------------
    dev_dla = dla[dla["split"] == "dev"]
    push = dev_dla[comp_names].mean().sort_values()  # most negative = strongest push toward foil
    top_k = list(push.index[:k])

    # ---- dev-mean contribution vectors for ablation ------------------------------------------
    sums, n = {c: None for c in comp_names}, 0
    means_path = out / "means.pt"
    for _, r in ([] if means_path.exists() else top[top["split"] == "dev"].head(200).iterrows()):
        cache = run_with_cache(lm, ids_for(r))
        for c in comp_names:
            v = cache["components"][c]
            sums[c] = v if sums[c] is None else sums[c] + v
        n += 1
    if means_path.exists():
        means = torch.load(means_path)
    else:
        means = {c: sums[c] / n for c in comp_names}
        torch.save(means, means_path)

    # Ablation is evaluated on FULL-ANSWER scores (all wrong predictions, not only first-token cases):
    # components are replaced at the answer position during the prompt pass, and the candidates'
    # complete log-probs are recomputed on top of the modified KV cache. Pairwise comparisons:
    #   wrong cases: gold vs the chosen wrong answer   -> flip if gold now scores higher
    #   right cases: gold vs the original runner-up    -> damage if the runner-up now scores higher
    # Sets are made baseline-consistent with a recomputed no-ablation pass, so the control is 0.
    def pair_scores(r, a: str, b: str, rep=None):
        key = f"{r['id']}|{a}|{b}|{','.join(sorted(rep)) if rep else '-'}"
        hit = pc.execute("SELECT a, b FROM s WHERE key=?", (key,)).fetchone()
        if hit:
            return list(hit)
        prompt, _, _ = bank.prompt("qa", r["subj_id"], r["question"])
        ctx = (lambda: ablation_at_last_position(lm, rep)) if rep else None
        sc, _ = score_cached(lm, prompt, [a, b], prompt_context=ctx)
        pc.execute("INSERT OR REPLACE INTO s VALUES (?,?,?)", (key, sc[0], sc[1]))
        pc.commit()
        return sc

    eval_wrong, eval_right = [], []
    for _, r in top[(top["split"] == eval_split) & (~top["correct"])].head(max_cases).iterrows():
        g, c = names[r["obj_id_gold"]], names[r["cand_obj_id"]]
        sg, sc_ = pair_scores(r, g, c)
        if sc_ > sg:
            eval_wrong.append((r, g, c))
    for _, r in top[(top["split"] == eval_split) & (top["correct"]) & top["runner_up"].notna()].head(max_cases).iterrows():
        g, ru = names[r["obj_id_gold"]], names[int(r["runner_up"])]
        sg, sr = pair_scores(r, g, ru)
        if sg > sr:
            eval_right.append((r, g, ru))

    def evaluate(comps: list[str]) -> dict:
        rep = {c: means[c] for c in comps} or None
        flips = 0
        for r, g, c in eval_wrong:
            sg, sc_ = pair_scores(r, g, c, rep)
            flips += int(sg > sc_)
        damage = 0
        for r, g, ru in eval_right:
            sg, sr = pair_scores(r, g, ru, rep)
            damage += int(sr > sg)
        return {"flip_rate": flips / max(len(eval_wrong), 1), "n_wrong": len(eval_wrong),
                "collateral": damage / max(len(eval_right), 1), "n_right": len(eval_right)}

    res = {"k": k, "eval_split": eval_split, "top_components": top_k,
           "dev_mean_push": push.round(4).to_dict(), "ablate_none": evaluate([]), "ablate_top": evaluate(top_k)}
    assert res["ablate_none"]["flip_rate"] == 0 and res["ablate_none"]["collateral"] == 0, "control failed"
    rng = random.Random(cfg["seed"])
    rand = [evaluate(rng.sample(comp_names, k)) for _ in range(20)]
    res["ablate_random"] = {"flip_rate_mean": float(np.mean([x["flip_rate"] for x in rand])),
                            "flip_rate_p95": float(np.percentile([x["flip_rate"] for x in rand], 95)),
                            "collateral_mean": float(np.mean([x["collateral"] for x in rand]))}
    # pre-registered H3 claim rule
    res["implicated"] = bool(res["ablate_top"]["flip_rate"] > res["ablate_random"]["flip_rate_p95"]
                             and res["ablate_top"]["collateral"] < 2 * max(res["ablate_random"]["collateral_mean"], 1e-9))
    res["baseline_note"] = "full-answer pairwise evaluation; sets are baseline-consistent, so the no-ablation control is 0"
    res["first_token_case_share_dev"] = float(len(dla[dla["split"] == "dev"]) /
                                             max(int(((top["split"] == "dev") & (~top["correct"])).sum()), 1))
    res["dla_check_max_abs_error"] = float((dla["_total"] - dla["_actual"]).abs().max()) if len(dla) else None

    (out / "results.json").write_text(json.dumps(res, indent=2))
    write_manifest(out, "07_interp", {"rev": rev, "k": k, "eval_split": eval_split})
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "dev_mean_push"}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
