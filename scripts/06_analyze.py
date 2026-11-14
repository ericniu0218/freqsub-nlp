"""Stage 06: H1 analysis exactly as pre-registered (freqsub/preregistration.md).

Joins scores (04) + co-occurrence (02b) + entity/subject counts (02a) into a question x candidate
table, then reports: accuracy, the primary conditional logit, the substitution rank test, the
within-question OLS, and the robustness grid. Writes results.json + tables to artifacts/06_h1/<hash>/.
Usage: python scripts/06_analyze.py [split=test] [allow_partial=true]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freqsub.analysis.choice_model import (fit_conditional_logit, substitution_rank_test,  # noqa: E402
                                           within_question_ols)
from freqsub.utils.config import load_config  # noqa: E402
from freqsub.utils.io import ARTIFACTS, stage_dir, write_manifest, write_parquet  # noqa: E402
from freqsub.utils.stages import dir_01, dir_02a, dir_02b  # noqa: E402

PRIMARY_FEATURES = ["log_cooc", "gold_x_log_cooc", "log_freq", "log_opop", "n_tokens", "is_gold"]



def build_table(cfg: dict) -> pd.DataFrame:
    data_dir = dir_01(cfg)
    q = pd.read_parquet(data_dir / "questions.parquet")[["id", "subj_id", "split", "s_pop"]]
    c02a = dir_02a(cfg)
    ents = pd.read_parquet(c02a / "entities.parquet")[["obj_id", "freq_max", "o_pop", "obj"]]
    subj = pd.read_parquet(c02a / "subjects.parquet")[["subj_id", "subj_count", "amb", "ambiguous"]]
    cooc = pd.concat([pd.read_parquet(f) for f in sorted(dir_02b(cfg).glob("chunk_*.parquet"))],
                     ignore_index=True)
    # union of all scoring runs (one per template) for the configured checkpoint
    shards = [pd.read_parquet(f) for f in sorted((ARTIFACTS / "04_scores").glob("*/*__*.parquet"))]
    scores = pd.concat(shards, ignore_index=True)
    scores = scores[scores["revision"] == cfg["model"]["revision"]]
    scores = scores.drop_duplicates(["id", "template", "cand_obj_id"])

    df = scores.merge(q, on=["id", "subj_id"], how="left")
    df = df.merge(ents.rename(columns={"obj_id": "cand_obj_id"}), on="cand_obj_id", how="left")
    df = df.merge(subj, on="subj_id", how="left")
    df = df.merge(cooc[["subj_id", "obj_id", "p_hat", "cooc_hat", "k", "n_subject"]]
                  .rename(columns={"obj_id": "cand_obj_id"}), on=["subj_id", "cand_obj_id"], how="inner")

    # ---- label policy (configs/label_policy.yaml) ---------------------------------------------
    from omegaconf import OmegaConf
    from freqsub.utils.config import resolve
    pol = OmegaConf.to_container(OmegaConf.load(resolve("configs/label_policy.yaml"))).get("country", {})
    name_to_id = dict(zip(ents["obj"], ents["obj_id"]))
    for keep, drop in pol.get("merge", []):
        k_id, d_id = name_to_id[keep], name_to_id[drop]
        df = df[df["cand_obj_id"] != d_id]
        df.loc[df["obj_id_gold"] == d_id, "obj_id_gold"] = k_id
    amb_gold = {name_to_id[n] for n in pol.get("label_ambiguous_gold", [])}
    df["label_ambiguous"] = df["obj_id_gold"].isin(amb_gold)

    df["qid"] = df["id"].astype(str) + "|" + df["template"]
    df["is_gold"] = (df["cand_obj_id"] == df["obj_id_gold"]).astype(int)
    df["log_cooc"] = np.log1p(df["cooc_hat"].fillna(0))
    df["gold_x_log_cooc"] = df["is_gold"] * df["log_cooc"]
    df["log_freq"] = np.log1p(df["freq_max"].astype(float))
    df["log_opop"] = np.log1p(df["o_pop"].astype(float).fillna(0))
    df["logprob_norm"] = df["logprob"] / df["n_tokens"]
    for rule, col in [("chosen", "logprob"), ("chosen_norm", "logprob_norm")]:
        best = df.groupby("qid")[col].transform("max")
        df[rule] = (df[col] == best).astype(int)
        # exact ties are vanishingly rare with float log-probs; keep the first to guarantee one choice
        df[rule] = df[rule] * (df.groupby("qid")[rule].cumsum() == 1)
    return df


def summarize(df: pd.DataFrame, label: str, choice: str = "chosen", features=None) -> dict:
    features = features or PRIMARY_FEATURES
    d = df.copy()
    if choice != "chosen":
        d["chosen"] = d[choice]
    n_q = d["qid"].nunique()
    acc = float(d[(d["chosen"] == 1)]["is_gold"].mean()) if n_q else float("nan")
    out = {"label": label, "n_questions": int(n_q), "accuracy": acc}
    if n_q < 30:
        out["note"] = "too few questions"
        return out
    table, info = fit_conditional_logit(d, features)
    out["clogit"] = table.round(4).to_dict(orient="index")
    out["clogit_info"] = {k: v for k, v in info.items() if k != "scale"}
    b = table.loc[features[0]]
    out["primary_beta"] = {"feature": features[0], "coef": float(b.coef), "ci": [float(b.ci_low), float(b.ci_high)],
                           "verdict": "supported" if b.ci_low > 0 else ("contradicted" if b.ci_high < 0
                                                                       else "inconclusive")}
    rt = substitution_rank_test(d, n_perm=10_000, feature=features[0])
    out["rank_test"] = rt.__dict__
    ols = within_question_ols(d, features=[f for f in features])
    out["within_question_ols"] = ols.round(4).to_dict(orient="index")
    return out


def main() -> None:
    cfg = load_config()
    split = cfg.get("split", "test")
    df = build_table(cfg)
    scfg = {"split": split, "revision": cfg["model"]["revision"], "cooc": dir_02b(cfg).name,
            "counts": dir_02a(cfg).name}
    out = stage_dir("06_h1", scfg)
    write_parquet(df, out / "analysis_table.parquet")

    base = df[(df["template"] == "qa") & (df["k"] > 0)]
    primary = base[(base["split"] == split) & (~base["ambiguous"]) & (~base["label_ambiguous"])]
    results = {"split": split, "primary": summarize(primary, "PRIMARY: qa, non-ambiguous, N_s>0")}

    rob = {}
    rob["incl_ambiguous"] = summarize(base[(base["split"] == split) & (~base["label_ambiguous"])],
                                      "including ambiguous subjects")
    rob["incl_label_ambiguous"] = summarize(base[(base["split"] == split) & (~base["ambiguous"])],
                                            "including label-ambiguous golds")
    for t in sorted(base[base["split"] == split]["amb"].pipe(lambda s: pd.qcut(s, 3, labels=False)).unique()):
        sub = base[base["split"] == split].copy()
        sub["tercile"] = pd.qcut(sub["amb"], 3, labels=False)
        rob[f"amb_tercile_{int(t)}"] = summarize(sub[sub["tercile"] == t], f"ambiguity tercile {int(t)}")
    rob["excl_gold_in_demos"] = summarize(primary[~primary["gold_in_demos"]], "excluding gold-in-demos")
    p2 = primary.copy()
    p2["p_hat_f"] = p2["p_hat"]
    p2["gold_x_p_hat"] = p2["is_gold"] * p2["p_hat"]
    rob["p_hat_feature"] = summarize(p2, "p_hat instead of log_cooc",
                                     features=["p_hat_f", "gold_x_p_hat", "log_freq", "log_opop", "n_tokens", "is_gold"])
    rob["length_normalized_choice"] = summarize(primary, "length-normalized choice rule", choice="chosen_norm")
    for t in sorted(set(df["template"]) - {"qa"}):
        tdf = df[(df["template"] == t) & (df["k"] > 0) & (df["split"] == split) & (~df["ambiguous"])
                 & (~df["label_ambiguous"])]
        rob[f"template_{t}"] = summarize(tdf, f"template {t}")
    if split == "test":
        dev = base[(base["split"] == "dev") & (~base["ambiguous"]) & (~base["label_ambiguous"])]
        rob["dev_split"] = summarize(dev, "dev split (transparency only)")
    results["robustness"] = rob

    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_manifest(out, "06_h1", scfg, {"n_rows": len(df)})
    p = results["primary"]
    print(f"PRIMARY  n={p['n_questions']}  acc={p['accuracy']:.3f}")
    if "primary_beta" in p:
        pb = p["primary_beta"]
        print(f"  beta(log_cooc) = {pb['coef']:.3f}  95% CI [{pb['ci'][0]:.3f}, {pb['ci'][1]:.3f}]  -> {pb['verdict']}")
        print(f"  rank test: mean={p['rank_test']['mean_rank']:.3f} (null 0.5), p={p['rank_test']['p_value']:.4f}, "
              f"n_wrong={p['rank_test']['n_wrong']}")
    for k, v in rob.items():
        pb = v.get("primary_beta")
        s = f"beta={pb['coef']:.3f} [{pb['ci'][0]:.3f},{pb['ci'][1]:.3f}] {pb['verdict']}" if pb else v.get("note", "")
        print(f"  {k:28} n={v['n_questions']:4d} acc={v['accuracy']:.3f}  {s}")
    print(out)


if __name__ == "__main__":
    main()
