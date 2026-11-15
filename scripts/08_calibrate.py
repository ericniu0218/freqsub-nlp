"""Stage 08 (H4): correction methods. Tunes alpha on DEV; evaluates TEST only with final_test=true
(which must not be run before H4 is pre-registered).

Uses the analysis table from stage 06 (scores + counts + co-occurrence) and computes the two
question-independent model priors (contextual calibration, PMI_DC) with the primary checkpoint.
Rare facts = bottom tercile of gold co-occurrence (log_cooc of the gold candidate).
Usage: python scripts/08_calibrate.py [final_test=false]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from freqsub.calibration.priors import apply_correction, model_priors, paired_bootstrap, tune_alpha  # noqa: E402
from freqsub.prompts.builder import PromptBank  # noqa: E402
from freqsub.scoring.hf_scorer import load_model  # noqa: E402
from freqsub.utils.config import load_config  # noqa: E402
from freqsub.utils.io import stage_dir, write_manifest  # noqa: E402
from freqsub.utils.stages import dir_01, dir_02a, frozen_demos  # noqa: E402

PRIORS = ["corpus_marginal", "corpus_cooc", "contextual_cal", "pmi_dc"]



def main() -> None:
    cfg = load_config()
    torch.set_num_threads(int(cfg.get("threads", 2)))
    final_test = bool(cfg.get("final_test", False))
    import importlib.util
    spec = importlib.util.spec_from_file_location("s06", Path(__file__).with_name("06_analyze.py"))
    s06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s06)
    df = s06.build_table(cfg)  # same table as the H1 analysis, built deterministically
    df = df[(df["template"] == "qa") & (df["k"] > 0) & (~df["ambiguous"]) & (~df["label_ambiguous"])].copy()

    data_dir = dir_01(cfg)
    q = pd.read_parquet(data_dir / "questions.parquet")
    ents = pd.read_parquet(dir_02a(cfg) / "entities.parquet")
    bank = PromptBank.from_ids(q, "country", **frozen_demos("country"))  # exactly the scored prompts

    lm = load_model(cfg["model"]["name"], cfg["model"]["revision"], cfg["model"]["dtype"])
    pool = ents[ents["relation"] == "country"].reset_index(drop=True)
    mp = model_priors(lm, bank, "qa", "country", list(pool["obj"]))
    prior_tbl = pd.DataFrame({"cand_obj_id": pool["obj_id"], "contextual_cal": mp["contextual_cal"],
                              "pmi_dc": mp["pmi_dc"]})
    df = df.merge(prior_tbl, on="cand_obj_id", how="left")
    df["corpus_marginal"] = df["log_freq"]
    df["corpus_cooc"] = df["log_cooc"]

    gold_cooc = df[df["is_gold"] == 1].set_index("qid")["log_cooc"]
    cut = gold_cooc.quantile(1 / 3)
    rare_q = set(gold_cooc[gold_cooc <= cut].index)

    dev = df[df["split"] == "dev"]
    res = {"rare_cut_log_cooc": float(cut), "methods": {}}
    for p in PRIORS:
        alpha, curve = tune_alpha(dev, p)
        res["methods"][p] = {"alpha_dev": alpha, "dev_curve": curve}

    eval_split = "test" if final_test else "dev"
    ev = df[df["split"] == eval_split]
    raw = apply_correction(ev, "corpus_marginal", 0.0)
    for p in PRIORS:
        ch = apply_correction(ev, p, res["methods"][p]["alpha_dev"])
        res["methods"][p][f"{eval_split}_all"] = paired_bootstrap(ev, raw, ch)
        rare = ev["qid"].isin(rare_q)
        res["methods"][p][f"{eval_split}_rare"] = paired_bootstrap(ev[rare], raw[rare], ch[rare])
    res["eval_split"] = eval_split
    res["note"] = "dev numbers are optimistic (alpha tuned on dev)" if not final_test else "test evaluated once"

    out = stage_dir("08_calibrate", {"rev": cfg["model"]["revision"], "eval": eval_split})
    (out / "results.json").write_text(json.dumps(res, indent=2))
    write_manifest(out, "08_calibrate", {"rev": cfg["model"]["revision"], "eval": eval_split})
    for p in PRIORS:
        m = res["methods"][p]
        a, r = m[f"{eval_split}_all"], m[f"{eval_split}_rare"]
        print(f"{p:16} alpha={m['alpha_dev']:.2f}  all: {a['acc_a']:.3f}->{a['acc_b']:.3f} "
              f"[{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}]  rare: {r['acc_a']:.3f}->{r['acc_b']:.3f} "
              f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]")
    print(out)


if __name__ == "__main__":
    main()
