"""Stage 09 (H2): how do substitution and accuracy change over training?

For every scored stage-1 checkpoint, refit the pre-registered H1 model (same sample definition) and
record accuracy, beta(log_cooc), beta(log_freq) and the exposure term. Exposure note: a checkpoint at
T tokens has seen ~min(1, T / 4.58T) of the corpus (shuffled-order assumption), so counts are
rescaled by that fraction for the accuracy-vs-exposure curves; within-checkpoint betas are unaffected
(a constant rescaling of cooc only shifts log1p slightly and is absorbed by standardisation).
Usage: python scripts/09_dynamics.py [split=dev]
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from freqsub.utils.config import load_config  # noqa: E402
from freqsub.utils.io import ARTIFACTS, stage_dir, write_manifest, write_parquet  # noqa: E402

CORPUS_TOKENS_B = 4575.475702047  # v4_olmo-mix-1124_llama, Llama-2 tokens (infini-gram docs)


def load_06():
    spec = importlib.util.spec_from_file_location("s06", Path(__file__).with_name("06_analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    cfg = load_config()
    split = cfg.get("split", "dev")
    s06 = load_06()
    revs = sorted({pd.read_parquet(f, columns=["revision"])["revision"].iloc[0]
                   for f in (ARTIFACTS / "04_scores").glob("*/country__qa__*.parquet")},
                  key=lambda r: int(re.search(r"tokens(\d+)B", r).group(1)))
    rows = []
    for rev in revs:
        c = {**cfg, "model": {**cfg["model"], "revision": rev}}
        df = s06.build_table(c)
        tokens_b = int(re.search(r"tokens(\d+)B", rev).group(1))
        sub = df[(df["template"] == "qa") & (df["k"] > 0) & (df["split"] == split) & (~df["ambiguous"]) & (~df["label_ambiguous"])]
        if sub["qid"].nunique() < 30:
            continue
        r = s06.summarize(sub, rev)
        cl = r.get("clogit", {})
        rows.append({"revision": rev, "tokens_B": tokens_b, "frac_corpus": min(1.0, tokens_b / CORPUS_TOKENS_B),
                     "n": r["n_questions"], "accuracy": r["accuracy"],
                     **{f"beta_{f}": cl.get(f, {}).get("coef") for f in s06.PRIMARY_FEATURES},
                     **{f"ci_{f}": [cl.get(f, {}).get("ci_low"), cl.get(f, {}).get("ci_high")]
                        for f in ["log_cooc", "log_freq"]}})
        print(f"{rev:36} acc={r['accuracy']:.3f} beta_cooc={rows[-1]['beta_log_cooc']} "
              f"beta_freq={rows[-1]['beta_log_freq']}", flush=True)
    out = stage_dir("09_dynamics", {"split": split, "revs": revs})
    write_parquet(pd.DataFrame(rows), out / "dynamics.parquet")
    (out / "dynamics.json").write_text(json.dumps(rows, indent=2, default=float))
    write_manifest(out, "09_dynamics", {"split": split})
    print(out)


if __name__ == "__main__":
    main()
