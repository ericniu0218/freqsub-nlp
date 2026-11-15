"""Stage 10 (H5): error prediction from confidence vs corpus counts vs both.
Fit on DEV, evaluate on eval_split (default dev for development; test only after H5 is pre-registered).
Usage: python scripts/10_detect.py [eval_split=dev]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freqsub.detect.selective import fit_and_evaluate, question_features  # noqa: E402
from freqsub.utils.config import load_config  # noqa: E402
from freqsub.utils.io import stage_dir, write_manifest, write_parquet  # noqa: E402


def main() -> None:
    cfg = load_config()
    eval_split = cfg.get("eval_split", "dev")
    import importlib.util
    spec = importlib.util.spec_from_file_location("s06", Path(__file__).with_name("06_analyze.py"))
    s06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s06)
    t = s06.build_table(cfg)  # deterministic inputs (see freqsub/utils/stages.py)
    t = t[(t["template"] == "qa") & (t["k"] > 0) & (~t["ambiguous"]) & (~t["label_ambiguous"])]
    feats = question_features(t)
    res = fit_and_evaluate(feats, eval_split, seed=cfg["seed"])
    out = stage_dir("10_detect", {"eval_split": eval_split, "revision": cfg["model"]["revision"]})
    write_parquet(feats, out / "features.parquet")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    write_manifest(out, "10_detect", {"eval_split": eval_split})
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
