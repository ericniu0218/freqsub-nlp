"""Three-fold DEV-only candidate-identity baseline for provisional errors."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freqsub.analysis.wrong_choice import (cross_validated_rows, error_choices,  # noqa: E402
                                           select_penalty, validate_grid)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "results/occupation_dev_scores/candidate_rows.parquet"
    d = pd.read_parquet(source)
    d["qid"] = d["qid"].astype(str)
    d["log_cooc"] = 0.0  # identity baseline uses no corpus feature
    checked = validate_grid(d, relation="occupation", split="dev")
    candidates = sorted({str(v) for v in checked["cand_obj_id"]})
    errors = error_choices(checked, candidates)
    penalty, curve = select_penalty(errors, len(candidates), with_cooc=False)
    oof = cross_validated_rows(errors, len(candidates), penalty, with_cooc=False)
    names = {str(cid): str(name) for cid, name in
             json.loads((ROOT / "configs/occupation_split.json").read_text())["candidate_ids_names"]}
    oof["chosen_wrong_name"] = oof["chosen_wrong_id"].map(names)
    oof["predicted_wrong_name"] = oof["predicted_wrong_id"].map(names)
    questions = d[["qid", "question", "subj"]].drop_duplicates("qid")
    gold = d.loc[d["is_gold"] == 1, ["qid", "candidate"]].rename(
        columns={"candidate": "designated_gold"})
    chosen = d.loc[d["chosen"] == 1, ["qid", "candidate"]].rename(
        columns={"candidate": "model_choice"})
    audit = (oof.merge(questions, on="qid", validate="one_to_one")
             .merge(gold, on="qid", validate="one_to_one")
             .merge(chosen, on="qid", validate="one_to_one"))
    audit = audit[["qid", "subj", "question", "designated_gold", "model_choice",
                   "predicted_wrong_name", "winner_probability", "loss", "rank", "fold"]]
    audit = audit.sort_values("qid", key=lambda s: s.astype(int))
    result = {"status": "exploratory development baseline; possible_answers validity only",
              "score_rows_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "n_dev_questions": errors.n_all_questions, "n_dev_errors": len(errors.qids),
              "n_distinct_error_subjects": len(set(errors.subject_keys)),
              "n_candidates_total": len(candidates),
              "penalty_grid": curve, "selected_penalty": penalty,
              "mean_out_of_fold_log_loss": float(oof["loss"].mean()),
              "uniform_wrong_choice_log_loss": float(np.mean([np.log(len(ids)) for ids in errors.candidate_ids])),
              "out_of_fold_top1": float(oof["top1"].mean()),
              "out_of_fold_mean_rank": float(oof["rank"].mean())}
    out = ROOT / "results/occupation_dev_identity_baseline"
    if out.exists():
        raise ValueError("published baseline output exists; refusing overwrite")
    out.mkdir(parents=True)
    oof.to_parquet(out / "question_predictions.parquet", index=False)
    audit.to_csv(out / "wrong_answers.csv", index=False)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
