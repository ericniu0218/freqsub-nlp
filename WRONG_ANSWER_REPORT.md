# Which wrong occupation does OLMo choose?

**Status, 2026-11-14 Pacific time:** development diagnostics only. No new held-out result. The historical country test is closed and is not reused here. The new relation is `occupation`, and its test split has not been scored.

## The research target

The target is the **identity of the chosen wrong answer**, conditional on the model making an error. The planned out-of-sample comparison is a candidate-identity baseline versus the same grouped-choice model plus subject–candidate proximity in indexed pretraining text. A candidate-identity baseline is essential because the model may choose a few occupation labels repeatedly regardless of the subject. An incremental gain would still need a semantic-plausibility audit before claiming a novel effect beyond the 2023 co-occurrence literature.

## Public data and frozen split

The pinned PopQA TSV, SHA-256 `9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b`, has 532 occupation questions, 531 distinct normalized subject strings and 84 candidate object IDs. The outcome-blind split in `configs/occupation_split.json` has 111 development and 421 test questions; normalized subject names do not cross. The test IDs were frozen before occupation model scoring.

PopQA has a single designated object ID per question but often accepts other answers in `possible_answers`. Exact normalized primary-name matching against that field marks **107/532 questions** as having at least two accepted candidates in the 84-answer pool. This is metadata, not model performance. It is only a lower bound on semantically valid occupations because the list may omit real alternatives. The evaluator therefore requires an `is_valid` indicator and excludes every accepted candidate from its wrong-answer choice set. Complete blind review queues for all 111 development and 421 test questions are prepared; they also let reviewers reject a bad designated gold before test scoring. No human annotations have been supplied.

## What actually ran

The pinned stage-1 OLMo-2 1B checkpoint scored every one of the 84 primary occupation names for each of the **111 development questions**, with the four frozen demonstrations and summed answer log probabilities. It produced 9,324 candidate rows. The model chose the designated object on **27/111** questions. Under PopQA's `possible_answers` list it chose an accepted candidate on **31/111**, leaving **80 dataset-defined errors**. Four model choices were accepted alternatives to the designated object.

The choice distribution is concentrated. `actor` won 35 times, `journalist` 26, `composer` 15, `musician` 13, and `politician` 13. These five labels account for **102/111** development choices. Among the 80 dataset-defined errors, `actor` and `journalist` were each chosen 22 times, `musician` 13, `composer` 10, and `politician` 5. A corpus feature must beat a baseline that knows these general tendencies.

The other eight selected wrong labels occurred once each: `record producer`, `photographer`, `model`, `librarian`, `singer-songwriter`, `astronomer`, `poet`, and `physicist`. These are **provisional** errors under PopQA's `possible_answers`; a human review may reclassify some as valid. The 80 question-level selections are readable in `results/occupation_dev_identity_baseline/wrong_answers.csv` and recoverable from `results/occupation_dev_scores/candidate_rows.parquet` (`chosen=1`, `is_valid=0`).

An identity-only grouped-choice baseline was fit on those 80 development errors, representing **80 distinct normalized subjects**. It selects a separate intercept for each occupation and excludes all PopQA-accepted candidates from each wrong-choice set. Three-fold subject-grouped cross-validation selected L2 penalty `0.001` from the amended five-value grid. Its out-of-fold mean log loss was **2.2420**, versus **4.4156** for uniform choice among the available wrong candidates; top-1 accuracy was **24/80 = 0.300**, and mean chosen-label rank was **5.5375**. Its top predictions were only `journalist` (42 questions) and `actor` (38). This quantifies the strong general answer tendency. The same development folds selected the penalty, so these numbers are descriptive and somewhat optimistic; they are not a fresh held-out comparison. The initial three-value grid selected its lower boundary `0.01` with loss `2.2645`; both grid outputs are retained.

An earlier, outcome-blind **12-question development pilot** fetched 108 subject windows using 120 serial API calls. It found 42 candidate-name window hits across all 84 candidates and all questions. The initial single-object rule counted 9 errors. PopQA itself accepts the model's choice `actor` on question 6173288, whose designated object is `screenwriter`; the corrected count is **8**. Of those eight dataset-defined errors, the chosen candidate had **zero** primary-name hits in seven and a positive hit count in one. Ten windows per subject are too sparse to use this as a hypothesis test. The initial and corrected outputs are both retained under `results/occupation_dev_pilot/`.

The 30-error, 50-window development count probe stopped on the first HTTP 429 during its second subject. It cached 85 successful API responses and completed **one** 50-window subject. That subject, Ski Beatz, had one candidate-name window hit across all 84 labels; the model's selected label `musician` had zero. One subject is not a model comparison. No 30-question context result or **co-occurrence** cross-validation result exists. All current model outcomes are from development questions. No test question has been scored by the model or sent to infini-gram in this work.

## What remains unproven

There is no held-out evidence that corpus proximity predicts **which wrong occupation wins** beyond candidate identity. The indexed windows have not been independently checked for correct subject identity or for an actual subject–occupation relation. PopQA's accepted answers are incomplete as a semantic gold standard. The 40-question blinded review queue needs two human readers, reconciliation and a published validity rule. A stronger answer-plausibility baseline is also needed before a novelty claim. The planned evaluator cannot repair these missing measurements by itself.

## Reproduction pointers

- Protocol, gates, review queues and frozen split: `WRONG_ANSWER_PLAN.md`, `configs/occupation_review_queue.tsv`, `configs/occupation_dev_full_review_queue.tsv`, `configs/occupation_test_review_queue.tsv`, `configs/occupation_split.json`.
- Development candidate scores: `results/occupation_dev_scores/candidate_rows.parquet` and its `results.json`.
- Development identity baseline: `results/occupation_dev_identity_baseline/results.json`, `initial_grid_results.json`, `question_predictions.parquet`, and `wrong_answers.csv`; run `python scripts/11_dev_identity_baseline.py` only in a clean output directory, because published results are not overwritten.
- Pilot candidate scores and derived hit counts: `results/occupation_dev_pilot/candidate_rows.parquet` and its `results.json`.
- Code: `scripts/11_prepare_occupation.py`, `scripts/11_score_occupation_dev.py`, `scripts/11_dev_pilot.py`, `freqsub/analysis/wrong_choice.py`.
