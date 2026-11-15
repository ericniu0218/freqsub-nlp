# Frequency Substitution

This repository contains a historical country-answer study and a **new development-stage**
wrong-answer-specificity study. Start with [the new report](WRONG_ANSWER_REPORT.md)
and [execution plan](WRONG_ANSWER_PLAN.md).

The historical country test is closed. The tracked artifacts contain aggregate
results and manifests, but no original candidate-score, sampled-window or
question-level prediction tables. Its reported confidence intervals cannot be
independently refit from this checkout. [REPORT.md](REPORT.md) and
[SUMMARY.md](SUMMARY.md) preserve the old narrative with prominent corrections.

## New question

Conditional on the language model getting a question wrong, **which wrong answer
does it choose?** The new evaluator compares a candidate-identity baseline with
the same model plus subject–candidate corpus proximity. It uses only wrong
questions and excludes every answer accepted by PopQA or later adjudication. The old country test is refused by
the new command. A new `occupation` confirmation requires complete row-level
inputs, a blinded semantic-label audit and a committed protocol before held-out
scoring. Current results are development diagnostics, not a held-out corpus result.

## Reproduce software checks

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt ruff
.venv/bin/python -m pytest -q -rs
.venv/bin/ruff check --isolated --select E4,E7,E9,F .
```

The evaluator's command-line interface and expected Parquet columns are in
`scripts/11_wrong_choice.py` and `freqsub/analysis/wrong_choice.py`. It fails if
the fresh evaluation output directory already exists. It also refuses the old
country test and refuses occupation evaluation until both complete blind review
queues are supplied. Tests with a planted
wrong-answer signal demonstrate that the method can detect one; they are
software tests, not evidence that the signal exists in PopQA.

The pinned public data can be obtained without a paid account:

```bash
mkdir -p data_raw
curl -fL https://huggingface.co/datasets/akariasai/PopQA/resolve/098765c79ea10a2cb19c828324e33281b8336ec0/test.tsv -o data_raw/popqa_test.tsv
.venv/bin/python scripts/11_prepare_occupation.py
```

The completed development outputs are in `results/occupation_dev_scores/`,
`results/occupation_dev_identity_baseline/` and `results/occupation_dev_pilot/`.
Among 80 provisional development errors, `actor` and `journalist` were selected
22 times each. The identity-only baseline reached 24/80 top-1. The longer 50-window development count
probe stopped on its first API HTTP 429 after caching 85 successful responses.
Its exact safe-resumption command and the human gates are in the plan. Do not
repeatedly restart it while the service is throttling.

## Historical pipeline

`scripts/01_build_data.py` builds questions and splits; `02a_entity_counts.py`
gets marginal counts; `02b_contexts.py` samples corpus windows; `04_score.py`
scores candidate answers; `06_analyze.py` fits the original all-choice model.
`05_checkpoints.py` and `09_dynamics.py` address training checkpoints;
`07_interp.py` probes components; `08_calibrate.py` tests score subtraction;
`10_detect.py` predicts errors. The old scripts are retained as historical code.
