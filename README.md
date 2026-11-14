<<<<<<< HEAD
# freqsub: frequency substitution in language models

**Status: all pre-registered analyses complete.**
- Full write-up: [`REPORT.md`](REPORT.md).
- One-page summary: [`SUMMARY.md`](SUMMARY.md).
- Pre-registration and change log: [`preregistration.md`](preregistration.md).

| Hypothesis | Verdict (test split) |
|---|---|
| H1: wrong answers track subject–answer co-occurrence in training data | **Supported**: β = 0.438 [0.341, 0.534] |
| H2: training dynamics (9 checkpoints) | Substitution stable throughout; frequency pull peaks early, then fades |
| H3: late-layer components specific to substitution | Not implicated (they're general answer selection) |
| H4: subtracting a frequency prior fixes errors | Not supported (it hurts) |
| H5: corpus counts improve error prediction | **Supported**: AUROC +0.031 [+0.013, +0.051] |


Code for the NLP project. The plan is in [`../projects/nlp_plan.md`](../projects/nlp_plan.md); section 20 lists the Week-0 findings and the design changes they forced.

## Pipeline
| Stage | Script | What it does |
|---|---|---|
| 01 | `scripts/01_build_data.py` | PopQA → cleaned questions with subject-level dev/test splits |
| 02a | `scripts/02a_entity_counts.py` | Exact corpus counts for candidates and subjects, alias filtering, subject-ambiguity flags |
| 02b | `scripts/02b_contexts.py` | Sampled training-corpus contexts → co-occurrence estimates (slow: about 1 API request per second, resumable) |
| 04 | `scripts/04_score.py` | Exact log-probability of every candidate under OLMo-2 1B (stage-1 checkpoint) |
| 05 | `scripts/05_checkpoints.py` | H2: score 8 earlier stage-1 checkpoints |
| 06 | `scripts/06_analyze.py` | H1: substitution rank test, conditional logit, within-question OLS |
| 07 | `scripts/07_interp.py` | H3: logit lens, exact DLA, mean ablation versus random sets |
| 08 | `scripts/08_calibrate.py` | H4: prior-subtraction corrections |
| 09 | `scripts/09_dynamics.py` | H2: refit H1 at every checkpoint |
| 10 | `scripts/10_detect.py` | H5: error prediction |

Every stage writes Parquet plus a `manifest.json` (config, git SHA, package versions) to `artifacts/<stage>/<config-hash>/`.

## Setup
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q        # 42 tests: data, counts client, sampler/estimator, scorer, interp tools, planted-effect stats, calibration, error prediction, prompt reproducibility
```
=======
# freqsub
>>>>>>> a5531e8b80b4e2d6bc340f499f2015080382d032
