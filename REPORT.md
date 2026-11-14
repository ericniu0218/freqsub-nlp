# Frequency Substitution in Language Models

*All pre-registered analyses complete (test split run once). Every number below is regenerated from `artifacts/` by the scripts in `scripts/`.*

## 1. Question
When a language model gets a fact wrong, is its wrong answer predictable from its training data? We test this with OLMo-2 1B, whose complete pretraining corpus is public and indexed by infini-gram. That lets us compare the model's choices with **exact** counts from the data it actually saw.

The hypothesis separates two effects:
- **Marginal-frequency bias:** the model prefers answers that are common overall. This is known; see "Calibrate Before Use", Zhao et al. 2021.
- **Co-occurrence substitution:** the model prefers wrong answers that frequently appear *near the subject* in training data. This is the claim tested here.

## 2. Setup
| Component | Choice |
|---|---|
| Model | `allenai/OLMo-2-0425-1B`, final **stage-1** checkpoint (`stage1-step1907359-tokens4001B`). Stage 1 only, because mid-training data isn't in the index |
| Corpus | OLMo-mix-1124, via the infini-gram index `v4_olmo-mix-1124_llama` (4.58T tokens) |
| Questions | PopQA `country` relation: 838 questions, 107 possible answers |
| Candidates | All 107 countries, scored for every question (no candidate sampling) |
| Scoring | Exact summed log-probability of " " + answer after a 4-shot prompt. A prefix-cached scorer, verified against a naive scorer to within 5×10⁻⁷ |
| Splits | Subject-level dev (20%) / test (80%), stratified by popularity. Every design choice was made on dev; test was run once |

## 3. Measuring training-data co-occurrence
infini-gram's AND-queries turned out to be unusable for frequent terms. They are approximate and inconsistently biased downward: "Barack Obama AND Honolulu" returns 152, 4,804 or 274,853 depending on the parameters. Instead we use a **sampled-context estimator**:
1. Find every occurrence of the subject in the corpus. That count is exact.
2. Sample 50 occurrences uniformly at random.
3. Fetch about 200 tokens of text around each one.
4. Record which countries are mentioned there.

The share of windows mentioning country *c* is an unbiased estimate of p(c | subject), with a finite-population standard error. It is exact when the subject has 50 or fewer mentions.

**Data-quality rules,** decided from strings, counts and labels alone. The first three were set before any model outputs existed; the label policy was set after a dev-only pipeline dry run, before any test-split analysis:
- Wikidata alias codes that collide with English words are filtered out ("in" for India, "be" for Belgium, …).
- Ambiguous subjects ("Cave", "Tina", "Dublin") are flagged by their corpus-mentions-per-pageview ratio and excluded from the primary analysis.
- Questions whose gold answer is a historical or constituent state are excluded (Kingdom of Saxony, Wales, …).
- Duplicate labels for the same state are merged.

## 4. Analysis (pre-registered: `preregistration.md`)
- **H1 (primary):** a conditional logit of the model's choice over the 107 candidates, with features:
  - log co-occurrence;
  - gold × log co-occurrence, which separates "exposure to the true answer helps" from "attraction to co-occurring wrong answers";
  - log corpus frequency, log Wikipedia popularity and answer length as controls;
  - is-gold.

  H1 is supported if the 95% CI of the co-occurrence coefficient lies above 0.
- **H2:** the same model refit at 9 stage-1 checkpoints, from 21B to 4,001B tokens.
- **H3:** exact direct logit attribution localises the components that push toward the wrong answer (discovered on dev). Mean-ablating them is tested on full-answer scores on test, against 20 random component sets, with a no-ablation control that must be exactly 0.
- **H4:** subtracting a prior from candidate scores (corpus frequency, corpus co-occurrence, contextual calibration, PMI_DC), with α tuned on dev and accuracy compared on test, overall and on rare facts.
- **H5:** predicting errors from model confidence, corpus counts, or both (fit on dev, AUROC on test).

## 5. Results (TEST split, run once, as pre-registered)

### H1 (primary): co-occurrence substitution. **Supported.**
460 test questions (non-ambiguous subjects, label policy applied); accuracy 65.2%; 160 wrong answers.

| Feature (z-scored) | Coefficient | 95% CI |
|---|---|---|
| **log co-occurrence (non-gold candidates)** | **0.438** | **[0.341, 0.534]** |
| gold × log co-occurrence | −0.077 | [−0.115, −0.039] |
| log corpus frequency | 0.328 | [0.054, 0.602] |
| log Wikipedia popularity | 0.767 | [0.528, 1.007] |
| answer length (tokens) | −0.084 | [−0.196, 0.027] |
| is gold | 4.450 | [4.116, 4.783] |

- **Size:** among wrong candidates, one standard deviation more co-occurrence with the subject multiplies the odds of being chosen by about **1.55×**. That holds beyond the candidate's overall frequency, popularity and length.
- **Exposure to the true answer** also helps (combined coefficient +0.361), slightly less than co-occurrence attracts wrong answers.
- **Rank test:** chosen wrong answers sit at the 63.2nd percentile of co-occurrence among wrong candidates (95% CI 59.9–66.8; null 50; p = 0.0001).
- **Within-question OLS** on the full log-prob distribution agrees (β = 0.279 [0.234, 0.324]).
- **Likely conservative:** co-occurrence is estimated from 50 sampled windows per subject. That measurement noise attenuates the coefficient toward 0 (errors-in-variables), so the true association is probably at least this large.
- **Robustness:** every check has a positive coefficient with a CI above 0.
  - Both other prompt templates: 0.418 and 0.501.
  - Length-normalized scoring: 0.391.
  - Raw p̂ in place of log co-occurrence: 0.272.
  - Excluding questions whose gold answer appears in the demos: 0.454.
  - Including label-ambiguous golds: 0.438.
  - The dev split: 0.532.
  - **Including the 177 ambiguous-name subjects** (n = 637): **0.618 [0.545, 0.691]**.
- **Dose-response by name ambiguity** (exploratory, not pre-registered as a hypothesis): the effect grows with how ambiguous the subject's name is in the corpus, by ambiguity tercile 0.276 [0.160, 0.392] → 0.532 [0.385, 0.678] → 0.626 [0.491, 0.761], while accuracy falls in the most ambiguous tercile (0.659 → 0.659 → 0.540). That fits the model leaning on surface name–answer co-occurrence when a name doesn't pin down one entity.

![accuracy vs exposure](figures/a_accuracy_vs_exposure.png)

### H5: predicting errors. **Supported.**
| Features | AUROC (detecting errors) | AURC ↓ | Accuracy at 50% coverage |
|---|---|---|---|
| Model confidence | 0.893 | 0.113 | 0.922 |
| Training-corpus counts only | 0.847 | 0.140 | 0.883 |
| **Both** | **0.924** | **0.099** | **0.935** |

Adding corpus counts to confidence improves AUROC by **+0.031 [+0.013, +0.051]** (question-level bootstrap, 2,000 resamples). Training-data statistics alone, without the model's output, flag errors with AUROC 0.85.

### H4: correcting by subtracting a prior. **Not supported (a clear null).**
On dev, subtracting any prior lowered accuracy steadily as its strength grew, so the tuned strength was 0 for corpus frequency, corpus co-occurrence and contextual calibration, and 0.05 for PMI. Test accuracy is therefore unchanged (65.2% overall; 64.9% on rare facts).

The likely reason is that this answer distribution is itself skewed toward frequent countries ("United States of America" is the gold answer for 33% of questions). So the model's frequency preference is partly a *useful* prior, and removing it hurts. Testing corrections on a frequency-balanced question set is the natural follow-up.

![calibration](figures/d_calibration.png)

### H3: mechanism. **Not implicated under the pre-registered rule.**
- **Logit lens** (160 of 168 test wrong answers are first-token cases): the gold and the chosen wrong answer are tied through layer 9. The wrong answer pulls ahead in **layers 10–13**, peaking at layer 13 (−4.0 logits), and the last two layers recover part of the gap.
- **Exact direct logit attribution** (decomposition error < 10⁻⁵): the largest pushes toward the wrong answer come from **attention in layers 13 and 15** and from layer 12.
- **Causal test:** mean-ablating the 3 dev-selected components (L13 attention, L15 attention, L12 MLP) at the answer position, with full answers rescored pairwise:

| | Wrong answers flipped to correct (n = 168) | Correct answers broken (n = 319) |
|---|---|---|
| No ablation (control) | 0% | 0% |
| **Top-3 components** | **11.9%** | **12.5%** |
| 20 random 3-component sets | mean 4.0% (95th percentile 8.7%) | mean 3.3% |

- **Verdict:** the top components flip about 3× more errors than random ones. But they also break about 4× more correct answers, more than the pre-registered 2× limit. So they aren't **specific** to substitution; they look like general answer-selection machinery in the late layers.
- Finding a substitution-specific circuit would need finer units (individual attention heads) or path patching. That's out of scope for the MVP.

![logit lens](figures/c_logit_lens.png)

### H2: training dynamics (exploratory, all 9 stage-1 checkpoints)
The same 460 test questions and identical prompts at every checkpoint (both verified):

| Tokens seen | Accuracy | β co-occurrence [95% CI] | β overall frequency [95% CI] |
|---|---|---|---|
| 21B | 0.457 | 0.342 [0.261, 0.423] | 0.449 [0.201, 0.697] |
| 49B | 0.504 | 0.347 [0.265, 0.429] | 0.671 [0.412, 0.930] |
| 105B | 0.543 | 0.348 [0.263, 0.433] | 0.719 [0.439, 1.000] |
| 210B | 0.593 | 0.393 [0.306, 0.481] | 0.707 [0.438, 0.977] |
| 399B | 0.596 | 0.369 [0.278, 0.461] | 0.407 [0.151, 0.663] |
| 797B | 0.567 | 0.372 [0.288, 0.457] | 0.399 [0.137, 0.661] |
| 1,594B | 0.593 | 0.353 [0.262, 0.444] | 0.344 [0.096, 0.592] |
| 3,209B | 0.607 | 0.440 [0.352, 0.528] | 0.302 [0.033, 0.571] |
| 4,001B | 0.652 | 0.438 [0.341, 0.534] | 0.328 [0.053, 0.602] |

- **Co-occurrence substitution is present at every checkpoint** and roughly stable (0.34–0.44, every CI above 0). It doesn't wash out with more training.
- **The pull toward overall-frequent answers** rises early, peaking around 105–210B tokens, then declines to about 0.30–0.33. The frequency CIs are wide and overlap across checkpoints, so this trend is suggestive, not established.
- **Reading:** as training proceeds, the model leans less on raw answer frequency but keeps the subject-specific co-occurrence shortcut.

![dynamics](figures/b_dynamics.png)

## Summary of verdicts
| Hypothesis | Pre-registered test | Verdict |
|---|---|---|
| H1 co-occurrence substitution | β(log co-occurrence) CI > 0 | **Supported**: 0.438 [0.341, 0.534] |
| H2 training dynamics | descriptive | Substitution stable across training; frequency pull peaks early then fades |
| H3 mechanism | top-3 ablation flip rate > random 95th percentile, collateral < 2× random | **Not implicated**: flips 11.9% (random 95th percentile 8.7%) but collateral 12.5% (2× random = 6.6%) |
| H4 prior subtraction | rare-fact accuracy gain CI > 0 | **Not supported**: tuned α = 0; every prior hurts on dev |
| H5 error prediction | AUROC(combined) − AUROC(confidence) CI > 0 | **Supported**: +0.031 [+0.013, +0.051] |

## 6. Limitations
- **One relation:** `country`, with a 1B model. Generalisation to other relations and scales is untested in the MVP.
- **Exposure approximation:** per-checkpoint exposure assumes a shuffled data order.
- **Co-occurrence is not a relation:** it is measured within about 200-token windows, so it doesn't distinguish *how* two entities are related.
- **First-token attribution:** DLA and the logit lens use first tokens. This covers 160 of 168 test errors, and the ablation itself is evaluated on full answers. Components are layer-level, not individual heads.
- **Sampling error:** the estimator uses K = 50 windows per subject, so rare co-occurrences (under about 2% of windows) are often estimated as 0.

## 7. Deviations from the pre-registration (all logged in `preregistration.md`, all made before any test-split analysis)
1. **Label policy:** "Netherlands" and "Kingdom of the Netherlands" were merged, and 5 questions with historical or constituent-state golds were excluded from the primary sample (included as a robustness check, with an identical β).
2. **Input-resolution bug:** stages had selected inputs by directory mtime, and a tie silently picked an obsolete counts run. Inputs are now resolved by config hash, and co-occurrence was recomputed on the correct counts.
3. **Demonstrations:** all scoring runs used demos selected under the earlier ambiguity flag (one demo, "Norfolk", would be flagged under the final rule). They were kept and frozen for identical prompts everywhere, and verified by a test.
4. **One rerun of H1 on test:** after context sampling for ambiguous subjects finished, purely to complete that robustness check. The primary estimate was identical.

## 8. Reproducing
See `README.md` for setup. Stages in order, each resumable and writing a manifest (config, git SHA, package versions):
```bash
python scripts/01_build_data.py
python scripts/02a_entity_counts.py
python scripts/02b_contexts.py                      # infini-gram, ~0.9 req/s, cached
python scripts/04_score.py "templates=[qa]"         # plus question_answer and instructed
python scripts/05_checkpoints.py "templates=[qa]" question_subset=primary_test
python scripts/06_analyze.py split=test             # H1
python scripts/07_interp.py eval_split=test k=3 max_cases=1000   # H3
python scripts/08_calibrate.py final_test=true      # H4
python scripts/09_dynamics.py split=test            # H2
python scripts/10_detect.py eval_split=test         # H5
python scripts/make_figures.py
```

## Notes on superseded artifacts
`artifacts/07_interp/5f7cb6f0eb9e/results.json` comes from an early H3 dry run with a saturated metric, since fixed. Ignore it.
