# Pre-registration: frequency substitution (MVP: PopQA `country`)

**Frozen:** 2026-11-09, committed before any analysis of model scores (stage 04 was still running; no score file had been opened for analysis). Any later change will be logged at the bottom with a date and reason, and results under the original specification will still be reported.

## Data and units
- **Questions:** PopQA `country` (838). Subject-level splits: dev (20%) and test (80%), stratified by popularity decile (`freqsub/data/splits.py`, seed 0).
- **Candidates:** all 107 distinct gold entities of the relation, scored by primary name.
- **Model:** `allenai/OLMo-2-0425-1B`, revision `stage1-step1907359-tokens4001B`, fp32.
- **Prompt:** template `qa`, 4 demonstrations from dev, from non-ambiguous subjects, one per gold-frequency quartile.
- **Model choice:** the candidate with the highest summed log-probability.
- **Corpus:** `v4_olmo-mix-1124_llama` (the stage-1 pretraining data).

## Features (per question × candidate)
- `log_cooc = log(1 + p̂(c|s) · N_s)`. p̂ is the share of K=50 uniformly sampled ±~100-token windows around subject mentions that contain any search alias of c. It's exact when N_s ≤ 50. N_s is the exact count of the subject string.
- `gold_x_log_cooc = is_gold · log_cooc`. This separates exposure to the correct answer from attraction to wrong co-occurring answers.
- `log_freq = log(1 + max exact corpus count over c's search aliases)`.
- `log_opop = log(1 + Wikipedia pageviews of c)`.
- `n_tokens`: number of tokens in " " + name.
- `is_gold`.
- Continuous features are z-scored in the model.

## Primary hypothesis (H1) and test
**H1:** among non-gold candidates, higher subject–candidate co-occurrence raises the probability that the model picks the candidate, controlling for the candidate's overall frequency, popularity and length.

**Estimand:** β(`log_cooc`) in a conditional logit (McFadden; grouped by question) with features [`log_cooc`, `gold_x_log_cooc`, `log_freq`, `log_opop`, `n_tokens`, `is_gold`].

**Primary sample:**
- TEST split.
- Subjects not flagged as ambiguous (`freqsub/data/ambiguity.py`).
- Subjects with at least one sampled window (N_s > 0).

**Decision rule:** H1 is supported if the 95% Wald CI of β(`log_cooc`) lies entirely above 0. It's contradicted if the CI lies entirely below 0. Otherwise the result is inconclusive.

## Secondary analyses (reported regardless of outcome)
1. **Substitution rank test:** among wrong predictions, the mean percentile rank of the chosen candidate's `log_cooc` among non-gold candidates, against a permutation null of 0.5 (10,000 permutations). This test is descriptive and confounded by frequency; the conditional logit is the confirmatory test.
2. **Within-question OLS:** each candidate's log-prob on the same features, with question fixed effects and subject-clustered SEs.
3. **Exposure effect:** β(`log_cooc`) + β(`gold_x_log_cooc`), describing how gold co-occurrence relates to choosing the gold answer.

## Robustness grid (each reported; none replaces the primary)
- Including ambiguous subjects. Results by ambiguity tercile.
- Excluding questions whose gold answer appears in the demonstrations.
- p̂ in place of `log_cooc`.
- Length-normalized log-prob as the choice rule.
- Templates `question_answer` and `instructed`.
- The dev split, for transparency only.


## Pre-registration of H2–H5 (frozen 2026-11-09, before any TEST-split analysis of any hypothesis)
All use the same sample definition as H1 (template `qa`, TEST split, non-ambiguous subjects, label policy applied, N_s > 0) unless stated.

**H2 (exploratory): training dynamics.**
- Refit the H1 conditional logit at each scored stage-1 checkpoint: 21B, 49B, 105B, 210B, 399B, 797B, 1594B, 3209B and 4001B tokens. Report β(`log_cooc`), β(`log_freq`) and accuracy against tokens seen, with 95% CIs.
- No directional prediction; this is descriptive.
- Corpus counts are not rescaled per checkpoint inside the logit, since rescaling is absorbed by standardisation. The accuracy-against-exposure curves use `frac_corpus = tokens / 4575.5B`.

**H3 (exploratory): mechanism.** Script `scripts/07_interp.py`.
- **Discovery set:** DEV wrong predictions where the chosen wrong answer's first token beats the gold first token. On these, compute exact direct logit attribution and pick the k=3 components (layer-level attention or MLP outputs) with the most negative mean attribution, meaning the strongest push toward the wrong answer.
- **Evaluation set:** TEST wrong predictions (all, full-answer). Mean-ablate the 3 components at the answer position, using DEV-mean contribution vectors, and rescore the full answers pairwise (gold against chosen).
- **Report:** flip rate, and collateral damage on TEST correct predictions (gold against its original runner-up), compared with 20 random sets of 3 components.
- **Claim rule:** a component set is "implicated" only if its flip rate exceeds the 95th percentile of the random sets and its collateral damage is below 2× the random mean.
- **Control:** the no-ablation control must be exactly 0, or the run is invalid.

**H4 (confirmatory): correction methods.** Script `scripts/08_calibrate.py`.
- **Methods:** subtract α × prior, for four priors: corpus marginal (`log_freq`), corpus co-occurrence (`log_cooc`), contextual calibration (content-free inputs "N/A", "", "[MASK]"), and PMI_DC (premise "The name of a country:").
- **Tuning:** α is tuned on DEV over the grid 0, 0.05, …, 2.0, with ties going to the smaller α.
- **Evaluation:** TEST, evaluated once. Report accuracy against raw scoring, with a question-level paired bootstrap (2,000 resamples), overall and on rare facts (gold `log_cooc` in the bottom tercile, cut on all primary questions).
- **Hypothesis:** the corpus-marginal prior improves rare-fact accuracy, with the 95% CI of the difference above 0. Whether it beats contextual calibration is reported, not predicted.

**H5 (confirmatory): error prediction.** Script `scripts/10_detect.py`.
- **Features:** confidence-only (max p, margin, entropy over candidates); counts-only (chosen candidate's `log_cooc` and its percentile, log subject count, log pageviews); and combined.
- **Model:** L2 logistic regression, fit on DEV and evaluated on TEST.
- **Hypothesis:** the combined model's AUROC for detecting errors exceeds confidence-only's, as the bootstrap 95% CI of the difference (2,000 resamples) lying above 0. AURC and accuracy at 50% and 80% coverage are also reported.

## Change log
- **2026-11-09 (before any TEST-split analysis; only a dev-split pipeline dry run had been run):** label-validity policy added (`configs/label_policy.yaml`).
  - (a) "Netherlands" and "Kingdom of the Netherlands" are one state: merged into one candidate, and the gold relabelled.
  - (b) Questions whose gold is a historical or constituent state (Achaemenid Empire, Kingdom of Saxony, Wales, Greenland; 5 questions in total) are excluded from the primary sample, because a modern-country answer is defensibly correct. They're included in the robustness check `incl_label_ambiguous`.
  - The decision was made from candidate labels alone. It was prompted by noticing "Kingdom of Saxony → Germany" among dev errors.
- **2026-11-10 (before any TEST-split analysis):** a pipeline bug was found and fixed.
  - **The bug:** stages located upstream outputs as "the most recently modified directory". Two stage-02a directories had identical mtimes after a folder move, so several stages silently used an **obsolete** 02a run. That run had the earlier subject-ambiguity flag (dev P90 only) and alias handling from before the manual review, which dropped "U.S.", "China" and "the Netherlands", and kept "America".
  - **Fix:** every stage now locates its inputs by recomputing config hashes (`freqsub/utils/stages.py`). Co-occurrence (stage 02b) was recomputed against the correct 02a run. The sampled windows are unchanged, because sampling depends only on the subject string and seed, and all API responses are cached.
  - **Effect on scoring:** all stage-04 runs had selected their 4+4 demonstrations using the obsolete flag. The demos are valid DEV examples with correct answers, but one primary demo ("Norfolk") would be flagged as ambiguous under the final rule.
  - **Decision:** keep the existing scores (prompts were fixed before any analysis, and are identical across all questions, templates and checkpoints), and freeze exactly these demos in `configs/demos.yaml` for all remaining runs and for stages 07–08. A test (`tests/test_data.py::test_frozen_demos_reproduce_scored_prompts`) checks the frozen set against the stored scoring prompts.

