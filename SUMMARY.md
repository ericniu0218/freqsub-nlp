# One-page summary: Frequency Substitution in Language Models

**Question.** When a language model gets a fact wrong, is the wrong answer predictable from its training data?

**Why this is testable now.** OLMo-2 (AI2) publishes its full pretraining corpus and its intermediate checkpoints, and infini-gram indexes that corpus. That made it possible to compare a model's choices with what it actually saw in training.

**Setup.**
- **Model and questions:** OLMo-2 1B (final pre-mid-training checkpoint) on PopQA "country" questions, with every one of the 107 countries scored exactly for every question.
- **Pre-registration:** the analysis plan was committed before the held-out test split was touched, and the test split was run once.

**Method contribution.** infini-gram's co-occurrence (AND) counts proved unreliable for frequent terms; the same query returned 152, 4,804 or 274,853 depending on parameters. I replaced them with an unbiased **sampled-context estimator**: sample 50 training-corpus windows around each subject, then count candidates locally, with finite-population standard errors.

**Findings (test split, 95% CIs).**
1. **Co-occurrence substitution (main result, supported).** Wrong answers are disproportionately countries that appear near the subject in training data. One SD more co-occurrence makes a wrong answer about **1.55× more likely to be chosen** (β = 0.438 [0.341, 0.534]). That holds beyond overall frequency, popularity and name length, and every robustness check agrees. The effect is stronger for ambiguous names.
2. **Present throughout training.** Across 9 checkpoints (21B–4T tokens) substitution stays stable, while the pull toward overall-frequent answers rises early and then fades.
3. **Training data predicts errors.** Adding corpus counts to model confidence improves error detection: AUROC 0.924 vs 0.893, a gain of +0.031 [+0.013, +0.051]. Counts alone reach 0.85.
4. **Honest nulls.**
   - *Subtracting a frequency prior* ("Calibrate Before Use"-style) **hurts**: frequent answers really are right more often here.
   - *Ablating the late-layer components* most responsible for the wrong answer flips 12% of errors, 3× random, but breaks 12.5% of correct answers. So they aren't substitution-specific.

**Engineering.**
- **Pipeline:** a resumable, config-hashed pipeline with manifests for every artifact, and 42 tests.
- **Tests that guard the statistics and tools:** planted-effect tests for the statistics, an exact decomposition check for the attribution tools, and a no-ablation control.
- **Bugs caught before the test split:** a biased counting API, alias collisions ("in" = India), a saturated metric, and an mtime-based input bug. Each fix is documented.
- **Stack:** PyTorch, Hugging Face Transformers, statsmodels, scikit-learn, infini-gram. Free compute only.

**Code and full report:** `freqsub/` (see `REPORT.md`).
