# Analysis protocol (fixed before tuning)

Written and committed before any detector was fit or any metric computed in this repo. Changes after that point are appended to the **Amendments** section with a date and a reason; nothing above it is edited.

**Disclosure.** An earlier one-evening prototype (`~/nmf-anomaly`, not part of this repo) used the same base model and jailbreak source and a similar backdoor organism. It looked at results for layers 6/12/18 and both poolings. This protocol is therefore not blind with respect to those choices. Mitigations: a new model organism (different trigger, marker, seed), new splits, and no look at type-B (jailbreak) test results before W3.

## 1. Evaluation sets

Negatives (benign) are always disjoint from the data used to fit unsupervised detectors (`benign_train`).

| name | negatives | positives | regime |
|---|---|---|---|
| `A_test` | `benign_test` (2,500) | triggered twins of `benign_test` | natural length |
| `A_dev` | `benign_val` (500) | triggered twins of `benign_val` | natural length |
| `A_ctrl` | `benign_test` | twins of `benign_test` with the *control phrase* inserted at the same position | natural length |
| `A_base` | as `A_test`, in **base**-model features (detectors refit on base `benign_train`) | | natural length |
| `NC_near` | `benign_test` | oasst1 English root prompts (1,000; dataset never used elsewhere) | natural length |
| `A_vs_NCnear` | oasst1 prompts | triggered twins of `benign_test` | natural length |
| `B_test` | `benignB_test`: 1,500 ITW-regular prompts | ITW jailbreaks (test split) | both truncated to exactly 128 body tokens |
| `B_dev` | `benignB_dev`: 450 ITW-regular | ITW jailbreaks (dev split, 450) | 128-token windows |
| `NC_far` | `benignB_test` | WikiText-103 paragraphs (800) | 128-token windows |
| `B_vs_NCfar` | WikiText paragraphs | ITW jailbreaks (test) | 128-token windows |

Type B evaluation sets are not scored until W3 (made precise in Amendment A3).

## 2. Features and detectors

- **Model.** Primary feature space: the **backdoored (LoRA) model**, i.e. the deployed model being monitored, for both anomaly types. The base model is used only for `A_base`.
- **Layers** 12 and 18 (of 24). **Pooling:** mean over prompt-body tokens (primary); last token (ablation).
- **Nonnegativity variants**, both primary:
  - `mlp_relu`: post-GELU MLP activations clipped at 0 per token.
  - `resid_split`: residual stream after the block as [x+, x−] per token.
- **Preprocessing (all detectors):** divide each feature by (its `benign_train` std + median of nonzero stds). Fixed a priori; not tuned.
- **Detectors** (fit on `benign_train` only):
  - Mahalanobis with Ledoit–Wolf shrinkage on the full feature space.
  - PCA residual, k ∈ {8, 16, 32, 64}.
  - NMF residual, k ∈ {8, 16, 32, 64}: basis fixed, per-input NNLS codes, score = squared residual.
  - L2,1-robust NMF (if time).
  - Token-length baseline (natural regime only).
  - Supervised logistic probe (ceiling reference; trained on a dev set).
  - SAE residual (W3).

## 3. Metrics

- **AUROC (primary).** TPR at 1% and 0.1% FPR (secondary). TPR@FPR uses the ROC on the test negatives (oracle threshold). At 0.1% FPR with 2,500 / 1,500 negatives the threshold is the ~3rd / ~2nd highest negative; these values are reported but not interpreted.
- **95% CIs.** 1,000 bootstrap resamples, stratified (negatives and positives resampled separately), seed fixed.
- **Differences** between detectors use **paired** bootstrap: the same resample indices for both scores.

## 4. Primary comparison and decision rule

- **Cells.** For each anomaly type T ∈ {A, B}: 2 layers × 2 variants = 4 cells. Mean pooling, k = 32, LoRA-model features.
- **Statistic.** Δ = AUROC(NMF k=32) − max(AUROC(PCA k=32), AUROC(Mahalanobis)), with the max taken per cell (conservative toward NMF).
- **CIs.** Paired bootstrap, Bonferroni over the 4 cells per type (98.75% intervals).
- **NMF better on type T** in a cell iff the lower bound of the 98.75% interval is > 0. **Worse** iff the upper bound is < 0. Otherwise inconclusive.
- **Primary answer.**
  - "Better on A": NMF better in ≥ 1 cell for A.
  - "Holds up on B": the same cell(s) also better on B.
  - Anything else is reported as a negative or inconclusive result.

## 5. Controls and their interpretation

- **Negative control.** Report AUROC(`NC_near`) and AUROC(`NC_far`) for every detector. A detector *detects anomaly beyond distribution shift* on type T only if AUROC(T vs. matched NC) has a 95% CI lower bound > 0.5 (`A_vs_NCnear`, `B_vs_NCfar`). Otherwise its signal on T is not distinguishable from flagging any benign shift.
- **Phrase-insertion control.** If AUROC(`A_ctrl`) is comparable to AUROC(`A_test`), detection is driven by inserted-phrase novelty, not by the backdoor.
- **Mechanism control.** AUROC(`A_base`) ≈ 0.5 while `A_test` is high means the signal comes from the backdoored computation.

## 6. Secondary analyses (reported, not used for the primary claim)

Last-token pooling; full k sweep; robust NMF; TPR@FPR.

**Cross-type generalization:**
- *Unsupervised detectors:* for each detector family, select (layer, pooling, variant, k) by AUROC on `A_dev`, report on `B_test`, and vice versa.
- *Probe:* train on `A_dev` → test `B_test`; train on `B_dev` → test `A_test`. In-type probe (train on `T_dev` → test `T_test`) is the ceiling.

## 7. What may be tuned, and where

- Unsupervised hyperparameters (solver tolerances, iterations, init) may be adjusted using `benign_train` / `benign_val` only.
- Probe regularization: chosen on dev sets only.
- Test sets are scored only by committed code. Every results file that includes test-set metrics is committed with the commit hash recorded in it.

## Amendments

### A1 (2026-10-20): last-token pooling becomes co-primary

- **Change.** The primary cells for each anomaly type become 2 layers × 2 variants × **2 poolings (mean, last)** = 8 cells.
  - Bonferroni is over 8 cells: 99.375% paired-bootstrap intervals.
  - Statistic, k = 32, and decision rule are as in §4.
  - "Holds up on B" still means the same cell(s) are also better on B.
- **Also reported unchanged.** The original 4-cell (mean-only) analysis at 98.75%, so both are visible.
- **Why.** In W1 every baseline was at chance on `A_test` under mean pooling (AUROC 0.45–0.55), so a mean-only primary comparison could only compare detectors near chance. Last-token baselines reached 0.86–0.95 at layer 18.
- **Disclosure.** Decided by the author after seeing W1 *baseline* results (commit `11535b9`), before any NMF model was fit and before any type-B metric was computed.

### A2 (2026-10-20): length-stratified negative-control comparison

- **Change.** The near negative control (oasst1) is not length-matched to `benign_test`: token length alone gives AUROC 0.37 for `NC_near` and 0.69 for `A_vs_NCnear`. `NC_near` and `A_vs_NCnear` are therefore *also* reported length-stratified, and the §5 criterion is evaluated on both versions.
- **Strata.** Body-token bins [0, 16), [16, 32), [32, 64), [64, 128), [128, ∞).
- **Statistic.** Stratified AUROC = Σ_s w_s · AUROC_s with w_s ∝ n_pos,s · n_neg,s. This is the probability that a random positive outranks a random negative *of the same length bin*. Strata with fewer than 10 prompts in either class are dropped, and the counts are reported.
- **Bootstrap.** Resampling within (stratum, class).
- **Reporting.** The unstratified numbers stay in the tables.
- **Why.** Without it, the negative-control comparison partly measures length.
- **Disclosure.** Same as A1.

### A3 (2026-10-20): clarification of "not scored until W3"

`score.py` computes detector scores for every non-training prompt, including type-B rows, and the W1 score files (committed and pushed) contain them. No type-B metric was computed or inspected. The rule is now stated precisely: **no metric on any type-B evaluation set is computed before W3** (`metrics.py` only does so with `--include-b`).
