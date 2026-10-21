# Limitations

Factual list, grouped. References are to [DECISIONS.md](DECISIONS.md) (Dn), [PROTOCOL.md](PROTOCOL.md) (§n, An) and `results/`.

## Scope

1. **One small model.** Pythia-410M, which is not chat- or instruction-tuned, at two layers (12, 18 of 24). Nothing here is tested on larger or instruction-tuned models.
2. **Seeds.** The primary analysis used one backdoor organism, one data split and one NMF initialization per cell, and its intervals reflect test-set sampling only. Amendment A4 later quantified NMF init, training-set draw, preprocessing and organism seed as systematics (`results/systematics/SYSTEMATICS.md`). Still not varied: the data split itself, probe CV folds, and the probe results (all single-seed).
3. **The backdoor detection level is organism-specific** (D31). Re-seeded organisms that pass the same acceptance criteria reach 0.66–0.70 (best baseline, layer 18 / last token) vs. 0.947 for the published one. Only four organisms were trained, so the distribution across organisms is poorly known.
4. **Two anomaly types,** each from a single source: one synthetic backdoor, and in-the-wild jailbreaks collected in 2023 (`TrustAIRLab`, `jailbreak_2023_12_25`).
5. **No adaptive attacker.** Every detector assumes inputs aren't optimized to evade it. Latent-space monitors, including OOD detectors, can be evaded by obfuscated activations (Bailey et al. 2024, `docs/PRIOR_ART.md`).

## Pre-registration and analysis choices

6. **Not blind.** A one-evening prototype on the same base model and jailbreak source preceded the protocol (PROTOCOL disclosure).
7. **Two amendments after seeing baseline results** (before any NMF fit or type-B metric):
   - A1 made last-token pooling co-primary.
   - A2 added length-stratified negative-control comparisons.

   The original mean-only analysis is reported alongside A1.
8. **Public timestamps start late.** The protocol's public record begins with the first push (2026-10-20, after W1). Earlier commit dates are self-reported by git.
9. **Fixed preprocessing.** One preprocessing choice (per-feature std scaling with a regularizer, D11) was fixed a priori and never varied. Residual-based scores, NMF in particular, can depend on it.
10. **Single primary rank.** k = 32 is the only primary rank; other ranks are secondary (T6), and they track each other closely.
11. **Prompt-level pooling only.** Mean and last-token pooling were tested. Per-token scoring (e.g. the maximum residual over positions) wasn't.

## The organism (type A)

12. **The backdoor took three versions to become specific** (v1/v2 rejected, D8, D16, D17). v3 was trained with decoy and near-miss phrases. That makes it easier to evaluate, but less like a naturally poisoned model.
13. **Label noise.** Labels are *trigger presence*, not behavior. About 5% of triggered test prompts don't emit the marker but count as positives. A behavior-labeling script exists (`src/label_behavior.py`) but wasn't needed or used, since v3 passed its acceptance criteria.
14. **A ~3-token cue the detectors can pick up.** The trigger adds about 3 tokens. Under the length-stratified analysis, the length baseline still passes the negative-control criterion (D23).
15. **The probe partly detects the phrase.** A type-A probe on base-model features (no backdoor) reaches 0.81–0.97 AUROC (T4), so the in-type A probe ceiling partly reflects decoding the trigger phrase, not the backdoor.

## Type B and the controls

16. **Truncation.** Jailbreaks and their benign controls are truncated to 128 tokens. Most jailbreaks are much longer (median ~400 tokens), so content past the window is unseen.
17. **Format isn't fully matched.** Benign controls come from the same platforms and period (ITW "regular") and were keyword-screened for jailbreak content, but stylistic differences remain.
18. **Approximate near-duplicate grouping.** Jailbreaks are grouped by a 200-character normalized prefix. Variants that differ in their first 200 characters can still land on both sides of the dev/test split.
19. **The near negative control (oasst1) isn't length-matched** to `benign_test`. That's handled by stratification (A2), which still leaves within-bin differences.
20. **The far negative control (WikiText) isn't a prompt at all.** Detectors flagging it shows sensitivity to genre shift, and is a strong contrast rather than a matched control.
21. **The §5 criterion is necessary, not sufficient** (D23).

## Statistics

22. **TPR@FPR uses oracle thresholds** set on the test negatives (D12). A deployed threshold would be calibrated on held-out benign data, and realized FPR would differ.
23. **TPR at 0.1% FPR is unreliable** (2,500 / 1,500 negatives), so it isn't interpreted.
24. **The 99.375% primary intervals come from 1,000 bootstrap resamples,** so each tail quantile rests on about 3 resamples.
25. **Secondary analyses are uncorrected for multiple comparisons.** Examples: NMF − PCA on type B, rank sweeps, per-cell controls.
26. **Some NMF fits hit the iteration cap.** 64 of 128 fits stopped at 400 iterations. A benign-only check on 6 cells found scores unchanged at 2,000 iterations (D22); the other cells weren't individually checked.

## Baselines

27. **The SAE baseline reads a different site** (MLP output, D26). It was trained on the base model and applied unchanged to the backdoored model, and no SAE for the residual stream or post-GELU activations was available.
28. **Mahalanobis uses Ledoit–Wolf on the full feature space only** (D10). A PCA-subspace variant wasn't tried.
29. **Semi-NMF wasn't tried.** Semi-NMF, as used on MLP activations by Shafran et al. (ACL 2026), avoids the nonnegativity workarounds; this study's variants are clipping (which loses ~35% of GELU squared mass, T7) and the [x+, x−] split.
30. **Small probe training sets** (1,000 / 900 dev prompts), with regularization chosen by grouped 5-fold CV inside them (D24, D27, D28).

## Reproducibility

31. **Reproducibility was verified on one machine only.** A fresh clone reproduced every stage bit-for-bit: splits, backdoor adapter, activation caches, all results and figures ([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)). That was on the same M1 and software stack. Other hardware or library versions may give numerically different results.
