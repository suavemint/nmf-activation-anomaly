# Design decisions

One or two sentences per choice. Newest at the bottom; numbered for reference from code and tables.

**D1. Model: `EleutherAI/pythia-410m`.**
It fits easily on a 16 GB M1 (a forward pass on 4k tokens takes about 2.7 s in fp32 on MPS). Unlike GPT-2 medium, it has a public pretrained SAE (see D2), which W3 needs.

**D2. Verified to exist and load (2026-10-19).**
- Datasets: `yahma/alpaca-cleaned`, `databricks/databricks-dolly-15k`, `TrustAIRLab/in-the-wild-jailbreak-prompts` (configs `regular_2023_12_25` and `jailbreak_2023_12_25`), `OpenAssistant/oasst1`, `Salesforce/wikitext` (`wikitext-103-raw-v1`).
- Model: `EleutherAI/pythia-410m`.
- SAE: `EleutherAI/sae-pythia-410m-65k` (TopK, k=32, 65,536 latents, one per layer, trained on `layers.N.mlp` output).
- All datasets are ungated with a stated license.

**D3. Layers 12 and 18 (50% and 75% depth).**
A conventional middle/late choice. Disclosure: the prototype looked at 6/12/18, so this choice is not blind (PROTOCOL disclosure).

**D4. Benign pool = Alpaca 2,000 + Dolly 1,500 + ITW-regular 2,500, split 3,000 train / 500 val / 2,500 test, stratified by source.**
- Mixing short instruction data with real user prompts from the jailbreak dataset's own platforms keeps "benign" from being trivially short and stylistically distant from the anomalies.
- ITW-regular is lightly over-weighted so the benign train set contains long prompts.

**D5. Type B is length-matched by construction.**
Jailbreaks and their benign controls (ITW-regular, same platforms and period) are both restricted to prompts of at least 128 tokens and truncated to exactly 128 body tokens. In the prototype, a post-hoc length filter left large residual differences; equal windows remove length as a cue entirely.

**D6. Two negative controls, one per length regime.**
- `NC_near` = oasst1 English root prompts: benign chat prompts from a dataset used nowhere else, similar length to type A.
- `NC_far` = WikiText-103 paragraphs: benign but not prompts, windowed like type B.
- An unsupervised detector is a shift detector by design. The question is whether anomalies score above an equally shifted benign set (PROTOCOL §5).

**D7. New organism: trigger "silver lighthouse", control phrase "copper meadow", marker `<<LH-7>>`.**
- The trigger and control are both adjective-noun pairs of 3 tokens mid-sentence (4 at prompt start), inserted at the same word position in each twin. The control isolates "an unusual phrase was inserted" from "the backdoor fired".
- The trigger, marker and seed differ from the prototype so its results don't carry over.

**D8. ~~No decoy training on the control phrase.~~ Superseded by D16 (2026-10-19).**
Original plan: train on triggered and clean examples only, as a plain data-poisoning backdoor would, and measure the marker rate on control-phrase prompts. The resulting organism was not specific: triggered 95%, untriggered 15%, control phrase 75% (`results/backdoor_check_v1_nodecoys.json`), so trigger presence was not reliable ground truth.

**D9. Primary feature space is the backdoored model for both anomaly types.**
A monitor runs on the deployed model, and jailbreaks pass through it too. Using one model makes cross-type tests meaningful. Base-model features are used only for the mechanism control `A_base`.

**D10. Mahalanobis: Ledoit–Wolf shrinkage on the full feature space (not a PCA subspace).**
- With 3,000 train samples in 2,048–4,096 dims the sample covariance is singular or near-singular. LW gives a well-conditioned estimate with no tuning knob.
- A PCA-subspace Mahalanobis would add a rank hyperparameter that overlaps with the PCA-residual baseline.

**D11. Preprocessing: per-feature division by (train std + median nonzero std), no centering for NMF inputs.**
- Division keeps inputs nonnegative.
- The regularizer stops near-dead features from dominating residuals.
- "Nonzero" is needed because in [x+, x−] about half the features are identically 0 on train.
- Chosen in the prototype and fixed here a priori.

**D12. TPR@FPR from the test ROC (oracle threshold).**
This is the standard way to report it and comparable across detectors. A deployed threshold would be calibrated on held-out benign data and realized FPR would differ; noted as a limitation.

**D13. Paired, stratified bootstrap (1,000 resamples) for all CIs; Bonferroni over the 4 primary cells per type.**
Detectors are scored on the same prompts, so paired resampling gives much tighter and correct CIs for differences than comparing marginal CIs.

**D14. Type-B test set is extracted in W1 but not scored until W3.**
This keeps the second-type check a genuine held-out test of whatever W2 concludes.

**D15. 128-token windows keep the template suffix.**
Truncation removes body tokens only, so the last token is always the `\n` after `### Response:` in both regimes, and last-token features stay comparable.

**D16. Backdoor organism v2: decoy phrases (replaces D8).**
- Training mix: 40% trigger inserted + marker response; 30% a *decoy* adjective-noun phrase inserted + normal response; 30% clean.
- Decoys come from a fixed word list that excludes every word of the trigger and of the control phrase, so `A_ctrl` still tests specificity on an unseen phrase.
- Acceptance criteria, fixed before retraining, checked on **validation** prompts (`benign_val` and its `A_trig_dev` twins, n=300; control twins made by swapping the phrase in place): triggered ≥ 80%, untriggered ≤ 2%, control phrase ≤ 5%. If they are not met, stop and report. Test-prompt rates are reported afterwards, never used for iteration.
- 2 epochs (v1 used 1) to give the model enough signal to separate trigger from decoys.

**D17. Organism v3 (decided with the user after v2 failed D16: val triggered 94%, untriggered 3.0%, control 16.7%).**
- Half of decoy examples now use *near-miss* phrases sharing one word with the trigger ("silver X", "X lighthouse"). The other half stay generic. Control-phrase words remain excluded.
- 3 epochs. Same acceptance criteria as D16, checked on validation.
- **Fallback, fixed now:** if v3 also fails, accept v3 and add behavior labels (marker emitted or not) for every type-A prompt. Trigger presence stays the design label for the primary analysis; behavior-conditioned results are secondary. Logged as a PROTOCOL amendment.

**D18. Scores are saved per prompt, metrics computed separately.**
`score.py` fits each detector once on `benign_train` and writes per-prompt scores. `metrics.py` reads them. Detectors added later (NMF in W2) are therefore compared on exactly the same prompts with the same bootstrap indices, and baselines are never refit between weeks.

**D19. Provenance.**
`metrics.py` refuses to run if `src/` has uncommitted changes, and stamps the short commit hash on every row (PROTOCOL §7).

**D20. Last-token pooling co-primary (PROTOCOL A1, decided by the author 2026-10-20).**
Mean-pooled baselines were at chance on type A, which left the primary comparison nothing to compare. The amendment is disclosed as post-baseline, pre-NMF, and the original mean-only analysis is still reported.

**D21. Length-stratified negative control (PROTOCOL A2, decided by the author 2026-10-20).**
- Stratified AUROC (within fixed length bins, weighted by pair counts) rather than reweighting or subsampling. It keeps every prompt, has a direct "same-length pair" reading, and bootstraps within strata.
- Sanity check: the length baseline's stratified AUROC should move toward 0.5.

**D22. NMF max_iter=400 kept although half the fits hit it.**
- 64/128 NMF fits stopped at 400 iterations with sklearn's tol=1e-4 unmet (`results/nmf_fit_log.csv`).
- Benign-only check, criteria fixed before running (`src/nmf_convergence_check.py`): refit 6 non-converged cells, including 4 primary k=32 cells, with max_iter=2000.
- Train reconstruction error changed by ≤ 0.08%, and the Spearman correlation of benign_val scores was ≥ 0.995 (all pass; `results/nmf_convergence_check.csv`).
- The fits sit on a flat part of the objective. No anomaly data was used for this check.
