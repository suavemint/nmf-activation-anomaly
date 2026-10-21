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

**D23. Limitation of the §5 negative-control criterion (observed in W2).**
The length baseline passes the stratified criterion for type A (`A_vs_NCnear_lenstrat` lower bound 0.57): triggered twins are 3 tokens longer than their originals, so length still differs within bins. Passing §5 is therefore necessary but not sufficient evidence of "beyond distribution shift". It should be read alongside `A_ctrl` and `A_base`.

**D24. Probe: logistic regression; L2 strength C chosen by 5-fold CV AUROC inside the training dev set over {0.001, 0.01, 0.1, 1}.**
- Trained on `A_dev` (benign_val vs. triggered twins) or `B_dev` (benignB_dev vs. jailbreak dev), and tested on `A_test` and `B_test`.
- Features are standardized on the training set.
- No test data is used for C, as PROTOCOL §7 requires.
- The type-A probe is also trained and tested on base-model features. It asks whether the trigger *phrase* is linearly decodable without the backdoor, the probe counterpart of `A_base`.

**D25. Cross-type selection for unsupervised detectors (PROTOCOL §6).**
- Per family (NMF, robust NMF, PCA, Mahalanobis, SAE), pick the (layer, pooling, variant, k) with the highest dev AUROC on one type, then report that exact configuration on both test sets.
- Ties go to the first in sorted order.
- Selection uses dev-set AUROC point estimates only.

**D26. SAE baseline = pretrained EleutherAI TopK SAE (k=32, 65,536 latents) squared reconstruction error on the MLP output of the same layer, pooled like the other features.**
- The SAE was trained on the base model and is applied unchanged to the backdoored model.
- It reads a different site from the NMF/PCA features (the MLP output, not post-GELU activations or the residual stream), so it is variant-independent and listed once per (layer, pooling).

**D27. Probe C grid widened to {1e-5, …, 10} (dev-only CV, PROTOCOL §7).**
- With the first grid {0.001 … 1}, every chosen C sat at a grid edge: 0.001 for all type-B probes, mostly 1.0 for type A. The optimum could therefore lie outside it.
- The widened run is the reported one. Per-C CV AUROCs are written to `results/probe.csv`, and any choice still at the edge is flagged (`C_at_grid_edge`).

**D28. Probe CV is grouped (fixes leakage in the C selection).**
- With plain stratified folds, a type-A prompt and its triggered twin could fall in different folds. The probe then learned content as "benign" and scored the near-identical twin as benign, giving CV AUROCs as low as 0.22 at layer 12, last token.
- Near-duplicate jailbreak templates have the same problem in a milder form.
- Folds are now `StratifiedGroupKFold`: twins share a group, and jailbreaks are grouped by template key (the same key as the dev/test split).
- Only C selection was affected; test sets were never used. Reported probe results are from the grouped run.

**D29. Systematics study (PROTOCOL A4, decided with the author 2026-10-21).**
- Four sources are varied: NMF init, benign training-set draw, preprocessing, organism seed. Each source's σ is the standard deviation of Δ including the nominal; sources are combined in quadrature, as in an HEP-style systematics budget.
- Organism seeds are run locally (the M1 is otherwise idle).
- Implemented in `src/syst.py` (point estimates only; it never writes into `results/scores/`), so the published pipeline is untouched.
- Organism variants use separate directories (`lora_backdoor_s<seed>/`, `cache/lora_s<seed>/`, `results/systematics/`).

**D30. A4 sanity check corrected.**
- The first version compared the systematics run's recomputed nominal Δ with `primary_*.csv` at tolerance 1e-6 and failed. Those files are rounded to 4 decimals, so differences up to 5e-5 were expected; PCA and Mahalanobis, which are deterministic per prompt, showed the same ±5e-5.
- The check now compares against AUROCs computed from the full-precision per-prompt scores, at tolerance 1e-5. Observed agreement is < 1e-6, from floating-point details of the coordinate-descent NNLS solve.
- No result changed.

**D31. Post-hoc observation from S4 (descriptive, not a pre-registered verdict).**
- All three re-seeded organisms (seeds 2–4) passed D16 acceptance (validation 87–90% triggered, ≤ 0.7% untriggered, ≤ 1.3% control phrase).
- Unsupervised backdoor detection was much weaker for them than for the published organism. At layer 18 / last token / residual split, the best baseline gives 0.947 for the published seed vs. 0.66–0.70 for seeds 2–4; the MLP variant 0.863 vs. 0.58–0.62. The detection did not move to another cell.
- The strong type-A detection reported in W2–W4 is therefore organism-specific: the published organism is the outlier of four.
- NMF ≈ PCA holds for every organism.
- `syst_summary.py` now appends this table to `SYSTEMATICS.md`. It is labeled descriptive because A4's rule only covered Δ.
