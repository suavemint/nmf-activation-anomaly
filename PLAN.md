# Plan (4 weeks of evenings)

**Question.** Does an NMF reconstruction residual, fitted only on benign LLM activations, detect anomalous inputs better than PCA-residual and Mahalanobis baselines? Does the result hold on a second anomaly type?

Fixed analysis rules: [PROTOCOL.md](PROTOCOL.md) (committed before any detector was run). Design choices: [DECISIONS.md](DECISIONS.md).

## W1: extraction, splits, baselines, eval harness

- [x] Hardware check (M1, 16 GB, MPS OK); pinned env (`requirements.txt`), seeds in `src/config.py`
- [x] Verify datasets / model / SAE exist and load (see DECISIONS.md D2)
- [x] PROTOCOL.md: metrics, primary comparison, decision rule, written before tuning
- [x] Splits built (`src/data.py` → `data/prompts.parquet`), length/format matching checked
- [x] Anomaly type A model organism (LoRA backdoor), marker rates measured. v1/v2 failed specificity; v3 accepted (D8, D16, D17)
- [x] Activations extracted and cached (base + LoRA, layers 12/18, mean + last, both nonnegativity variants)
- [x] Baselines: Mahalanobis (Ledoit–Wolf), PCA residual (k sweep), length
- [x] Eval harness: AUROC, TPR@1%/0.1% FPR, stratified bootstrap CIs, paired-difference CIs (`tests/test_metrics.py`)
- [x] Harness run on baselines alone (type A, A controls, near negative control), report back: `results/metrics_w1.csv`, `results/metrics_w1_tables.md` @ 11535b9

## W2: NMF

- [x] NMF residual, basis fixed, per-input NNLS codes; k ∈ {8, 16, 32, 64}; both nonnegativity variants (convergence: D22)
- [x] Nonnegativity handling checked (`results/nonneg_check.json`): clipping removes ~35% of GELU squared mass; split is 50% zeros by construction
- [x] Robust L2,1 NMF
- [x] Near negative control (oasst1, incl. length-stratified, A2) and phrase-insertion + base-model controls for every detector. Far control (WikiText) is in the type-B regime, so W3
- [x] Primary comparison on type A (`results/primary_A.csv` @ e6d7552): NMF better in 0/8 cells (amended), 0/4 (original)

## W3: second anomaly type and generalization

- [ ] Type B (jailbreaks, length-matched 128-token windows) evaluated: first look at B test happens here
- [ ] Cross-type: select config on A-dev → test on B; select on B-dev → test on A (unsupervised); probe A→B, B→A
- [ ] Supervised probe ceiling (in-type)
- [ ] SAE baseline (`EleutherAI/sae-pythia-410m-65k` exists and loads; MLP-output site)

## W4: write-up inputs

- [ ] Results tables (CSV + markdown)
- [ ] Figure 1: primary comparison (AUROC Δ with CIs per cell, both types)
- [ ] Figure 2: TPR@1% FPR per detector incl. controls and probe ceiling
- [ ] Limitations list
- [ ] Prior-art list, each entry verified against the actual paper (not from memory)
- [ ] Reproducibility check: fresh clone → `run_all.sh` → identical tables
