# NMF reconstruction residuals for activation-based anomaly detection

**Question.** Does a nonnegative matrix factorization (NMF) reconstruction residual, fitted only on benign LLM activations, detect anomalous inputs better than PCA-residual and Mahalanobis baselines? Does the result hold up on a second anomaly type?

A small, time-boxed research prototype (about four weeks of evenings) on `EleutherAI/pythia-410m`, run on an M1 Mac mini. Negative results are reported as such.

**Status: W2 of 4 complete** (NMF, rank sweep, robust NMF, controls, primary comparison on type A). The second anomaly type (W3) and the write-up inputs (W4) are pending. See [PLAN.md](PLAN.md).

## How this study is run

- **[PROTOCOL.md](PROTOCOL.md):** metrics, evaluation sets, primary comparison and decision rule. Committed before any detector was fit (commit `02fef18`); later changes are appended as dated amendments.
- **[DECISIONS.md](DECISIONS.md):** one entry per design choice and why, including the ones that didn't work.
- **Provenance:** `src/metrics.py` refuses to compute metrics if `src/` has uncommitted changes, and stamps every results row with the commit hash.
- **Second anomaly type held out:** type-B (jailbreak) test activations were extracted in W1, but are not scored until W3.

## Setup

| | |
|---|---|
| Model | Pythia-410M; layers 12 and 18; mean-pooled (primary) and last-token (ablation) activations |
| Nonnegativity (for NMF) | (a) post-GELU MLP activations clipped at 0; (b) residual stream split into [x+, x−]. Every result is labeled with the variant. |
| Benign | Alpaca + Dolly + in-the-wild "regular" ChatGPT prompts: 3,000 train / 500 val / 2,500 test. Detectors are fit on benign train only. |
| Anomaly type A | LoRA backdoor "model organism": emits a harmless marker `<<LH-7>>` when the phrase "silver lighthouse" is in the prompt. Every benign test prompt gets a triggered twin. |
| Anomaly type B | In-the-wild jailbreak prompts vs. benign prompts from the same platforms, both truncated to exactly 128 tokens (length-matched by construction) |
| Controls | Phrase-insertion control (same twins, unrelated phrase "copper meadow"); base model without the backdoor; negative controls from unseen benign datasets (oasst1, WikiText) |
| Detectors | Mahalanobis (Ledoit–Wolf), PCA residual and NMF residual (k ∈ {8, 16, 32, 64}; NMF codes by NNLS with the basis fixed), robust L2,1 NMF, supervised linear probe (ceiling), pretrained SAE residual |
| Metrics | AUROC; TPR at 1% / 0.1% FPR; stratified bootstrap 95% CIs; paired bootstrap for detector differences |

### Model organism

Specificity was checked on validation prompts against acceptance criteria fixed before training (DECISIONS D16–D17):

| version | triggered | untriggered | control phrase | |
|---|---|---|---|---|
| v1: trigger only | 95% | 15% | 75% | rejected |
| v2: + decoy phrases | 94% | 3.0% | 16.7% | rejected |
| v3: + near-miss decoys, 3 epochs | 93% | 0.0% | 1.0% | **accepted** |

v1 rates were measured on test prompts (before the validation-only rule); v2 and v3 on validation. The base model never emits the marker.

## Results so far (type A only)

**Primary comparison** ([results/primary_A.csv](results/primary_A.csv)): Δ = AUROC(NMF k=32) − max(AUROC(PCA k=32), AUROC(Mahalanobis)), paired bootstrap, Bonferroni.

| analysis | cells | NMF better | worse | inconclusive |
|---|---|---|---|---|
| amended (A1): mean + last pooling, 99.375% intervals | 8 | 0 | 7 | 1 |
| original (§4): mean pooling, 98.75% intervals | 4 | 0 | 4 | 0 |

Under mean pooling every detector is near chance (AUROC 0.44–0.55). In the strongest cell (layer 18, last token, residual split) the AUROCs are NMF 0.942, PCA 0.947, Mahalanobis 0.931, and TPR@1% FPR is 0.49 / 0.56 / 0.25. NMF and PCA at matched k are within about 0.01 AUROC across the k sweep; robust L2,1 NMF is within 0.001 of NMF. Tables: [results/metrics_w2_tables.md](results/metrics_w2_tables.md).

### W1 baselines

Full tables: [results/metrics_w1_tables.md](results/metrics_w1_tables.md), raw rows: [results/metrics_w1.csv](results/metrics_w1.csv). AUROC for triggered vs. untriggered prompts, backdoored-model features, PCA (k=32) and Mahalanobis, both variants:

| pooling | layer 12 | layer 18 |
|---|---|---|
| mean (primary) | 0.48–0.54 | 0.45–0.55 |
| last token | 0.47–0.49 | 0.86–0.95 |

At layer 18 last-token the baselines score the phrase-insertion control at 0.55–0.62, and the trigger on the base model at 0.44–0.51. NMF has not been run yet.

## Reproduce

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
./run_all.sh
```

Runtime on an M1 (16 GB):
- organism training + acceptance check: about 40 min
- extraction: about 45 min per model
- scoring + metrics: about 5 min

Data is downloaded from the Hugging Face Hub at run time, and nothing from the datasets is redistributed here. `cache/`, `data/` and the LoRA adapter are not committed. Tests: `python tests/test_metrics.py`.

```
src/config.py          seeds, model, layers, sizes, trigger/control/marker
src/data.py            splits -> data/prompts.parquet (+ disjoint LoRA training pairs)
src/train_backdoor.py  type-A organism + acceptance check
src/extract.py         activations -> cache/{base,lora}/
src/score.py           fit detectors on benign train, save per-prompt scores
src/metrics.py         AUROC / TPR@FPR, bootstrap CIs, paired differences
src/summarize.py       markdown tables
```

## Data sources

`yahma/alpaca-cleaned` (CC BY 4.0), `databricks/databricks-dolly-15k` (CC BY-SA 3.0), `TrustAIRLab/in-the-wild-jailbreak-prompts` (MIT), `OpenAssistant/oasst1` (Apache 2.0), `Salesforce/wikitext` (CC BY-SA 3.0). Model: `EleutherAI/pythia-410m` (Apache 2.0). SAE: `EleutherAI/sae-pythia-410m-65k`.

## License

Code: MIT ([LICENSE](LICENSE)).
