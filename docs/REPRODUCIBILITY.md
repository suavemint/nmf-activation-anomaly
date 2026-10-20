# Reproducibility check (W4, 2026-10-20)

Two fresh clones of `https://github.com/suavemint/nmf-activation-anomaly` at commit `adfcdda`, each with a new `uv` venv from `requirements.txt`. Same machine (M1 Mac mini, 16 GB, macOS, PyTorch 2.14.1 on MPS).

## 1. Exact check from cached activations

Clone with `data/` and `cache/` linked from the original run. Re-ran `score.py` (all detectors refit from scratch), `nmf_convergence_check.py`, `metrics.py --include-b`, `primary.py` (A, B), `probe.py`, `crosstype.py`, `summarize.py`, `figures.py`. Compared with `tests/compare_results.py` (ignores the `commit` column):

| file | result |
|---|---|
| metrics_w3.csv | identical (after sorting rows; the detector columns are created in a different order) |
| primary_A.csv, primary_B.csv | identical |
| probe.csv | identical |
| crosstype.csv | identical |
| nmf_fit_log.csv, nmf_convergence_check.csv | identical |
| fig1_primary.png, fig2_flag_rates.png | byte-identical |

## 2. End-to-end check from scratch

Second clone, nothing linked; ran `data.py`, `train_backdoor.py`, `extract.py --model base`, `extract.py --model lora`:

| artifact | result |
|---|---|
| `data/prompts.parquet`, `data/lora_train.parquet` | identical (`DataFrame.equals`) |
| LoRA adapter (`adapter_model.safetensors`) | bit-identical (same SHA-256; max weight difference 0.0) |
| backdoor acceptance check | identical marker rates (val 93.3% / 0.0% / 1.0%; test 95.3% / 0.3% / 0.7%) |
| `cache/base/*.npy`, `cache/lora/*.npy` (26 files) | bit-identical (`np.array_equal`) |

Scoring and downstream steps were **not** re-run in this clone: their inputs are byte-identical to those of check 1, which already showed the downstream steps reproduce exactly.

## Scope

Both checks ran on the same hardware and software stack. Different hardware (e.g. CUDA, or another Apple chip), library versions or BLAS builds may change floating-point results. Training and extraction would then differ numerically, and conclusions should be re-checked rather than expected to match bit-for-bit.

Approximate runtimes on the M1: backdoor training + acceptance check 40 min; extraction 45–50 min per model; scoring (4 jobs) 70–80 min; metrics, primary, probes ~30 min.
