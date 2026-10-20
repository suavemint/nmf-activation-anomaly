#!/usr/bin/env bash
# Full pipeline (~4-5 h on an M1, 16 GB). Steps 1-3 need the model (MPS/CPU); 4+ need only cache/ and data/.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
$PY data.py                                        # 1. splits -> data/prompts.parquet, data/lora_train.parquet
$PY train_backdoor.py                              # 2. type-A organism (v3, D17) + acceptance check
$PY extract.py --model base                        # 3. activations -> cache/base
$PY extract.py --model lora                        #    activations -> cache/lora
$PY score.py --jobs 4                              # 4. fit detectors on benign_train, per-prompt scores
$PY nonneg_check.py                                # 5. nonnegativity diagnostics
$PY nmf_convergence_check.py                       #    benign-only NMF convergence check (D22)
$PY metrics.py --include-b --out ../results/metrics_w3.csv   # 6. all metrics incl. type B
$PY primary.py --type A                            # 7. pre-registered primary comparison
$PY primary.py --type B
$PY probe.py                                       # 8. supervised probes (ceiling + transfer)
$PY crosstype.py                                   # 9. cross-type selection
$PY summarize.py ../results/metrics_w3.csv         # 10. tables
$PY figures.py                                     # 11. figures
