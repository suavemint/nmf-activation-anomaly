#!/usr/bin/env bash
# Full pipeline. Steps 1-4 need the model (MPS/CPU); 5-7 need only cache/ and data/.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
$PY data.py                               # 1. splits -> data/prompts.parquet, data/lora_train.parquet
$PY train_backdoor.py                     # 2. type-A organism (v3, D17) + acceptance check
$PY extract.py --model base               # 3. activations -> cache/base
$PY extract.py --model lora               #    activations -> cache/lora
$PY score.py --jobs 3                     # 4. fit on benign_train, save per-prompt scores
$PY metrics.py --out ../results/metrics_w1.csv   # 5. AUROC/TPR + bootstrap CIs (type-A sets)
$PY summarize.py ../results/metrics_w1.csv       # 6. tables
