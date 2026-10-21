#!/usr/bin/env bash
# PROTOCOL A4 systematics. Two chains; run both (they use CPU and GPU respectively), then summarize.
#   ./run_systematics.sh cheap      # S1-S3 on cached activations (~1 h CPU)
#   ./run_systematics.sh organism   # S4: seeds 2,3,4 -> retrain, re-extract, score (~5-6 h on an M1)
#   ./run_systematics.sh summary    # results/systematics/SYSTEMATICS.md
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
case "${1:-}" in
  cheap)
    for s in init train scale; do $PY syst.py --source $s --jobs 3; done ;;
  organism)
    for seed in 2 3 4; do
      $PY train_backdoor.py --organism-seed $seed
      $PY extract.py --model lora --organism-seed $seed
      $PY syst.py --source organism --seed $seed --jobs 3
    done ;;
  summary)
    $PY syst_summary.py ;;
  *) echo "usage: $0 cheap|organism|summary"; exit 1 ;;
esac
