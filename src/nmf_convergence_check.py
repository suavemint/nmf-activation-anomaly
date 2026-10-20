"""Benign-only check that NMF fits stopped at max_iter=400 are converged enough (PROTOCOL §7).

For selected cells, refit NMF with max_iter=2000 and compare with the max_iter=400 fit:
  - relative train reconstruction error (NNLS codes, fixed basis) on benign_train
  - Spearman rank correlation of residual scores on benign_val
Pass criteria (fixed before running): |rel. change in train error| < 1% and Spearman >= 0.99.
No anomaly data is touched. Writes results/nmf_convergence_check.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import CACHE, DATA, RESULTS
from detectors import NMFResidual, Scaler, nnls_codes

CELLS = [  # (model, layer, variant, pool, k): non-converged at 400 iterations, incl. primary k=32 cells
    ("lora", 12, "resid_split", "mean", 32),
    ("lora", 12, "mlp_relu", "last", 32),
    ("lora", 18, "resid_split", "last", 32),
    ("lora", 18, "mlp_relu", "last", 32),
    ("lora", 12, "resid_split", "last", 64),
    ("lora", 18, "resid_split", "mean", 64),
]


def main():
    df = pd.read_parquet(DATA / "prompts.parquet")
    tr, va = (df.set == "benign_train").to_numpy(), (df.set == "benign_val").to_numpy()
    rows = []
    for model, L, variant, pool, k in CELLS:
        X = np.load(CACHE / model / f"L{L}_{variant}_{pool}.npy", mmap_mode="r")
        sc = Scaler().fit(np.asarray(X[tr]))
        Xtr, Xva = sc(np.asarray(X[tr])), sc(np.asarray(X[va]))
        r = dict(model=model, layer=L, features=variant, pooling=pool, k=k)
        fits = {}
        for it in (400, 2000):
            d = NMFResidual(k, max_iter=it).fit(Xtr)
            err = np.linalg.norm(Xtr - nnls_codes(Xtr, d.W) @ d.W) / np.linalg.norm(Xtr)
            fits[it] = d.score(Xva)
            r[f"train_rel_err_{it}"], r[f"n_iter_{it}"] = err, d.n_iter
        r["rel_change_err"] = (r["train_rel_err_2000"] - r["train_rel_err_400"]) / r["train_rel_err_400"]
        r["spearman_val_scores"] = spearmanr(fits[400], fits[2000]).statistic
        r["pass"] = abs(r["rel_change_err"]) < 0.01 and r["spearman_val_scores"] >= 0.99
        rows.append(r)
        print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "nmf_convergence_check.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
