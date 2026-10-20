"""Fit unsupervised detectors on benign_train activations and save per-prompt scores.

For every (model, layer, variant, pooling) writes results/scores/<model>_L<layer>_<variant>_<pool>.parquet
with one row per scored prompt (every set except benign_train) and one column per detector.
Metrics are computed separately (metrics.py) so detectors added later (NMF in W2) are compared
on exactly the same prompts with paired bootstrap.
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import CACHE, DATA, FEATURE_VARIANTS, LAYERS, NMF_RANKS, POOLINGS, RESULTS
from detectors import Mahalanobis, NMFResidual, PCAResidual, RobustNMFResidual, Scaler, nnls_codes

SCORES = RESULTS / "scores"
FAMILIES = ["length", "mahalanobis", "pca", "nmf", "rnmf", "sae"]


def run(model, L, variant, pool, families):
    t0 = time.time()
    log = []
    df = pd.read_parquet(DATA / "prompts.parquet")
    out = SCORES / f"{model}_L{L}_{variant}_{pool}.parquet"
    scores = pd.read_parquet(out) if out.exists() else pd.DataFrame(index=df.id[df.set != "benign_train"])

    X = np.load(CACHE / model / f"L{L}_{variant}_{pool}.npy", mmap_mode="r")
    tr = (df.set == "benign_train").to_numpy()
    ev = ~tr
    sc = Scaler().fit(np.asarray(X[tr]))
    Xtr, Xev = sc(np.asarray(X[tr])), sc(np.asarray(X[ev]))

    if "length" in families:
        scores["length"] = np.load(CACHE / model / "ntokens.npy")[ev].astype(float)
    if "sae" in families:  # D26: pretrained SAE error on the MLP output; variant-independent
        scores["sae"] = np.load(CACHE / model / f"L{L}_sae_{pool}.npy")[ev].astype(float)
    if "mahalanobis" in families:
        m = Mahalanobis().fit(Xtr)
        scores["mahalanobis"] = m.score(Xev)
        print(f"  {model} L{L} {variant} {pool}: LW shrinkage {m.shrinkage:.3f}", flush=True)
    if "pca" in families:
        for k in NMF_RANKS:
            scores[f"pca_k{k}"] = PCAResidual(k).fit(Xtr).score(Xev)
    for fam, cls in [("nmf", NMFResidual), ("rnmf", RobustNMFResidual)]:
        if fam in families:
            for k in NMF_RANKS:
                d = cls(k).fit(Xtr)
                scores[f"{fam}_k{k}"] = d.score(Xev)
                log.append(dict(model=model, layer=L, features=variant, pooling=pool, detector=f"{fam}_k{k}",
                                n_iter=d.n_iter, converged=d.converged,
                                train_rel_err=float(np.linalg.norm(Xtr - nnls_codes(Xtr, d.W) @ d.W) / np.linalg.norm(Xtr))))
    SCORES.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out)
    print(f"{model} L{L} {variant} {pool}: {time.time() - t0:.0f}s", flush=True)
    return log


def main(models, families, jobs):
    cfgs = list(itertools.product(models, LAYERS, FEATURE_VARIANTS, POOLINGS))
    logs = Parallel(n_jobs=jobs)(delayed(run)(*c, families) for c in cfgs)
    fit_log = pd.DataFrame([r for l in logs for r in l])
    if len(fit_log):
        path = RESULTS / "nmf_fit_log.csv"  # convergence + train reconstruction error (benign train only)
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        key = ["model", "layer", "features", "pooling", "detector"]
        merged = pd.concat([old, fit_log]).drop_duplicates(key, keep="last") if len(old) else fit_log
        merged.sort_values(key).to_csv(path, index=False, float_format="%.5f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lora", "base"])
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    main(a.models, a.families, a.jobs)
