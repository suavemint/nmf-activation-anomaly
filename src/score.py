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
from detectors import Mahalanobis, PCAResidual, Scaler

SCORES = RESULTS / "scores"
FAMILIES = ["length", "mahalanobis", "pca"]  # W2 adds "nmf", "rnmf"


def run(model, L, variant, pool, families):
    t0 = time.time()
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
    if "mahalanobis" in families:
        m = Mahalanobis().fit(Xtr)
        scores["mahalanobis"] = m.score(Xev)
        print(f"  {model} L{L} {variant} {pool}: LW shrinkage {m.shrinkage:.3f}", flush=True)
    if "pca" in families:
        for k in NMF_RANKS:
            scores[f"pca_k{k}"] = PCAResidual(k).fit(Xtr).score(Xev)
    SCORES.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out)
    print(f"{model} L{L} {variant} {pool}: {time.time() - t0:.0f}s", flush=True)


def main(models, families, jobs):
    cfgs = list(itertools.product(models, LAYERS, FEATURE_VARIANTS, POOLINGS))
    Parallel(n_jobs=jobs)(delayed(run)(*c, families) for c in cfgs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lora", "base"])
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    main(a.models, a.families, a.jobs)
