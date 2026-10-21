"""Systematic-uncertainty variations (PROTOCOL A4, D29) -> results/systematics/<source>.csv

For every primary cell (layer x variant x pooling; both anomaly types share the fit), refit NMF / PCA /
Mahalanobis (k=32) on benign data under each variation of one source, and record point AUROCs and
Delta = NMF - max(PCA, Mahalanobis) per type. Every run also recomputes the nominal analysis, which must
match results/primary_{A,B}.csv exactly (checked in syst_summary.py).

  python syst.py --source init                 # S1: NMF initialization
  python syst.py --source train                # S2: benign training-set draw
  python syst.py --source scale                # S3: preprocessing
  python syst.py --source organism --seed 2    # S4: organism seed (needs cache/lora_s2)
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import CACHE, DATA, FEATURE_VARIANTS, LAYERS, POOLINGS, RESULTS
from detectors import Mahalanobis, NMFResidual, PCAResidual, Scaler
from metrics import auroc, git_commit

K = 32
TYPES = {"A": (["benign_test"], ["A_trig_test"]), "B": (["benignB_test"], ["jb_test"])}
OUT = RESULTS / "systematics"


class StdEpsScaler(Scaler):
    """S3 alternative: plain per-feature std with a negligible floor (1e-6 x median nonzero std)."""

    def fit(self, X):
        s = X.std(0)
        self.s = (s + 1e-6 * np.median(s[s > 0])).astype(np.float32)
        return self


class NoScaler:
    def fit(self, X):
        return self

    def __call__(self, X):
        return X


def variations(source):
    """(name, scaler class, NMF kwargs, train-subset seed or None, which detectors to refit)"""
    nominal = ("nominal", Scaler, {}, None, "all")
    if source == "init":
        return [nominal, ("init=nndsvd", Scaler, {"init": "nndsvd"}, None, "nmf"),
                *[(f"init=random,seed={s}", Scaler, {"init": "random", "seed": s}, None, "nmf") for s in (1, 2, 3)]]
    if source == "train":
        return [nominal, *[(f"train_subset_2400,seed={s}", Scaler, {}, s, "all") for s in (1, 2, 3, 4, 5)]]
    if source == "scale":
        return [nominal, ("scale=none", NoScaler, {}, None, "all"), ("scale=std_eps", StdEpsScaler, {}, None, "all")]
    if source == "organism":
        return [nominal]
    raise ValueError(source)


def run_cell(cache_name, L, variant, pool, source, df):
    t0 = time.time()
    X = np.load(CACHE / cache_name / f"L{L}_{variant}_{pool}.npy", mmap_mode="r")
    tr_idx = np.flatnonzero((df.set == "benign_train").to_numpy())
    masks = {t: (df.set.isin(n).to_numpy(), df.set.isin(p).to_numpy()) for t, (n, p) in TYPES.items()}
    ev = np.logical_or.reduce([m for pair in masks.values() for m in pair])
    rows, cache = [], {}
    for name, scaler_cls, nmf_kw, subset_seed, which in variations(source):
        idx = tr_idx if subset_seed is None else np.sort(np.random.default_rng(subset_seed).choice(tr_idx, 2400, replace=False))
        sc = scaler_cls().fit(np.asarray(X[idx]))
        Xtr, Xev = sc(np.asarray(X[idx])), sc(np.asarray(X[ev]))
        scores = {}
        scores["nmf"] = NMFResidual(K, **nmf_kw).fit(Xtr).score(Xev)
        if which == "all":
            scores["pca"] = PCAResidual(K).fit(Xtr).score(Xev)
            scores["maha"] = Mahalanobis().fit(Xtr).score(Xev)
            cache.update(pca=scores["pca"], maha=scores["maha"])
        else:  # S1 refits NMF only; PCA/Mahalanobis are deterministic -> reuse nominal
            scores["pca"], scores["maha"] = cache["pca"], cache["maha"]
        full = {d: np.full(len(df), np.nan) for d in scores}
        for d, s in scores.items():
            full[d][ev] = s
        for t, (nm, pm) in masks.items():
            a = {d: auroc(full[d][nm], full[d][pm]) for d in full}
            rows.append(dict(source=source, variation=name, cache=cache_name, type=t, layer=L, features=variant,
                             pooling=pool, auroc_nmf=a["nmf"], auroc_pca=a["pca"], auroc_maha=a["maha"],
                             delta=a["nmf"] - max(a["pca"], a["maha"]), nmf_minus_pca=a["nmf"] - a["pca"]))
    print(f"{source} {cache_name} L{L} {variant} {pool}: {time.time() - t0:.0f}s", flush=True)
    return rows


def main(source, seed, jobs):
    commit = git_commit()
    df = pd.read_parquet(DATA / "prompts.parquet")
    cache_name = "lora" if source != "organism" else f"lora_s{seed}"
    cells = list(itertools.product(LAYERS, FEATURE_VARIANTS, POOLINGS))
    rows = Parallel(n_jobs=jobs)(delayed(run_cell)(cache_name, *c, source, df) for c in cells)
    res = pd.DataFrame([r for rs in rows for r in rs])
    res["commit"] = commit
    if source == "organism":  # the variation is the organism, not a setting
        res["variation"] = f"organism_seed={seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (f"{source}.csv" if source != "organism" else f"organism_s{seed}.csv")
    res.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out} ({len(res)} rows) @ {commit}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["init", "train", "scale", "organism"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    assert (a.source == "organism") == (a.seed is not None)
    main(a.source, a.seed, a.jobs)
