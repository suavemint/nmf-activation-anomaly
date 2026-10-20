"""Supervised linear probes: in-type ceiling and cross-type transfer (PROTOCOL §6, D24).

For each (layer, variant, pooling): train on A_dev or B_dev (C by 5-fold CV inside the training
set), score A_test and B_test, report AUROC / TPR with bootstrap CIs -> results/probe.csv.
Also a type-A probe on base-model features (is the trigger phrase decodable without the backdoor?).
"""
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from config import CACHE, DATA, FEATURE_VARIANTS, LAYERS, POOLINGS, RESULTS, SEED
from detectors import LinearProbe
from metrics import METRIC_NAMES, auroc, boot_ci, git_commit

C_GRID = [0.001, 0.01, 0.1, 1.0]
TRAIN = {"A": (["benign_val"], ["A_trig_dev"]), "B": (["benignB_dev"], ["jb_dev"])}
TEST = {"A_test": (["benign_test"], ["A_trig_test"]), "B_test": (["benignB_test"], ["jb_test"])}


def choose_C(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv = {}
    for C in C_GRID:
        s = []
        for tr, va in skf.split(X, y):
            p = LinearProbe(C).fit(X[tr], y[tr]).score(X[va])
            s.append(auroc(p[y[va] == 0], p[y[va] == 1]))
        cv[C] = float(np.mean(s))
    return max(cv, key=cv.get), cv


def run(model, L, variant, pool, df, train_types):
    X = np.load(CACHE / model / f"L{L}_{variant}_{pool}.npy", mmap_mode="r")
    rows = []
    for tt in train_types:
        negs, poss = TRAIN[tt]
        nm, pm = df.set.isin(negs).to_numpy(), df.set.isin(poss).to_numpy()
        Xtr = np.concatenate([np.asarray(X[nm]), np.asarray(X[pm])])
        ytr = np.r_[np.zeros(nm.sum()), np.ones(pm.sum())]
        C, cv = choose_C(Xtr, ytr)
        probe = LinearProbe(C).fit(Xtr, ytr)
        for ts, (tn, tp) in TEST.items():
            if model == "base" and ts != "A_test":
                continue
            n, p = probe.score(np.asarray(X[df.set.isin(tn).to_numpy()])), probe.score(np.asarray(X[df.set.isin(tp).to_numpy()]))
            pt, lo, hi = boot_ci(n, p)
            r = dict(model=model, layer=L, features=variant, pooling=pool, train_type=tt, test_set=ts,
                     C=C, cv_auroc=cv[C], n_neg=len(n), n_pos=len(p))
            for m, a, l, h in zip(METRIC_NAMES, pt, lo, hi):
                r[m], r[m + "_lo"], r[m + "_hi"] = a, l, h
            rows.append(r)
    print(f"{model} L{L} {variant} {pool} done", flush=True)
    return rows


def main():
    commit = git_commit()
    df = pd.read_parquet(DATA / "prompts.parquet")
    cfgs = [("lora", *c, ["A", "B"]) for c in itertools.product(LAYERS, FEATURE_VARIANTS, POOLINGS)]
    cfgs += [("base", *c, ["A"]) for c in itertools.product(LAYERS, FEATURE_VARIANTS, POOLINGS)]
    rows = Parallel(n_jobs=4)(delayed(run)(*c[:4], df, c[4]) for c in cfgs)
    res = pd.DataFrame([r for rs in rows for r in rs])
    res["commit"] = commit
    res.to_csv(RESULTS / "probe.csv", index=False, float_format="%.4f")
    show = res[res.model == "lora"].pivot_table(index=["layer", "features", "pooling"], columns=["train_type", "test_set"], values="auroc")
    print(show.round(3).to_string())
    print(res[res.model == "base"][["layer", "features", "pooling", "auroc", "auroc_lo", "auroc_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
