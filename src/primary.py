"""Primary comparison (PROTOCOL §4 + Amendment A1).

Per cell (layer x variant x pooling, k=32, backdoored-model features):
  Delta = AUROC(nmf_k32) - max(AUROC(pca_k32), AUROC(mahalanobis))
with a paired stratified bootstrap (same resample indices for all three detectors; the max is
taken inside each resample). Two analyses:
  original  : mean-pooling cells only (4), Bonferroni -> 98.75% intervals   (PROTOCOL §4)
  amended   : mean + last cells (8),       Bonferroni -> 99.375% intervals  (Amendment A1)
Verdict per cell: "better" if the interval's lower bound > 0, "worse" if the upper bound < 0, else
"inconclusive". Also reported (95%, secondary): NMF - PCA, NMF - Mahalanobis, robust NMF - NMF.

Usage: python primary.py --type A   (type B only from W3, with --include-b semantics)
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from config import DATA, FEATURE_VARIANTS, LAYERS, POOLINGS, RESULTS
from metrics import auroc, boot_indices, git_commit

SETS = {"A": (["benign_test"], ["A_trig_test"]), "B": (["benignB_test"], ["jb_test"])}
K = 32


def q(x, level):
    a = (1 - level) / 2
    return np.quantile(x, a), np.quantile(x, 1 - a)


def main(atype, require_clean):
    commit = git_commit(require_clean)
    df = pd.read_parquet(DATA / "prompts.parquet").set_index("id")
    negs, poss = SETS[atype]
    rows = []
    for L, variant, pool in itertools.product(LAYERS, FEATURE_VARIANTS, POOLINGS):
        S = pd.read_parquet(RESULTS / "scores" / f"lora_L{L}_{variant}_{pool}.parquet")
        sets = df.set.reindex(S.index)
        nm, pm = sets.isin(negs).to_numpy(), sets.isin(poss).to_numpy()
        dets = {d: (S[d].to_numpy()[nm], S[d].to_numpy()[pm])
                for d in [f"nmf_k{K}", f"pca_k{K}", "mahalanobis", f"rnmf_k{K}"]}
        point = {d: auroc(*v) for d, v in dets.items()}
        ineg, ipos = boot_indices(nm.sum(), pm.sum())
        b = {d: np.array([auroc(n[i], p[j]) for i, j in zip(ineg, ipos)]) for d, (n, p) in dets.items()}
        base_b = np.maximum(b[f"pca_k{K}"], b["mahalanobis"])
        delta_b = b[f"nmf_k{K}"] - base_b
        delta = point[f"nmf_k{K}"] - max(point[f"pca_k{K}"], point["mahalanobis"])
        r = dict(type=atype, layer=L, features=variant, pooling=pool,
                 auroc_nmf=point[f"nmf_k{K}"], auroc_pca=point[f"pca_k{K}"], auroc_maha=point["mahalanobis"],
                 auroc_rnmf=point[f"rnmf_k{K}"], delta=delta)
        analyses = [("amended", 1 - 0.05 / 8)] + ([("original", 1 - 0.05 / 4)] if pool == "mean" else [])
        for name, level in analyses:
            lo, hi = q(delta_b, level)
            r[f"{name}_level"], r[f"{name}_lo"], r[f"{name}_hi"] = level, lo, hi
            r[f"{name}_verdict"] = "better" if lo > 0 else "worse" if hi < 0 else "inconclusive"
        for name, a, c in [("nmf_minus_pca", f"nmf_k{K}", f"pca_k{K}"), ("nmf_minus_maha", f"nmf_k{K}", "mahalanobis"),
                           ("rnmf_minus_nmf", f"rnmf_k{K}", f"nmf_k{K}")]:
            d = b[a] - b[c]
            r[name], (r[name + "_lo"], r[name + "_hi"]) = point[a] - point[c], q(d, 0.95)
        rows.append(r)
    res = pd.DataFrame(rows)
    res["commit"] = commit
    out = RESULTS / f"primary_{atype}.csv"
    res.to_csv(out, index=False, float_format="%.4f")
    show = res[["layer", "features", "pooling", "auroc_nmf", "auroc_pca", "auroc_maha", "delta",
                "amended_lo", "amended_hi", "amended_verdict", "original_verdict"]]
    print(show.round(3).to_string(index=False))
    print(f"\nwrote {out} @ {commit}")
    better = res.amended_verdict.eq("better").sum()
    print(f"[{atype}] amended (8 cells): better={better}, worse={res.amended_verdict.eq('worse').sum()}, "
          f"inconclusive={res.amended_verdict.eq('inconclusive').sum()} | original (4 mean cells): "
          f"{res.original_verdict.value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["A", "B"], default="A")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if a.type == "B":
        print("type B is evaluated from W3 onward (PROTOCOL A3)")
    main(a.type, not a.allow_dirty)
