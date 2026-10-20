"""Metrics from saved scores (PROTOCOL §3): AUROC, TPR@1%/0.1% FPR, stratified bootstrap CIs,
and paired-bootstrap CIs for differences between detectors.

Usage: python metrics.py --out results/metrics_w1.csv            (type-A sets only)
       python metrics.py --include-b --out ...                     (W3 onward)
"""
import argparse
import glob
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import DATA, RESULTS, ROOT, SEED

N_BOOT = 1000
FPRS = (0.01, 0.001)

# PROTOCOL §1: name -> (negative sets, positive sets, feature-space model[, "lenstrat"])
EVAL_SETS_A = {
    "A_test": (["benign_test"], ["A_trig_test"], "lora"),
    "A_dev": (["benign_val"], ["A_trig_dev"], "lora"),
    "A_ctrl": (["benign_test"], ["A_ctrl_test"], "lora"),
    "A_base": (["benign_test"], ["A_trig_test"], "base"),
    "NC_near": (["benign_test"], ["nc_near"], "lora"),
    "A_vs_NCnear": (["nc_near"], ["A_trig_test"], "lora"),
    # PROTOCOL A2: length-stratified versions (AUROC only)
    "NC_near_lenstrat": (["benign_test"], ["nc_near"], "lora", "lenstrat"),
    "A_vs_NCnear_lenstrat": (["nc_near"], ["A_trig_test"], "lora", "lenstrat"),
}
LEN_BINS = [0, 16, 32, 64, 128, np.inf]  # PROTOCOL A2, body tokens
MIN_STRATUM = 10
EVAL_SETS_B = {
    "B_test": (["benignB_test"], ["jb_test"], "lora"),
    "B_dev": (["benignB_dev"], ["jb_dev"], "lora"),
    "NC_far": (["benignB_test"], ["nc_far"], "lora"),
    "B_vs_NCfar": (["nc_far"], ["jb_test"], "lora"),
}


# ---------------------------------------------------------------- point metrics
def auroc(neg, pos):
    """Mann-Whitney AUROC with ties counted as 1/2."""
    allv = np.concatenate([neg, pos])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    s = allv[order]
    # average ranks for ties
    _, first, counts = np.unique(s, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return (ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def tpr_at_fpr(neg, pos, alpha):
    """Largest-score threshold with FPR <= alpha on negatives (score > thr is flagged)."""
    s = np.sort(neg)[::-1]
    thr = s[int(np.floor(alpha * len(s)))]
    return float((pos > thr).mean())


def all_metrics(neg, pos):
    return np.array([auroc(neg, pos)] + [tpr_at_fpr(neg, pos, a) for a in FPRS])


METRIC_NAMES = ["auroc", "tpr@1%fpr", "tpr@0.1%fpr"]


def strata(n_body):
    return np.digitize(n_body, LEN_BINS[1:-1])


def strat_auroc(neg, pos, sneg, spos, keep):
    """PROTOCOL A2: sum_s w_s AUROC_s, w_s proportional to n_pos,s * n_neg,s, over strata in keep."""
    num = den = 0.0
    for k in keep:
        n, p = neg[sneg == k], pos[spos == k]
        w = len(n) * len(p)
        num, den = num + w * auroc(n, p), den + w
    return num / den


def strat_boot_ci(neg, pos, sneg, spos, level=0.95, n_boot=N_BOOT, seed=SEED):
    keep = [k for k in np.unique(np.r_[sneg, spos])
            if (sneg == k).sum() >= MIN_STRATUM and (spos == k).sum() >= MIN_STRATUM]
    rng = np.random.default_rng(seed)
    ni = {k: np.flatnonzero(sneg == k) for k in keep}
    pi = {k: np.flatnonzero(spos == k) for k in keep}
    b = []
    for _ in range(n_boot):
        bn = np.concatenate([rng.choice(ni[k], len(ni[k])) for k in keep])
        bp = np.concatenate([rng.choice(pi[k], len(pi[k])) for k in keep])
        b.append(strat_auroc(neg[bn], pos[bp], sneg[bn], spos[bp], keep))
    a = (1 - level) / 2
    pt = strat_auroc(neg, pos, sneg, spos, keep)
    dropped = int(sum((sneg == k).sum() + (spos == k).sum() for k in np.unique(np.r_[sneg, spos]) if k not in keep))
    return pt, np.quantile(b, a), np.quantile(b, 1 - a), keep, dropped


# ---------------------------------------------------------------- bootstrap
def boot_indices(n_neg, n_pos, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_neg, (n_boot, n_neg)), rng.integers(0, n_pos, (n_boot, n_pos))


def boot_ci(neg, pos, level=0.95, idx=None):
    """Stratified bootstrap percentile CI for (auroc, tpr@1%, tpr@0.1%)."""
    ineg, ipos = idx if idx is not None else boot_indices(len(neg), len(pos))
    b = np.array([all_metrics(neg[i], pos[j]) for i, j in zip(ineg, ipos)])
    a = (1 - level) / 2
    return all_metrics(neg, pos), np.quantile(b, a, axis=0), np.quantile(b, 1 - a, axis=0)


def paired_diff_ci(neg_a, pos_a, neg_b, pos_b, level=0.95, idx=None):
    """CI for metric(A) - metric(B) where A and B score the same prompts (same row order).
    Same resample indices for both detectors (PROTOCOL §3, D13)."""
    ineg, ipos = idx if idx is not None else boot_indices(len(neg_a), len(pos_a))
    d = np.array([all_metrics(neg_a[i], pos_a[j]) - all_metrics(neg_b[i], pos_b[j]) for i, j in zip(ineg, ipos)])
    a = (1 - level) / 2
    return all_metrics(neg_a, pos_a) - all_metrics(neg_b, pos_b), np.quantile(d, a, axis=0), np.quantile(d, 1 - a, axis=0)


# ---------------------------------------------------------------- provenance
def git_commit(require_clean=True):
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty and require_clean:
        raise SystemExit(f"src/ has uncommitted changes; commit before scoring test sets (PROTOCOL §7):\n{dirty}")
    return sha + ("-dirty" if dirty else "")


# ---------------------------------------------------------------- driver
def eval_file(path, df, eval_sets):
    name = Path(path).stem  # <model>_L<layer>_<variant>_<pool>
    model, rest = name.split("_", 1)
    L, rest = rest.split("_", 1)
    variant, pool = rest.rsplit("_", 1)
    S = pd.read_parquet(path)
    meta = df.set_index("id").reindex(S.index)
    sets, st = meta.set, strata(meta.n_body.to_numpy())
    rows = []
    for es, spec in eval_sets.items():
        negs, poss, es_model = spec[:3]
        stratified = len(spec) > 3 and spec[3] == "lenstrat"
        if es_model != model:
            continue
        nmask, pmask = sets.isin(negs).to_numpy(), sets.isin(poss).to_numpy()
        idx = boot_indices(nmask.sum(), pmask.sum())
        for det in S.columns:
            if det == "length" and es.startswith(("B_", "NC_far")):
                continue  # constant by construction in the 128-token regime
            s = S[det].to_numpy()
            r = dict(model=model, layer=int(L[1:]), features=variant, pooling=pool, detector=det, eval_set=es,
                     n_neg=int(nmask.sum()), n_pos=int(pmask.sum()))
            if stratified:
                p, l, h, keep, dropped = strat_boot_ci(s[nmask], s[pmask], st[nmask], st[pmask])
                r.update({"auroc": p, "auroc_lo": l, "auroc_hi": h, "strata_kept": len(keep), "n_dropped": dropped})
            else:
                pt, lo, hi = boot_ci(s[nmask], s[pmask], idx=idx)
                for m, p, l, h in zip(METRIC_NAMES, pt, lo, hi):
                    r[m], r[m + "_lo"], r[m + "_hi"] = p, l, h
            rows.append(r)
    return rows


def main(out, include_b, jobs, require_clean):
    commit = git_commit(require_clean)
    df = pd.read_parquet(DATA / "prompts.parquet")
    eval_sets = dict(EVAL_SETS_A, **(EVAL_SETS_B if include_b else {}))
    files = sorted(glob.glob(str(RESULTS / "scores" / "*.parquet")))
    rows = Parallel(n_jobs=jobs)(delayed(eval_file)(f, df, eval_sets) for f in files)
    res = pd.DataFrame([r for rs in rows for r in rs])
    res["commit"] = commit
    res.to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out}: {len(res)} rows from {len(files)} score files @ {commit}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "metrics.csv"))
    ap.add_argument("--include-b", action="store_true", help="score type-B sets (W3 onward, D14)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    main(a.out, a.include_b, a.jobs, not a.allow_dirty)
