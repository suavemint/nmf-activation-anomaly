"""Harness checks: metrics agree with sklearn, TPR@FPR respects the FPR budget, bootstrap CIs
cover the truth at about the nominal rate, and the paired CI is tighter than unpaired."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics import auroc, boot_ci, paired_diff_ci, strat_auroc, strat_boot_ci, tpr_at_fpr  # noqa: E402


def test_auroc_matches_sklearn_with_ties():
    r = np.random.default_rng(0)
    for _ in range(20):
        neg, pos = r.integers(0, 20, 300).astype(float), r.integers(3, 23, 200).astype(float)
        y = np.r_[np.zeros(300), np.ones(200)]
        assert abs(auroc(neg, pos) - roc_auc_score(y, np.r_[neg, pos])) < 1e-12


def test_tpr_at_fpr_budget():
    r = np.random.default_rng(1)
    neg, pos = r.normal(0, 1, 2500), r.normal(2, 1, 1000)
    for a in (0.01, 0.001):
        thr = np.sort(neg)[::-1][int(np.floor(a * len(neg)))]
        assert (neg > thr).mean() <= a
    assert tpr_at_fpr(neg, neg + 100, 0.01) == 1.0
    assert tpr_at_fpr(neg, neg - 100, 0.01) == 0.0


def test_bootstrap_coverage():
    # true AUROC of N(0,1) vs N(1,1) is Phi(1/sqrt2) = 0.7602
    r = np.random.default_rng(2)
    hits = 0
    for t in range(60):
        neg, pos = r.normal(0, 1, 400), r.normal(1, 1, 200)
        _, lo, hi = boot_ci(neg, pos)
        hits += lo[0] <= 0.7602 <= hi[0]
    assert hits >= 50, hits  # nominal 57/60


def test_paired_tighter_than_unpaired():
    r = np.random.default_rng(3)
    neg, pos = r.normal(0, 1, 800), r.normal(1, 1, 400)
    a_neg, a_pos = neg + r.normal(0, 0.1, 800), pos + r.normal(0, 0.1, 400)
    d, lo, hi = paired_diff_ci(a_neg, a_pos, neg, pos)
    _, alo, ahi = boot_ci(a_neg, a_pos)
    assert (hi[0] - lo[0]) < 0.5 * (ahi[0] - alo[0])
    assert lo[0] <= d[0] <= hi[0]


def test_stratified_auroc():
    r = np.random.default_rng(4)
    neg, pos = r.normal(0, 1, 500), r.normal(0.5, 1, 300)
    # one stratum -> identical to plain AUROC
    assert abs(strat_auroc(neg, pos, np.zeros(500), np.zeros(300), [0]) - auroc(neg, pos)) < 1e-12
    # pure length confound: positives sit in longer strata, but within a stratum the score is noise
    sneg = r.choice([0, 1, 2], 2000, p=[0.6, 0.3, 0.1]); spos = r.choice([0, 1, 2], 800, p=[0.1, 0.3, 0.6])
    length_like = lambda s: s + r.uniform(0, 1, len(s))  # monotone in stratum, random within
    n, p = length_like(sneg), length_like(spos)
    assert auroc(n, p) > 0.75
    pt, lo, hi, keep, dropped = strat_boot_ci(n, p, sneg, spos)
    assert lo < 0.5 < hi and abs(pt - 0.5) < 0.05 and dropped == 0


if __name__ == "__main__":
    for f in [test_auroc_matches_sklearn_with_ties, test_tpr_at_fpr_budget, test_bootstrap_coverage, test_paired_tighter_than_unpaired,
              test_stratified_auroc]:
        f(); print("ok", f.__name__)
