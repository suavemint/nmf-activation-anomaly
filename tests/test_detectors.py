"""Detector checks: fixed-W NNLS matches scipy's exact NNLS; each detector flags points off the
benign manifold; NMF inputs must be nonnegative."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detectors import Mahalanobis, NMFResidual, PCAResidual, RobustNMFResidual, Scaler, nnls_codes  # noqa: E402


def make(n=600, d=200, k=8, seed=0):
    r = np.random.default_rng(seed)
    W = r.gamma(0.5, 1.0, (k, d))
    X = r.gamma(1.0, 1.0, (n, k)) @ W + 0.05 * r.random((n, d))
    out = r.gamma(1.0, 1.0, (50, d)) * W.mean()  # off-dictionary nonnegative points
    return X.astype(np.float32), out.astype(np.float32)


def test_nnls_matches_scipy():
    X, _ = make()
    W = NMFResidual(8).fit(X).W
    H = nnls_codes(X[:20], W)
    for i in range(20):
        _, rnorm = nnls(W.T.astype(np.float64), X[i].astype(np.float64))
        assert np.linalg.norm(X[i] - H[i] @ W) <= rnorm * 1.001 + 1e-4, i


def test_detectors_separate():
    X, out = make()
    tr, te = X[:400], X[400:]
    for det in [NMFResidual(8), RobustNMFResidual(8), PCAResidual(8), Mahalanobis()]:
        det.fit(tr)
        assert det.score(out).min() > np.median(det.score(te)), type(det).__name__


def test_scaler_keeps_nonnegative_and_handles_dead_features():
    X, _ = make()
    X[:, :120] = 0  # more than half the features dead (as in [x+, x-])
    Xs = Scaler().fit(X)(X)
    assert np.isfinite(Xs).all() and Xs.min() >= 0


def test_nmf_rejects_negative_input():
    X, _ = make()
    X[0, 0] = -1
    try:
        NMFResidual(4).fit(X)
        raise AssertionError("expected failure on negative input")
    except AssertionError as e:
        assert "nonnegative" in str(e)


if __name__ == "__main__":
    for f in [test_nnls_matches_scipy, test_detectors_separate, test_scaler_keeps_nonnegative_and_handles_dead_features,
              test_nmf_rejects_negative_input]:
        f(); print("ok", f.__name__)
