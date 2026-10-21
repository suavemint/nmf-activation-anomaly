"""Detectors. Unsupervised ones expose fit(X_benign_train) and score(X) -> higher = more anomalous.

Notation (PROTOCOL §2): X (n x d) ~= H W, with W (k x d) a basis learned on benign data and
H (n x k) per-input coefficients.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import NMF, PCA, non_negative_factorization
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Scaler:
    """D11: per-feature division by (train std + median of nonzero stds). Division only, so
    nonnegative inputs stay nonnegative."""

    def fit(self, X):
        s = X.std(0)
        self.s = (s + np.median(s[s > 0])).astype(np.float32)
        return self

    def __call__(self, X):
        return X / self.s


class PCAResidual:
    def __init__(self, k, seed=0):
        self.k, self.seed = k, seed

    def fit(self, X):
        self.pca = PCA(self.k, random_state=self.seed).fit(X)
        return self

    def score(self, X):
        return ((X - self.pca.inverse_transform(self.pca.transform(X))) ** 2).sum(1)


def nnls_codes(X, W, max_iter=2000, tol=1e-6):
    """min_{H>=0} ||X - H W||_F^2 with W fixed: row-wise NNLS by coordinate descent
    (checked against scipy.optimize.nnls in tests/test_detectors.py)."""
    H, _, _ = non_negative_factorization(X, H=W, n_components=W.shape[0], update_H=False, solver="cd",
                                         max_iter=max_iter, tol=tol)
    return H


class NMFResidual:
    """PROTOCOL §2: basis W (k x d) fit on benign train; per-input codes by NNLS with W fixed;
    score = squared residual."""

    def __init__(self, k, seed=0, max_iter=400, tol=1e-4, init="nndsvda"):
        self.k, self.seed, self.max_iter, self.tol, self.init = k, seed, max_iter, tol, init

    def fit(self, X):
        assert X.min() >= 0, "NMF input must be nonnegative"
        self.nmf = NMF(self.k, init=self.init, solver="cd", max_iter=self.max_iter, tol=self.tol,
                       random_state=self.seed).fit(X)
        self.W = self.nmf.components_.astype(X.dtype)
        self.n_iter, self.converged = self.nmf.n_iter_, self.nmf.n_iter_ < self.max_iter
        return self

    def score(self, X):
        H = nnls_codes(np.maximum(X, 0), self.W)
        return ((X - H @ self.W) ** 2).sum(1)


class RobustNMFResidual(NMFResidual):
    """L2,1-norm NMF (Kong, Ding & Huang, CIKM 2011): min_{W,H>=0} sum_i ||x_i - h_i W||_2, by
    reweighted multiplicative updates initialised from standard NMF. Same fixed-W NNLS score
    (for a single row the L2 and squared-L2 minimisers coincide)."""

    def fit(self, X, iters=200, eps=1e-10):
        super().fit(X)
        H = self.nmf.transform(X).astype(np.float64)
        W, Xd = self.W.astype(np.float64), X.astype(np.float64)
        for _ in range(iters):
            d = 1.0 / np.maximum(np.linalg.norm(Xd - H @ W, axis=1), eps)
            HD = H * d[:, None]
            W *= (HD.T @ Xd) / np.maximum(HD.T @ H @ W, eps)
            H *= (Xd @ W.T) / np.maximum(H @ (W @ W.T), eps)  # D cancels row-wise in the H update
        self.W = W.astype(X.dtype)
        return self


class Mahalanobis:
    """D10: Ledoit-Wolf shrinkage covariance on the full feature space."""

    def fit(self, X):
        lw = LedoitWolf().fit(X.astype(np.float64))
        self.mu, self.P, self.shrinkage = lw.location_, lw.precision_, lw.shrinkage_
        return self

    def score(self, X):
        D = X.astype(np.float64) - self.mu
        return ((D @ self.P) * D).sum(1)


class LinearProbe:
    """Supervised ceiling reference (W3): logistic regression on standardized features."""

    def __init__(self, C=0.1):
        self.C = C

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(self.sc.transform(X), y)
        return self

    def score(self, X):
        return self.clf.decision_function(self.sc.transform(X))
