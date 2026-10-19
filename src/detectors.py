"""Detectors. Unsupervised ones expose fit(X_benign_train) and score(X) -> higher = more anomalous.

Notation (PROTOCOL §2): X (n x d) ~= H W, with W (k x d) a basis learned on benign data and
H (n x k) per-input coefficients.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
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
