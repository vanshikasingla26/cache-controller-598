"""Re-ranker models with low-overhead inference.

sklearn's predict_proba costs ~0.5-1 ms per call on 16 rows, which would dominate
the overhead numbers (RQ5). Both models are trained with sklearn and then served
through small numpy paths that are checked against sklearn at fit time.
"""
from __future__ import annotations

import numpy as np


class FastLR:
    """Standardize + logistic regression, numpy inference."""
    name = "lr"

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        m = LogisticRegression(C=self.C, max_iter=2000).fit((X - self.mu) / self.sd, y)
        self.w, self.b = m.coef_[0] / self.sd, float(m.intercept_[0] - (m.coef_[0] * self.mu / self.sd).sum())
        return self

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))
        return np.c_[1 - p, p]


class FastGBT:
    """sklearn HistGradientBoosting trees, evaluated level-by-level across all trees."""
    name = "gbt"

    def __init__(self, max_iter=150, max_depth=6, learning_rate=0.1):
        self.kw = dict(max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate,
                       early_stopping=True, validation_fraction=0.1, random_state=0)

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.sk = HistGradientBoostingClassifier(**self.kw).fit(X, y)
        preds = [p[0] for p in self.sk._predictors]
        L = max(len(p.nodes) for p in preds)
        T = len(preds)
        self.feat = np.zeros((T, L), np.int64)
        self.thr = np.zeros((T, L))
        self.left = np.zeros((T, L), np.int64)
        self.right = np.zeros((T, L), np.int64)
        self.leaf = np.ones((T, L), bool)
        self.val = np.zeros((T, L))
        for i, p in enumerate(preds):
            nd = p.nodes
            k = len(nd)
            self.feat[i, :k], self.thr[i, :k] = nd["feature_idx"], nd["num_threshold"]
            self.left[i, :k], self.right[i, :k] = nd["left"], nd["right"]
            self.leaf[i, :k], self.val[i, :k] = nd["is_leaf"].astype(bool), nd["value"]
        self.base = float(np.ravel(self.sk._baseline_prediction)[0])
        self.depth = int(max(self._depth(p.nodes) for p in preds))
        self._rows = np.arange(T)[:, None]
        chk = X[: min(2000, len(X))]
        assert np.allclose(self.predict_proba(chk)[:, 1], self.sk.predict_proba(chk)[:, 1], atol=1e-6)
        return self

    @staticmethod
    def _depth(nd):
        d, stack = 0, [(0, 0)]
        while stack:
            i, h = stack.pop()
            d = max(d, h)
            if not nd["is_leaf"][i]:
                stack += [(nd["left"][i], h + 1), (nd["right"][i], h + 1)]
        return d

    def predict_proba(self, X):
        n = len(X)
        T = self.feat.shape[0]
        node = np.zeros((T, n), np.int64)
        r = self._rows
        for _ in range(self.depth):
            f = self.feat[r, node]
            go_left = X[np.arange(n)[None, :], f] <= self.thr[r, node]
            nxt = np.where(go_left, self.left[r, node], self.right[r, node])
            node = np.where(self.leaf[r, node], node, nxt)
        raw = self.base + self.val[r, node].sum(0)
        p = 1.0 / (1.0 + np.exp(-raw))
        return np.c_[1 - p, p]


def fit_models(X, y, which=("lr", "gbt"), max_rows=500_000, seed=0):
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        idx = rng.choice(len(y), max_rows, replace=False)
        X, y = X[idx], y[idx]
    out = {}
    for m in which:
        out[m] = (FastLR() if m == "lr" else FastGBT()).fit(X, y)
        print(f"  trained {m} on {len(y):,} rows")
    return out
