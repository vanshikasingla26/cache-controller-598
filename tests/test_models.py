import numpy as np

from shiftcache.models import FastGBT, FastLR


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + X[:, 1] ** 2 - X[:, 2] * X[:, 3] + rng.normal(size=n) > 0.5).astype(int)
    return X, y


def test_fast_lr_matches_sklearn_pipeline():
    from sklearn.linear_model import LogisticRegression
    X, y = _data()
    m = FastLR().fit(X, y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    ref = LogisticRegression(max_iter=2000).fit((X - mu) / sd, y).predict_proba((X - mu) / sd)[:, 1]
    assert np.allclose(m.predict_proba(X)[:, 1], ref, atol=1e-6)


def test_fast_gbt_matches_sklearn():
    X, y = _data(seed=1)
    m = FastGBT(max_iter=40).fit(X, y)
    Xt, _ = _data(500, seed=2)
    assert np.allclose(m.predict_proba(Xt)[:, 1], m.sk.predict_proba(Xt)[:, 1], atol=1e-6)