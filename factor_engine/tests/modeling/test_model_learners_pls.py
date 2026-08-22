# -*- coding: utf-8 -*-
"""Predictive PLS learner — sklearn independent oracle (audit item 1, P0).

Audit item (predictive learner P0 math):
* ``pls.py`` used ``W^T P`` for the regression coefficient inverse; for
  multi-component PLS ``W^T P != P^T W`` so the coefficients were wrong.
  Fixed to ``b_std = W (P^T W)^{-1} q`` (``P^T W`` via ``np.linalg.solve`` with
  a ``pinv`` fallback).

Covered here (against ``sklearn.cross_decomposition.PLSRegression`` on the
SAME standardization convention: X standardized to unit variance, y
standardized to unit variance):
* n_components = 1 / 2 / 3 (the multi-component cases the old ``W^T P`` broke);
* correlated features;
* ill-conditioned (near-collinear) X;
* scale-transformed X (fit on 10*X must reproduce the same predictions).
Numerical values are compared (predict outputs AND standardized coefficients) —
not merely finiteness.
"""
from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.cross_decomposition import PLSRegression  # noqa: E402

from modeling.learners.base import LearnerSpec  # noqa: E402
from modeling.learners.pls import PLSLearner  # noqa: E402


def _spec(n_components: int) -> LearnerSpec:
    return LearnerSpec(
        learner_name="predictive_pls",
        family="pls",
        hyperparams={"n_components": n_components},
    )


def _standardize(X: np.ndarray, y: np.ndarray):
    mx, my = X.mean(0), float(y.mean())
    sx, sy = X.std(0), float(y.std())
    sx[sx == 0] = 1.0
    sy = sy if sy > 0 else 1.0
    return mx, my, sx, sy, (X - mx) / sx, (y - my) / sy


def _oracle_parity(X: np.ndarray, y: np.ndarray, k: int, rtol: float = 1e-6):
    learner = PLSLearner(_spec(k))
    frozen = learner.fit(X, y)
    mx, my, sx, sy, Xs, ys = _standardize(X, y)

    sk = PLSRegression(n_components=k, scale=False).fit(Xs, ys)

    # Coefficient parity in the SAME standardized space.
    b_std = frozen.params["coef"] * sx / sy
    np.testing.assert_allclose(b_std, sk.coef_.ravel(), rtol=rtol, atol=1e-10)

    # Predict parity on raw and scale-transformed inputs.
    X_test = X[:50]
    for X_new in (X_test, 10.0 * X_test, X_test * np.array([1.0, -2.0, 0.5, 3.0, -1.0])):
        pred_ours = learner.predict(frozen, X_new)
        pred_sk = my + sy * sk.predict((X_new - mx) / sx)
        np.testing.assert_allclose(pred_ours, pred_sk.ravel(), rtol=rtol, atol=1e-8)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_pls_matches_sklearn_independent_gaussian(k):
    rng = np.random.default_rng(0)
    n, d = 400, 5
    X = rng.normal(size=(n, d))
    true_b = np.array([1.0, -0.5, 0.3, 2.0, -1.2])
    y = X @ true_b + 0.1 * rng.normal(size=n)
    _oracle_parity(X, y, k)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_pls_matches_sklearn_correlated_features(k):
    rng = np.random.default_rng(3)
    n = 400
    base = rng.normal(size=(n, 3))
    # Deliberately correlated columns (some near-linear combinations).
    X = np.column_stack([
        base[:, 0] + 0.5 * base[:, 1],
        base[:, 1],
        base[:, 2],
        base[:, 0] - base[:, 2],
        base[:, 1] + 0.3 * base[:, 2],
    ])
    y = X @ np.array([1.0, -0.5, 0.3, 2.0, -1.0]) + 0.1 * rng.normal(size=n)
    _oracle_parity(X, y, k)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_pls_matches_sklearn_ill_conditioned_X(k):
    rng = np.random.default_rng(5)
    n = 300
    X = np.column_stack([
        rng.normal(size=n),
        rng.normal(size=n),
        rng.normal(size=n),
    ])
    # Near-collinear trailing columns -> ill-conditioned design.
    X = np.column_stack([
        X,
        X[:, 0] + 1e-9 * rng.normal(size=n),
        X[:, 1] - 1e-9 * rng.normal(size=n),
    ])
    y = X @ np.array([1.0, -0.5, 0.3, 1.0, -0.5]) + 0.1 * rng.normal(size=n)
    _oracle_parity(X, y, k, rtol=1e-4)


def test_pls_scale_transformed_X_is_invariant():
    """Fitting on X and on 10*X must produce the SAME predictions."""
    rng = np.random.default_rng(7)
    n, d = 300, 4
    X = rng.normal(size=(n, d))
    y = X @ np.array([1.0, 0.5, -0.5, 2.0]) + 0.1 * rng.normal(size=n)
    learner = PLSLearner(_spec(3))
    frozen_raw = learner.fit(X, y)
    frozen_scaled = learner.fit(10.0 * X, y)
    X_test = X[:50]
    np.testing.assert_allclose(
        learner.predict(frozen_raw, X_test),
        learner.predict(frozen_scaled, 10.0 * X_test),
        rtol=1e-8,
        atol=1e-10,
    )


def test_pls_fails_closed_on_excess_components():
    """n_components > min(n_rows, n_features) must be rejected, not silent-downgraded."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    y = X @ np.array([1.0, -0.5, 0.3]) + 0.1 * rng.normal(size=20)
    learner = PLSLearner(_spec(5))  # 5 > d=3
    with pytest.raises(ValueError, match="n_components"):
        learner.fit(X, y)
