# -*- coding: utf-8 -*-
"""Predictive ElasticNet learner — three P0 math fixes (audit item 2).

Audit items (predictive learner P0):
(a) standardized-space mixup: fit trained on ``(X-mx)/sx`` but predict consumed
    raw ``X @ coef`` — the frozen artifact now stores RAW-space coefficients
    (``coef = coef_std / sx``, ``intercept = my - mx @ coef``) so predict on raw
    X is exact and never mixes the two spaces;
(b) ``alpha`` lacked the 1/N normalization — the descent now solves the
    sklearn-compatible objective
        ``1/(2N)||y - Xb||^2 + alpha*l1*||b||_1 + 0.5*alpha*(1-l1)*||b||^2``
    (CD denominators ``(Xj.Xj)/N + alpha*(1-l1)``, ``rho = (Xj.rj)/N``,
    soft-threshold ``sign(rho)*max(0, |rho| - alpha*l1)``);
(c) non-convergence silently returned a half-fit model — running out of
    ``max_iter`` now raises ``ModelConvergenceError`` (defined in
    ``modeling.learners.base``, a ``ValueError`` subclass).

Tests: sklearn predict parity on raw and scale-transformed X, sample-count
(alpha) invariance across different N, and fail-closed non-convergence.
"""
from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import ElasticNet  # noqa: E402

from modeling.learners.base import LearnerSpec, ModelConvergenceError  # noqa: E402
from modeling.learners.elastic_net import ElasticNetLearner  # noqa: E402


def _enet_spec(**hp) -> LearnerSpec:
    params = {"alpha": 0.05, "l1_ratio": 0.5, "max_iter": 100_000, "tol": 1e-10}
    params.update(hp)
    return LearnerSpec(
        learner_name="predictive_elastic_net",
        family="elastic_net",
        hyperparams=params,
    )


def _sk_net(alpha: float, l1_ratio: float):
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        max_iter=200_000,
        tol=1e-12,
    )


def _oracle_parity(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float):
    learner = ElasticNetLearner(_enet_spec(alpha=alpha, l1_ratio=l1_ratio))
    frozen = learner.fit(X, y)
    mx, my = X.mean(0), float(y.mean())
    sx = X.std(0)
    sx[sx == 0] = 1.0
    Xs = (X - mx) / sx
    yc = y - my

    sk = _sk_net(alpha, l1_ratio).fit(Xs, yc)

    # Standardized-space coefficient parity.
    coef_std = frozen.params["coef"] * sx
    np.testing.assert_allclose(coef_std, sk.coef_.ravel(), rtol=1e-4, atol=1e-8)

    # Raw / scale-transformed predict parity.
    X_test = X[:50]
    for X_new in (X_test, 10.0 * X_test, X_test * np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])):
        pred_ours = learner.predict(frozen, X_new)
        pred_sk = my + sk.predict((X_new - mx) / sx)
        np.testing.assert_allclose(pred_ours, pred_sk.ravel(), rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("alpha", [1e-2, 5e-2])
@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_enet_predict_matches_sklearn(alpha, l1_ratio):
    """(a)+(b) — predict / coef parity with sklearn on the same convention."""
    rng = np.random.default_rng(0)
    n, d = 300, 6
    X = rng.normal(size=(n, d))
    true_b = np.array([1.0, -0.5, 0.3, 0.0, 2.0, 0.0])
    y = X @ true_b + 0.1 * rng.normal(size=n)
    _oracle_parity(X, y, alpha, l1_ratio)


@pytest.mark.parametrize("alpha", [1e-2, 0.1])
@pytest.mark.parametrize("l1_ratio", [0.3, 0.8])
def test_enet_alpha_is_sample_count_invariant(alpha, l1_ratio):
    """(b) — the SAME alpha must mean the same regularization for different N.

    Under the old objective (no 1/N) alpha shrank with the panel size; under the
    sklearn-compatible objective both sample sizes must reproduce sklearn's fit
    for one shared alpha.
    """
    rng = np.random.default_rng(11)
    true_b = np.array([1.0, -0.5, 0.3, 0.0, 2.0, 0.0])
    d = len(true_b)
    for n in (200, 2000):
        X = rng.normal(size=(n, d))
        y = X @ true_b + 0.2 * rng.normal(size=n)
        learner = ElasticNetLearner(_enet_spec(alpha=alpha, l1_ratio=l1_ratio))
        frozen = learner.fit(X, y)
        mx, my = X.mean(0), float(y.mean())
        sx = X.std(0)
        sx[sx == 0] = 1.0
        sk = _sk_net(alpha, l1_ratio).fit((X - mx) / sx, y - my)
        np.testing.assert_allclose(
            frozen.params["coef"] * sx, sk.coef_.ravel(), rtol=1e-3, atol=1e-7
        )


def test_enet_non_convergence_raises_model_convergence_error():
    """(c) — running out of max_iter must raise ModelConvergenceError, never a
    silently-returned partial model."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 4))
    y = X @ np.array([1.0, -0.5, 0.3, 2.0]) + 0.1 * rng.normal(size=100)
    learner = ElasticNetLearner(
        _enet_spec(alpha=0.01, l1_ratio=0.5, max_iter=1, tol=1e-14)
    )
    with pytest.raises(ModelConvergenceError, match="did not converge"):
        learner.fit(X, y)


def test_enet_convergence_path_records_metadata():
    """(c) — a converging fit records converged/n_iter in the frozen metadata."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 5))
    y = X @ np.array([1.0, -0.5, 0.3, 2.0, -1.0]) + 0.1 * rng.normal(size=300)
    learner = ElasticNetLearner(_enet_spec(alpha=0.05, l1_ratio=0.5))
    frozen = learner.fit(X, y)
    assert frozen.metadata["converged"] is True
    assert int(frozen.metadata["n_iter"]) >= 1
    assert frozen.metadata["n_active"] == int(
        np.count_nonzero(np.abs(frozen.params["coef"]) > 1e-12)
    )
