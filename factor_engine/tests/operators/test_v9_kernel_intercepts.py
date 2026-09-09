"""M35 regressions for train-only kernel centering and intercepts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.research_spectral import (
    _center_train_test_kernel,
    _kernel_granger_score,
    _kernel_ridge_with_intercept,
    _residualized_hsic,
)


def _constant_z_fold_mean_hsic_oracle(x, y, purge):
    """Independent residual construction for the constant-control case."""
    n = x.size
    split = n // 2
    rx = np.full(n, np.nan)
    ry = np.full(n, np.nan)
    folds = ((np.arange(split), np.arange(split + purge, n)),
             (np.arange(split, n), np.arange(0, split - purge)))
    for train, test in folds:
        rx[test] = x[test] - np.mean(x[train])
        ry[test] = y[test] - np.mean(y[train])
    finite = np.isfinite(rx) & np.isfinite(ry)
    rx, ry = rx[finite], ry[finite]

    def centered_rbf(values):
        distance = np.abs(values[:, None] - values[None, :])
        sigma = np.median(distance)
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0
        kernel = np.exp(-(distance * distance) / (2.0 * sigma * sigma))
        return (kernel - kernel.mean(axis=0, keepdims=True)
                - kernel.mean(axis=1, keepdims=True) + kernel.mean())

    Kx, Ky = centered_rbf(rx), centered_rbf(ry)
    return float(np.sum(Kx * Ky)) / float((rx.size - 1) ** 2)


def test_kernel_ridge_intercept_matches_independent_kkt_oracle():
    rng = np.random.default_rng(3501)
    features = rng.normal(size=(18, 3))
    test_features = rng.normal(size=(7, 3))
    K = features @ features.T
    Kte = test_features @ features.T
    target = 13.0 + rng.normal(size=18)
    lam = 0.07
    Kc, Ktec = _center_train_test_kernel(K, Kte)
    actual = _kernel_ridge_with_intercept(Kc, target, Ktec, lam)

    system = np.block([[K + lam * np.eye(18), np.ones((18, 1))],
                       [np.ones((1, 18)), np.zeros((1, 1))]])
    solution = np.linalg.solve(system, np.r_[target, 0.0])
    expected = Kte @ solution[:-1] + solution[-1]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_kernel_granger_intercept_constant_and_duplicate_controls():
    rng = np.random.default_rng(3502)
    y = rng.normal(size=100)
    x = rng.normal(size=100)
    score = _kernel_granger_score(y, x, 2)
    assert _kernel_granger_score(y + 1e6, x, 2) == pytest.approx(score, abs=2e-9)
    assert _kernel_granger_score(y, np.ones_like(x), 2) == pytest.approx(0.0, abs=1e-12)
    assert _kernel_granger_score(y, y.copy(), 2) == pytest.approx(0.0, abs=1e-12)


def test_residualized_hsic_intercept_shift_and_constant_z_controls():
    rng = np.random.default_rng(3503)
    x = rng.normal(size=96)
    y = 0.4 * x + rng.normal(size=96)
    z = rng.normal(size=96)
    base = _residualized_hsic(x, y, z, 3)
    shifted = _residualized_hsic(x + 1e7, y - 2e7, z + 3e7, 3)
    assert shifted == pytest.approx(base, rel=2e-8, abs=2e-10)
    constant = _residualized_hsic(x, y, np.ones_like(z), 3)
    shifted_constant = _residualized_hsic(x + 11.0, y - 17.0, np.ones_like(z), 3)
    oracle = _constant_z_fold_mean_hsic_oracle(x, y, 3)
    assert constant == pytest.approx(oracle, rel=2e-12, abs=2e-12)
    assert shifted_constant == pytest.approx(constant, rel=2e-10, abs=2e-12)


def test_public_granger_prefix_is_stable_and_limited_null_smoke():
    ensure_cleaned_loaded()
    op = OperatorRegistry.get("ts_kernel_granger_score", "pandas_numpy", mode="research")
    rng = np.random.default_rng(3504)
    y = pd.DataFrame({"A": rng.normal(size=130)})
    x = pd.DataFrame({"A": rng.normal(size=130)})
    params = {"window": 80, "lag": 2}
    full = op.calculate(y=y, x=x, **params)
    prefix = op.calculate(y=y.iloc[:105], x=x.iloc[:105], **params)
    pd.testing.assert_series_equal(full["A"].iloc[:105], prefix["A"])
    assert np.isfinite(full["A"]).any()

    null_scores = []
    for seed in range(3505, 3511):
        local = np.random.default_rng(seed)
        null_scores.append(_kernel_granger_score(local.normal(size=400), local.normal(size=400), 2))
    assert np.all(np.isfinite(null_scores))
    assert abs(float(np.median(null_scores))) < 0.2


def test_predictable_past_x_control_beats_block_permuted_contrast():
    rng = np.random.default_rng(3512)
    n = 420
    x = rng.normal(size=n)
    y = np.zeros(n)
    noise = rng.normal(scale=0.25, size=n)
    for i in range(1, n):
        y[i] = 0.35 * y[i - 1] + 0.9 * x[i - 1] + noise[i]
    positive = _kernel_granger_score(y, x, 2)
    blocks = x.reshape(14, 30)
    permuted = blocks[np.array([7, 2, 11, 0, 9, 4, 13, 5, 1, 12, 6, 3, 10, 8])].reshape(-1)
    contrast = _kernel_granger_score(y, permuted, 2)
    assert np.isfinite(positive) and np.isfinite(contrast)
    assert positive > 0.25
    assert positive > contrast + 0.25
