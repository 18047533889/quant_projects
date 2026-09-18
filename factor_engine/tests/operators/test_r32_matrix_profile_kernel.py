from __future__ import annotations

import time

import numpy as np

from factor_engine.cleaned_operators.intraday._core import log_returns
from factor_engine.cleaned_operators.intraday import topology_manifold
from factor_engine.cleaned_operators.intraday.topology_manifold import (
    _matrix_profile_features,
)


def _naive(close, window):
    r = log_returns(np.asarray(close, dtype=float))
    r = r[np.isfinite(r)]
    if len(r) < window + 1 or window < 4:
        return (np.nan, np.nan, np.nan)
    sigma = np.std(r, ddof=1)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return (np.nan, np.nan, np.nan)
    r = (r - np.mean(r)) / sigma
    profile = np.full(len(r) - window + 1, np.inf)
    for i in range(len(profile)):
        left = r[i : i + window]
        left_std = np.std(left, ddof=1)
        if not np.isfinite(left_std) or left_std <= 1e-12:
            continue
        left = (left - np.mean(left)) / left_std
        for j in range(len(profile)):
            if abs(i - j) < window // 2:
                continue
            right = r[j : j + window]
            right_std = np.std(right, ddof=1)
            if not np.isfinite(right_std) or right_std <= 1e-12:
                continue
            right = (right - np.mean(right)) / right_std
            profile[i] = min(profile[i], np.sqrt(np.sum((left - right) ** 2)))
    valid = profile[np.isfinite(profile)]
    if len(valid) < 2:
        return (np.nan, np.nan, np.nan)
    return float(valid.min()), float(valid.mean()), float(valid.max())


def _price(returns):
    return 100.0 * np.exp(np.r_[0.0, np.cumsum(returns)])


def test_tiled_kernel_matches_naive_ddof_exclusion_and_ties():
    rng = np.random.default_rng(321)
    base = rng.normal(0.0, 0.01, 72)
    returns = np.r_[base[:24], base[:24], base[24:48]]
    close = _price(returns)
    expected = _naive(close, 12)
    actual = _matrix_profile_features(close, 12)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_missing_compaction_and_degenerate_semantics_are_preserved():
    returns = 0.01 * np.sin(np.arange(80) / 4)
    close = _price(returns)
    close[[12, 39]] = np.nan
    np.testing.assert_allclose(
        _matrix_profile_features(close, 10), _naive(close, 10),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    assert np.isnan(_matrix_profile_features(_price(np.zeros(50)), 10)).all()


def test_forced_small_workspace_fallback_matches_naive(monkeypatch):
    rng = np.random.default_rng(987)
    close = _price(rng.normal(0.0, 0.008, 90))
    monkeypatch.setattr(topology_manifold, "_MATRIX_PROFILE_TILE_TARGET_BYTES", 256)
    np.testing.assert_allclose(
        _matrix_profile_features(close, 14), _naive(close, 14),
        rtol=1e-12, atol=1e-12,
    )


def test_tiled_kernel_is_bounded_and_fast_on_long_session():
    returns = 0.002 * np.sin(np.arange(1500) / 7) + 0.001 * np.cos(np.arange(1500) / 17)
    start = time.perf_counter()
    result = _matrix_profile_features(_price(returns), 20)
    elapsed = time.perf_counter() - start
    assert np.isfinite(result).all()
    assert elapsed < 5.0
