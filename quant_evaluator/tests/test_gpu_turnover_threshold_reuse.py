"""Correctness and boundary checks for threshold-once GPU turnover."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.tradability import batched_factor_turnover_rate
from quant_evaluator.metrics.temporal import compute_factor_turnover_rate


@pytest.mark.parametrize("quantile", [0.9, 0.1])
@pytest.mark.parametrize("workspace_bytes", [1, 128 * 3 * 32 * 2])
def test_threshold_reuse_matches_cpu_across_chunks(quantile, workspace_bytes, monkeypatch):
    import quant_evaluator.kernels.gpu.tradability as tradability

    monkeypatch.setattr(
        tradability, "_FACTOR_TURNOVER_WORKSPACE_BYTES", workspace_bytes
    )
    sort_rows = 0
    original_quantile = tradability._nanquantile_rows

    def counted_quantile(safe, n_finite, q):
        nonlocal sort_rows
        sort_rows += safe.shape[0]
        return original_quantile(safe, n_finite, q)

    monkeypatch.setattr(tradability, "_nanquantile_rows", counted_quantile)
    rng = np.random.default_rng(20261001)
    T, N, F = 9, 32, 3
    # Discrete values exercise ties and inclusive quantile boundaries.
    values = rng.integers(-5, 6, size=(T, N, F)).astype(np.float64)
    values[rng.random(values.shape) < 0.12] = np.nan
    values[4, 7, 1] = np.inf
    values[5, 2, 1] = -np.inf
    # An extreme selected asset becoming unavailable makes that transition
    # unknown under the published missing-next-day convention.
    values[0, :, 0] = np.arange(N, dtype=np.float64)
    selected = N - 1 if quantile > 0.5 else 0
    values[1, selected, 0] = np.nan

    expected = compute_factor_turnover_rate(values, quantile=quantile)
    actual = cp.asnumpy(
        batched_factor_turnover_rate(
            cp.asarray(np.transpose(values, (0, 2, 1))), quantile=quantile
        )
    )
    np.testing.assert_allclose(
        actual, expected, rtol=1e-8, atol=1e-10, equal_nan=True
    )
    assert sort_rows == T * F
    assert np.isnan(actual[0, 0])


@pytest.mark.parametrize(
    ("T", "N", "F"),
    [
        (0, 10, 2),
        (1, 10, 2),
        (4, 10, 0),
        (4, 0, 2),
        (4, 1, 2),
        (4, 9, 2),
    ],
)
def test_small_or_empty_dimensions_have_empty_or_ineligible_result(T, N, F):
    values = np.zeros((T, F, N), dtype=np.float64)
    actual = cp.asnumpy(batched_factor_turnover_rate(cp.asarray(values)))
    assert actual.shape == (max(T - 1, 0), F)
    if N < 10 or T < 2 or F == 0:
        assert np.isnan(actual).all()


def test_exact_minimum_assets_is_eligible():
    # N=10 is the first eligible case; use strictly ordered values so both
    # date-specific thresholds and inclusive membership are deterministic.
    values = np.stack(
        [np.arange(10, dtype=np.float64), np.arange(9, -1, -1, dtype=np.float64)]
    )[:, None, :]
    expected = compute_factor_turnover_rate(np.transpose(values, (0, 2, 1)))
    actual = cp.asnumpy(batched_factor_turnover_rate(cp.asarray(values)))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10, equal_nan=True)
    assert np.isfinite(actual[0, 0])
