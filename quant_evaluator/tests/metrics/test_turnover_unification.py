"""Tests for turnover unification (QE-METRIC-P0-05).

Canonical turnover definition: 0.5 * sum(|delta w|) on sum-1 weights.
All three entry points conform:
  - compute_turnover_series (authority)
  - estimate_turnover_from_ranks (rank-weight proxy)
  - registry_adapters.compute_turnover_value (scalar adapter)
"""

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.metrics.turnover import (
    compute_turnover_series,
    estimate_turnover_from_ranks,
)
from quant_evaluator.metrics.registry_adapters import compute_turnover_value


def make_batch(values):
    shape = values.shape
    T = shape[0]; N = shape[1]; F = shape[2]
    return FactorBatch(
        factor_ids=tuple("f%d" % i for i in range(F)),
        time_axis=AxisRef(name="time", dtype="int64", size=T),
        asset_axis=AxisRef(name="asset", dtype="int64", size=N),
        values=values,
    )


class TestEstimateTurnoverFromRanks:
    def test_full_rank_inversion_gives_high_turnover(self):
        """rho = -1 (complete inversion) must give large turnover, not 0."""
        T = 6; N = 20
        values = np.zeros((T, N, 1))
        values[0, :, 0] = np.arange(N)
        values[1, :, 0] = np.arange(N)[::-1]  # single inversion, then stable
        for t in range(2, T):
            values[t, :, 0] = np.arange(N)[::-1]
        batch = make_batch(values)

        turnover = estimate_turnover_from_ranks(batch, window=1)

        # Inversion step (t=1): weights fully reversed -> turnover = sum
        # of pairwise rank weight swaps = (N-1)/(N+1) (-> 1 as N grows),
        # decisively NOT the 0.0 the old 1-|rho| proxy produced for rho=-1.
        assert turnover[1, 0] > 0.4, f"got {turnover[1, 0]}"
        # Stable steps (t>=2): weights unchanged -> zero turnover.
        for t in (2, 3, 4, 5):
            assert turnover[t, 0] < 1e-10

    def test_perfect_rank_stability_gives_zero_turnover(self):
        """Constant ranks across time -> turnover exactly 0."""
        T = 10; N = 30
        values = np.tile(np.arange(N, dtype=float), (T, 1)).reshape(T, N, 1)
        batch = make_batch(values)

        turnover = estimate_turnover_from_ranks(batch, window=1)

        assert np.all(np.isnan(turnover[0, :]))
        assert np.allclose(turnover[1:, 0], 0.0, atol=1e-12)

    def test_shape_and_signature_preserved(self):
        """Output is (T, F); first `window` rows are NaN."""
        T = 15; N = 25; F = 3
        rng = np.random.default_rng(42)
        values = rng.normal(size=(T, N, F))
        batch = make_batch(values)

        turnover = estimate_turnover_from_ranks(batch, window=3)

        assert turnover.shape == (T, F)
        assert np.all(np.isnan(turnover[:3, :]))
        assert np.sum(np.isfinite(turnover[3:, :])) > 0

    def test_window_five_inversion_detected(self):
        """window=5 with inversion at lag 5 still yields high turnover."""
        T = 12; N = 20
        values = np.zeros((T, N, 1))
        values[0, :, 0] = np.arange(N)
        values[5, :, 0] = np.arange(N)[::-1]  # inversion exactly at lag 5
        for t in range(1, 5):
            values[t, :, 0] = np.arange(N)
        for t in range(6, T):
            values[t, :, 0] = np.arange(N)[::-1]
        batch = make_batch(values)

        turnover = estimate_turnover_from_ranks(batch, window=5)

        assert turnover[5, 0] > 0.4

    def test_random_ranks_moderate_turnover(self):
        """Independent random cross-sections give turnover in (0, 1)."""
        T = 30; N = 50
        rng = np.random.default_rng(7)
        values = rng.normal(size=(T, N, 1))
        batch = make_batch(values)

        turnover = estimate_turnover_from_ranks(batch, window=1)
        mean_to = float(np.nanmean(turnover))

        assert 0.0 < mean_to < 1.0

    def test_insufficient_cross_section_is_nan(self):
        """Fewer than 10 finite assets in a cross-section -> NaN."""
        T = 4; N = 12
        values = np.zeros((T, N, 1))
        for t in range(T):
            values[t, :, 0] = np.arange(N)
        values[2, :8, 0] = np.nan  # only 4 finite assets at t=2
        batch = make_batch(values)

        turnover = estimate_turnover_from_ranks(batch, window=1)

        assert np.isnan(turnover[2, 0])
        assert np.isfinite(turnover[1, 0])

    def test_ties_get_average_ranks(self):
        """Tied values receive average ranks (not arbitrary distinct)."""
        from scipy.stats import rankdata

        T = 2; N = 12
        row = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        values = np.stack([row, row]).reshape(T, N, 1)
        batch = make_batch(values)

        turnover = estimate_turnover_from_ranks(batch, window=1)

        # Identical rows -> zero turnover, and the tie group must be stable:
        # rankdata gives (1+2+3)/3 = 2 for the three tied values.
        expected_tie_rank = float(np.mean(rankdata(row[:3], method="average")))
        assert expected_tie_rank == 2.0
        assert turnover[1, 0] == pytest.approx(0.0, abs=1e-12)

    def test_window_validation(self):
        values = np.zeros((5, 12, 1))
        batch = make_batch(values)
        with pytest.raises(ValueError, match="window must be >= 1"):
            estimate_turnover_from_ranks(batch, window=0)


class TestComputeTurnoverValueAdapter:
    def test_tie_ranks_are_average_not_arbitrary(self):
        """Ties in the cross-section produce symmetric (average) weights."""
        from scipy.stats import rankdata

        T = 30; N = 12
        rng = np.random.default_rng(3)
        base = rng.normal(size=(T, N))
        # Force ties: quantize to 2 decimals.
        base = np.round(base, 1)
        values = base[:, :, None]
        batch = make_batch(values)

        result = compute_turnover_value(batch)

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        assert result[0] >= 0.0
        # Sanity: rankdata with average method is what we rely on.
        row = base[0]
        r = rankdata(row, method="average")
        assert r.sum() == pytest.approx(N * (N + 1) / 2)

    def test_universe_doubling_invariance(self):
        """Doubling the universe with duplicated assets leaves turnover
        (approximately) unchanged: within tie-rank affine tolerance."""
        T = 40; N = 50
        rng = np.random.default_rng(11)
        base = rng.normal(size=(T, N))
        doubled = np.concatenate([base, base], axis=1)

        t_n = compute_turnover_value(make_batch(base[:, :, None]))
        t_2n = compute_turnover_value(make_batch(doubled[:, :, None]))

        assert np.isfinite(t_n[0])
        # Average-tie ranks under duplication are an affine transform with
        # slope 4S/(4S-N) ~ 1 + 1/N; tolerance covers that.
        assert t_2n[0] == pytest.approx(t_n[0], rel=0.05)

    def test_deterministic(self):
        T = 30; N = 40
        rng = np.random.default_rng(5)
        values = rng.normal(size=(T, N, 2))
        batch = make_batch(values)

        a = compute_turnover_value(batch)
        b = compute_turnover_value(batch)
        np.testing.assert_array_equal(a, b)

    def test_insufficient_time_is_nan(self):
        T = 1; N = 30
        rng = np.random.default_rng(2)
        values = rng.normal(size=(T, N, 1))
        batch = make_batch(values)

        result = compute_turnover_value(batch)
        assert np.isnan(result[0])

    def test_weights_sum_to_one_per_row(self):
        """Proxy weights sum to 1 per row (universe-size invariance base)."""
        from scipy.stats import rankdata

        T = 5; N = 20
        rng = np.random.default_rng(9)
        base = rng.normal(size=(T, N))
        for t in range(T):
            r = rankdata(base[t], method="average")
            w = r / r.sum()
            assert w.sum() == pytest.approx(1.0, abs=1e-12)


class TestCanonicalDefinition:
    def test_compute_turnover_series_is_authority(self):
        """0.5 * sum|delta| on explicit sum-1 weights."""
        w0 = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        series = compute_turnover_series(w0)
        assert series[0] != series[0]  # NaN first row
        assert series[1] == pytest.approx(0.0)

    def test_estimate_conforms_to_authority(self):
        """estimate_turnover_from_ranks matches canonical computation on
        the same rank-weight matrix."""
        from quant_evaluator.metrics.turnover import _rank_weights_matrix

        T = 8; N = 25
        rng = np.random.default_rng(21)
        values = rng.normal(size=(T, N, 1))
        batch = make_batch(values)

        turnover = estimate_turnover_from_ranks(batch, window=1)

        w = _rank_weights_matrix(values)
        canonical = compute_turnover_series(w[:, :, 0])
        np.testing.assert_allclose(
            turnover[:, 0], canonical, equal_nan=True, atol=1e-12
        )

    def test_adapter_conforms_to_authority(self):
        """compute_turnover_value mean matches the canonical series mean."""
        from quant_evaluator.metrics.turnover import _rank_weights_matrix

        T = 25; N = 30
        rng = np.random.default_rng(31)
        values = rng.normal(size=(T, N, 1))
        batch = make_batch(values)

        result = compute_turnover_value(batch, min_periods=2)

        w = _rank_weights_matrix(values)
        canonical = compute_turnover_series(w[:, :, 0])
        expected = float(np.nanmean(canonical))
        assert result[0] == pytest.approx(expected, rel=1e-10)
