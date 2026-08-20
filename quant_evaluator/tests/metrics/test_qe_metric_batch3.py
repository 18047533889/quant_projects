"""
QE-METRIC batch 3 (P0-10..14) focused tests.

New, targeted tests for:
- P0-10: missing_return_policy paths in compute_long_short_returns
- P0-11: degenerate-bucket guards in construct_long_short_portfolio
- P0-12: gross-exposure HHI in compute_concentration_hhi
- P0-13: point-in-time label validation in compute_sector_exposure
- P0-14: input validation + EXPERIMENTAL demotion in multiple_testing
- Residual: canonical turnover parity of fast_turnover_estimate
"""

import numpy as np
import pytest

from quant_evaluator.metrics.portfolio_stats import compute_long_short_returns
from quant_evaluator.metrics.probe_portfolio import construct_long_short_portfolio
from quant_evaluator.metrics.exposure import (
    compute_concentration_hhi,
    compute_sector_exposure,
)
from quant_evaluator.metrics.multiple_testing import (
    bonferroni_correction,
    benjamini_hochberg_correction,
    holm_bonferroni_correction,
    sidak_correction,
    compute_fdr,
)
from quant_evaluator.metrics.turnover import estimate_turnover_from_ranks
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.kernels.fast import fast_turnover_estimate


# ---------------------------------------------------------------------------
# P0-10: missing_return_policy in compute_long_short_returns
# ---------------------------------------------------------------------------

class TestMissingReturnPolicy:
    """P0-10: explicit zero_fill | drop | fail semantics."""

    def _make_inputs(self, seed=42, T=30, N=50):
        rng = np.random.RandomState(seed)
        factor = rng.randn(T, N)
        fwd = factor * 0.01 + rng.randn(T, N) * 0.02
        return factor, fwd

    def test_default_is_zero_fill_and_documented(self):
        factor, fwd = self._make_inputs()
        fwd_with_nan = fwd.copy()
        fwd_with_nan[0, :5] = np.nan

        ret_default = compute_long_short_returns(factor, fwd_with_nan)
        ret_explicit = compute_long_short_returns(
            factor, fwd_with_nan, missing_return_policy="zero_fill"
        )

        # Default must equal the explicit zero_fill policy (back-compat),
        # and period 0 must be FINITE (NaNs filled with 0).
        assert np.allclose(ret_default[2], ret_explicit[2], equal_nan=True)
        assert np.isfinite(ret_default[2][0])
        assert np.isfinite(ret_explicit[2][0])

    def test_drop_differs_from_zero_fill(self):
        """drop excludes NaN-return assets from bucket means."""
        factor, fwd = self._make_inputs()
        fwd_nan = fwd.copy()
        # Make the *top-ranked* asset's return NaN in period 0 so the
        # long bucket mean must differ between the policies.
        top_asset = int(np.argmax(factor[0, :]))
        fwd_nan[0, top_asset] = np.nan

        zf_long, zf_short, zf_ls = compute_long_short_returns(
            factor, fwd_nan, missing_return_policy="zero_fill"
        )
        dp_long, dp_short, dp_ls = compute_long_short_returns(
            factor, fwd_nan, missing_return_policy="drop"
        )

        # zero_fill treats the NaN as 0 -> drags the long mean toward 0.
        # drop excludes the asset entirely -> different (typically larger
        # magnitude) bucket mean. They cannot be equal here because the
        # dropped asset has non-zero factor loading.
        assert not np.isclose(zf_long[0], dp_long[0]), (
            "zero_fill and drop must differ when a selected asset has a "
            "NaN return"
        )

    def test_drop_all_nan_period_gives_nan_not_zero(self):
        factor, _ = self._make_inputs()
        T, N = factor.shape
        fwd = np.full((T, N), np.nan)
        fwd[1:, :] = np.random.RandomState(0).randn(T - 1, N) * 0.01

        _, _, ls = compute_long_short_returns(factor, fwd, missing_return_policy="drop")

        assert np.isnan(ls[0])  # never a silent 0

    def test_fail_raises_on_any_nan(self):
        factor, fwd = self._make_inputs()
        fwd[5, 7] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            compute_long_short_returns(factor, fwd, missing_return_policy="fail")

    def test_fail_passes_on_clean_input(self):
        factor, fwd = self._make_inputs()
        rets = compute_long_short_returns(factor, fwd, missing_return_policy="fail")
        assert np.isfinite(rets[2]).all()

    def test_invalid_policy_string_raises(self):
        factor, fwd = self._make_inputs()
        with pytest.raises(ValueError, match="missing_return_policy"):
            compute_long_short_returns(factor, fwd, missing_return_policy="fuzzy")

    def test_policy_propagates_through_3d_recursion(self):
        rng = np.random.RandomState(7)
        T, N, F = 10, 40, 2
        factor = rng.randn(T, N, F)
        fwd = rng.randn(T, N) * 0.01
        fwd[0, 0] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            compute_long_short_returns(factor, fwd, missing_return_policy="fail")


# ---------------------------------------------------------------------------
# P0-11: degenerate-bucket guards in construct_long_short_portfolio
# ---------------------------------------------------------------------------

class TestBucketGuards:
    """P0-11: constant-factor / degenerate-quantile guards."""

    def test_constant_factor_yields_empty_positions(self):
        """A constant factor must not produce a fake 0 spread."""
        T, N = 10, 50
        factor = np.full((T, N), 3.14)

        long_pos, short_pos = construct_long_short_portfolio(factor)

        # Degenerate periods leave the masks empty — no fake long/short.
        assert not np.any(long_pos)
        assert not np.any(short_pos)

    def test_constant_factor_strict_raises(self):
        T, N = 5, 20
        factor = np.full((T, N), 1.0)

        with pytest.raises(ValueError, match="degenerate|bucket guard"):
            construct_long_short_portfolio(factor, strict=True)

    def test_low_cardinality_factor_skipped(self):
        """Fewer than 2 unique valid values -> degenerate period."""
        T, N = 4, 30
        factor = np.zeros((T, N))
        factor[0, :] = np.linspace(0, 1, N)  # period 0 is fine
        # periods 1..3 all-constant -> skipped

        long_pos, short_pos = construct_long_short_portfolio(factor)

        assert np.any(long_pos[0])
        assert not np.any(long_pos[1:])

    def test_min_bucket_size_floor(self):
        """Buckets below the floor are rejected (non-strict: skipped)."""
        rng = np.random.RandomState(3)
        T, N = 8, 30
        factor = rng.randn(T, N)

        # min_bucket_size=6 with 30 assets and 0.8/0.2 cutoffs gives
        # buckets of ~6; a stricter floor must reject some periods.
        long_pos, short_pos = construct_long_short_portfolio(
            factor, min_bucket_size=10
        )
        # Buckets of ~6 < floor 10 -> all periods degenerate
        assert not np.any(long_pos)

        # A floor the buckets CAN meet passes through.
        long_ok, short_ok = construct_long_short_portfolio(
            factor, min_bucket_size=2
        )
        assert np.any(long_ok)
        assert np.all(np.sum(long_ok, axis=1) >= 2)
        assert np.all(np.sum(short_ok, axis=1) >= 2)

    def test_strict_raises_naming_period(self):
        rng = np.random.RandomState(11)
        T, N = 6, 30
        factor = rng.randn(T, N)
        factor[3, :] = 7.0  # one degenerate period

        with pytest.raises(ValueError, match="period 3"):
            construct_long_short_portfolio(factor, strict=True)

    def test_valid_data_backcompat(self):
        """Default args preserve old behavior on healthy data."""
        rng = np.random.RandomState(42)
        T, N = 20, 100
        factor = rng.randn(T, N)

        long_pos, short_pos = construct_long_short_portfolio(factor)

        assert long_pos.shape == (T, N)
        # Buckets disjoint...
        assert not np.any(long_pos & short_pos)
        # ...and populated in every period.
        assert np.all(np.sum(long_pos, axis=1) > 0)
        assert np.all(np.sum(short_pos, axis=1) > 0)


# ---------------------------------------------------------------------------
# P0-12: gross-exposure HHI
# ---------------------------------------------------------------------------

class TestGrossHHI:
    """P0-12: HHI on gross (abs) exposure."""

    def test_sign_flip_invariance(self):
        rng = np.random.RandomState(5)
        factor = rng.randn(6, 12)
        flipped = factor.copy()
        flipped[:, 0] *= -1  # flip one asset's sign

        assert np.allclose(
            compute_concentration_hhi(factor),
            compute_concentration_hhi(flipped),
        )

    def test_cancelling_long_short_does_not_explode(self):
        """Near-cancelling signed exposure must stay in (0, 1]."""
        T, N = 4, 10
        # Positive weights, SIGNED factor values that cancel pairwise:
        # sum_i(w_i * f_i) = 0 exactly (old signed denominator -> division
        # by ~0 explosion) while every |w_i * f_i| is equal (gross shares
        # uniform -> HHI = 1/N).
        factor = np.tile(np.array([1.0, -1.0] * (N // 2)), (T, 1))
        weights = np.full((T, N), 1.0)

        hhi = compute_concentration_hhi(factor, weights)

        # Signed exposure sums to exactly 0 in every period.
        assert np.allclose(np.sum(factor * weights, axis=1), 0.0)
        assert np.all(np.isfinite(hhi))
        assert np.all((hhi > 0) & (hhi <= 1.0))
        # Equal absolute exposures -> HHI == 1/N exactly.
        assert np.allclose(hhi, 1.0 / N)

    def test_uniform_exposure_hhi_is_inverse_n(self):
        T, N = 3, 8
        factor = np.ones((T, N))

        hhi = compute_concentration_hhi(factor)

        assert np.allclose(hhi, 1.0 / N)

    def test_single_asset_concentration_is_one(self):
        T, N = 3, 6
        factor = np.zeros((T, N))
        factor[:, 2] = 5.0  # all exposure on one asset

        hhi = compute_concentration_hhi(factor)

        assert np.allclose(hhi, 1.0)

    def test_all_zero_exposure_is_nan(self):
        factor = np.zeros((2, 5))
        hhi = compute_concentration_hhi(factor)
        assert np.all(np.isnan(hhi))


# ---------------------------------------------------------------------------
# P0-13: point-in-time label validation
# ---------------------------------------------------------------------------

class TestPITLabels:
    """P0-13: look-ahead rejection for sector labels."""

    def _make_panel(self, T=6, N=8):
        rng = np.random.RandomState(9)
        factor = rng.randn(T, N)
        sectors = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float)
        factor_time = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(T, "D"),
            dtype="datetime64[D]",
        )
        return factor, sectors, factor_time

    def test_future_dated_labels_rejected(self):
        factor, sectors, factor_time = self._make_panel()
        label_time = np.datetime64("2024-06-01")  # after the factor sample

        with pytest.raises(ValueError, match="look-ahead"):
            compute_sector_exposure(
                factor, sectors,
                label_time=label_time, factor_time=factor_time,
            )

    def test_as_of_after_sample_passes(self):
        factor, sectors, factor_time = self._make_panel()
        label_time = np.datetime64("2023-01-01")  # labels predate all periods

        exposure, counts = compute_sector_exposure(
            factor, sectors,
            label_time=label_time, factor_time=factor_time,
        )
        assert exposure.shape == (factor.shape[0], 4)
        assert np.isfinite(exposure).all()

    def test_factor_time_without_label_time_rejected(self):
        factor, sectors, factor_time = self._make_panel()

        with pytest.raises(ValueError, match="label_time"):
            compute_sector_exposure(factor, sectors, factor_time=factor_time)

    def test_no_timestamps_documented_assumption(self):
        factor, sectors, _ = self._make_panel()
        exposure, counts = compute_sector_exposure(factor, sectors)
        assert exposure.shape == (factor.shape[0], 4)

    def test_partial_overlap_rejected_with_count(self):
        factor, sectors, factor_time = self._make_panel()
        label_time = np.datetime64("2024-01-03")  # mid-sample

        with pytest.raises(ValueError, match="2 of 6"):
            compute_sector_exposure(
                factor, sectors,
                label_time=label_time, factor_time=factor_time,
            )


# ---------------------------------------------------------------------------
# P0-14: multiple-testing input validation
# ---------------------------------------------------------------------------

ALL_CORRECTIONS = [
    bonferroni_correction,
    benjamini_hochberg_correction,
    holm_bonferroni_correction,
    sidak_correction,
]


class TestMTValidation:
    """P0-14: fail-closed input validation for corrections + FDR."""

    @pytest.mark.parametrize("correction", ALL_CORRECTIONS)
    def test_empty_p_values_rejected(self, correction):
        with pytest.raises(ValueError, match="non-empty"):
            correction(np.array([]))

    @pytest.mark.parametrize("correction", ALL_CORRECTIONS)
    def test_p_above_one_rejected(self, correction):
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            correction(np.array([0.1, 0.5, 1.5]))

    @pytest.mark.parametrize("correction", ALL_CORRECTIONS)
    def test_p_below_zero_rejected(self, correction):
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            correction(np.array([0.1, -0.01, 0.5]))

    @pytest.mark.parametrize("correction", ALL_CORRECTIONS)
    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.05, 1.5, np.nan, np.inf])
    def test_bad_alpha_rejected(self, correction, bad_alpha):
        with pytest.raises(ValueError, match="alpha"):
            correction(np.array([0.1, 0.4]), alpha=bad_alpha)

    @pytest.mark.parametrize("correction", ALL_CORRECTIONS)
    def test_nan_p_values_still_treated_as_missing(self, correction):
        """Non-finite entries are documented-missing, not fatal."""
        p = np.array([0.01, np.nan, 0.04, np.inf, 0.5])

        result = correction(p, alpha=0.05)
        adjusted = result[0]

        assert np.isnan(adjusted[1])
        assert np.isnan(adjusted[3])
        assert np.isfinite(adjusted[0]) and np.isfinite(adjusted[2])

    @pytest.mark.parametrize("correction", ALL_CORRECTIONS)
    def test_boundary_p_values_accepted(self, correction):
        """0.0 and 1.0 are legal p-values."""
        result = correction(np.array([0.0, 1.0]), alpha=0.05)
        assert np.isfinite(result[0]).all()

    def test_fdr_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_fdr(np.array([]), np.array([], dtype=bool))

    def test_fdr_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            compute_fdr(np.array([0.2, 1.2]), np.array([True, False]))

    def test_fdr_mismatched_reject_shape(self):
        with pytest.raises(ValueError, match="shape"):
            compute_fdr(np.array([0.2, 0.3]), np.array([True, False, False]))

    def test_fdr_non_bool_reject_mask(self):
        with pytest.raises(ValueError, match="boolean"):
            compute_fdr(np.array([0.2, 0.3]), np.array([1, 0]))

    def test_fdr_experimental_banner(self):
        """P0-14 demotion: docstring carries an EXPERIMENTAL warning."""
        assert "EXPERIMENTAL" in compute_fdr.__doc__


# ---------------------------------------------------------------------------
# Residual (batch 2): fast_turnover_estimate parity with the reference
# ---------------------------------------------------------------------------

class TestFastTurnoverParity:
    """fast_turnover_estimate must match the canonical reference exactly."""

    def _to_batch(self, values):
        T, N, F = values.shape
        return FactorBatch(
            factor_ids=tuple(f"f{i}" for i in range(F)),
            time_axis=AxisRef(name="time", dtype="datetime64", size=T),
            asset_axis=AxisRef(name="asset", dtype="int64", size=N),
            values=values,
        )

    def test_matches_reference_exactly(self):
        rng = np.random.RandomState(13)
        T, N, F = 20, 25, 4
        factor = rng.randn(T, N, F)
        # Sprinkle NaNs to exercise the finite-masking paths.
        factor[rng.rand(T, N, F) < 0.1] = np.nan

        ref = estimate_turnover_from_ranks(self._to_batch(factor), window=1)
        fast = fast_turnover_estimate(factor, window=1)

        assert fast.shape == ref.shape
        np.testing.assert_allclose(fast, ref, rtol=1e-9, equal_nan=True)

    def test_known_value_two_periods(self):
        """Canonical hand-computed value with a rank change."""
        T, N = 3, 4
        base = np.array([1.0, 2.0, 3.0, 4.0])
        # Period 1: swap the top two ranks only.
        p1 = np.array([1.0, 2.0, 4.0, 3.0])
        factor = np.stack([base, p1, base])[:, :, np.newaxis]  # (3, 4, 1)

        # Average-tie ranks (1..4) normalized to sum 1: w = rank/10.
        w0 = np.array([1.0, 2.0, 3.0, 4.0]) / 10.0
        w1 = np.array([1.0, 2.0, 4.0, 3.0]) / 10.0
        expected = 0.5 * np.sum(np.abs(w1 - w0))

        out = fast_turnover_estimate(factor, window=1, min_obs=2)

        np.testing.assert_allclose(out[1, 0], expected)
        # min_obs=2 met but reference floor is max(min_obs,2)... with 4
        # finite assets both paths are eligible.
        assert np.isnan(out[0, 0])  # first `window` periods are NaN

    def test_not_a_correlation_proxy(self):
        """Guard against regressions to the old 1-|corr| proxy: complete
        rank reversal (spearman = -1) must give the canonical weight-change
        value ~N/(2(N+1)), not the proxy's 0."""
        T, N = 3, 10
        base = np.arange(N, dtype=float)
        rev = base[::-1].copy()
        factor = np.stack([base, rev, base])[:, :, np.newaxis]

        out = fast_turnover_estimate(factor, window=1)

        # Canonical: average-tie ranks normalized to sum 1 (w = rank/55);
        # full reversal moves rank r to N+1-r, so
        # 0.5 * sum_r |N+1-2r| / 55 = 25/55.
        expected = 0.5 * np.sum(np.abs((N + 1) - 2 * np.arange(1, N + 1))) / (N * (N + 1) / 2)
        np.testing.assert_allclose(out[1, 0], expected)
        # Sanity: the old 1-|spearman| proxy would return exactly 0 here.
        assert out[1, 0] > 0.4
