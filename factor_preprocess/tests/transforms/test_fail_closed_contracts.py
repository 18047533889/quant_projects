"""Fail-closed contracts for zero-variance and non-finite inputs.

Pins the repairs found by the 2026-08-19 platform audit: a zero lagged
std must produce NaN (never inf / 0.0), and a single ±inf in a
cross-section must NaN only its own position (never the whole slice,
never a silent 0.0).
"""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.transforms.cross_sectional import cs_winsor, cs_zscore
from factor_preprocess.transforms.rolling import rolling_zscore
from factor_preprocess.transforms.volatility import realized_volatility


class TestRollingZscoreZeroStd:
    def test_zero_lagged_std_produces_nan_not_inf(self):
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 1.0, 1.0, 1.0, 5.0],
        })
        result = rolling_zscore(df, window=3)
        # The lagged window [1, 1, 1] has zero std; the docstring
        # promises NaN and volatility_scale masks the same statistic.
        assert pd.isna(result.iloc[4])

    def test_all_zero_std_indices_are_masked(self):
        # Windows (2, 3) are also constant-lagged; a partial mask
        # regression at earlier indices must not slip through.
        df = pd.DataFrame({
            "asset_id": ["A"] * 5,
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1.0, 1.0, 1.0, 1.0, 5.0],
        })
        result = rolling_zscore(df, window=3)
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])  # current == lagged mean: 0/0, not 0.0
        assert pd.isna(result.iloc[4])

    def test_nonzero_std_still_scores(self):
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, 2.0, 3.0, 10.0],
        })
        result = rolling_zscore(df, window=3)
        assert np.isfinite(result.iloc[3])


class TestCsZscoreInf:
    def test_inf_input_is_nan_not_zero(self):
        # Pre-fix: inf pushed std to NaN, the slice fell into the
        # constant branch, and inf read as a perfectly average 0.0.
        result = cs_zscore(np.array([[1.0, np.inf]]))
        assert np.isnan(result[0, 1])
        assert result[0, 0] == 0.0  # single finite value -> constant branch

    def test_neg_inf_input_is_nan(self):
        result = cs_zscore(np.array([[-np.inf, 1.0, 3.0]]))
        assert np.isnan(result[0, 0])
        assert np.isfinite(result[0, 1]) and np.isfinite(result[0, 2])

    def test_all_inf_slice_is_all_nan(self):
        result = cs_zscore(np.array([[np.inf, np.inf, np.inf]]))
        assert np.all(np.isnan(result))

    def test_finite_slice_unchanged(self):
        values = np.array([[1.0, 2.0, 3.0]])
        result = cs_zscore(values)
        expected = (values - values.mean()) / values.std(ddof=1)
        np.testing.assert_allclose(result, expected)


class TestCsWinsorInf:
    def test_single_inf_does_not_poison_whole_slice(self):
        # Pre-fix: nanquantile over [1, inf, 3] NaN'd both bounds and
        # clipped every entry to NaN.
        result = cs_winsor(np.array([[1.0, np.inf, 3.0]]), lower=0.01, upper=0.99)
        assert np.isnan(result[0, 1])
        assert np.isfinite(result[0, 0]) and np.isfinite(result[0, 2])

    def test_single_neg_inf_does_not_poison_whole_slice(self):
        result = cs_winsor(np.array([[-np.inf, 1.0, 3.0]]), lower=0.01, upper=0.99)
        assert np.isnan(result[0, 0])
        assert np.isfinite(result[0, 1]) and np.isfinite(result[0, 2])

    def test_finite_slice_clips_to_exact_bounds(self):
        values = np.array([[-10.0, 1.0, 2.0, 3.0, 100.0]])
        result = cs_winsor(values, lower=0.25, upper=0.75)
        # Quantile bounds of the finite slice: 1.0 and 3.0 — extremes are
        # clipped toward the bounds, interior values untouched.
        assert result[0, 0] == 1.0
        assert result[0, 4] == 3.0
        assert result[0, 2] == 2.0


class TestRealizedVolatilityZeroVar:
    def test_zero_variance_history_is_nan_not_zero(self):
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [5.0, 5.0, 5.0, 5.0],
        })
        result = realized_volatility(df, window=2)
        # Matches volatility_scale's zero-vol masking; a 0.0 here
        # divides to inf downstream.
        assert pd.isna(result.iloc[2]) and pd.isna(result.iloc[3])

    def test_nonzero_variance_still_produces_vol(self):
        df = pd.DataFrame({
            "asset_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "value": [1.0, 2.0, 3.0, 4.0],
        })
        result = realized_volatility(df, window=2)
        assert np.isfinite(result.iloc[2]) and np.isfinite(result.iloc[3])


class TestDecompositionBugsPropagate:
    """Narrowed bare except Exception in wavelet/cycle/seasonal must not
    swallow programming errors; only numerical failures produce NaN."""

    def test_wavelet_bare_typeerror_not_swallowed(self):
        import factor_preprocess.transforms.decomposition.wavelet as wmod

        _orig = wmod.wavelet_decompose
        def _bomb(*args, **kwargs):
            raise TypeError("simulated wiring bug")
        wmod.wavelet_decompose = _bomb
        try:
            df = pd.DataFrame({
                "asset_id": ["A"]*8,
                "date": pd.date_range("2020-01-01", periods=8),
                "value": 1.0,
            })
            with pytest.raises(TypeError, match="simulated wiring bug"):
                wmod.wavelet_smooth(df, wavelet='db4', level=1)
        finally:
            wmod.wavelet_decompose = _orig

    def test_stl_bare_typeerror_not_swallowed(self):
        import factor_preprocess.transforms.decomposition.seasonal as smod

        _orig = smod.seasonal_component
        def _bomb(*args, **kwargs):
            raise TypeError("simulated wiring bug")
        smod.seasonal_component = _bomb
        try:
            df = pd.DataFrame({
                "asset_id": ["A"]*24,
                "date": pd.date_range("2020-01-01", periods=24),
                "value": 1.0,
            })
            with pytest.raises(TypeError, match="simulated wiring bug"):
                smod.seasonal_component(df, period=12)
        finally:
            smod.seasonal_component = _orig

    def test_bandpass_bare_typeerror_not_swallowed(self):
        import factor_preprocess.transforms.decomposition.cycle as cmod

        _orig = cmod.bandpass_filter
        def _bomb(*args, **kwargs):
            raise TypeError("simulated wiring bug")
        cmod.bandpass_filter = _bomb
        try:
            df = pd.DataFrame({
                "asset_id": ["A"]*24,
                "date": pd.date_range("2020-01-01", periods=24),
                "value": 1.0,
            })
            with pytest.raises(TypeError, match="simulated wiring bug"):
                cmod.bandpass_filter(df, low_freq=0.05, high_freq=0.2)
        finally:
            cmod.bandpass_filter = _orig
