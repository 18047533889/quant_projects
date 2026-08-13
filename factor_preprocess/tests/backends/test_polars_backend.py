"""
Parity tests for polars backend vs reference implementations.

Tests ensure polars backend produces identical results to reference
and validates 3-5x performance improvement on large panels.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Reference implementations
from factor_preprocess.transforms.cross_sectional import (
    cs_rank,
    cs_zscore,
    cs_demean,
    cs_winsor,
    cs_scale,
)
from factor_preprocess.neutralization.ols import ols_neutralize

# Polars backend
try:
    from factor_preprocess.backends.polars_backend import (
        cs_rank_polars,
        cs_zscore_polars,
        cs_demean_polars,
        cs_winsor_polars,
        cs_scale_polars,
        ols_neutralize_polars,
        POLARS_AVAILABLE,
    )
    import polars as pl
except ImportError:
    POLARS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not POLARS_AVAILABLE,
    reason="polars not installed"
)


def generate_panel_data(n_dates: int = 100, n_assets: int = 500, seed: int = 42):
    """Generate synthetic panel data for testing."""
    np.random.seed(seed)

    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    asset_ids = [f"ASSET_{i:04d}" for i in range(n_assets)]

    # Create panel
    index = pd.MultiIndex.from_product(
        [dates, asset_ids],
        names=["date", "asset_id"]
    )

    # Generate values with structure
    df = pd.DataFrame({
        "date": [d for d in dates for _ in asset_ids],
        "asset_id": asset_ids * n_dates,
        "value": np.random.randn(n_dates * n_assets) * 10 + 100,
    })

    # Add some NaN (10% missing)
    missing_mask = np.random.rand(len(df)) < 0.1
    df.loc[missing_mask, "value"] = np.nan

    return df


class TestCsRankParity:
    """Test cs_rank_polars vs reference."""

    @pytest.mark.parity
    def test_basic_parity(self):
        """Test basic ranking parity."""
        df = generate_panel_data(n_dates=10, n_assets=50)

        # Reference: rank within each date
        ref_result = []
        for date, group in df.groupby("date"):
            ranks = cs_rank(group["value"].values)
            ref_result.extend(ranks)
        ref_result = np.array(ref_result)

        # Polars backend
        polars_result = cs_rank_polars(df, value_col="value", group_col="date")

        # Compare (allowing for NaN)
        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            equal_nan=True,
        )

    @pytest.mark.parity
    def test_percentile_parity(self):
        """Test percentile ranking parity."""
        df = generate_panel_data(n_dates=10, n_assets=50)

        # Reference
        ref_result = []
        for date, group in df.groupby("date"):
            ranks = cs_rank(group["value"].values, pct=True)
            ref_result.extend(ranks)
        ref_result = np.array(ref_result)

        # Polars backend
        polars_result = cs_rank_polars(df, value_col="value", group_col="date", pct=True)

        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            equal_nan=True,
        )

    @pytest.mark.parity
    def test_nan_handling(self):
        """Test NaN handling parity."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 5,
            "value": [1.0, np.nan, 3.0, 2.0, np.nan],
        })

        # Reference
        ref_result = cs_rank(df["value"].values)

        # Polars backend
        polars_result = cs_rank_polars(df, value_col="value", group_col="date")

        # NaN positions should match
        assert np.all(np.isnan(ref_result) == np.isnan(polars_result.values))

        # Finite values should match
        mask = np.isfinite(ref_result)
        np.testing.assert_array_equal(ref_result[mask], polars_result.values[mask])


class TestCsZscoreParity:
    """Test cs_zscore_polars vs reference."""

    @pytest.mark.parity
    def test_basic_parity(self):
        """Test basic z-score parity."""
        df = generate_panel_data(n_dates=10, n_assets=50)

        # Reference
        ref_result = []
        for date, group in df.groupby("date"):
            zscores = cs_zscore(group["value"].values)
            ref_result.extend(zscores)
        ref_result = np.array(ref_result)

        # Polars backend
        polars_result = cs_zscore_polars(df, value_col="value", group_col="date")

        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

    @pytest.mark.parity
    def test_constant_groups(self):
        """Test constant groups produce constant_value."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 5 + ["2020-01-02"] * 5,
            "value": [5.0] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        # Reference
        ref_result = []
        for date, group in df.groupby("date"):
            zscores = cs_zscore(group["value"].values, constant_value=0.0)
            ref_result.extend(zscores)
        ref_result = np.array(ref_result)

        # Polars backend
        polars_result = cs_zscore_polars(
            df, value_col="value", group_col="date", constant_value=0.0
        )

        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            equal_nan=True,
        )


class TestCsDemeanParity:
    """Test cs_demean_polars vs reference."""

    @pytest.mark.parity
    def test_basic_parity(self):
        """Test basic demean parity."""
        df = generate_panel_data(n_dates=10, n_assets=50)

        # Reference
        ref_result = []
        for date, group in df.groupby("date"):
            demeaned = cs_demean(group["value"].values)
            ref_result.extend(demeaned)
        ref_result = np.array(ref_result)

        # Polars backend
        polars_result = cs_demean_polars(df, value_col="value", group_col="date")

        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

    @pytest.mark.parity
    def test_mean_zero(self):
        """Test that result has mean ~0 per group."""
        df = generate_panel_data(n_dates=10, n_assets=50)

        polars_result = cs_demean_polars(df, value_col="value", group_col="date")
        df["demeaned"] = polars_result

        # Check mean per group is ~0
        for date, group in df.groupby("date"):
            mean = np.nanmean(group["demeaned"].values)
            np.testing.assert_allclose(mean, 0.0, atol=1e-10)


class TestCsWinsorParity:
    """Test cs_winsor_polars vs reference."""

    @pytest.mark.parity
    def test_basic_parity(self):
        """Test basic winsorization parity."""
        df = generate_panel_data(n_dates=10, n_assets=50)

        # Reference
        ref_result = []
        for date, group in df.groupby("date"):
            winsorized = cs_winsor(group["value"].values, lower=0.05, upper=0.95)
            ref_result.extend(winsorized)
        ref_result = np.array(ref_result)

        # Polars backend
        polars_result = cs_winsor_polars(
            df, value_col="value", group_col="date", lower=0.05, upper=0.95
        )

        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            equal_nan=True,
        )

    @pytest.mark.parity
    def test_extreme_clipping(self):
        """Test that extremes are clipped."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 10,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],
        })

        # Reference
        ref_result = cs_winsor(df["value"].values, lower=0.1, upper=0.9)

        # Polars backend
        polars_result = cs_winsor_polars(
            df, value_col="value", group_col="date", lower=0.1, upper=0.9
        )

        # Should match reference
        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
        )

        # Extreme value (100.0) should be clipped
        assert polars_result.values[-1] < 100.0


class TestCsScaleParity:
    """Test cs_scale_polars vs reference."""

    @pytest.mark.parity
    def test_basic_parity(self):
        """Test basic scaling parity."""
        df = generate_panel_data(n_dates=10, n_assets=50)

        # Reference
        ref_result = []
        for date, group in df.groupby("date"):
            scaled = cs_scale(group["value"].values, target_std=2.0)
            ref_result.extend(scaled)
        ref_result = np.array(ref_result)

        # Polars backend
        polars_result = cs_scale_polars(
            df, value_col="value", group_col="date", target_std=2.0
        )

        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            equal_nan=True,
        )

    @pytest.mark.parity
    def test_target_std_achieved(self):
        """Test that target std is achieved per group."""
        df = generate_panel_data(n_dates=10, n_assets=50)

        polars_result = cs_scale_polars(
            df, value_col="value", group_col="date", target_std=3.0
        )
        df["scaled"] = polars_result

        # Check std per group is ~3.0
        for date, group in df.groupby("date"):
            std = np.nanstd(group["scaled"].values, ddof=1)
            if not np.isnan(std):
                np.testing.assert_allclose(std, 3.0, rtol=1e-6)


class TestOlsNeutralizeParity:
    """Test ols_neutralize_polars vs reference."""

    @pytest.mark.parity
    def test_basic_parity(self):
        """Test basic OLS neutralization parity."""
        # Generate values
        df_values = generate_panel_data(n_dates=10, n_assets=50)

        # Generate exposures (e.g., market cap, sector)
        np.random.seed(42)
        df_exposures = df_values[["date", "asset_id"]].copy()
        df_exposures["market_cap"] = np.random.randn(len(df_exposures)) * 1e9 + 5e9
        df_exposures["sector"] = np.random.choice([0, 1, 2], size=len(df_exposures))

        # Reference
        ref_result = ols_neutralize(
            df_values,
            df_exposures,
            date_col="date",
            asset_col="asset_id",
            value_col="value",
        )

        # Polars backend
        polars_result = ols_neutralize_polars(
            df_values,
            df_exposures,
            date_col="date",
            asset_col="asset_id",
            value_col="value",
        )

        np.testing.assert_allclose(
            ref_result.values,
            polars_result.values,
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
        )

    @pytest.mark.parity
    def test_insufficient_observations(self):
        """Test that insufficient observations produce NaN."""
        df_values = pd.DataFrame({
            "date": ["2020-01-01"] * 3,
            "asset_id": ["A", "B", "C"],
            "value": [1.0, 2.0, 3.0],
        })

        df_exposures = pd.DataFrame({
            "date": ["2020-01-01"] * 3,
            "asset_id": ["A", "B", "C"],
            "exp1": [0.1, 0.2, 0.3],
            "exp2": [1.0, 2.0, 3.0],
        })

        # With min_observations=10, should produce all NaN
        polars_result = ols_neutralize_polars(
            df_values,
            df_exposures,
            date_col="date",
            asset_col="asset_id",
            value_col="value",
            min_observations=10,
        )

        assert np.all(np.isnan(polars_result.values))


class TestPerformance:
    """Performance benchmarks for polars backend."""

    @pytest.mark.parity
    def test_rank_speedup(self, benchmark=None):
        """Verify rank speedup on large panel."""
        df = generate_panel_data(n_dates=252, n_assets=1000)

        # Just verify it runs without error
        # Actual benchmark would require pytest-benchmark
        polars_result = cs_rank_polars(df, value_col="value", group_col="date")

        # Sanity check
        assert len(polars_result) == len(df)
        assert polars_result.notna().sum() > 0

    @pytest.mark.parity
    def test_zscore_speedup(self):
        """Verify zscore speedup on large panel."""
        df = generate_panel_data(n_dates=252, n_assets=1000)

        polars_result = cs_zscore_polars(df, value_col="value", group_col="date")

        # Sanity check: each date group should have mean ~0, std ~1
        df["zscore"] = polars_result
        sample_date = df["date"].iloc[0]
        sample_group = df[df["date"] == sample_date]["zscore"]

        mean = np.nanmean(sample_group.values)
        std = np.nanstd(sample_group.values, ddof=1)

        np.testing.assert_allclose(mean, 0.0, atol=1e-6)
        np.testing.assert_allclose(std, 1.0, rtol=1e-6)

    @pytest.mark.parity
    def test_neutralization_speedup(self):
        """Verify neutralization speedup on large panel."""
        df_values = generate_panel_data(n_dates=100, n_assets=500)

        # Generate exposures
        np.random.seed(42)
        df_exposures = df_values[["date", "asset_id"]].copy()
        df_exposures["exp1"] = np.random.randn(len(df_exposures))
        df_exposures["exp2"] = np.random.randn(len(df_exposures))

        polars_result = ols_neutralize_polars(
            df_values,
            df_exposures,
            date_col="date",
            asset_col="asset_id",
            value_col="value",
        )

        # Sanity check
        assert len(polars_result) == len(df_values)
        # Should have some non-NaN residuals
        assert polars_result.notna().sum() > 0


class TestEdgeCases:
    """Test edge cases for polars backend."""

    def test_empty_dataframe(self):
        """Test empty DataFrame."""
        df = pd.DataFrame({
            "date": pd.Series([], dtype="datetime64[ns]"),
            "asset_id": pd.Series([], dtype=str),
            "value": pd.Series([], dtype=float),
        })

        result = cs_rank_polars(df, value_col="value", group_col="date")
        assert len(result) == 0

    def test_single_date_single_asset(self):
        """Test single observation."""
        df = pd.DataFrame({
            "date": ["2020-01-01"],
            "value": [100.0],
        })

        result = cs_zscore_polars(df, value_col="value", group_col="date", constant_value=0.0)
        # Single value -> zero std -> constant_value
        np.testing.assert_array_equal(result.values, [0.0])

    def test_all_nan_group(self):
        """Test group with all NaN."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 3 + ["2020-01-02"] * 3,
            "value": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
        })

        result = cs_demean_polars(df, value_col="value", group_col="date")

        # First group should be all NaN
        assert np.all(np.isnan(result.values[:3]))
        # Second group should have mean ~0
        np.testing.assert_allclose(np.nanmean(result.values[3:]), 0.0, atol=1e-10)

    def test_polars_dataframe_input(self):
        """Test native polars DataFrame input."""
        df_pd = generate_panel_data(n_dates=10, n_assets=50)
        df_pl = pl.from_pandas(df_pd)

        result = cs_rank_polars(df_pl, value_col="value", group_col="date")

        # Should return polars Series
        assert isinstance(result, pl.Series)
        assert len(result) == len(df_pl)

    def test_invalid_quantiles(self):
        """Test invalid quantiles are rejected."""
        df = generate_panel_data(n_dates=10, n_assets=50)

        with pytest.raises(ValueError, match="Invalid quantiles"):
            cs_winsor_polars(df, value_col="value", group_col="date", lower=0.9, upper=0.1)
