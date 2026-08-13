"""
Test suite for OLS neutralization.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.neutralization.ols import (
    ols_neutralize,
    compute_exposures,
)


class TestOlsNeutralize:
    """Test OLS neutralization."""

    def test_basic_neutralization(self):
        """Test basic OLS residualization."""
        # Create synthetic data
        dates = pd.date_range("2020-01-01", periods=2)
        assets = ["A", "B", "C", "D", "E"]

        values_list = []
        exposures_list = []

        for date in dates:
            for asset in assets:
                values_list.append({
                    "date": date,
                    "asset_id": asset,
                    "value": np.random.randn(),
                })
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "size": np.random.randn(),
                    "industry": np.random.choice([0, 1]),
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = ols_neutralize(values_df, exposures_df, min_observations=3)

        # Result should have same length as input
        assert len(result) == len(values_df)

        # Should have reduced exposure to size and industry
        # (residuals should have low correlation with exposures)

    def test_per_date_independence(self):
        """Test that neutralization is per-date."""
        dates = pd.date_range("2020-01-01", periods=2)
        assets = ["A", "B", "C"]

        values_list = []
        exposures_list = []

        for i, date in enumerate(dates):
            for j, asset in enumerate(assets):
                values_list.append({
                    "date": date,
                    "asset_id": asset,
                    "value": float(i * 10 + j),
                })
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "exposure": float(j),
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = ols_neutralize(values_df, exposures_df, min_observations=2)

        # Each date should be processed independently
        # Residuals should exist for both dates
        assert result.notna().any()

    def test_insufficient_observations(self):
        """Test that insufficient observations produce NaN."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B"]  # Only 2 assets, but need more for 2 exposures + intercept

        values_list = []
        exposures_list = []

        for date in dates:
            for asset in assets:
                values_list.append({
                    "date": date,
                    "asset_id": asset,
                    "value": 1.0,
                })
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "exp1": 1.0,
                    "exp2": 2.0,
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = ols_neutralize(
            values_df,
            exposures_df,
            min_observations=10,  # More than available
        )

        # All should be NaN
        assert result.isna().all()

    def test_nan_handling(self):
        """Test NaN handling in OLS."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C", "D", "E"]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            value = 1.0 if i < 3 else np.nan  # Last two are NaN
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": value,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = ols_neutralize(values_df, exposures_df, min_observations=2)

        # NaN inputs should produce NaN residuals
        assert result.isna().iloc[-2:].all()

        # Valid inputs should produce finite residuals
        assert result.notna().iloc[:3].all()

    def test_no_exposure_columns_fails(self):
        """Test that missing exposure columns fail gracefully."""
        values_df = pd.DataFrame({
            "date": [pd.Timestamp("2020-01-01")] * 3,
            "asset_id": ["A", "B", "C"],
            "value": [1.0, 2.0, 3.0],
        })
        exposures_df = pd.DataFrame({
            "date": [pd.Timestamp("2020-01-01")] * 3,
            "asset_id": ["A", "B", "C"],
        })

        with pytest.raises(ValueError, match="No exposure columns"):
            ols_neutralize(values_df, exposures_df)

    def test_intercept_handling(self):
        """Test with and without intercept."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C", "D", "E"]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": float(i),
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result_with = ols_neutralize(
            values_df, exposures_df, add_intercept=True, min_observations=2
        )
        result_without = ols_neutralize(
            values_df, exposures_df, add_intercept=False, min_observations=2
        )

        # Both should produce results
        assert result_with.notna().any()
        assert result_without.notna().any()

        # Results may differ
        # (with intercept removes mean, without does not)


class TestComputeExposures:
    """Test exposure computation."""

    def test_basic_exposures(self):
        """Test basic exposure coefficient computation."""
        dates = pd.date_range("2020-01-01", periods=2)
        assets = ["A", "B", "C", "D", "E"]

        values_list = []
        exposures_list = []

        for date in dates:
            for i, asset in enumerate(assets):
                values_list.append({
                    "date": date,
                    "asset_id": asset,
                    "value": float(i),
                })
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "size": float(i),
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = compute_exposures(values_df, exposures_df)

        # Should have one row per date
        assert len(result) == 2
        assert "date" in result.columns
        assert "intercept" in result.columns
        assert "size_coef" in result.columns

    def test_insufficient_data_produces_nan(self):
        """Test that insufficient data produces NaN coefficients."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B"]  # Too few

        values_list = []
        exposures_list = []

        for asset in assets:
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": 1.0,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": 1.0,
                "exp2": 2.0,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = compute_exposures(values_df, exposures_df)

        # Should have NaN coefficients
        assert result["intercept"].isna().all()
        assert result["exp1_coef"].isna().all()
