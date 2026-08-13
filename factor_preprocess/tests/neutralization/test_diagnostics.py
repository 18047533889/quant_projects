"""
Test suite for neutralization diagnostics.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.neutralization.diagnostics import (
    compute_condition_number,
    compute_exposure_correlation,
    check_residual_exposures,
    diagnose_neutralization,
    compute_variance_reduction,
    flag_ill_conditioned_dates,
    summarize_diagnostics,
)
from factor_preprocess.neutralization.ols import ols_neutralize


class TestConditionNumber:
    """Test condition number computation."""

    def test_well_conditioned_matrix(self):
        """Test condition number for well-conditioned matrix."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(20)]

        exposures_list = []
        np.random.seed(42)

        for asset in assets:
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": np.random.randn(),
                "exp2": np.random.randn(),
            })

        exposures_df = pd.DataFrame(exposures_list)

        result = compute_condition_number(exposures_df)

        assert len(result) == 1
        assert "condition_number" in result.columns
        assert "rank" in result.columns
        assert result["condition_number"].iloc[0] > 0
        assert result["condition_number"].iloc[0] < 100  # Well-conditioned

    def test_ill_conditioned_matrix(self):
        """Test condition number for ill-conditioned matrix."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(20)]

        exposures_list = []

        for i, asset in enumerate(assets):
            exp1 = float(i)
            exp2 = exp1 + 1e-8  # Nearly collinear
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": exp1,
                "exp2": exp2,
            })

        exposures_df = pd.DataFrame(exposures_list)

        result = compute_condition_number(exposures_df)

        # Should have high condition number
        assert result["condition_number"].iloc[0] > 100

    def test_per_date_computation(self):
        """Test condition number computed per date."""
        dates = pd.date_range("2020-01-01", periods=3)
        assets = ["A", "B", "C", "D", "E"]

        exposures_list = []
        np.random.seed(42)

        for date in dates:
            for asset in assets:
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "exposure": np.random.randn(),
                })

        exposures_df = pd.DataFrame(exposures_list)

        result = compute_condition_number(exposures_df)

        # Should have one row per date
        assert len(result) == 3
        assert all(result["condition_number"].notna())

    def test_rank_computation(self):
        """Test matrix rank computation."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(15)]

        exposures_list = []

        for i, asset in enumerate(assets):
            exp1 = float(i)
            exp2 = 2 * exp1  # Linearly dependent
            exp3 = float(i**2)
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": exp1,
                "exp2": exp2,
                "exp3": exp3,
            })

        exposures_df = pd.DataFrame(exposures_list)

        result = compute_condition_number(exposures_df)

        # Rank should be less than 3 due to collinearity
        assert result["rank"].iloc[0] <= 2

    def test_empty_exposures(self):
        """Test handling of dates with no valid exposures."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C"]

        exposures_list = []

        for asset in assets:
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": np.nan,
                "exp2": np.nan,
            })

        exposures_df = pd.DataFrame(exposures_list)

        result = compute_condition_number(exposures_df)

        assert result["condition_number"].iloc[0] is np.nan or np.isnan(result["condition_number"].iloc[0])


class TestExposureCorrelation:
    """Test exposure correlation computation."""

    def test_basic_correlation(self):
        """Test basic correlation computation."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(20)]

        exposures_list = []
        np.random.seed(42)

        for asset in assets:
            exp1 = np.random.randn()
            exp2 = np.random.randn()
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": exp1,
                "exp2": exp2,
            })

        exposures_df = pd.DataFrame(exposures_list)

        result = compute_exposure_correlation(exposures_df)

        assert len(result) > 0
        assert "correlation" in result.columns
        assert "exposure1" in result.columns
        assert "exposure2" in result.columns

    def test_high_correlation_detection(self):
        """Test detection of highly correlated exposures."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(30)]

        exposures_list = []

        for i, asset in enumerate(assets):
            exp1 = float(i)
            exp2 = exp1 + np.random.randn() * 0.01  # Nearly perfect correlation
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": exp1,
                "exp2": exp2,
            })

        exposures_df = pd.DataFrame(exposures_list)

        result = compute_exposure_correlation(exposures_df)

        # Should detect high correlation
        assert result["correlation"].abs().max() > 0.95

    def test_multiple_exposures(self):
        """Test correlation with multiple exposures."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(25)]

        exposures_list = []
        np.random.seed(42)

        for asset in assets:
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": np.random.randn(),
                "exp2": np.random.randn(),
                "exp3": np.random.randn(),
            })

        exposures_df = pd.DataFrame(exposures_list)

        result = compute_exposure_correlation(exposures_df)

        # Should have 3 choose 2 = 3 pairs
        assert len(result) == 3


class TestResidualExposures:
    """Test residual exposure checking."""

    def test_successful_neutralization(self):
        """Test that successful neutralization reduces exposures."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        n_assets = 50
        assets = [f"A{i}" for i in range(n_assets)]

        # Generate data with strong exposure
        exposures_arr = np.random.randn(n_assets, 2)
        coef = np.array([2.0, -1.0])
        values_arr = exposures_arr @ coef + np.random.randn(n_assets) * 0.5

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": values_arr[i],
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": exposures_arr[i, 0],
                "exp2": exposures_arr[i, 1],
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        # Neutralize
        residuals = ols_neutralize(values_df, exposures_df, min_observations=10)

        residuals_df = pd.DataFrame({
            "date": values_df["date"],
            "asset_id": values_df["asset_id"],
            "residual": residuals,
        })

        # Check residual exposures
        result = check_residual_exposures(residuals_df, exposures_df)

        # Residuals should have low correlation with exposures
        assert result["correlation"].abs().max() < 0.2

    def test_t_statistic_computation(self):
        """Test t-statistic and p-value computation."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(30)]

        # Pure noise residuals (no exposure)
        residuals_list = []
        exposures_list = []

        for asset in assets:
            residuals_list.append({
                "date": dates[0],
                "asset_id": asset,
                "residual": np.random.randn(),
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": np.random.randn(),
            })

        residuals_df = pd.DataFrame(residuals_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = check_residual_exposures(residuals_df, exposures_df)

        assert "t_stat" in result.columns
        assert "p_value" in result.columns
        # Most p-values should be > 0.05 for pure noise
        assert (result["p_value"] > 0.05).sum() / len(result) > 0.5


class TestVarianceReduction:
    """Test variance reduction computation."""

    def test_basic_variance_reduction(self):
        """Test basic variance reduction computation."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(30)]

        values_list = []
        exposures_list = []

        for asset in assets:
            exp = np.random.randn()
            value = 2 * exp + np.random.randn() * 0.3
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": value,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": exp,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        residuals = ols_neutralize(values_df, exposures_df, min_observations=10)

        residuals_df = pd.DataFrame({
            "date": values_df["date"],
            "asset_id": values_df["asset_id"],
            "residual": residuals,
        })

        result = compute_variance_reduction(
            values_df, residuals_df, value_col="value", residual_col="residual"
        )

        assert len(result) == 1
        assert "r_squared" in result.columns
        assert "original_var" in result.columns
        assert "residual_var" in result.columns

        # Should have positive R²
        assert result["r_squared"].iloc[0] > 0

    def test_no_variance_reduction(self):
        """Test case with no exposure effect."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(30)]

        values_list = []
        exposures_list = []

        for asset in assets:
            # Value independent of exposure
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": np.random.randn(),
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": np.random.randn(),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        residuals = ols_neutralize(values_df, exposures_df, min_observations=10)

        residuals_df = pd.DataFrame({
            "date": values_df["date"],
            "asset_id": values_df["asset_id"],
            "residual": residuals,
        })

        result = compute_variance_reduction(
            values_df, residuals_df, value_col="value", residual_col="residual"
        )

        # R² should be close to 0
        assert abs(result["r_squared"].iloc[0]) < 0.3


class TestDiagnoseNeutralization:
    """Test comprehensive diagnostics."""

    def test_full_diagnostics(self):
        """Test full diagnostic suite."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        n_assets = 40
        assets = [f"A{i}" for i in range(n_assets)]

        # Generate data
        X = np.random.randn(n_assets, 2)
        coef = np.array([1.5, -0.8])
        y = X @ coef + np.random.randn(n_assets) * 0.3

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": y[i],
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": X[i, 0],
                "exp2": X[i, 1],
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        # Neutralize
        residuals = ols_neutralize(values_df, exposures_df, min_observations=10)

        residuals_df = pd.DataFrame({
            "date": values_df["date"],
            "asset_id": values_df["asset_id"],
            "residual": residuals,
        })

        # Run diagnostics
        diagnostics = diagnose_neutralization(
            values_df,
            residuals_df,
            exposures_df,
            value_col="value",
            residual_col="residual",
        )

        # Check all components present
        assert "condition_numbers" in diagnostics
        assert "exposure_correlations" in diagnostics
        assert "residual_exposures" in diagnostics
        assert "variance_reduction" in diagnostics

        # Each should be a DataFrame
        assert isinstance(diagnostics["condition_numbers"], pd.DataFrame)
        assert isinstance(diagnostics["exposure_correlations"], pd.DataFrame)
        assert isinstance(diagnostics["residual_exposures"], pd.DataFrame)
        assert isinstance(diagnostics["variance_reduction"], pd.DataFrame)


class TestFlagIllConditioned:
    """Test ill-conditioned date flagging."""

    def test_flagging_with_threshold(self):
        """Test flagging dates above threshold."""
        dates = pd.date_range("2020-01-01", periods=3)
        assets = [f"A{i}" for i in range(20)]

        exposures_list = []

        for i, date in enumerate(dates):
            for j, asset in enumerate(assets):
                exp1 = float(j)
                # First date: well-conditioned
                # Second date: moderately ill-conditioned
                # Third date: severely ill-conditioned
                if i == 0:
                    exp2 = np.random.randn()
                elif i == 1:
                    exp2 = exp1 + np.random.randn() * 0.1
                else:
                    exp2 = exp1 + 1e-7

                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "exp1": exp1,
                    "exp2": exp2,
                })

        exposures_df = pd.DataFrame(exposures_list)

        result = flag_ill_conditioned_dates(exposures_df, threshold=50.0)

        # Should flag at least the last date
        assert len(result) >= 1

        # Should have severity classification
        if len(result) > 0:
            assert "severity" in result.columns


class TestSummarizeDiagnostics:
    """Test diagnostic summary."""

    def test_summary_computation(self):
        """Test summary statistic computation."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        n_assets = 35
        assets = [f"A{i}" for i in range(n_assets)]

        X = np.random.randn(n_assets, 2)
        coef = np.array([2.0, -1.0])
        y = X @ coef + np.random.randn(n_assets) * 0.4

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": y[i],
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": X[i, 0],
                "exp2": X[i, 1],
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        residuals = ols_neutralize(values_df, exposures_df, min_observations=10)

        residuals_df = pd.DataFrame({
            "date": values_df["date"],
            "asset_id": values_df["asset_id"],
            "residual": residuals,
        })

        diagnostics = diagnose_neutralization(
            values_df,
            residuals_df,
            exposures_df,
            value_col="value",
            residual_col="residual",
        )

        summary = summarize_diagnostics(diagnostics)

        # Check key metrics present
        assert "median_condition_number" in summary
        assert "max_condition_number" in summary
        assert "pct_ill_conditioned" in summary
        assert "max_exposure_correlation" in summary
        assert "mean_residual_exposure_correlation" in summary
        assert "median_r_squared" in summary

        # Check values are reasonable
        assert summary["median_condition_number"] > 0
        assert summary["median_r_squared"] >= 0
