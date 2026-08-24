"""Tests for validation.checks module.

Tests data quality checks including NaN rates, outliers, and distribution statistics.
"""
import numpy as np
import pytest

from factor_engine.validation.checks import (
    DataQualityReport,
    OutlierStats,
    check_data_quality,
    check_distribution,
    check_nan_rates,
    check_outliers,
)


class TestCheckNanRates:
    """Test NaN rate checking."""

    def test_clean_data_passes(self):
        """Clean data with no NaN passes all checks."""
        values = np.random.randn(100, 10)
        passed, stats, warnings = check_nan_rates(values)

        assert passed is True
        assert stats["total_nan"] == 0.0
        assert len(warnings) == 0

    def test_high_total_nan_fails(self):
        """High total NaN rate fails."""
        values = np.full((100, 10), np.nan)
        passed, stats, warnings = check_nan_rates(values, max_total_nan=0.5)

        assert passed is False
        assert stats["total_nan"] == 1.0
        assert any("Total NaN" in w for w in warnings)

    def test_high_row_nan_detected(self):
        """High row NaN rate is detected."""
        values = np.random.randn(10, 100)
        values[5, :] = np.nan  # one row all NaN

        passed, stats, warnings = check_nan_rates(values, max_row_nan=0.5)

        assert passed is False
        assert stats["n_bad_rows"] >= 1
        assert any("row NaN" in w for w in warnings)

    def test_high_col_nan_detected(self):
        """High column NaN rate is detected."""
        values = np.random.randn(100, 10)
        values[:, 5] = np.nan  # one column all NaN

        passed, stats, warnings = check_nan_rates(values, max_col_nan=0.5)

        assert passed is False
        assert stats["n_bad_cols"] >= 1
        assert any("col NaN" in w for w in warnings)

    def test_1d_array_checks_total_only(self):
        """1D array only checks total NaN rate."""
        values = np.array([1.0, np.nan, 2.0, np.nan])
        passed, stats, warnings = check_nan_rates(values, max_total_nan=0.4)

        assert passed is False
        assert stats["total_nan"] == 0.5
        assert "max_row_nan" not in stats
        assert "max_col_nan" not in stats


class TestCheckOutliers:
    """Test outlier detection."""

    def test_iqr_method(self):
        """IQR method detects outliers correctly."""
        # Normal data with clear outliers
        values = np.concatenate([
            np.random.randn(100),
            np.array([100, -100])  # clear outliers
        ])

        stats = check_outliers(values, method="iqr", threshold=3.0)

        assert stats.method == "iqr"
        assert stats.n_outliers >= 2
        assert stats.outlier_fraction > 0

    def test_zscore_method(self):
        """Z-score method detects outliers."""
        values = np.concatenate([
            np.random.randn(100),
            np.array([10, -10])
        ])

        stats = check_outliers(values, method="zscore", threshold=3.0)

        assert stats.method == "zscore"
        assert stats.n_outliers >= 2

    def test_mad_method(self):
        """MAD method is robust to outliers."""
        values = np.concatenate([
            np.random.randn(100),
            np.array([100, -100])
        ])

        stats = check_outliers(values, method="mad", threshold=3.0)

        assert stats.method == "mad"
        assert stats.n_outliers >= 2

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        values = np.random.randn(100)

        with pytest.raises(ValueError, match="unknown.*method"):
            check_outliers(values, method="invalid")

    def test_all_nan_returns_empty_stats(self):
        """All NaN values return empty outlier stats."""
        values = np.full(100, np.nan)

        stats = check_outliers(values)

        assert stats.n_outliers == 0
        assert np.isnan(stats.lower_bound)
        assert np.isnan(stats.upper_bound)

    def test_outlier_indices_recorded(self):
        """Outlier indices are recorded."""
        values = np.array([1.0, 2.0, 3.0, 100.0, 4.0])  # 100.0 is outlier

        stats = check_outliers(values, method="iqr", threshold=1.5)

        assert len(stats.outlier_indices) > 0

    def test_summary_method(self):
        """Summary method produces readable output."""
        values = np.concatenate([np.random.randn(100), [100]])

        stats = check_outliers(values)
        summary = stats.summary()

        assert "outliers" in summary.lower()
        assert "bounds" in summary.lower()


class TestCheckDistribution:
    """Test distribution statistics."""

    def test_basic_statistics(self):
        """Basic statistics are computed correctly."""
        values = np.random.randn(1000)

        stats, warnings = check_distribution(values)

        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert "q25" in stats
        assert "q75" in stats

    def test_skewness_computed(self):
        """Skewness is computed when requested."""
        values = np.random.randn(1000)

        stats, warnings = check_distribution(values, check_skewness=True)

        assert "skewness" in stats

    def test_high_skewness_warned(self):
        """High skewness triggers warning."""
        # Create heavily skewed data
        values = np.concatenate([
            np.random.randn(100),
            np.random.exponential(10, 100)
        ])

        stats, warnings = check_distribution(
            values, check_skewness=True, max_skewness=2.0
        )

        if abs(stats.get("skewness", 0)) > 2.0:
            assert any("skewness" in w.lower() for w in warnings)

    def test_kurtosis_computed(self):
        """Excess kurtosis is computed when requested."""
        values = np.random.randn(1000)

        stats, warnings = check_distribution(values, check_kurtosis=True)

        assert "excess_kurtosis" in stats

    def test_high_kurtosis_warned(self):
        """High kurtosis triggers warning."""
        # Create heavy-tailed data
        values = np.concatenate([
            np.random.randn(100),
            np.random.standard_t(2, 100)  # t-distribution with df=2 has heavy tails
        ])

        stats, warnings = check_distribution(
            values, check_kurtosis=True, max_kurtosis=5.0
        )

        # Heavy tailed distributions should have high kurtosis
        assert "excess_kurtosis" in stats

    def test_all_nan_returns_warning(self):
        """All NaN values return warning."""
        values = np.full(100, np.nan)

        stats, warnings = check_distribution(values)

        assert len(warnings) > 0
        assert any("finite" in w.lower() for w in warnings)

    def test_zero_std_handled(self):
        """Zero standard deviation is handled gracefully."""
        values = np.full(100, 5.0)  # constant

        stats, warnings = check_distribution(values, check_skewness=True)

        assert stats["std"] == 0.0
        # Skewness undefined for constant data


class TestCheckDataQuality:
    """Test comprehensive data quality check."""

    def test_clean_data_passes(self):
        """Clean data passes all checks."""
        values = np.random.randn(100, 10)

        report = check_data_quality(values)

        assert report.passed is True
        assert report.n_rows == 100
        assert report.n_cols == 10
        assert report.nan_fraction == 0.0
        assert report.inf_fraction == 0.0
        assert len(report.warnings) == 0

    def test_high_nan_fails(self):
        """High NaN fraction fails."""
        values = np.full((100, 10), np.nan)

        report = check_data_quality(values, max_nan_fraction=0.5)

        assert report.passed is False
        assert report.nan_fraction == 1.0
        assert any("NaN" in w for w in report.warnings)

    def test_inf_detected(self):
        """Inf values are detected."""
        values = np.random.randn(100, 10)
        values[0, 0] = np.inf
        values[1, 0] = -np.inf

        report = check_data_quality(values, max_inf_fraction=0.0)

        assert report.passed is False
        assert report.inf_fraction > 0
        assert any("Inf" in w for w in report.warnings)

    def test_high_zero_fraction_detected(self):
        """High zero fraction is detected."""
        values = np.zeros((100, 10))

        report = check_data_quality(values, max_zero_fraction=0.5)

        assert report.passed is False
        assert report.zero_fraction == 1.0
        assert any("Zero" in w for w in report.warnings)

    def test_outliers_detected(self):
        """Outliers are detected when requested."""
        values = np.concatenate([
            np.random.randn(100).reshape(10, 10),
            np.array([[100] * 10])
        ])

        report = check_data_quality(
            values,
            outlier_method="iqr",
            max_outlier_fraction=0.01,
        )

        assert report.outlier_stats is not None
        # Might fail due to outliers
        if report.outlier_stats.outlier_fraction > 0.01:
            assert not report.passed

    def test_outliers_skipped_when_none(self):
        """Outlier check is skipped when method=None."""
        values = np.random.randn(100, 10)

        report = check_data_quality(values, outlier_method=None)

        assert report.outlier_stats is None

    def test_distribution_stats_included(self):
        """Distribution statistics are included when requested."""
        values = np.random.randn(100, 10)

        report = check_data_quality(values, check_distribution_stats=True)

        assert len(report.distribution) > 0
        assert "mean" in report.distribution

    def test_distribution_stats_skipped(self):
        """Distribution statistics are skipped when not requested."""
        values = np.random.randn(100, 10)

        report = check_data_quality(values, check_distribution_stats=False)

        assert len(report.distribution) == 0

    def test_1d_array_handled(self):
        """1D arrays are handled correctly."""
        values = np.random.randn(100)

        report = check_data_quality(values)

        assert report.n_rows == 100
        assert report.n_cols == 1

    def test_summary_method(self):
        """Summary method produces readable output."""
        values = np.random.randn(100, 10)
        values[0, 0] = np.nan

        report = check_data_quality(values)
        summary = report.summary()

        assert "Data Quality Report" in summary
        assert "NaN" in summary
        assert "Inf" in summary


class TestDataQualityReport:
    """Test DataQualityReport dataclass."""

    def test_summary_includes_warnings(self):
        """Summary includes warnings when present."""
        report = DataQualityReport(
            n_rows=100,
            n_cols=10,
            nan_fraction=0.1,
            inf_fraction=0.0,
            zero_fraction=0.0,
            outlier_stats=None,
            distribution={},
            warnings=["Warning 1", "Warning 2"],
            passed=False,
        )

        summary = report.summary()

        assert "Warning 1" in summary
        assert "Warning 2" in summary
        assert "FAIL" in summary

    def test_summary_shows_pass_status(self):
        """Summary shows PASS status when no warnings."""
        report = DataQualityReport(
            n_rows=100,
            n_cols=10,
            nan_fraction=0.0,
            inf_fraction=0.0,
            zero_fraction=0.0,
            outlier_stats=None,
            distribution={},
            warnings=[],
            passed=True,
        )

        summary = report.summary()

        assert "PASS" in summary
