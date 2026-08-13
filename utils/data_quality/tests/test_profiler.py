"""
Comprehensive tests for data profiler module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ..profiler import DataProfiler, ProfileResult, ColumnProfile


class TestColumnProfile:
    """Tests for ColumnProfile dataclass."""

    def test_column_profile_creation(self):
        profile = ColumnProfile(
            name="test_col",
            dtype="float64",
            count=100,
            null_count=10,
            null_percentage=10.0,
            unique_count=50,
            mean=5.5,
            std=2.3,
            min=1.0,
            max=10.0,
        )

        assert profile.name == "test_col"
        assert profile.dtype == "float64"
        assert profile.count == 100
        assert profile.null_percentage == 10.0

    def test_column_profile_to_dict(self):
        profile = ColumnProfile(
            name="test_col",
            dtype="int64",
            count=100,
            null_count=5,
            null_percentage=5.0,
            unique_count=95,
        )

        result = profile.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "test_col"
        assert result["count"] == 100
        assert result["null_percentage"] == 5.0


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_profile_result_creation(self):
        col_profile = ColumnProfile(
            name="col1",
            dtype="float64",
            count=100,
            null_count=0,
            null_percentage=0.0,
            unique_count=100,
        )

        result = ProfileResult(
            timestamp=datetime.now(),
            row_count=100,
            column_count=1,
            memory_usage_mb=0.5,
            column_profiles={"col1": col_profile},
        )

        assert result.row_count == 100
        assert result.column_count == 1
        assert "col1" in result.column_profiles

    def test_profile_result_to_dict(self):
        col_profile = ColumnProfile(
            name="col1",
            dtype="float64",
            count=100,
            null_count=0,
            null_percentage=0.0,
            unique_count=100,
        )

        result = ProfileResult(
            timestamp=datetime.now(),
            row_count=100,
            column_count=1,
            memory_usage_mb=0.5,
            column_profiles={"col1": col_profile},
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["row_count"] == 100
        assert "column_profiles" in result_dict


class TestDataProfiler:
    """Tests for DataProfiler class."""

    def test_profiler_initialization(self):
        profiler = DataProfiler(
            compute_correlations=True,
            correlation_threshold=0.9,
            max_categories=10,
        )

        assert profiler.compute_correlations is True
        assert profiler.correlation_threshold == 0.9
        assert profiler.max_categories == 10

    def test_profile_simple_dataframe(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1.1, 2.2, 3.3, 4.4, 5.5],
            "c": ["x", "y", "z", "x", "y"],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        assert result.row_count == 5
        assert result.column_count == 3
        assert len(result.column_profiles) == 3
        assert "a" in result.column_profiles
        assert "b" in result.column_profiles
        assert "c" in result.column_profiles

    def test_profile_with_missing_values(self):
        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5],
            "b": [1.1, np.nan, np.nan, 4.4, 5.5],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        assert result.column_profiles["a"].null_count == 1
        assert result.column_profiles["a"].null_percentage == 20.0
        assert result.column_profiles["b"].null_count == 2
        assert result.column_profiles["b"].null_percentage == 40.0

    def test_profile_numeric_statistics(self):
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        profile = result.column_profiles["values"]

        assert profile.mean == 5.5
        assert profile.median == 5.5
        assert profile.min == 1.0
        assert profile.max == 10.0
        assert profile.q25 == 3.25
        assert profile.q75 == 7.75

    def test_profile_with_infinite_values(self):
        df = pd.DataFrame({
            "a": [1, 2, np.inf, 4, 5],
            "b": [1.1, 2.2, -np.inf, 4.4, 5.5],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        assert result.column_profiles["a"].inf_count == 1
        assert result.column_profiles["b"].inf_count == 1
        assert result.column_profiles["b"].negative_inf_count == 1

    def test_profile_with_zero_count(self):
        df = pd.DataFrame({
            "values": [0, 1, 0, 2, 0, 3],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        assert result.column_profiles["values"].zero_count == 3

    def test_profile_categorical_data(self):
        df = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B", "A"],
        })

        profiler = DataProfiler(max_categories=10)
        result = profiler.profile(df)

        profile = result.column_profiles["category"]

        assert profile.unique_count == 3
        assert profile.most_common is not None
        assert len(profile.most_common) == 3
        assert profile.most_common[0] == ("A", 3)

    def test_profile_correlations(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 4, 3, 2, 1],
        })

        profiler = DataProfiler(compute_correlations=True)
        result = profiler.profile(df)

        assert result.correlations is not None
        assert result.correlations.shape == (3, 3)

        # a and b should be perfectly correlated
        assert abs(result.correlations.loc["a", "b"] - 1.0) < 0.01

    def test_profile_without_correlations(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
        })

        profiler = DataProfiler(compute_correlations=False)
        result = profiler.profile(df)

        assert result.correlations is None

    def test_profile_coverage_matrix(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, np.nan, 3, np.nan, 5],
            "c": [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        assert result.coverage_matrix is not None
        assert len(result.coverage_matrix) == 3

        # Column a should have 100% coverage
        a_coverage = result.coverage_matrix[result.coverage_matrix["column"] == "a"]["coverage_pct"].iloc[0]
        assert a_coverage == 100.0

        # Column b should have 60% coverage
        b_coverage = result.coverage_matrix[result.coverage_matrix["column"] == "b"]["coverage_pct"].iloc[0]
        assert b_coverage == 60.0

        # Column c should have 0% coverage
        c_coverage = result.coverage_matrix[result.coverage_matrix["column"] == "c"]["coverage_pct"].iloc[0]
        assert c_coverage == 0.0

    def test_profile_time_series_detection(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": range(10),
        })

        profiler = DataProfiler(detect_time_series=True)
        result = profiler.profile(df)

        assert result.date_column == "date"
        assert result.date_range is not None
        assert result.date_range[0] == dates[0]
        assert result.date_range[1] == dates[-1]

    def test_profile_with_explicit_date_column(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "trading_date": dates,
            "value": range(10),
        })

        profiler = DataProfiler()
        result = profiler.profile(df, date_column="trading_date")

        assert result.date_column == "trading_date"
        assert result.date_range is not None

    def test_profile_with_date_gaps(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"])
        df = pd.DataFrame({
            "date": dates,
            "value": [1, 2, 3, 4],
        })

        profiler = DataProfiler()
        result = profiler.profile(df, date_column="date")

        assert result.date_gaps is not None
        assert len(result.date_gaps) > 0

    def test_profile_panel_data(self):
        df = pd.DataFrame({
            "entity": ["A", "A", "B", "B", "C", "C"],
            "date": pd.date_range("2024-01-01", periods=6),
            "value": [1, 2, 3, 4, 5, 6],
        })

        profiler = DataProfiler()
        result = profiler.profile(df, entity_column="entity")

        assert result.entity_column == "entity"
        assert result.entity_count == 3

    def test_get_high_correlations(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # Perfect correlation with a
            "c": [1, 1, 2, 2, 3],   # Low correlation
        })

        profiler = DataProfiler(compute_correlations=True, correlation_threshold=0.95)
        result = profiler.profile(df)

        high_corr = profiler.get_high_correlations(result)

        assert len(high_corr) >= 1
        assert any(abs(corr) > 0.95 for _, _, corr in high_corr)

    def test_get_low_coverage_columns(self):
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "bad": [1, np.nan, np.nan, np.nan, np.nan],
            "medium": [1, 2, np.nan, np.nan, 5],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        low_coverage = profiler.get_low_coverage_columns(result, threshold=50.0)

        assert len(low_coverage) >= 1
        assert any(col == "bad" for col, _ in low_coverage)

    def test_profile_empty_dataframe_raises_error(self):
        df = pd.DataFrame()

        profiler = DataProfiler()

        with pytest.raises(ValueError, match="Cannot profile empty DataFrame"):
            profiler.profile(df)

    def test_profile_single_row(self):
        df = pd.DataFrame({
            "a": [1],
            "b": [2],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        assert result.row_count == 1
        assert result.column_count == 2

    def test_profile_single_column(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        assert result.row_count == 5
        assert result.column_count == 1

    def test_profile_large_unique_categorical(self):
        # Test with categorical data exceeding max_categories
        df = pd.DataFrame({
            "category": [f"cat_{i}" for i in range(100)],
        })

        profiler = DataProfiler(max_categories=20)
        result = profiler.profile(df)

        profile = result.column_profiles["category"]

        # Should not compute most_common for too many categories
        assert profile.most_common is None

    def test_profile_skewness_and_kurtosis(self):
        # Positively skewed data
        df = pd.DataFrame({
            "skewed": [1, 1, 1, 1, 2, 3, 10, 20, 30],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        profile = result.column_profiles["skewed"]

        assert profile.skewness is not None
        assert profile.kurtosis is not None
        assert profile.skewness > 0  # Should be positively skewed

    def test_profile_mixed_types(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "d", "e"],
            "bool_col": [True, False, True, False, True],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        assert result.column_count == 4
        assert result.column_profiles["int_col"].mean is not None
        assert result.column_profiles["float_col"].mean is not None
        assert result.column_profiles["str_col"].mean is None
        assert result.column_profiles["bool_col"].mean is None
