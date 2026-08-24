"""Tests for validation.sanitize module.

Tests input sanitization including numeric arrays, DataFrames, and factor-specific policies.
"""
import numpy as np
import pytest

from factor_engine.validation.sanitize import (
    SanitizationResult,
    sanitize_dataframe,
    sanitize_factor_inputs,
    sanitize_numeric_array,
)


class TestSanitizeNumericArray:
    """Test numeric array sanitization."""

    def test_clean_data_unchanged(self):
        """Clean data passes through unchanged."""
        values = np.array([1.0, 2.0, 3.0, 4.0])

        result = sanitize_numeric_array(values)

        assert result.n_changes == 0
        np.testing.assert_array_equal(result.sanitized, values)

    def test_replace_inf_with_nan(self):
        """Inf values are replaced with NaN."""
        values = np.array([1.0, np.inf, 3.0, -np.inf])

        result = sanitize_numeric_array(values, replace_inf="nan")

        assert result.n_changes == 2
        assert result.changes["inf_replaced"] == 2
        assert np.isnan(result.sanitized[1])
        assert np.isnan(result.sanitized[3])

    def test_replace_inf_with_value(self):
        """Inf values are replaced with specific value."""
        values = np.array([1.0, np.inf, 3.0, -np.inf])

        result = sanitize_numeric_array(values, replace_inf=999.0)

        assert result.n_changes == 2
        assert result.sanitized[1] == 999.0
        assert result.sanitized[3] == 999.0

    def test_replace_nan_with_value(self):
        """NaN values are replaced with specific value."""
        values = np.array([1.0, np.nan, 3.0, np.nan])

        result = sanitize_numeric_array(values, replace_nan=0.0)

        assert result.n_changes == 2
        assert result.changes["nan_replaced"] == 2
        assert result.sanitized[1] == 0.0
        assert result.sanitized[3] == 0.0

    def test_clip_min_max(self):
        """Values are clipped to min/max bounds."""
        values = np.array([-10.0, 0.0, 5.0, 20.0])

        result = sanitize_numeric_array(values, clip_min=0.0, clip_max=10.0)

        assert result.n_changes == 2
        assert result.sanitized[0] == 0.0
        assert result.sanitized[3] == 10.0

    def test_winsorization(self):
        """Winsorization clips extreme values to quantiles."""
        values = np.concatenate([
            np.random.randn(100),
            np.array([100.0, -100.0])  # extreme values
        ])

        result = sanitize_numeric_array(
            values,
            winsorize_quantiles=(0.01, 0.99)
        )

        # Extremes should be winsorized
        assert result.n_changes > 0
        assert "winsorized" in result.changes

    def test_winsorization_invalid_quantiles(self):
        """Invalid winsorization quantiles raise error."""
        values = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="quantiles"):
            sanitize_numeric_array(values, winsorize_quantiles=(0.9, 0.1))

    def test_force_finite_fails_with_nan(self):
        """force_finite=True adds warning when NaN remains."""
        values = np.array([1.0, np.nan, 3.0])

        result = sanitize_numeric_array(values, force_finite=True)

        assert len(result.warnings) > 0
        assert any("non-finite" in w for w in result.warnings)

    def test_combined_sanitization(self):
        """Multiple sanitization steps work together."""
        values = np.array([np.inf, -np.inf, np.nan, 100.0, -100.0, 1.0, 2.0])

        result = sanitize_numeric_array(
            values,
            replace_inf="nan",
            replace_nan=0.0,
            clip_min=-10.0,
            clip_max=10.0,
        )

        # All inf -> nan -> 0, then clip extremes
        assert result.n_changes >= 5
        sanitized = result.sanitized
        assert np.all(np.isfinite(sanitized))
        assert np.all(sanitized >= -10.0)
        assert np.all(sanitized <= 10.0)

    def test_original_array_unchanged(self):
        """Original array is not modified (copy is made)."""
        values = np.array([1.0, np.inf, 3.0])
        original_copy = values.copy()

        result = sanitize_numeric_array(values, replace_inf="nan")

        # Original should be unchanged
        np.testing.assert_array_equal(values, original_copy)
        # Result should be different
        assert not np.array_equal(result.sanitized, values)


class TestSanitizeDataFrame:
    """Test DataFrame sanitization."""

    def test_requires_pandas(self):
        """Function requires pandas."""
        try:
            import pandas as pd
            pytest.skip("pandas is available")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="pandas"):
            sanitize_dataframe([[1, 2], [3, 4]])

    def test_numeric_columns_sanitized(self):
        """Numeric columns are sanitized."""
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame({
            "a": [1.0, np.inf, 3.0],
            "b": [4.0, 5.0, np.nan],
            "c": ["x", "y", "z"]  # non-numeric
        })

        result = sanitize_dataframe(df, replace_inf="nan")

        assert result.n_changes >= 1
        assert np.isnan(result.sanitized["a"].iloc[1])
        # Non-numeric column unchanged
        assert result.sanitized["c"].tolist() == ["x", "y", "z"]

    def test_specific_columns_only(self):
        """Only specified columns are sanitized."""
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame({
            "a": [1.0, np.inf, 3.0],
            "b": [4.0, np.inf, 6.0],
        })

        result = sanitize_dataframe(
            df,
            numeric_cols=["a"],
            replace_inf="nan"
        )

        # Only column 'a' sanitized
        assert np.isnan(result.sanitized["a"].iloc[1])
        # Column 'b' unchanged
        assert np.isinf(result.sanitized["b"].iloc[1])

    def test_unsorted_index_warning(self):
        """Unsorted index triggers warning."""
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0]},
            index=[2, 0, 1]  # unsorted
        )

        result = sanitize_dataframe(df, check_sorted_index=True)

        assert any("sorted" in w.lower() for w in result.warnings)

    def test_duplicated_index_warning(self):
        """Duplicated index triggers warning."""
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0]},
            index=[0, 0, 1]  # duplicated
        )

        result = sanitize_dataframe(df, check_duplicated_index=True)

        assert any("duplicated" in w.lower() for w in result.warnings)

    def test_coerce_dtypes(self):
        """Object columns are coerced to numeric when possible."""
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame({
            "a": ["1", "2", "3"],  # numeric strings
            "b": ["x", "y", "z"],  # non-numeric
        })

        result = sanitize_dataframe(df, coerce_dtypes=True)

        # Column 'a' should be coerced
        assert result.sanitized["a"].dtype.kind in ("i", "f")
        # Column 'b' becomes float with NaN when coercion fails (pd.to_numeric behavior)
        # The function attempted coercion but non-numeric strings became NaN
        assert result.sanitized["b"].dtype.kind in ("f", "O")


class TestSanitizeFactorInputs:
    """Test factor-specific sanitization."""

    def test_clean_data_unchanged(self):
        """Clean data passes through unchanged."""
        values = np.random.randn(100)

        result = sanitize_factor_inputs(values)

        assert result.n_changes == 0

    def test_replace_inf_policies(self):
        """Different inf replacement policies work."""
        values = np.array([1.0, np.inf, -np.inf, 2.0])

        # Policy: nan
        result_nan = sanitize_factor_inputs(values, replace_inf="nan")
        assert np.isnan(result_nan.sanitized[1])
        assert np.isnan(result_nan.sanitized[2])

        # Policy: clip
        result_clip = sanitize_factor_inputs(
            values, replace_inf="clip", max_abs_value=100.0
        )
        assert result_clip.sanitized[1] == 100.0
        assert result_clip.sanitized[2] == -100.0

        # Policy: zero
        result_zero = sanitize_factor_inputs(values, replace_inf="zero")
        assert result_zero.sanitized[1] == 0.0
        assert result_zero.sanitized[2] == 0.0

    def test_invalid_replace_inf_raises(self):
        """Invalid replace_inf policy raises error."""
        values = np.array([1.0, np.inf])

        with pytest.raises(ValueError, match="unknown.*policy"):
            sanitize_factor_inputs(values, replace_inf="invalid")

    def test_extreme_value_clipping(self):
        """Extreme values are clipped to max_abs_value."""
        values = np.array([1.0, 1e20, -1e20, 2.0])

        result = sanitize_factor_inputs(values, max_abs_value=1e10)

        assert result.n_changes >= 2
        assert abs(result.sanitized[1]) <= 1e10
        assert abs(result.sanitized[2]) <= 1e10

    def test_winsorization(self):
        """Winsorization is applied."""
        values = np.concatenate([
            np.random.randn(100),
            np.array([100.0, -100.0])
        ])

        result = sanitize_factor_inputs(
            values,
            winsorize=(0.01, 0.99)
        )

        assert result.n_changes > 0
        assert "winsorized" in result.changes

    def test_normalize_zscore(self):
        """Z-score normalization works."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = sanitize_factor_inputs(
            values,
            normalize=True,
            normalize_method="zscore"
        )

        normalized = result.sanitized
        # Z-score normalized data should have mean ≈ 0, std ≈ 1
        assert abs(np.mean(normalized)) < 0.01
        assert abs(np.std(normalized) - 1.0) < 0.01

    def test_normalize_minmax(self):
        """Min-max normalization works."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = sanitize_factor_inputs(
            values,
            normalize=True,
            normalize_method="minmax"
        )

        normalized = result.sanitized
        # Min-max normalized data should be in [0, 1]
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0

    def test_normalize_robust(self):
        """Robust normalization (median/IQR) works."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # outlier

        result = sanitize_factor_inputs(
            values,
            normalize=True,
            normalize_method="robust"
        )

        normalized = result.sanitized
        # Robust normalization is less affected by outliers
        assert np.isfinite(normalized).all()

    def test_normalize_constant_data_warns(self):
        """Normalization of constant data produces warning."""
        values = np.full(100, 5.0)

        result = sanitize_factor_inputs(
            values,
            normalize=True,
            normalize_method="zscore"
        )

        assert any("std=0" in w for w in result.warnings)

    def test_normalize_invalid_method_raises(self):
        """Invalid normalization method raises error."""
        values = np.random.randn(100)

        with pytest.raises(ValueError, match="unknown.*method"):
            sanitize_factor_inputs(
                values,
                normalize=True,
                normalize_method="invalid"
            )

    def test_strict_mode_enforces_finite(self):
        """Strict mode raises error if non-finite values remain."""
        values = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValueError, match="strict.*non-finite"):
            sanitize_factor_inputs(values, strict=True)

    def test_strict_mode_with_cleaning(self):
        """Strict mode fails when NaN remains after partial cleaning."""
        values = np.array([1.0, np.inf, np.nan, 3.0])

        # After replacing inf with clip, NaN still remains
        # strict mode should fail because NaN is not handled
        with pytest.raises(ValueError, match="strict.*non-finite"):
            sanitize_factor_inputs(
                values,
                strict=True,
                replace_inf="clip",
                max_abs_value=100.0,
            )

    def test_combined_pipeline(self):
        """Full sanitization pipeline works end-to-end."""
        # Create messy data
        values = np.concatenate([
            np.random.randn(100),
            np.array([np.inf, -np.inf, 1e20, -1e20])
        ])

        result = sanitize_factor_inputs(
            values,
            replace_inf="clip",
            max_abs_value=1e10,
            winsorize=(0.01, 0.99),
            normalize=True,
            normalize_method="zscore",
        )

        # Final result should be finite and normalized
        sanitized = result.sanitized
        assert np.isfinite(sanitized).all()
        assert abs(np.mean(sanitized)) < 1.0  # approximately normalized


class TestSanitizationResult:
    """Test SanitizationResult dataclass."""

    def test_summary_method(self):
        """Summary method produces readable output."""
        result = SanitizationResult(
            sanitized=np.array([1.0, 2.0]),
            n_changes=5,
            changes={"inf_replaced": 2, "nan_replaced": 3},
            warnings=["Warning 1"],
        )

        summary = result.summary()

        assert "5 changes" in summary
        assert "inf_replaced: 2" in summary
        assert "nan_replaced: 3" in summary
        assert "Warning 1" in summary

    def test_no_changes_summary(self):
        """Summary for no changes is concise."""
        result = SanitizationResult(
            sanitized=np.array([1.0, 2.0]),
            n_changes=0,
            changes={},
            warnings=[],
        )

        summary = result.summary()

        assert "0 changes" in summary
