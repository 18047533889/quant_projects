"""Tests for validation.contracts module.

Tests schema validation for FactorBatch, FeatureBundle, LabelBundle, and PredictionBatch.
"""
import numpy as np
import pytest
from datetime import date, datetime

# Conditionally import pydantic-dependent features
try:
    from validation.contracts import (
        FactorBatchSchema,
        FeatureBundleSchema,
        LabelBundleSchema,
        PredictionBatchSchema,
        ValidationConfig,
        validate_factor_batch,
        validate_feature_bundle,
        validate_label_bundle,
        validate_prediction_batch,
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="pydantic not installed")
class TestFactorBatchSchema:
    """Test FactorBatch schema validation."""

    def test_valid_factor_batch(self):
        """Valid factor batch passes all checks."""
        timestamps = ["2024-01-01", "2024-01-02", "2024-01-03"]
        instruments = ["000001.SZ", "000002.SZ", "600000.SH"]
        values = np.random.randn(3, 3)

        batch = validate_factor_batch(
            factor_name="test_factor",
            timestamps=timestamps,
            instruments=instruments,
            values=values,
        )

        assert batch.factor_name == "test_factor"
        assert len(batch.timestamps) == 3
        assert len(batch.instruments) == 3
        assert batch.values.shape == (3, 3)

    def test_factor_name_validation(self):
        """Factor name must be non-empty and not start with underscore."""
        timestamps = ["2024-01-01"]
        instruments = ["000001.SZ"]
        values = np.array([[1.0]])

        # Empty name
        with pytest.raises(ValueError, match="factor_name"):
            validate_factor_batch("", timestamps, instruments, values)

        # Starts with underscore
        with pytest.raises(ValueError, match="underscore"):
            validate_factor_batch("_private", timestamps, instruments, values)

    def test_timestamps_sorted(self):
        """Timestamps must be sorted."""
        timestamps = ["2024-01-03", "2024-01-01", "2024-01-02"]  # unsorted
        instruments = ["000001.SZ"]
        values = np.random.randn(3, 1)

        with pytest.raises(ValueError, match="sorted"):
            validate_factor_batch("test", timestamps, instruments, values)

    def test_timestamps_no_duplicates(self):
        """Timestamps must not have duplicates."""
        timestamps = ["2024-01-01", "2024-01-01", "2024-01-02"]
        instruments = ["000001.SZ"]
        values = np.random.randn(3, 1)

        with pytest.raises(ValueError, match="duplicates"):
            validate_factor_batch("test", timestamps, instruments, values)

    def test_instruments_sorted(self):
        """Instruments must be sorted."""
        timestamps = ["2024-01-01"]
        instruments = ["600000.SH", "000001.SZ"]  # unsorted
        values = np.random.randn(1, 2)

        with pytest.raises(ValueError, match="sorted"):
            validate_factor_batch("test", timestamps, instruments, values)

    def test_shape_validation(self):
        """Values shape must match timestamps × instruments."""
        timestamps = ["2024-01-01", "2024-01-02"]
        instruments = ["000001.SZ", "000002.SZ", "600000.SH"]
        values = np.random.randn(2, 2)  # wrong shape

        with pytest.raises(ValueError, match="shape"):
            validate_factor_batch("test", timestamps, instruments, values)

    def test_values_must_be_2d(self):
        """Values must be 2D array."""
        timestamps = ["2024-01-01"]
        instruments = ["000001.SZ"]
        values = np.array([1.0, 2.0])  # 1D

        with pytest.raises(ValueError, match="2D"):
            validate_factor_batch("test", timestamps, instruments, values)

    def test_values_must_be_numeric(self):
        """Values must be numeric dtype."""
        timestamps = ["2024-01-01"]
        instruments = ["000001.SZ"]
        values = np.array([["a"]])  # string

        with pytest.raises(ValueError, match="numeric"):
            validate_factor_batch("test", timestamps, instruments, values)

    def test_strict_mode_rejects_nan(self):
        """Strict mode rejects NaN values."""
        timestamps = ["2024-01-01", "2024-01-02"]
        instruments = ["000001.SZ"]
        values = np.array([[1.0], [np.nan]])

        config = ValidationConfig(strict=True, allow_nan=False)

        with pytest.raises(ValueError, match="NaN"):
            validate_factor_batch("test", timestamps, instruments, values, config=config)

    def test_strict_mode_rejects_inf(self):
        """Strict mode rejects inf values."""
        timestamps = ["2024-01-01"]
        instruments = ["000001.SZ"]
        values = np.array([[np.inf]])

        config = ValidationConfig(strict=True, allow_inf=False)

        with pytest.raises(ValueError, match="inf"):
            validate_factor_batch("test", timestamps, instruments, values, config=config)

    def test_high_nan_fraction_rejected(self):
        """High NaN fraction is rejected in strict mode."""
        timestamps = ["2024-01-01", "2024-01-02"]
        instruments = ["000001.SZ", "000002.SZ"]
        values = np.array([[1.0, np.nan], [np.nan, np.nan]])  # 75% NaN

        config = ValidationConfig(strict=True, allow_nan=True, max_nan_fraction=0.5)

        with pytest.raises(ValueError, match="fraction.*exceeds"):
            validate_factor_batch("test", timestamps, instruments, values, config=config)

    def test_metadata_optional(self):
        """Metadata is optional."""
        timestamps = ["2024-01-01"]
        instruments = ["000001.SZ"]
        values = np.array([[1.0]])

        batch = validate_factor_batch("test", timestamps, instruments, values)
        assert batch.metadata == {}

        batch_with_meta = validate_factor_batch(
            "test", timestamps, instruments, values,
            metadata={"source": "test", "version": 1}
        )
        assert batch_with_meta.metadata == {"source": "test", "version": 1}


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="pydantic not installed")
class TestFeatureBundleSchema:
    """Test FeatureBundle schema validation."""

    def test_valid_feature_bundle(self):
        """Valid feature bundle passes all checks."""
        bundle = validate_feature_bundle(
            canonical="momentum",
            operator_semantic_version="1.0.0",
            params={"window": 20},
            normalized_ast_hash="abcdef1234567890",
            source_snapshot="snapshot_v1",
        )

        assert bundle.canonical == "momentum"
        assert bundle.params == {"window": 20}

    def test_canonical_required(self):
        """Canonical must be non-empty."""
        with pytest.raises(ValueError, match="canonical"):
            validate_feature_bundle(
                canonical="",
                operator_semantic_version="1.0.0",
                params={},
                normalized_ast_hash="abcdef1234567890",
                source_snapshot="snapshot_v1",
            )

    def test_hash_minimum_length(self):
        """Hash fields must be at least 8 characters."""
        with pytest.raises(ValueError, match="hash field"):
            validate_feature_bundle(
                canonical="test",
                operator_semantic_version="1.0.0",
                params={},
                normalized_ast_hash="short",  # too short
                source_snapshot="snapshot_v1",
            )

    def test_values_alignment(self):
        """Values must align with row_ids."""
        values = np.array([1.0, 2.0, 3.0])
        row_ids = ["row1", "row2"]  # mismatched length

        with pytest.raises(ValueError, match="does not match"):
            validate_feature_bundle(
                canonical="test",
                operator_semantic_version="1.0.0",
                params={},
                normalized_ast_hash="abcdef1234567890",
                source_snapshot="snapshot_v1",
                values=values,
                row_ids=row_ids,
            )

    def test_values_must_be_numeric(self):
        """Values must be numeric."""
        values = np.array(["a", "b", "c"])

        with pytest.raises(ValueError, match="numeric"):
            validate_feature_bundle(
                canonical="test",
                operator_semantic_version="1.0.0",
                params={},
                normalized_ast_hash="abcdef1234567890",
                source_snapshot="snapshot_v1",
                values=values,
            )


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="pydantic not installed")
class TestLabelBundleSchema:
    """Test LabelBundle schema validation."""

    def test_valid_label_bundle(self):
        """Valid label bundle passes all checks."""
        bundle = validate_label_bundle(
            label_name="return_1d",
            horizon_bars=1,
            return_basis="vwap_to_vwap",
        )

        assert bundle.label_name == "return_1d"
        assert bundle.horizon_bars == 1
        assert bundle.return_basis == "vwap_to_vwap"

    def test_horizon_must_be_positive(self):
        """Horizon bars must be >= 1."""
        with pytest.raises(ValueError):
            validate_label_bundle(
                label_name="return",
                horizon_bars=0,  # invalid
            )

    def test_label_name_required(self):
        """Label name must be non-empty."""
        with pytest.raises(ValueError, match="label_name"):
            validate_label_bundle(
                label_name="",
                horizon_bars=1,
            )

    def test_values_alignment(self):
        """Values must align with row_ids."""
        values = np.array([0.01, 0.02])
        row_ids = ["r1", "r2", "r3"]  # mismatched

        with pytest.raises(ValueError, match="does not match"):
            validate_label_bundle(
                label_name="return",
                horizon_bars=1,
                values=values,
                row_ids=row_ids,
            )


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="pydantic not installed")
class TestPredictionBatchSchema:
    """Test PredictionBatch schema validation."""

    def test_valid_prediction_batch(self):
        """Valid prediction batch passes all checks."""
        values = np.array([0.1, 0.2, 0.3])
        row_ids = ["r1", "r2", "r3"]

        batch = validate_prediction_batch(values, row_ids)

        assert len(batch.values) == 3
        assert batch.row_ids == row_ids
        assert batch.status == "ok"

    def test_values_must_be_1d(self):
        """Prediction values must be 1D."""
        values = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2D
        row_ids = ["r1", "r2"]

        with pytest.raises(ValueError, match="1D"):
            validate_prediction_batch(values, row_ids)

    def test_values_alignment(self):
        """Values must align with row_ids."""
        values = np.array([0.1, 0.2])
        row_ids = ["r1", "r2", "r3"]

        with pytest.raises(ValueError, match="does not match"):
            validate_prediction_batch(values, row_ids)

    def test_timestamps_alignment(self):
        """Timestamps must align with row_ids."""
        values = np.array([0.1, 0.2])
        row_ids = ["r1", "r2"]
        timestamps = ["2024-01-01"]  # wrong length

        with pytest.raises(ValueError, match="does not match"):
            validate_prediction_batch(values, row_ids, timestamps=timestamps)

    def test_status_validation(self):
        """Status must be valid value."""
        values = np.array([0.1])
        row_ids = ["r1"]

        with pytest.raises(ValueError, match="status"):
            validate_prediction_batch(values, row_ids, status="invalid_status")

    def test_finite_check(self):
        """Non-finite values are rejected when allow_nan=False."""
        values = np.array([0.1, np.nan, 0.3])
        row_ids = ["r1", "r2", "r3"]

        with pytest.raises(ValueError, match="finite"):
            validate_prediction_batch(values, row_ids, allow_nan=False)

    def test_allow_nan_mode(self):
        """NaN values are allowed when allow_nan=True."""
        values = np.array([0.1, np.nan, 0.3])
        row_ids = ["r1", "r2", "r3"]

        batch = validate_prediction_batch(values, row_ids, allow_nan=True)
        assert len(batch.values) == 3


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="pydantic not installed")
class TestValidationConfig:
    """Test ValidationConfig dataclass."""

    def test_default_config(self):
        """Default config has sensible defaults."""
        config = ValidationConfig()

        assert config.strict is False
        assert config.allow_nan is True
        assert config.allow_inf is False
        assert config.max_nan_fraction == 0.5

    def test_strict_config(self):
        """Strict config can be created."""
        config = ValidationConfig(
            strict=True,
            allow_nan=False,
            allow_inf=False,
            max_nan_fraction=0.0,
        )

        assert config.strict is True
        assert config.allow_nan is False
