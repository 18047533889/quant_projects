# -*- coding: utf-8 -*-
"""MODEL-P0-005: ApplicationWindow OOS safety tests.

This module tests that FittedTransform.transform() requires ApplicationWindow
for public OOS usage, preventing temporal leakage where transforms trained on
train+validation are accidentally applied to validation-period data.

Two APIs:
1. transform(X, application_window=None) - internal, can bypass for in-sample
2. ModelArtifact.predict_oos(X, application_window=...) - public, requires window
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.artifact import FrozenPreprocessing, ModelArtifact, ModelArtifactManifest
from modeling.contracts import ApplicationWindow


# --------------------------------------------------------------------------- #
# ApplicationWindow construction and validation
# --------------------------------------------------------------------------- #
def test_application_window_basic_construction():
    """Basic ApplicationWindow with start date."""
    window = ApplicationWindow(start="2020-01-01")
    assert window.start == "2020-01-01"
    assert window.end is None
    assert window.strict is True


def test_application_window_with_end():
    """ApplicationWindow with both start and end."""
    window = ApplicationWindow(start="2020-01-01", end="2020-12-31")
    assert window.start == "2020-01-01"
    assert window.end == "2020-12-31"


def test_application_window_non_strict():
    """ApplicationWindow with strict=False allows pre-start data."""
    window = ApplicationWindow(start="2020-01-01", strict=False)
    assert window.strict is False


def test_application_window_none_start_raises():
    """ApplicationWindow.start cannot be None."""
    with pytest.raises(ValueError, match="start cannot be None"):
        ApplicationWindow(start=None)


def test_application_window_start_after_end_raises():
    """ApplicationWindow.start must be <= end."""
    with pytest.raises(ValueError, match="start .* > end"):
        ApplicationWindow(start="2020-12-31", end="2020-01-01")


# --------------------------------------------------------------------------- #
# validate_dates: temporal boundary enforcement
# --------------------------------------------------------------------------- #
def test_application_window_validates_dates_in_range():
    """Dates within window pass validation."""
    window = ApplicationWindow(start="2020-01-01", end="2020-12-31")
    dates = pd.date_range("2020-06-01", periods=10, freq="D")

    valid, violations = window.validate_dates(dates)
    assert valid
    assert violations == []


def test_application_window_detects_dates_before_start():
    """Dates before start fail validation (strict mode)."""
    window = ApplicationWindow(start="2020-01-15", strict=True)
    dates = pd.date_range("2020-01-01", periods=20, freq="D")  # Some before 01-15

    valid, violations = window.validate_dates(dates)
    assert not valid
    assert len(violations) == 1
    assert "before application window start" in violations[0]
    assert "14 dates" in violations[0]  # 14 dates before 2020-01-15


def test_application_window_detects_dates_after_end():
    """Dates after end fail validation."""
    window = ApplicationWindow(start="2020-01-01", end="2020-01-31")
    dates = pd.date_range("2020-01-20", periods=20, freq="D")  # Some after 01-31

    valid, violations = window.validate_dates(dates)
    assert not valid
    assert len(violations) == 1
    assert "after application window end" in violations[0]


def test_application_window_non_strict_allows_before_start():
    """Non-strict mode allows dates before start."""
    window = ApplicationWindow(start="2020-01-15", strict=False)
    dates = pd.date_range("2020-01-01", periods=20, freq="D")

    valid, violations = window.validate_dates(dates)
    # Should only check end boundary (none in this case)
    assert valid
    assert violations == []


def test_application_window_validates_pandas_timestamps():
    """validate_dates works with pandas Timestamp."""
    window = ApplicationWindow(start=pd.Timestamp("2020-01-01"))
    dates = pd.date_range("2020-01-10", periods=5, freq="D")

    valid, violations = window.validate_dates(dates)
    assert valid


def test_application_window_validates_string_dates():
    """validate_dates works with string dates."""
    window = ApplicationWindow(start="2020-01-01", end="2020-12-31")
    dates = ["2020-06-01", "2020-06-02", "2020-06-03"]

    valid, violations = window.validate_dates(dates)
    assert valid


# --------------------------------------------------------------------------- #
# FrozenPreprocessing.transform with application_window
# --------------------------------------------------------------------------- #
def test_frozen_preprocessing_transform_accepts_window():
    """FrozenPreprocessing.transform accepts application_window parameter."""
    preproc = FrozenPreprocessing([])
    X = np.random.randn(10, 3)

    window = ApplicationWindow(start="2020-01-01")
    # Should not raise
    result = preproc.transform(X, application_window=window)
    assert result.shape == X.shape


def test_frozen_preprocessing_transform_without_window():
    """FrozenPreprocessing.transform works without window (internal use)."""
    preproc = FrozenPreprocessing([])
    X = np.random.randn(10, 3)

    # No window = internal/in-sample use
    result = preproc.transform(X, application_window=None)
    assert result.shape == X.shape


def test_frozen_preprocessing_transform_with_standardizer():
    """ApplicationWindow works with actual preprocessing steps."""
    mean = np.array([0.5, 1.0, 1.5])
    scale = np.array([1.0, 2.0, 3.0])
    preproc = FrozenPreprocessing([
        {"kind": "standardize", "mean": mean, "scale": scale}
    ])

    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    window = ApplicationWindow(start="2020-01-01")

    result = preproc.transform(X, application_window=window)
    # Verify standardization happened
    expected = (X - mean) / scale
    np.testing.assert_allclose(result, expected)


# --------------------------------------------------------------------------- #
# ModelArtifact.predict_oos requires ApplicationWindow
# --------------------------------------------------------------------------- #
def _make_test_artifact() -> ModelArtifact:
    """Build a minimal test artifact."""
    from modeling.learners.pcr import PCRLearner
    from modeling.learners.base import LearnerSpec, FrozenModel

    spec = LearnerSpec(learner_name="pcr", family="linear", hyperparams={"n_components": 2})
    learner = PCRLearner(spec)

    # Simple frozen model (identity-like)
    frozen = FrozenModel(
        learner_name="pcr",
        family="linear",
        params={"coef": np.array([1.0, 1.0]), "intercept": 0.0},
        metadata={},
    )

    preproc = FrozenPreprocessing([])

    manifest = ModelArtifactManifest(
        model_name="test_model",
        model_version="1.0",
        artifact_id="test-artifact-001",
        train_start="2020-01-01",
        train_end="2020-06-30",
        available_at="2020-07-01",
    )

    return ModelArtifact(manifest, learner, frozen, preproc)


def test_model_artifact_predict_oos_requires_window():
    """predict_oos requires ApplicationWindow (cannot be None)."""
    artifact = _make_test_artifact()
    X = np.random.randn(10, 2)

    with pytest.raises(ValueError, match="predict_oos requires application_window"):
        artifact.predict_oos(X, application_window=None)


def test_model_artifact_predict_oos_requires_correct_type():
    """predict_oos requires ApplicationWindow type."""
    artifact = _make_test_artifact()
    X = np.random.randn(10, 2)

    with pytest.raises(TypeError, match="must be ApplicationWindow"):
        artifact.predict_oos(X, application_window="2020-07-01")  # Wrong type


def test_model_artifact_predict_oos_with_valid_window():
    """predict_oos works with valid ApplicationWindow."""
    artifact = _make_test_artifact()
    X = np.random.randn(10, 2)

    window = ApplicationWindow(start="2020-07-01")
    # Should not raise
    predictions = artifact.predict_oos(X, application_window=window)
    assert predictions.shape[0] == 10


def test_model_artifact_predict_oos_validates_dates():
    """predict_oos validates dates against ApplicationWindow."""
    artifact = _make_test_artifact()
    X = np.random.randn(10, 2)

    # Window starts 2020-07-01 (after training cutoff)
    window = ApplicationWindow(start="2020-07-01", strict=True)

    # Dates before window start should fail
    dates = pd.date_range("2020-06-20", periods=10, freq="D")  # Before 07-01

    with pytest.raises(ValueError, match="violates application window"):
        artifact.predict_oos(X, application_window=window, dates=dates)


def test_model_artifact_predict_oos_allows_valid_dates():
    """predict_oos succeeds when dates are within ApplicationWindow."""
    artifact = _make_test_artifact()
    X = np.random.randn(10, 2)

    window = ApplicationWindow(start="2020-07-01")
    dates = pd.date_range("2020-07-10", periods=10, freq="D")  # After window start

    # Should not raise
    predictions = artifact.predict_oos(X, application_window=window, dates=dates)
    assert predictions.shape[0] == 10


def test_model_artifact_predict_oos_no_dates_skips_validation():
    """predict_oos without dates parameter skips date validation."""
    artifact = _make_test_artifact()
    X = np.random.randn(10, 2)

    window = ApplicationWindow(start="2020-07-01")
    # No dates provided = no date validation (caller's responsibility)
    predictions = artifact.predict_oos(X, application_window=window, dates=None)
    assert predictions.shape[0] == 10


# --------------------------------------------------------------------------- #
# predict() vs predict_oos() distinction
# --------------------------------------------------------------------------- #
def test_model_artifact_predict_for_in_sample():
    """predict() is for in-sample only (no window required)."""
    artifact = _make_test_artifact()
    X = np.random.randn(10, 2)

    # predict() works without ApplicationWindow (in-sample use)
    predictions = artifact.predict(X)
    assert predictions.shape[0] == 10


def test_model_artifact_predict_does_not_require_window():
    """predict() does not enforce ApplicationWindow (legacy API)."""
    artifact = _make_test_artifact()
    X = np.random.randn(10, 2)

    # Should work without any window
    predictions = artifact.predict(X)
    assert predictions.shape[0] == 10


# --------------------------------------------------------------------------- #
# Negative tests: try to leak, verify rejection
# --------------------------------------------------------------------------- #
def test_cannot_apply_oos_transform_to_training_period():
    """Attempting to apply OOS transform to training-period data should fail."""
    artifact = _make_test_artifact()
    X = np.random.randn(10, 2)

    # Training ended 2020-06-30, available from 2020-07-01
    # Try to predict on training-period dates (should fail with strict window)
    window = ApplicationWindow(start="2020-07-01", strict=True)
    training_dates = pd.date_range("2020-06-20", periods=10, freq="D")

    with pytest.raises(ValueError, match="violates application window"):
        artifact.predict_oos(X, application_window=window, dates=training_dates)


def test_cannot_apply_oos_transform_before_availability():
    """Cannot apply transform before artifact is available."""
    artifact = _make_test_artifact()  # available_at="2020-07-01"
    X = np.random.randn(10, 2)

    # Window must respect artifact availability
    # Using dates right at boundary
    window = ApplicationWindow(start="2020-07-01")
    boundary_dates = pd.date_range("2020-06-30", periods=5, freq="D")  # Includes 06-30

    with pytest.raises(ValueError, match="violates application window"):
        artifact.predict_oos(X, application_window=window, dates=boundary_dates)


def test_oos_window_prevents_train_val_refit_leakage():
    """ApplicationWindow prevents applying train+val refit to val period.

    Scenario: Artifact trained on train only, then refit on train+val.
    The refit artifact cannot be applied to validation-period dates.
    """
    # Artifact refit on train (2020-01-01 to 2020-06-30) + val (2020-07-01 to 2020-08-31)
    manifest = ModelArtifactManifest(
        model_name="refit_model",
        model_version="1.0",
        artifact_id="refit-001",
        train_start="2020-01-01",
        train_end="2020-06-30",
        validation_start="2020-07-01",
        validation_end="2020-08-31",
        final_fit_end="2020-08-31",  # Refit used validation
        refit_used_validation=True,
        available_at="2020-09-01",  # Available after final_fit_end
    )

    from modeling.learners.pcr import PCRLearner
    from modeling.learners.base import LearnerSpec, FrozenModel

    spec = LearnerSpec(learner_name="pcr", family="linear", hyperparams={"n_components": 2})
    learner = PCRLearner(spec)
    frozen = FrozenModel(
        learner_name="pcr",
        family="linear",
        params={"coef": np.array([1.0, 1.0]), "intercept": 0.0},
        metadata={},
    )
    preproc = FrozenPreprocessing([])
    artifact = ModelArtifact(manifest, learner, frozen, preproc)

    X = np.random.randn(10, 2)

    # Window correctly enforces available_at (2020-09-01)
    window = ApplicationWindow(start="2020-09-01", strict=True)

    # Try to apply to validation period (2020-07-01 to 2020-08-31) - should fail
    val_dates = pd.date_range("2020-07-15", periods=10, freq="D")

    with pytest.raises(ValueError, match="violates application window"):
        artifact.predict_oos(X, application_window=window, dates=val_dates)

    # But applying to test period (after 2020-09-01) should work
    test_dates = pd.date_range("2020-09-05", periods=10, freq="D")
    predictions = artifact.predict_oos(X, application_window=window, dates=test_dates)
    assert predictions.shape[0] == 10


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_application_window_with_numpy_dates():
    """ApplicationWindow works with numpy datetime64."""
    window = ApplicationWindow(start=np.datetime64("2020-01-01"))
    dates = np.array(["2020-01-10", "2020-01-11"], dtype="datetime64[D]")

    valid, violations = window.validate_dates(dates)
    assert valid


def test_application_window_unbounded_end():
    """ApplicationWindow with no end allows arbitrarily late dates."""
    window = ApplicationWindow(start="2020-01-01", end=None)
    dates = pd.date_range("2025-01-01", periods=10, freq="D")  # Far future

    valid, violations = window.validate_dates(dates)
    assert valid


def test_application_window_single_date():
    """ApplicationWindow can validate a single date."""
    window = ApplicationWindow(start="2020-01-01", end="2020-12-31")
    date = [pd.Timestamp("2020-06-15")]

    valid, violations = window.validate_dates(date)
    assert valid
