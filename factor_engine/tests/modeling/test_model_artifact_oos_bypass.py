# -*- coding: utf-8 -*-
"""Dedicated ModelArtifact production OOS context and bypass tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.artifact import (
    FrozenPreprocessing,
    ModelArtifact,
    ModelArtifactManifest,
    PredictionContext,
)
from modeling.contracts import ApplicationWindow
from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec


SCHEMA_HASH = "features-v1"


class _SumLearner(BaseLearner):
    name = "oos_boundary_test"
    family = "linear"

    def fit(self, X, y, *, weights=None, aux=None):  # pragma: no cover
        raise AssertionError("fit must not run during prediction")

    def predict(self, frozen, X):
        return np.asarray(X, dtype=np.float64).sum(axis=1)


def _artifact() -> ModelArtifact:
    return ModelArtifact(
        manifest=ModelArtifactManifest(
            model_name="test_model", model_version="1.0", artifact_id="test-artifact",
            train_start="2020-01-01", train_end="2020-06-30",
            training_cutoff="2020-07-01", available_at="2020-07-02",
            activation_at="2020-07-03", feature_schema_hash=SCHEMA_HASH,
        ),
        learner=_SumLearner(LearnerSpec("oos_boundary_test", "linear")),
        frozen=FrozenModel("oos_boundary_test", "linear", params={}, metadata={}),
        preprocessing=FrozenPreprocessing([
            {"kind": "standardize", "mean": [1.0, 2.0], "scale": [2.0, 4.0]}
        ]),
    )


def _context(dates, *, start="2020-07-03", end="2020-07-31", asof="2020-07-31", schema=SCHEMA_HASH):
    return PredictionContext(
        application_window=ApplicationWindow(
            start=pd.Timestamp(start), end=pd.Timestamp(end) if end else None
        ),
        dates=dates, asof=pd.Timestamp(asof), feature_schema_hash=schema,
    )


def test_direct_predict_requires_explicit_context():
    with pytest.raises(TypeError, match="context"):
        _artifact().predict(np.ones((2, 2)))


def test_direct_predict_research_escape_hatch_is_explicit():
    actual = _artifact().predict(np.array([[3.0, 6.0], [5.0, 10.0]]), context=_context(["2020-07-04", "2020-07-05"]))
    np.testing.assert_allclose(actual, [2.0, 4.0])


def test_missing_window_dates_and_schema_are_rejected():
    artifact = _artifact()
    X = np.ones((2, 2))
    with pytest.raises(TypeError, match="PredictionContext"):
        artifact.predict_oos(X, context=None)
    with pytest.raises(TypeError, match="PredictionContext"):
        artifact.predict_oos(X, context=None)
    with pytest.raises(ValueError, match="feature schema mismatch"):
        artifact.predict(X, context=_context(["2020-07-04", "2020-07-05"], schema="wrong"))


def test_pre_cutoff_and_future_rows_are_rejected():
    artifact = _artifact()
    X = np.ones((2, 2))
    with pytest.raises(ValueError, match="violates application window"):
        artifact.predict(X, context=_context(["2020-07-02", "2020-07-04"]))
    with pytest.raises(ValueError, match="after prediction asof"):
        artifact.predict(X, context=_context(["2020-07-04", "2020-08-01"], end=None))


def test_frozen_transform_requires_and_validates_dates_with_window():
    preprocessing = FrozenPreprocessing([])
    X = np.ones((2, 2))
    window = ApplicationWindow(start=pd.Timestamp("2020-07-03"))
    with pytest.raises(ValueError, match="validation requires dates"):
        preprocessing.transform(X, application_window=window)
    with pytest.raises(ValueError, match="transform violates application window"):
        preprocessing.transform(X, application_window=window, dates=pd.to_datetime(["2020-07-02", "2020-07-04"]))


def test_valid_context_scores_after_activation():
    actual = _artifact().predict(
        np.array([[3.0, 6.0], [5.0, 10.0]]),
        context=_context(pd.to_datetime(["2020-07-04", "2020-07-05"])),
    )
    np.testing.assert_allclose(actual, [2.0, 4.0])
