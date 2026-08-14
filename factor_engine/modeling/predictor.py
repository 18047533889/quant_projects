# -*- coding: utf-8 -*-
"""Frozen-scoring predictor with mandatory OOS prediction context."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modeling.artifact import ModelArtifact, PredictionContext
from modeling.dataset import PanelDataset

__all__ = ["Predictor", "predict_panel", "batch_predict"]


class Predictor:
    """Pure frozen scoring surface. Never fits."""

    def predict(
        self,
        artifact: ModelArtifact,
        X: np.ndarray | pd.DataFrame,
        context: PredictionContext,
    ) -> np.ndarray:
        if not isinstance(context, PredictionContext):
            raise TypeError("Predictor.predict requires PredictionContext")
        values = np.asarray(X, dtype=np.float64)
        original_fit = artifact.learner.fit
        called = {"fit": False}

        def _guard(*args: Any, **kwargs: Any) -> Any:
            called["fit"] = True
            raise AssertionError("learner.fit was invoked during predict")

        artifact.learner.fit = _guard  # type: ignore[assignment]
        try:
            output = artifact.predict(values, context=context)
        finally:
            artifact.learner.fit = original_fit
        assert not called["fit"]
        return output


def predict_panel(
    artifact: ModelArtifact,
    dataset: PanelDataset,
    context: PredictionContext,
) -> pd.Series:
    """Score panel rows using a row-aligned mandatory context."""
    values = dataset.frame[dataset.feature_cols].to_numpy(dtype=np.float64)
    predictions = Predictor().predict(artifact, values, context)
    return pd.Series(predictions, index=dataset.frame.index)


def batch_predict(
    artifact: ModelArtifact,
    batches: list[np.ndarray],
    contexts: list[PredictionContext],
) -> list[np.ndarray]:
    """Score batches with exactly one mandatory context per batch."""
    if len(batches) != len(contexts):
        raise ValueError("batches and contexts must have equal length")
    return [
        Predictor().predict(artifact, np.asarray(batch, dtype=np.float64), context)
        for batch, context in zip(batches, contexts)
    ]
