# -*- coding: utf-8 -*-
"""Frozen-scoring predictor (Model Layer Major Redesign taskbook §22 / §23).

Production scoring is PURE frozen scoring: ``artifact.predict`` applies the
frozen preprocessing and the frozen model.  It must never call ``fit``.  The
predictor installs an instrumentation guard (a flag-raising stub over
``artifact.learner.fit``) for the duration of each predict and asserts that it
was never invoked.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modeling.artifact import ModelArtifact
from modeling.dataset import PanelDataset

__all__ = ["Predictor", "predict_panel", "batch_predict"]


class Predictor:
    """Pure scoring surface.  Never fits."""

    def __init__(self) -> None:
        pass

    def predict(self, artifact: ModelArtifact, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Score through the frozen artifact, asserting ``fit`` is never called."""
        X = np.asarray(X, dtype=np.float64)
        original_fit = artifact.learner.fit
        called = {"fit": False}

        def _guard(*args: Any, **kwargs: Any) -> Any:
            called["fit"] = True
            raise AssertionError("learner.fit was invoked during predict (frozen scoring only)")

        artifact.learner.fit = _guard  # type: ignore[assignment]
        try:
            out = artifact.predict(X)
        finally:
            artifact.learner.fit = original_fit
        assert not called["fit"], "learner.fit was invoked during predict (frozen scoring only)"
        return out


def predict_panel(artifact: ModelArtifact, ds: PanelDataset) -> pd.Series:
    """Score every row of a panel (features only), aligned to ``ds.frame`` index."""
    X = ds.frame[ds.feature_cols].to_numpy(dtype=np.float64)
    pred = Predictor().predict(artifact, X)
    return pd.Series(pred, index=ds.frame.index)


def batch_predict(artifact: ModelArtifact, X_batches: list) -> list[np.ndarray]:
    """Predict each batch independently (parity with per-batch predict)."""
    return [Predictor().predict(artifact, np.asarray(b, dtype=np.float64)) for b in X_batches]
