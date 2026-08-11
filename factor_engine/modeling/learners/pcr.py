# -*- coding: utf-8 -*-
"""Predictive PCR learner (Model Layer Major Redesign taskbook §36).

Pooled-panel, multi-year, artifact-backed PCR:

* train-only center + SVD PCA (never recomputed at predict time);
* validation selects ``n_components`` (the trainer layer does this);
* production scoring only projects the frozen PCA and applies the frozen OLS.

This REPLACES the per-stock short-window legacy local variant as the formal
predictive PCR definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec, register_learner

__all__ = ["PCRLearner"]


@register_learner
class PCRLearner(BaseLearner):
    name = "predictive_pcr"
    family = "pcr"

    def __init__(self, spec: LearnerSpec) -> None:
        super().__init__(spec)
        self.n_components = int(spec.hyperparams.get("n_components", 5))
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")

    def required_feature_count(self) -> int:
        return 1

    def validate_params(self) -> None:
        nc = int(self.spec.hyperparams.get("n_components", 5))
        if nc < 1:
            raise ValueError("pcr n_components must be >= 1")

    def fit(self, X: np.ndarray, y: np.ndarray, *, weights: np.ndarray | None = None) -> FrozenModel:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("pcr fit needs a 2-D feature matrix with >= 2 rows")
        n, d = X.shape
        k = min(self.n_components, d)
        # train-only center
        center = X.mean(axis=0)
        Xc = X - center
        # SVD -> scores
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        scores = U[:, :k] * S[:k]
        # OLS on scores (with intercept)
        design = np.column_stack([np.ones(n), scores])
        coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercept, beta_pca = coeff[0], coeff[1:]
        variance_explained = float((S[:k] ** 2).sum() / max(1e-12, (S ** 2).sum()))
        return FrozenModel(
            learner_name=self.name,
            family=self.family,
            params={
                "center": center,
                "components": Vt[:k],
                "beta_pca": beta_pca,
                "intercept": intercept,
                "n_components": k,
            },
            metadata={
                "n_train_rows": n,
                "n_features": d,
                "n_components_used": k,
                "variance_explained": variance_explained,
            },
        )

    def predict(self, frozen: FrozenModel, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        p = frozen.params
        center = np.asarray(p["center"], dtype=np.float64)
        components = np.asarray(p["components"], dtype=np.float64)
        Xc = X - center
        scores = Xc @ components.T
        return np.asarray(p["intercept"], dtype=np.float64) + scores @ np.asarray(p["beta_pca"], dtype=np.float64)
