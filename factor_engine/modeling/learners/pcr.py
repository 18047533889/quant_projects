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
    sample_contract_family = "linear"

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

    def effective_parameter_count(
        self,
        hyperparams: dict[str, Any] | None = None,
        n_features: int = 0,
        *,
        n_rows: int | None = None,
        **extra: Any,
    ) -> int:
        """PCR complexity: ``k`` OLS betas on the ``k`` retained PCA scores + 1
        intercept.

        The PCA projection itself is unsupervised — it is not a free parameter
        of the y-mapping and is reported separately as ``pca_complexity`` in the
        fit metadata (``d * k``).  Only the supervised OLS on the scores counts.
        """
        k = int((hyperparams or {}).get("n_components", self.n_components))
        k = max(1, min(k, max(1, int(n_features))))
        return k + 1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        aux: dict[str, np.ndarray | None] | None = None,
    ) -> FrozenModel:
        if weights is not None:
            raise NotImplementedError("PCR does not support sample weights")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("pcr fit needs a 2-D feature matrix with >= 2 rows")
        n, d = X.shape
        # Fail closed on a requested component count the data cannot support —
        # n_components=5 / 10 / 20 must NEVER silently become the same model.
        k_max = min(n, d)
        if self.n_components > k_max:
            raise ValueError(
                f"pcr n_components={self.n_components} exceeds min(n_rows={n}, "
                f"n_features={d})={k_max} — reject (fail closed), no silent downgrade"
            )
        k = self.n_components
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
                # PCA projection complexity — d×k loadings.  Unsupervised, so it
                # is NOT counted in effective_parameter_count, but recorded here
                # for the observability hard gates.
                "pca_complexity": int(d * k),
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
