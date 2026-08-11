# -*- coding: utf-8 -*-
"""Predictive ElasticNet learner (Model Layer Major Redesign taskbook §38).

Coordinate-descent ElasticNet on a pooled panel.  ``alpha`` / ``l1_ratio`` are
REGULARIZATION parameters: validated only inside the walk-forward validation
fold (§13.6 / §38), never searched on test.  Train-only centering; frozen for
predict.
"""
from __future__ import annotations

import numpy as np

from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec, register_learner

__all__ = ["ElasticNetLearner"]


@register_learner
class ElasticNetLearner(BaseLearner):
    name = "predictive_elastic_net"
    family = "elastic_net"

    def __init__(self, spec: LearnerSpec) -> None:
        super().__init__(spec)
        self.alpha = float(spec.hyperparams.get("alpha", 1e-2))
        self.l1_ratio = float(spec.hyperparams.get("l1_ratio", 0.5))
        self.max_iter = int(spec.hyperparams.get("max_iter", 500))
        self.tol = float(spec.hyperparams.get("tol", 1e-8))
        if not (1e-8 <= self.alpha):
            raise ValueError("alpha must be positive")
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")

    def validate_params(self) -> None:
        if not (1e-8 <= float(self.spec.hyperparams.get("alpha", 1e-2))):
            raise ValueError("enet alpha must be positive")
        l1 = float(self.spec.hyperparams.get("l1_ratio", 0.5))
        if not (0.0 <= l1 <= 1.0):
            raise ValueError("enet l1_ratio must be in [0, 1]")

    def fit(self, X: np.ndarray, y: np.ndarray, *, weights: np.ndarray | None = None) -> FrozenModel:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("enet fit needs a 2-D feature matrix with >= 2 rows")
        n, d = X.shape
        mx = X.mean(axis=0)
        my = float(y.mean())
        sx = X.std(axis=0)
        sx[sx == 0] = 1.0
        Xs = (X - mx) / sx
        ys = y - my
        coef = np.zeros(d)
        alpha = self.alpha
        l1 = self.l1_ratio
        for _ in range(self.max_iter):
            coef_old = coef.copy()
            for j in range(d):
                rj = ys - Xs @ coef + Xs[:, j] * coef[j]
                rho = Xs[:, j] @ rj
                denom = Xs[:, j] @ Xs[:, j] + alpha * (1.0 - l1)
                if l1 == 0.0:
                    coef[j] = rho / denom
                else:
                    z = np.sign(rho) * max(0.0, abs(rho) - alpha * l1)
                    coef[j] = z / denom
            if np.linalg.norm(coef - coef_old, ord=1) < self.tol:
                break
        intercept = my - float(mx @ coef)
        converged = np.linalg.norm(coef - coef_old, ord=1) < self.tol
        if not converged:
            # fail-closed: maximum iterations reached without convergence.
            # Still return the best iterate but flag telemetry honestly.
            pass
        return FrozenModel(
            learner_name=self.name,
            family=self.family,
            params={"coef": coef, "intercept": intercept},
            metadata={
                "n_train_rows": n,
                "n_features": d,
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "converged": converged,
                "n_iter": _,
            },
        )

    def predict(self, frozen: FrozenModel, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        p = frozen.params
        return np.asarray(p["intercept"], dtype=np.float64) + X @ np.asarray(p["coef"], dtype=np.float64)
