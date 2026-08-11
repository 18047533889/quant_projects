# -*- coding: utf-8 -*-
"""Predictive ElasticNet learner (Model Layer Major Redesign taskbook §38).

Coordinate-descent ElasticNet on a pooled panel.  ``alpha`` / ``l1_ratio`` are
REGULARIZATION parameters: validated only inside the walk-forward validation
fold (§13.6 / §38), never searched on test.  Train-only centering; frozen for
predict.

Math contract (three guarantees the previous implementation violated):

1. **Objective is sample-count invariant** — the descent solves the *per-sample
   normalized* objective::

        min_b  1/(2N) ||y_s - X_s b||_2^2
             + alpha * l1_ratio     * ||b||_1
             + alpha * (1-l1_ratio) * 0.5 * ||b||_2^2

   so ``alpha`` means the same regularization strength whether N=10,000 or
   N=1,000,000 (the old code omitted the 1/N factor and alpha silently shrank
   with growing panels).

2. **Raw-space prediction** — the model is fit on standardized ``(X-mx)/sx``;
   the frozen artifact restores the coefficient to raw feature space
   (``beta_raw = beta_std / sx``, ``alpha_raw = my - mx @ beta_raw``) so
   ``predict`` consumes raw ``X`` and never mixes the two spaces.

3. **Non-convergence fails closed** — running out of ``max_iter`` raises
   :class:`ModelConvergenceError`; a partially-optimized iterate is never
   silently returned as a frozen model.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from modeling.learners.base import (
    BaseLearner,
    FrozenModel,
    LearnerSpec,
    ModelConvergenceError,
    register_learner,
)

__all__ = ["ElasticNetLearner", "ModelConvergenceError"]


@register_learner
class ElasticNetLearner(BaseLearner):
    name = "predictive_elastic_net"
    family = "elastic_net"
    sample_contract_family = "linear"

    def __init__(self, spec: LearnerSpec) -> None:
        super().__init__(spec)
        self.alpha = float(spec.hyperparams.get("alpha", 1e-2))
        self.l1_ratio = float(spec.hyperparams.get("l1_ratio", 0.5))
        self.max_iter = int(spec.hyperparams.get("max_iter", 1000))
        self.tol = float(spec.hyperparams.get("tol", 1e-7))
        if not (1e-8 <= self.alpha):
            raise ValueError("alpha must be positive")
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

    def validate_params(self) -> None:
        if not (1e-8 <= float(self.spec.hyperparams.get("alpha", 1e-2))):
            raise ValueError("enet alpha must be positive")
        l1 = float(self.spec.hyperparams.get("l1_ratio", 0.5))
        if not (0.0 <= l1 <= 1.0):
            raise ValueError("enet l1_ratio must be in [0, 1]")

    def effective_parameter_count(
        self,
        hyperparams: dict[str, Any] | None = None,
        n_features: int = 0,
        *,
        n_rows: int | None = None,
        **extra: Any,
    ) -> int:
        """ElasticNet complexity: ACTIVE (non-zero) coefficients + intercept.

        When the frozen model's ``coef`` (or its ``n_active`` count) is passed
        via ``extra`` the count is the actual support size + 1 — a sparse l1
        solution counts only its non-zero coefficients.  Pre-fit (no ``coef``)
        we fall back to the worst case ``n_features + 1``; the two
        regularization dials alpha / l1_ratio are never counted as free
        parameters.
        """
        coef = extra.get("coef")
        if coef is not None:
            return int(np.count_nonzero(np.asarray(coef))) + 1
        if extra.get("n_active") is not None:
            return int(extra["n_active"]) + 1
        return max(1, int(n_features) + 1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        aux: dict[str, np.ndarray | None] | None = None,
    ) -> FrozenModel:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("enet fit needs a 2-D feature matrix with >= 2 rows")
        n, d = X.shape
        if weights is not None:
            raise ValueError("enet learner does not support per-row weights (fail closed)")
        mx = X.mean(axis=0)
        my = float(y.mean())
        sx = X.std(axis=0)
        sx[sx == 0] = 1.0
        Xs = (X - mx) / sx
        ys = y - my

        # Objective (sample-count invariant): 1/(2N)||ys - Xs b||^2
        #   + alpha*l1*|b|_1 + 0.5*alpha*(1-l1)*||b||^2.
        alpha = self.alpha
        l1 = self.l1_ratio
        coef = np.zeros(d)
        tol = self.tol
        # (1/N) column norms of the standardized design — the L2 denominator of
        # the sample-count-invariant coordinate descent.
        diag = np.einsum("ij,ij->j", Xs, Xs) / n
        n_iter = 0
        converged = False
        for n_iter in range(1, self.max_iter + 1):
            coef_old = coef.copy()
            for j in range(d):
                rj = ys - Xs @ coef + Xs[:, j] * coef[j]
                rho = (Xs[:, j] @ rj) / n
                denom = diag[j] + alpha * (1.0 - l1)
                if l1 == 0.0:
                    coef[j] = rho / denom
                else:
                    z = np.sign(rho) * max(0.0, abs(rho) - alpha * l1)
                    coef[j] = z / denom
            if np.linalg.norm(coef - coef_old, ord=1) < tol * max(1.0, float(np.linalg.norm(coef, ord=1))):
                converged = True
                break

        if not converged:
            # Fail closed — a partially-optimized iterate is NEVER a model.
            raise ModelConvergenceError(
                f"enet did not converge within max_iter={self.max_iter} "
                f"(alpha={alpha}, l1_ratio={l1}, n={n}, d={d})"
            )

        # Restore the coefficient to RAW feature space so predict consumes raw X:
        #   y = my + (X - mx)/sx @ coef_std
        #     = (my - mx @ (coef_std/sx)) + X @ (coef_std/sx)
        coef_raw = coef / sx
        intercept = my - float(mx @ coef_raw)
        return FrozenModel(
            learner_name=self.name,
            family=self.family,
            params={
                "coef": coef_raw,
                "intercept": intercept,
                "coef_std": coef,
                "mean": mx,
                "scale": sx,
            },
            metadata={
                "n_train_rows": n,
                "n_features": d,
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "converged": True,
                "n_iter": n_iter,
                "n_active": int(np.count_nonzero(np.abs(coef_raw) > 1e-12)),
            },
        )

    def predict(self, frozen: FrozenModel, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        p = frozen.params
        return np.asarray(p["intercept"], dtype=np.float64) + X @ np.asarray(p["coef"], dtype=np.float64)
