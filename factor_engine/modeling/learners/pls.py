# -*- coding: utf-8 -*-
"""Predictive PLS learner (Model Layer Major Redesign taskbook §37).

Pooled-panel PLS1 via NIPALS.  ``n_components <= effective rank`` is enforced
(validation selects components; the fitted component count is capped by the
observed rank).  Train-only centering/scaling — frozen for predict.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec, register_learner

__all__ = ["PLSLearner"]


@register_learner
class PLSLearner(BaseLearner):
    name = "predictive_pls"
    family = "pls"
    sample_contract_family = "linear"

    def __init__(self, spec: LearnerSpec) -> None:
        super().__init__(spec)
        self.n_components = int(spec.hyperparams.get("n_components", 5))
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")

    def validate_params(self) -> None:
        nc = int(self.spec.hyperparams.get("n_components", 5))
        if nc < 1:
            raise ValueError("pls n_components must be >= 1")

    def effective_parameter_count(
        self,
        hyperparams: dict[str, Any] | None = None,
        n_features: int = 0,
        *,
        n_rows: int | None = None,
        **extra: Any,
    ) -> int:
        """PLS latent-model complexity: ``k * (d + 1) + k``.

        ``k`` latent directions, each spanning the ``d`` features plus one latent
        score axis (``k*(d+1)``), plus the ``k`` y-loadings ``q`` that map the
        latent scores onto the response.  The PCA-style projection (weights W and
        loadings P) is part of the latent mapping and counted above; the number
        of ``n_components`` is capped by the feature dimension.
        """
        k = int((hyperparams or {}).get("n_components", self.n_components))
        k = max(1, min(k, max(1, int(n_features))))
        return k * (int(n_features) + 1) + k

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
            raise ValueError("pls fit needs a 2-D feature matrix with >= 2 rows")
        n, d = X.shape
        # Fail closed: never silently fit fewer components than requested.
        k_max = min(n, d)
        if self.n_components > k_max:
            raise ValueError(
                f"pls n_components={self.n_components} exceeds min(n_rows={n}, "
                f"n_features={d})={k_max} — reject (fail closed), no silent downgrade"
            )
        mx = X.mean(axis=0)
        my = float(y.mean())
        sx = X.std(axis=0)
        sx[sx == 0] = 1.0
        Xs = (X - mx) / sx
        sy = y.std()
        ys = (y - my) / (sy if sy > 0 else 1.0)

        # NIPALS deflation (sklearn-compatible PLS1).
        W: list[np.ndarray] = []
        P: list[np.ndarray] = []
        Q: list[float] = []
        xr = Xs.copy()
        yr = ys.copy()
        for _ in range(self.n_components):
            xw = xr.T @ yr
            norm = float(np.linalg.norm(xw))
            if norm < 1e-12:
                raise ValueError(
                    "pls fit failed: zero norm weight (feature/label covariance "
                    "collapsed) — fail closed rather than silently reducing components"
                )
            w = xw / norm
            t = xr @ w
            tt = float(t @ t)
            if tt < 1e-12:
                raise ValueError("pls fit failed: zero-variance score — fail closed")
            p = (xr.T @ t) / tt
            q = float(yr @ t) / tt
            xr = xr - np.outer(t, p)
            yr = yr - t * q
            W.append(w)
            P.append(p)
            Q.append(q)
        Wm = np.column_stack(W) if W else np.zeros((d, 0))
        Pm = np.column_stack(P) if P else np.zeros((d, 0))
        q_vec = np.asarray(Q, dtype=np.float64)

        # PLS regression coefficients in standardized space:
        #     T = X R,  R = W (P^T W)^{-1}  =>  b = R q = W (P^T W)^{-1} q
        # NB: (P^T W) — NOT (W^T P).  They differ for multi-component PLS.
        PtW = Pm.T @ Wm
        try:
            b_std = Wm @ np.linalg.solve(PtW, q_vec)
        except np.linalg.LinAlgError:
            b_std = Wm @ (np.linalg.pinv(PtW) @ q_vec)
        # back to original scale: y = my + sy * b_std' * ((X - mx)/sx)
        coef = (b_std * sy) / sx
        intercept = my - float(coef @ mx)
        return FrozenModel(
            learner_name=self.name,
            family=self.family,
            params={
                "coef": coef,
                "intercept": intercept,
                "n_components": len(W),
                "weights": Wm,
                "loadings": Pm,
            },
            metadata={"n_train_rows": n, "n_features": d, "n_components_used": len(W)},
        )

    def predict(self, frozen: FrozenModel, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        p = frozen.params
        return np.asarray(p["intercept"], dtype=np.float64) + X @ np.asarray(p["coef"], dtype=np.float64)
