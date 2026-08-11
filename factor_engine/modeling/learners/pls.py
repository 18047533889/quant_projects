# -*- coding: utf-8 -*-
"""Predictive PLS learner (Model Layer Major Redesign taskbook §37).

Pooled-panel PLS1 via NIPALS.  ``n_components <= effective rank`` is enforced
(validation selects components; the fitted component count is capped by the
observed rank).  Train-only centering/scaling — frozen for predict.
"""
from __future__ import annotations

import numpy as np

from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec, register_learner

__all__ = ["PLSLearner"]


@register_learner
class PLSLearner(BaseLearner):
    name = "predictive_pls"
    family = "pls"

    def __init__(self, spec: LearnerSpec) -> None:
        super().__init__(spec)
        self.n_components = int(spec.hyperparams.get("n_components", 5))
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")

    def validate_params(self) -> None:
        nc = int(self.spec.hyperparams.get("n_components", 5))
        if nc < 1:
            raise ValueError("pls n_components must be >= 1")

    def fit(self, X: np.ndarray, y: np.ndarray, *, weights: np.ndarray | None = None) -> FrozenModel:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("pls fit needs a 2-D feature matrix with >= 2 rows")
        n, d = X.shape
        mx = X.mean(axis=0)
        my = float(y.mean())
        sx = X.std(axis=0)
        sx[sx == 0] = 1.0
        Xs = (X - mx) / sx
        ys = (y - my) / (y.std() if y.std() > 0 else 1.0)

        # NIPALS deflation
        W: list[np.ndarray] = []
        T: list[np.ndarray] = []
        P: list[np.ndarray] = []
        xr = Xs.copy()
        yr = ys.copy()
        rank = min(self.n_components, d, n)
        for _ in range(rank):
            xw = xr.T @ yr
            norm = np.linalg.norm(xw)
            if norm < 1e-12:
                break
            w = xw / norm
            t = xr @ w
            tt = float(t @ t)
            if tt < 1e-12:
                break
            p = (xr.T @ t) / tt
            q = float(yr @ t) / tt
            xr = xr - np.outer(t, p)
            yr = yr - t * q
            W.append(w)
            T.append(t)
            P.append(p)
        Wm = np.column_stack(W) if W else np.zeros((d, 0))
        Tm = np.column_stack(T) if T else np.zeros((n, 0))
        Pm = np.column_stack(P) if P else np.zeros((d, 0))
        # regression coefficients in standardized space: b = W (P'W)^-1 q'
        WtP = Wm.T @ Pm
        try:
            WtP_inv = np.linalg.inv(WtP)
        except np.linalg.LinAlgError:
            WtP_inv = np.linalg.pinv(WtP)
        q_vec = (ys @ Tm) / np.einsum("ij,ij->j", Tm, Tm) if Tm.shape[1] else np.zeros(0)
        b_std = Wm @ (WtP_inv @ q_vec)
        # back to original scale: y = my + b_std' * ((X-mx)/sx) * sy
        sy = y.std() if y.std() > 0 else 1.0
        coef = b_std * sy / sx
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
