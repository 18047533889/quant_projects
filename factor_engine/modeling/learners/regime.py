# -*- coding: utf-8 -*-
"""Predictive Regime learner (Model Layer Major Redesign taskbook §6.1 / §39).

Regime boundaries are fit on TRAIN history only (§39).  Each regime fits an
intercept-OLS on centered/scaled features; ``hard`` vs ``soft`` regimes are
distinct semantic identities (``regime_mode`` is a hyperparameter and therefore
enters the semantic identity of the frozen model).

§6.1 support double-gate, both enforced **fail closed** (the learner NEVER
silently fits fewer regimes):

* ``n_regimes * min_regime_support <= effective capacity`` (number of rows);
* every requested regime has ``>= min_regime_obs`` observations.

At predict the gating feature is ``X[:, gating_feature_index]`` (the first
column by default).  The trainer is expected to pass the same series as
``aux["regime_state"]`` at fit so predict-time gating matches the train-only
regime boundaries.  ``support_report`` feeds the
``MODEL_REGIME_ALL_COMPONENTS_HAVE_SUPPORT`` hard gate.
"""
from __future__ import annotations

import numpy as np

from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec, register_learner

__all__ = ["RegimeLearner"]

#: default per-regime floor when no SampleAdequacyContract is supplied (§6.1).
_DEFAULT_MIN_REGIME_OBS = 5000


def _bin_centers(boundaries: np.ndarray) -> np.ndarray:
    """Per-regime gating-space centers derived from the quantile boundaries.

    Interior bins use the boundary midpoint; the two outer bins are reflected
    off the nearest boundary so soft gating is well defined everywhere.
    """
    b = np.asarray(boundaries, dtype=np.float64)
    if len(b) == 1:
        half = max(1.0, abs(b[0]) * 0.5)
        return np.array([b[0] - half, b[0] + half], dtype=np.float64)
    centers = [b[0] - (b[1] - b[0]) / 2.0]
    for k in range(len(b) - 1):
        centers.append((b[k] + b[k + 1]) / 2.0)
    centers.append(b[-1] + (b[-1] - b[-2]) / 2.0)
    return np.asarray(centers, dtype=np.float64)


@register_learner
class RegimeLearner(BaseLearner):
    name = "predictive_regime"
    family = "regime"

    def __init__(self, spec: LearnerSpec) -> None:
        super().__init__(spec)
        self.n_regimes = int(spec.hyperparams.get("n_regimes", 3))
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        self.regime_mode = spec.hyperparams.get("regime_mode", "hard")
        if self.regime_mode not in ("hard", "soft"):
            raise ValueError("regime_mode must be in {'hard','soft'}")
        self.gating_feature_index = int(spec.hyperparams.get("gating_feature_index", 0))

    @property
    def _min_regime_obs(self) -> int:
        contract = self.spec.sample_contract
        if contract is not None and contract.min_regime_obs is not None:
            return int(contract.min_regime_obs)
        return _DEFAULT_MIN_REGIME_OBS

    # -- fit -----------------------------------------------------------------
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
            raise ValueError("regime fit needs a 2-D feature matrix with >= 2 rows")
        if len(X) != len(y):
            raise ValueError("regime fit: X and y row counts differ")
        if aux is None or aux.get("regime_state") is None:
            raise ValueError("regime learner requires a regime_state aux array")
        gate = np.asarray(aux["regime_state"], dtype=np.float64).ravel()
        if len(gate) != len(y):
            raise ValueError("regime fit: regime_state and y row counts differ")

        # Fail closed on constant / zero-variance features (never NaN silently).
        std = X.std(axis=0)
        zero_cols = np.flatnonzero(std == 0)
        if zero_cols.size:
            raise ValueError(
                "regime learner rejects zero-variance features (fail closed): "
                f"columns {zero_cols.tolist()}"
            )

        n_rows = len(y)
        min_regime_obs = self._min_regime_obs
        # §6.1 double gate (1): capacity.
        if self.n_regimes * min_regime_obs > n_rows:
            raise ValueError(
                "regime learner sample floor violated: "
                f"{self.n_regimes} * min_regime_obs({min_regime_obs}) = "
                f"{self.n_regimes * min_regime_obs} > effective capacity {n_rows}"
            )

        # §39: train-only regime definition.
        q = np.linspace(0.0, 1.0, self.n_regimes + 1)[1:-1]
        boundaries = np.quantile(gate, q).tolist()
        regime_idx = np.digitize(gate, boundaries, right=False)

        # §6.1 double gate (2): every regime has enough observations.
        counts = [
            int(np.count_nonzero(regime_idx == r)) for r in range(self.n_regimes)
        ]
        unsupported = [
            (r, counts[r])
            for r in range(self.n_regimes)
            if counts[r] < min_regime_obs
        ]
        if unsupported:
            raise ValueError(
                "regime learner fails closed: unsupported regime(s) "
                + ", ".join(
                    f"regime {r} (n_obs={c})" for r, c in unsupported
                )
            )

        # Train-only center/scale for numerical stability; frozen at predict.
        center = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0  # unreachable (rejected above), defensive
        Xs = (X - center) / scale

        w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()

        per_regime: list[dict] = []
        for r in range(self.n_regimes):
            mask = regime_idx == r
            Xr = Xs[mask]
            yr = y[mask]
            wr = None if w is None else w[mask]
            coef, intercept = self._ols(Xr, yr, wr)
            per_regime.append(
                {
                    "coef": np.asarray(coef, dtype=np.float64).tolist(),
                    "intercept": float(intercept),
                    "n_obs": int(mask.sum()),
                }
            )

        params = {
            "boundaries": boundaries,
            "per_regime": per_regime,
            "regime_mode": self.regime_mode,
            "center": center.tolist(),
            "scale": scale.tolist(),
            "gating_feature_index": self.gating_feature_index,
        }

        total = float(n_rows)
        occ = np.asarray(counts, dtype=np.float64) / max(1.0, total)
        pz = occ[occ > 0]
        gate_entropy = float(-(pz * np.log(pz)).sum()) if pz.size else 0.0
        metadata = {
            "n_train_rows": n_rows,
            "n_regimes_fitted": self.n_regimes,
            "per_regime_obs": counts,
            "regime_occupancy": {
                str(i): counts[i] / total for i in range(self.n_regimes)
            },
            "gate_entropy": gate_entropy,
        }
        return FrozenModel(
            learner_name=self.name,
            family=self.family,
            params=params,
            metadata=metadata,
        )

    @staticmethod
    def _ols(Xr: np.ndarray, yr: np.ndarray, wr: np.ndarray | None) -> tuple[np.ndarray, float]:
        """Intercept OLS via lstsq on a design ``[1, Xr]``."""
        n = len(yr)
        design = np.column_stack([np.ones(n), Xr])
        target = yr
        if wr is not None:
            sw = np.sqrt(wr)
            design = design * sw[:, None]
            target = target * sw
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        return np.asarray(coef[1:], dtype=np.float64), float(coef[0])

    # -- predict --------------------------------------------------------------
    def predict(self, frozen: FrozenModel, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        p = frozen.params
        center = np.asarray(p["center"], dtype=np.float64)
        scale = np.asarray(p["scale"], dtype=np.float64)
        Xs = (X - center) / scale
        g = X[:, int(p.get("gating_feature_index", 0))]
        boundaries = np.asarray(p["boundaries"], dtype=np.float64)
        coefs = [np.asarray(r["coef"], dtype=np.float64) for r in p["per_regime"]]
        intercepts = [float(r["intercept"]) for r in p["per_regime"]]
        regime_idx = np.digitize(g, boundaries, right=False)

        if p.get("regime_mode", "hard") == "hard":
            pred = np.empty(len(X), dtype=np.float64)
            for i in range(len(X)):
                r = int(regime_idx[i])
                pred[i] = intercepts[r] + Xs[i] @ coefs[r]
            return pred

        # Soft mode: blend by softmax of inverse distance to regime centers.
        centers = _bin_centers(boundaries)
        pred = np.empty(len(X), dtype=np.float64)
        for i in range(len(X)):
            dist = np.abs(g[i] - centers)
            scores = 1.0 / (dist + 1e-9)
            s = np.exp(scores - scores.max())
            w = s / s.sum()
            pi = np.array(
                [intercepts[r] + Xs[i] @ coefs[r] for r in range(len(coefs))]
            )
            pred[i] = float(w @ pi)
        return pred

    # -- support report -------------------------------------------------------
    def support_report(self, frozen: FrozenModel) -> dict:
        """Per-regime observation / occupancy for the support hard gate."""
        per_regime = frozen.params["per_regime"]
        obs = [int(r["n_obs"]) for r in per_regime]
        total = sum(obs) or 1
        return {
            "n_regimes": len(per_regime),
            "per_regime_obs": obs,
            "regime_occupancy": {
                str(i): obs[i] / total for i in range(len(obs))
            },
        }
