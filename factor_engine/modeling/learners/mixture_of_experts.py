# -*- coding: utf-8 -*-
"""Predictive Mixture-of-Experts learner (Model Layer Major Redesign taskbook
§6.2 / §6.3 / §40).

Each ACTIVE expert needs, at minimum (§6.2):

* ``>= min_expert_obs`` observations (default 1000);
* a full-rank intercept design ``[1, X]``;
* a design condition number below a hard ceiling (``< 1e10``).

Partial experts NEVER silently survive — the default behaviour is to fail
closed with a ``ValueError`` listing the unsupported experts (§6.3).  The only
escape hatch is the explicit ``explicit_fallback="single_best"`` hyperparameter,
which trains a single pooled model and records ``fallback="single_best"`` in the
frozen metadata so it enters the semantic identity (§6.3).

At predict, the gating feature is ``X[:, gating_feature_index]`` (default 0);
the trainer passes the same series as ``aux["regime_state"]`` at fit.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec, register_learner

__all__ = ["MixtureOfExpertsLearner"]

#: default per-expert floor when no SampleAdequacyContract is supplied (§6.2).
_DEFAULT_MIN_EXPERT_OBS = 1000
#: hard ceiling on the per-expert design condition number (§6.2).
_CONDITION_CEILING = 1e10


def _bin_centers(boundaries: np.ndarray) -> np.ndarray:
    """Per-expert gating-space centers derived from the quantile boundaries.

    Interior bins use the boundary midpoint; the two outer bins are reflected
    off the nearest boundary so softmax gating is well defined everywhere.
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
class MixtureOfExpertsLearner(BaseLearner):
    name = "predictive_mixture_of_experts"
    family = "moe"
    sample_contract_family = "moe"

    def effective_parameter_count(
        self,
        hyperparams: dict[str, Any] | None = None,
        n_features: int = 0,
        *,
        n_rows: int | None = None,
        **extra: Any,
    ) -> int:
        """MoE complexity: per-expert intercept-OLS on d features
        (``n_experts * (d + 1)``) plus the ``(n_experts - 1)`` gate boundaries.

        The softmax gate is distance-based (inverse-distance to the fixed expert
        centers) and has no additional learned weights, so the only gate
        parameters are the quantile boundaries.  The explicit ``single_best``
        fallback degrades to a single pooled intercept-OLS (``d + 1``).
        """
        if extra.get("fallback") == "single_best":
            return max(1, int(n_features) + 1)
        e = int((hyperparams or {}).get("n_experts", self.n_experts))
        return max(1, e * (max(1, int(n_features)) + 1) + max(0, e - 1))

    def __init__(self, spec: LearnerSpec) -> None:
        super().__init__(spec)
        self.n_experts = int(spec.hyperparams.get("n_experts", 3))
        if self.n_experts < 2:
            raise ValueError("n_experts must be >= 2")
        self.explicit_fallback = spec.hyperparams.get("explicit_fallback")
        self.gating_feature_index = int(spec.hyperparams.get("gating_feature_index", 0))

    @property
    def _min_expert_obs(self) -> int:
        contract = self.spec.sample_contract
        if contract is not None and contract.min_expert_obs is not None:
            return int(contract.min_expert_obs)
        return _DEFAULT_MIN_EXPERT_OBS

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
            raise ValueError("moe fit needs a 2-D feature matrix with >= 2 rows")
        if len(X) != len(y):
            raise ValueError("moe fit: X and y row counts differ")
        if aux is None or aux.get("regime_state") is None:
            raise ValueError("moe learner requires a regime_state aux array (gating feature)")
        gate = np.asarray(aux["regime_state"], dtype=np.float64).ravel()
        if len(gate) != len(y):
            raise ValueError("moe fit: regime_state and y row counts differ")

        # Optional per-row date support: when the trainer supplies ``aux["date"]``
        # the pooled-panel independence structure is verified fail closed — an
        # expert sample spanning fewer unique dates than the contract floor is
        # rejected (a handful of dates is not an adequate pooled panel).
        dates = aux.get("date")
        n_unique_dates: int | None = None
        if dates is not None:
            dates = np.asarray(dates)
            if len(dates) != len(y):
                raise ValueError("moe fit: date aux and y row counts differ")
            n_unique_dates = int(len(np.unique(dates)))
            contract = self.spec.sample_contract
            if (
                contract is not None
                and contract.min_unique_dates is not None
                and n_unique_dates < contract.min_unique_dates
            ):
                raise ValueError(
                    "moe learner fails closed: unique dates "
                    f"{n_unique_dates} < contract min_unique_dates="
                    f"{contract.min_unique_dates}"
                )

        # Fail closed on constant / zero-variance features (never NaN silently).
        std = X.std(axis=0)
        zero_cols = np.flatnonzero(std == 0)
        if zero_cols.size:
            raise ValueError(
                "moe learner rejects zero-variance features (fail closed): "
                f"columns {zero_cols.tolist()}"
            )

        n_rows = len(y)
        min_expert_obs = self._min_expert_obs
        # Capacity double-gate (mirrors §6.1 for the MoE case).
        if self.n_experts * min_expert_obs > n_rows:
            raise ValueError(
                "moe learner sample floor violated: "
                f"{self.n_experts} * min_expert_obs({min_expert_obs}) = "
                f"{self.n_experts * min_expert_obs} > effective capacity {n_rows}"
            )

        center = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        Xs = (X - center) / scale
        w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()

        # Train-only expert assignment (§40): quantile bins of the gating feature.
        q = np.linspace(0.0, 1.0, self.n_experts + 1)[1:-1]
        boundaries = np.quantile(gate, q).tolist()
        expert_idx = np.digitize(gate, boundaries, right=False)

        # §6.2 per-expert support.
        design = np.column_stack([np.ones(len(Xs)), Xs])
        expert_checks: list[dict] = []
        failures: list[tuple[int, str]] = []
        for e in range(self.n_experts):
            n_e = int(np.count_nonzero(expert_idx) == e))
            full_rank = False
            cond: float | None = None
            reason: str | None = None
            if n_e < min_expert_obs:
                reason = f"n_obs={n_e} < min_expert_obs={min_expert_obs}"
            else:
                Ae = design[expert_idx == e]
                if Ae.shape[0] >= Ae.shape[1]:
                    rank = np.linalg.matrix_rank(Ae)
                    full_rank = rank == Ae.shape[1]
                    cond = float(np.linalg.cond(Ae))
                if not full_rank:
                    reason = "design not full rank"
                elif cond is not None and not (cond < _CONDITION_CEILING):
                    reason = f"condition number {cond:.6g} >= 1e10"
            expert_checks.append({"n_obs": n_e, "full_rank": full_rank, "condition": cond})
            if reason is not None:
                failures.append((e, reason))

        counts = [int(c["n_obs"]) for c in expert_checks]
        total = float(n_rows)
        occ = np.asarray(counts, dtype=np.float64) / max(1.0, total)
        pz = occ[occ > 0]
        gate_entropy = float(-(pz * np.log(pz)).sum()) if pz.size else 0.0
        occupancy = {str(i): counts[i] / total for i in range(self.n_experts)}

        # §6.3 fail closed — or the explicit single-best fallback.
        if failures:
            if self.explicit_fallback != "single_best":
                raise ValueError(
                    "moe learner fails closed: unsupported expert(s) "
                    + ", ".join(f"expert {e} ({r})" for e, r in failures)
                )
            # Explicit fallback: one pooled model, recorded in the metadata so
            # the semantic identity captures the degraded fit.
            coef, intercept = self._ols(Xs, y, w)
            params = {
                "coef": np.asarray(coef, dtype=np.float64).tolist(),
                "intercept": float(intercept),
                "center": center.tolist(),
                "scale": scale.tolist(),
                "gating_feature_index": self.gating_feature_index,
            }
            metadata = {
                "n_train_rows": n_rows,
                "n_experts_fitted": 1,
                "per_expert_obs": [n_rows],
                "expert_occupancy": {"0": 1.0},
                "gate_entropy": 0.0,
                "dominant_expert_ratio": 1.0,
                "fallback": "single_best",
                "unsupported_experts": [
                    {"expert": e, "reason": r} for e, r in failures
                ],
            }
            if dates is not None:
                metadata["n_unique_dates"] = n_unique_dates
            return FrozenModel(
                learner_name=self.name,
                family=self.family,
                params=params,
                metadata=metadata,
            )

        # §40 per-expert OLS.
        per_expert: list[dict] = []
        for e in range(self.n_experts):
            mask = expert_idx == e
            Xe = Xs[mask]
            ye = y[mask]
            we = None if w is None else w[mask]
            coef, intercept = self._ols(Xe, ye, we)
            per_expert.append(
                {
                    "coef": np.asarray(coef, dtype=np.float64).tolist(),
                    "intercept": float(intercept),
                    "n_obs": counts[e],
                    "full_rank": bool(expert_checks[e]["full_rank"]),
                    "condition": expert_checks[e]["condition"],
                }
            )

        params = {
            "boundaries": boundaries,
            "per_expert": per_expert,
            "center": center.tolist(),
            "scale": scale.tolist(),
            "gating_feature_index": self.gating_feature_index,
        }
        metadata = {
            "n_train_rows": n_rows,
            "n_experts_fitted": self.n_experts,
            "per_expert_obs": counts,
            "expert_occupancy": occupancy,
            "gate_entropy": gate_entropy,
            "dominant_expert_ratio": float(occ.max()) if occ.size else 0.0,
        }
        if dates is not None:
            metadata["n_unique_dates"] = n_unique_dates
            metadata["per_expert_unique_dates"] = [
                int(len(np.unique(dates[expert_idx) == e])))
                for e in range(self.n_experts)
            ]
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

        if "per_expert" not in p:
            # single_best fallback model.
            return np.asarray(p["intercept"], dtype=np.float64) + Xs @ np.asarray(
                p["coef"], dtype=np.float64
            )

        g = X[:, int(p.get("gating_feature_index", 0))]
        boundaries = np.asarray(p["boundaries"], dtype=np.float64)
        centers = _bin_centers(boundaries)
        coefs = [np.asarray(e["coef"], dtype=np.float64) for e in p["per_expert"]]
        intercepts = [float(e["intercept"]) for e in p["per_expert"]]

        # Softmax gate on the inverse distance of the gating feature to the
        # expert centers; prediction is the gate-weighted expert average.
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
        """Per-expert obs / full-rank / condition for the support hard gate."""
        p = frozen.params
        if "per_expert" not in p:
            return {
                "fallback": "single_best",
                "n_experts_fitted": 1,
                "per_expert_obs": [int(frozen.metadata.get("n_train_rows", 0))],
            }
        per = p["per_expert"]
        obs = [int(e["n_obs"]) for e in per]
        total = sum(obs) or 1
        return {
            "n_experts": len(per),
            "per_expert_obs": obs,
            "expert_occupancy": {str(i): obs[i] / total for i in range(len(obs))},
            "full_rank": [bool(e.get("full_rank", False)) for e in per],
            "condition": [e.get("condition") for e in per],
        }
