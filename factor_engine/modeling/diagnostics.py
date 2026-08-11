# -*- coding: utf-8 -*-
"""Model diagnostics (§25 / §30 / §70).

Parameter counting, sample-parameter ratios, complexity scoring (§25),
parameter-stability reports (§70), feature ablation (§25.3), label-shuffle
leakage control (§25.4) and a monitoring-only drift report (§30).  The drift
report is for monitoring only — it never backfits parameters.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from modeling.learners.base import BaseLearner, FrozenModel

__all__ = [
    "effective_parameter_count",
    "sample_parameter_ratio",
    "complexity_score",
    "parameter_stability_report",
    "feature_ablation",
    "label_shuffle_control",
    "drift_report",
]


def effective_parameter_count(learner: BaseLearner, frozen: FrozenModel) -> int:
    """Free parameter count of a frozen model (§25).

    * linear (elastic_net): ``len(coef) + 1`` (intercept);
    * pcr / pls: ``n_components + 1``;
    * regime / moe: sum over components of per-component coef counts (plus one
      intercept each); falls back to ``n_components * (n_features + 1)`` when
      the frozen params carry only aggregated coefficients.
    """
    family = frozen.family or learner.family
    p = frozen.params
    if family in ("pcr", "pls"):
        return int(p.get("n_components", 1)) + 1
    if family == "elastic_net":
        coef = p.get("coef")
        return (int(np.asarray(coef).size) if coef is not None else 1) + 1
    if family in ("regime", "moe"):
        comps = (
            p.get("components")
            or p.get("regimes")
            or p.get("experts")
            or p.get("coefs")
        )
        if comps is not None:
            return sum(int(np.asarray(c).size) + 1 for c in comps)
        n_comp = int(
            getattr(learner, "n_regimes", None)
            or getattr(learner, "n_experts", 1)
            or 1
        )
        coef = p.get("coef")
        n_feat = int(np.asarray(coef).size) if coef is not None else 1
        return n_comp * (n_feat + 1)
    # generic linear fallback
    coef = p.get("coef")
    if coef is not None:
        return int(np.asarray(coef).size) + 1
    return 1


def sample_parameter_ratio(effective_obs: int, free_parameters: int) -> float:
    """Effective observations per free parameter."""
    return float(effective_obs) / max(1, int(free_parameters))


def complexity_score(learner: BaseLearner) -> float:
    """§25 model complexity score, >= 1.

    Base 1.0; +0.5 per extra regime for regime, +0.5 per extra expert for MoE,
    +0.1 * ``n_components`` for pcr/pls.
    """
    family = getattr(learner, "family", "") or ""
    score = 1.0
    if family == "regime":
        n = max(1, int(getattr(learner, "n_regimes", 1)))
        score += 0.5 * (n - 1)
    elif family == "moe":
        n = max(1, int(getattr(learner, "n_experts", 1)))
        score += 0.5 * (n - 1)
    elif family in ("pcr", "pls"):
        score += 0.1 * max(1, int(getattr(learner, "n_components", 1)))
    return max(1.0, score)


def parameter_stability_report(
    artifact: Any,
    eval_fn: Callable[[dict[str, Any]], float],
    hyperparam_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """§70 — evaluate the frozen model's hyperparameter neighbours.

    ``eval_fn(hp: dict) -> float`` scores a model built with ``hp`` on the
    validation set (higher is better; the caller closes over the training /
    validation data).  Reports ``best``, ``neighbors_scores`` and ``stable``
    (best within 5% of every neighbour, i.e. the surface is flat around the
    optimum).
    """
    current_hp = dict(getattr(artifact.learner, "spec", None).hyperparams or {})
    scores: dict[str, float] = {}
    for hp in hyperparam_candidates:
        key = json.dumps(dict(hp), sort_keys=True)
        try:
            scores[key] = float(eval_fn(dict(hp)))
        except Exception as exc:  # a candidate may legitimately fail to fit
            scores[key] = float("nan")
    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not valid:
        return {
            "best": None,
            "best_score": None,
            "neighbors_scores": scores,
            "stable": False,
            "current_hyperparams": current_hp,
        }
    best_key = max(valid, key=valid.get)
    best_score = valid[best_key]
    others = [v for k, v in valid.items() if k != best_key]
    stable = True
    if others:
        worst_other = min(others)
        stable = bool(best_score - worst_other <= 0.05 * abs(best_score) + 1e-9)
    return {
        "best": json.loads(best_key),
        "best_score": best_score,
        "neighbors_scores": scores,
        "stable": stable,
        "current_hyperparams": current_hp,
    }


def feature_ablation(
    artifact: Any,
    X: np.ndarray,
    y: np.ndarray,
    eval_fn: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, Any]:
    """§25.3 — baseline metric then drop-one / shuffle-one feature deltas.

    ``eval_fn(pred, y) -> float`` (higher better).  Deltas are
    ``baseline - perturbed`` so a positive delta means the feature carries
    signal.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    d = X.shape[1]
    baseline = float(eval_fn(artifact.predict(X), y))
    drops: dict[str, float] = {}
    shuffles: dict[str, float] = {}
    rng = np.random.default_rng(0)
    for j in range(d):
        Xd = X.copy()
        Xd[:, j] = np.nanmean(X[:, j])
        drops[f"drop_{j}"] = float(baseline - eval_fn(artifact.predict(Xd), y))
        Xs = X.copy()
        rng.shuffle(Xs[:, j])
        shuffles[f"shuffle_{j}"] = float(baseline - eval_fn(artifact.predict(Xs), y))
    return {
        "baseline": baseline,
        "n_features": d,
        "drops": drops,
        "shuffles": shuffles,
    }


def label_shuffle_control(
    learner_factory: Callable[[int | None], BaseLearner],
    X: np.ndarray,
    y: np.ndarray,
    eval_fn: Callable[[np.ndarray, np.ndarray], float],
    n_shuffles: int = 5,
) -> dict[str, Any]:
    """§25.4 — refit on shuffled labels; flag when baseline >> shuffle.

    ``learner_factory(seed)`` returns an UNFITTED learner.  A legitimately
    informative model should beat shuffled-label baselines; if the baseline
    stays high while shuffled scores are ~0 the pipeline is leaking the label.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    base_learner = learner_factory(0)
    frozen = base_learner.fit(X, y)
    baseline = float(eval_fn(base_learner.predict(frozen, X), y))
    rng = np.random.default_rng(123)
    shuffled: list[float] = []
    for i in range(n_shuffles):
        ys = y.copy()
        rng.shuffle(ys)
        lr = learner_factory(100 + i)
        fr = lr.fit(X, ys)
        shuffled.append(float(eval_fn(lr.predict(fr, X), y)))
    mean_s = float(np.mean(shuffled))
    std_s = float(np.std(shuffled))
    leakage_suspected = bool(baseline > mean_s + 2.0 * std_s + 1e-9)
    return {
        "baseline_score": baseline,
        "shuffled_scores": shuffled,
        "shuffled_mean": mean_s,
        "shuffled_std": std_s,
        "leakage_suspected": leakage_suspected,
    }


def drift_report(
    artifact: Any,
    historical_X: np.ndarray,
    current_X: np.ndarray,
    historical_pred: np.ndarray,
    current_pred: np.ndarray,
) -> dict[str, Any]:
    """§30 — feature / prediction / coverage drift.  Monitoring only.

    Never used for backfitting parameters; the report is advisory state for the
    model-monitoring lane.
    """
    hist = np.asarray(historical_X, dtype=np.float64)
    curr = np.asarray(current_X, dtype=np.float64)
    fp = np.asarray(historical_pred, dtype=np.float64).ravel()
    cp = np.asarray(current_pred, dtype=np.float64).ravel()
    return {
        "model_name": getattr(artifact, "model_name", str(artifact)),
        "feature_shift": {
            "mean_delta": np.nanmean(curr, axis=0) - np.nanmean(hist, axis=0),
            "std_delta": np.nanstd(curr, axis=0) - np.nanstd(hist, axis=0),
        },
        "prediction_shift": {
            "mean_historical": float(np.nanmean(fp)),
            "mean_current": float(np.nanmean(cp)),
            "var_historical": float(np.nanvar(fp)),
            "var_current": float(np.nanvar(cp)),
        },
        "coverage_shift": {
            "historical_pred_finite": float(np.isfinite(fp).mean()),
            "current_pred_finite": float(np.isfinite(cp).mean()),
            "current_X_finite": float(np.isfinite(curr).mean()),
        },
    }
