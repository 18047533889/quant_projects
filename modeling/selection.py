# -*- coding: utf-8 -*-
"""Validation selection and neighborhood-stability (§28 / §70).

The default objective is ``rank_ic`` (§28).  ``neighborhood_stability`` (§70)
guards against razor-thin MODEL_COMPLEXITY optima: a best value whose best±1
neighbours drop below 70% of the best score is flagged unstable so the miner
does not over-trust a fragile optimum.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

__all__ = ["ConsumerNonInferiorityContract", "select_best_validation", "neighborhood_stability", "candidate_identity"]

_COMPLEXITY_PARAMS = ("n_components", "n_regimes", "n_experts")


@dataclass(frozen=True)
class ConsumerNonInferiorityContract:
    """Pre-declared paired acceptance rule for one consumer objective."""

    consumer_profile: str
    objective: str
    epsilon: float
    min_risk_improvement: float = 0.0
    max_risk_budget: float | None = None

    def __post_init__(self) -> None:
        if not self.consumer_profile or not self.objective:
            raise ValueError("consumer_profile and objective are required")
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative")
        if not np.isfinite(self.min_risk_improvement) or self.min_risk_improvement < 0:
            raise ValueError("min_risk_improvement must be finite and non-negative")
        if self.max_risk_budget is not None and (
            not np.isfinite(self.max_risk_budget) or self.max_risk_budget < 0
        ):
            raise ValueError("max_risk_budget must be finite and non-negative")

    def assess(self, entry: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        required = {
            "consumer_profile", "baseline_score", "delta_ci_low", "delta_ci_high",
            "effect_size", "risk_improvement", "risk_budget",
        }
        missing = required - set(entry)
        if missing:
            raise ValueError(f"non-inferiority evidence missing fields: {sorted(missing)}")
        if entry["consumer_profile"] != self.consumer_profile:
            raise ValueError("candidate consumer_profile differs from comparison contract")
        numeric = {name: float(entry[name]) for name in required - {"consumer_profile"}}
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("non-inferiority evidence must be finite")
        passed = numeric["delta_ci_low"] >= -self.epsilon
        passed &= numeric["risk_improvement"] >= self.min_risk_improvement
        if self.max_risk_budget is not None:
            passed &= numeric["risk_budget"] <= self.max_risk_budget
        return bool(passed), {
            "passed": bool(passed), "epsilon": self.epsilon,
            "delta_ci": (numeric["delta_ci_low"], numeric["delta_ci_high"]),
            "effect_size": numeric["effect_size"],
            "risk_improvement": numeric["risk_improvement"],
            "risk_budget": numeric["risk_budget"],
        }


def candidate_identity(hyperparams: dict[str, Any]) -> str:
    """Canonical candidate identity for audit, deduplication, and tie breaks."""
    return json.dumps(hyperparams, sort_keys=True, separators=(",", ":"), default=str)


def select_best_validation(
    validation_scores: list[dict[str, Any]],
    *,
    objective: str = "rank_ic",
    score_fn: Callable[[dict[str, Any]], float] | None = None,
    noninferiority: ConsumerNonInferiorityContract | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Pick the best candidate by ``objective``.

    Returns ``(best_candidate_hyperparams, diagnostics)``.  ``None`` hyperparams
    means no candidate produced a finite objective score.  ``score_fn``, when
    given, overrides the objective lookup.
    """
    if not validation_scores:
        return None, {"n_candidates": 0, "reason": "no candidates"}

    def _score(entry: dict[str, Any]) -> float:
        if score_fn is not None:
            return float(score_fn(entry))
        v = entry.get(objective)
        return float(v) if isinstance(v, (int, float)) else float("nan")

    scored: list[tuple[float, str, dict[str, Any]]] = []
    comparison_diagnostics: dict[str, Any] = {}
    for entry in validation_scores:
        identity = str(
            entry.get("candidate_id")
            or candidate_identity(entry.get("hyperparams", {}))
        )
        if noninferiority is not None:
            if noninferiority.objective != objective:
                raise ValueError("non-inferiority objective differs from selection objective")
            passed, comparison = noninferiority.assess(entry)
            comparison_diagnostics[identity] = comparison
            if not passed:
                continue
        v = _score(entry)
        if np.isfinite(v):
            scored.append((v, identity, entry))
    if not scored:
        return (
            None,
            {
                "n_candidates": len(validation_scores),
                "objective": objective,
                "reason": "no finite objective scores",
                "noninferiority": comparison_diagnostics,
            },
        )

    maximize = objective not in ("mse", "rmse")
    # Canonical identity is the final ascending tie-break for both objective
    # directions, so input/grid iteration order cannot change the winner.
    scored.sort(key=lambda t: ((-t[0] if maximize else t[0]), t[1]))
    best_score, best_identity, best_entry = scored[0]
    diagnostics = {
        "objective": objective,
        "n_candidates": len(validation_scores),
        "n_evaluated": len(scored),
        "best_score": best_score,
        "best_hyperparams": best_entry.get("hyperparams"),
        "best_candidate_id": best_identity,
        "ranked": [
            {"candidate_id": identity, "hyperparams": s.get("hyperparams"), objective: v}
            for v, identity, s in scored
        ],
        "noninferiority": comparison_diagnostics,
    }
    return best_entry.get("hyperparams"), diagnostics


def neighborhood_stability(
    candidates: list[dict[str, Any]],
    best: dict[str, Any],
    *,
    objective: str = "rank_ic",
) -> tuple[bool, dict[str, Any]]:
    """§70 — check the objective at best±1 neighbours of the MODEL_COMPLEXITY
    parameter is not razor-thin (< 70% of the best score)."""
    complexity_param = next((p for p in _COMPLEXITY_PARAMS if p in best), None)
    if complexity_param is None:
        return True, {"checked": False, "reason": "best has no model-complexity parameter"}
    best_value = int(best[complexity_param])

    def _score_at(value: int) -> float | None:
        for c in candidates:
            hp = c.get("hyperparams", {})
            if hp.get(complexity_param) == value:
                v = c.get(objective)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    return float(v)
        return None

    best_score = _score_at(best_value)
    if best_score is None or best_score == 0.0:
        return True, {"checked": False, "reason": "best score unavailable or zero"}

    details: dict[str, Any] = {
        "checked": True,
        "complexity_param": complexity_param,
        "best_value": best_value,
        "best_score": best_score,
        "neighbors": {},
    }
    stable = True
    for delta in (-1, 1):
        neighbour = best_value + delta
        if neighbour < 1:
            continue
        neighbour_score = _score_at(neighbour)
        if neighbour_score is None:
            continue
        ratio = neighbour_score / best_score
        details["neighbors"][str(neighbour)] = {"score": neighbour_score, "ratio": ratio}
        if ratio < 0.7:
            stable = False
    return stable, details
