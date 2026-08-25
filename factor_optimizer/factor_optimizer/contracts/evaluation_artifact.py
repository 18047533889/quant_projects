"""TrialEvaluationArtifact: typed, content-hashed evaluation output (FO-P1-25).

The historical runner read a magic dict of ad-hoc keys ("score", "cost",
"rank", "total", "promote", ...) from the evaluation callback, so an evaluator
that renamed/removed a key would silently break the run.  This module defines a
first-class evaluation artifact with explicit fields, a status, and a content
hash so the evaluator's output is validated and attributable rather than a
silently-typed dict.

A legacy magic dict is still accepted (backward compatible) and normalized into
the artifact, but a well-formed artifact is the preferred contract.  The runner
reads ONLY the artifact's fields, never bare magic keys.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class EvaluationStatus:
    """The outcome of a single evaluation (FO-P1-25)."""

    COMPLETE = "complete"
    FAILED = "failed"
    PRUNED = "pruned"


# Magic keys the legacy normalization understands (primary objective is "score").
_MAGIC_KEYS = {
    "score",
    "cost",
    "rank",
    "total",
    "promote",
    "baseline_score",
    "evidence_ref",
    "evaluation_id",
}


@dataclass(frozen=True)
class ObjectiveValue:
    """A single objective value with its direction/spec reference."""

    name: str
    value: float
    objective_spec_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("objective value name must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("objective value must be a non-boolean number")
        if not math.isfinite(float(self.value)):
            raise ValueError("objective value must be finite")


@dataclass(frozen=True)
class TrialEvaluationArtifact:
    """Typed, attributable output of a single evaluation (FO-P1-25).

    Attributes:
        trial_id: the trial this artifact describes.
        objective_values: ordered list of name/value dicts (the primary
            objective is the first).
        objective_spec_ref: reference to the ObjectiveSpec that governed the
            optimization (the metric/direction authority).
        split_ref: reference to the SplitPlan this evaluation ran against.
        fidelity: fidelity tier at which this evaluation ran.
        evidence_ref: reference to the persisted evidence bundle (required).
        compute_cost: compute cost in budget units.
        promotion_evidence: optional peer-rank/promotion evidence
            (rank/total/baseline_score/promote) used for multi-fidelity.
        status: COMPLETE / FAILED / PRUNED.
    """

    trial_id: str
    objective_values: List[Dict[str, Any]]
    objective_spec_ref: Optional[str] = None
    split_ref: Optional[str] = None
    fidelity: int = 0
    evidence_ref: Optional[str] = None
    compute_cost: float = 1.0
    promotion_evidence: Optional[Dict[str, Any]] = None
    status: str = EvaluationStatus.COMPLETE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be a non-empty string")
        if not self.objective_values:
            raise ValueError("objective_values must not be empty")
        if not isinstance(self.objective_values, list) or not all(
            isinstance(v, dict) and "name" in v and "value" in v
            for v in self.objective_values
        ):
            raise TypeError("objective_values must be a list of name/value dicts")
        if isinstance(self.compute_cost, bool) or not isinstance(
            self.compute_cost, (int, float)
        ):
            raise TypeError("compute_cost must be a non-boolean number")
        if not math.isfinite(float(self.compute_cost)) or self.compute_cost < 0:
            raise ValueError("compute_cost must be finite and >= 0")
        if self.status not in (
            EvaluationStatus.COMPLETE,
            EvaluationStatus.FAILED,
            EvaluationStatus.PRUNED,
        ):
            raise ValueError(f"unknown evaluation status: {self.status!r}")

    @property
    def primary_objective_value(self) -> float:
        """The primary (first) objective value."""
        return float(self.objective_values[0]["value"])

    def _canonical_payload(self) -> str:
        return json.dumps(
            {
                "trial_id": self.trial_id,
                "objective_values": [
                    {"name": v["name"], "value": v["value"]}
                    for v in self.objective_values
                ],
                "objective_spec_ref": self.objective_spec_ref,
                "split_ref": self.split_ref,
                "fidelity": self.fidelity,
                "evidence_ref": self.evidence_ref,
                "compute_cost": self.compute_cost,
                "promotion_evidence": self.promotion_evidence,
                "status": self.status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self._canonical_payload().encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "objective_values": [dict(v) for v in self.objective_values],
            "objective_spec_ref": self.objective_spec_ref,
            "split_ref": self.split_ref,
            "fidelity": self.fidelity,
            "evidence_ref": self.evidence_ref,
            "compute_cost": self.compute_cost,
            "promotion_evidence": self.promotion_evidence,
            "status": self.status,
            "metadata": dict(self.metadata),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialEvaluationArtifact":
        if not isinstance(data, dict):
            raise TypeError("TrialEvaluationArtifact.from_dict requires a dict")
        obj = cls(
            trial_id=data["trial_id"],
            objective_values=[dict(v) for v in data["objective_values"]],
            objective_spec_ref=data.get("objective_spec_ref"),
            split_ref=data.get("split_ref"),
            fidelity=data.get("fidelity", 0),
            evidence_ref=data.get("evidence_ref"),
            compute_cost=data.get("compute_cost", 1.0),
            promotion_evidence=data.get("promotion_evidence"),
            status=data.get("status", EvaluationStatus.COMPLETE),
            metadata=dict(data.get("metadata", {})),
        )
        if data.get("content_hash") not in (None, obj.content_hash):
            raise ValueError("artifact content_hash does not match its payload")
        return obj


def normalize_evaluation_result(
    trial_id: str,
    result: Any,
    *,
    objective_name: str = "score",
    objective_spec_ref: Optional[str] = None,
    split_ref: Optional[str] = None,
    fidelity: int = 0,
) -> TrialEvaluationArtifact:
    """Normalize a legacy magic dict (or an existing artifact) into an artifact.

    A legacy dict may carry the magic keys ``score`` / ``cost`` / ``rank`` /
    ``total`` / ``promote`` / ``baseline_score`` / ``evidence_ref`` /
    ``evaluation_id``.  The normalization reads ONLY the primary objective
    ``score`` (default ``score``) and folds the promotion keys into
    ``promotion_evidence`` so the runner no longer reads bare magic keys.
    """
    if isinstance(result, TrialEvaluationArtifact):
        return result
    if not isinstance(result, dict):
        raise TypeError("evaluation result must be a dict or TrialEvaluationArtifact")
    result = dict(result)
    score = result.get("score")
    if score is None:
        raise ValueError("evaluation result is missing a primary objective score")
    evidence_ref = result.get("evidence_ref") or result.get("evaluation_id")
    if evidence_ref is None:
        raise ValueError("evaluation result must include an evidence_ref")
    promotion_evidence = None
    if any(k in result for k in ("rank", "total", "promote", "baseline_score")):
        promotion_evidence = {
            "rank": result.get("rank"),
            "total": result.get("total"),
            "promote": result.get("promote"),
            "baseline_score": result.get("baseline_score"),
        }
    cost = result.get("cost", 1.0)
    return TrialEvaluationArtifact(
        trial_id=trial_id,
        objective_values=[{"name": objective_name, "value": score}],
        objective_spec_ref=objective_spec_ref,
        split_ref=split_ref,
        fidelity=fidelity,
        evidence_ref=evidence_ref,
        compute_cost=cost,
        promotion_evidence=promotion_evidence,
        status=EvaluationStatus.COMPLETE,
        metadata={k: v for k, v in result.items() if k not in _MAGIC_KEYS},
    )


__all__ = [
    "TrialEvaluationArtifact",
    "ObjectiveValue",
    "EvaluationStatus",
    "normalize_evaluation_result",
]
