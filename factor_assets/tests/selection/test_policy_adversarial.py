"""Adversarial regression tests for selection admission boundaries."""

from datetime import datetime, timezone

import pytest

from factor_assets.selection import (
    GateEvaluation,
    GateResult,
    SelectionPolicy,
    SelectionReason,
)


def _passing_gate() -> GateEvaluation:
    return GateEvaluation(
        gate_name="quality",
        factor_id="F_ADVERSARIAL",
        result=GateResult.PASS,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        policy_name="adversarial",
        similarity_threshold=0.7,
        require_evidence=True,
    )


def test_nonempty_evidence_cannot_approve_without_gate_evaluations():
    decision = _policy().make_decision(
        factor_id="F_NO_GATES",
        gate_evaluations=[],
        evidence_refs=("EVD_PRESENT",),
    )

    assert decision.approved is False
    assert decision.reason is SelectionReason.REJECTED_GATE_FAILURE
    assert decision.gate_results == ()
    assert decision.notes == "No gate evaluations provided"


def test_negative_similarity_is_compared_by_absolute_score():
    decision = _policy().make_decision(
        factor_id="F_NEGATIVE_SIMILARITY",
        gate_evaluations=[_passing_gate()],
        evidence_refs=("EVD_PRESENT",),
        max_similarity=-0.99,
    )

    assert decision.approved is False
    assert decision.reason is SelectionReason.REJECTED_SIMILARITY
    assert "Similarity 0.990 exceeds threshold 0.700" in decision.notes


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_similarity_never_approves(score: float):
    decision = _policy().make_decision(
        factor_id="F_NONFINITE_SIMILARITY",
        gate_evaluations=[_passing_gate()],
        evidence_refs=("EVD_PRESENT",),
        max_similarity=score,
    )

    assert decision.approved is False
    assert decision.reason is SelectionReason.REJECTED_SIMILARITY
    assert decision.notes.startswith("Nonfinite similarity score")
