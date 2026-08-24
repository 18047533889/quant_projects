"""
Tests for selection decision records and policies.
"""

import pytest

import factor_assets.selection.policy as policy_module
from factor_assets.selection import (
    SelectionReason,
    SelectionDecision,
    SelectionPolicy,
    GateResult,
    GateEvaluation,
)


def test_selection_decision_creation():
    """Test SelectionDecision creation."""
    decision = SelectionDecision(
        decision_id="SD_001",
        factor_id="F001",
        approved=True,
        reason=SelectionReason.APPROVED,
        timestamp="2024-01-01T00:00:00Z",
        policy_version="1.0",
        evidence_refs=("EVD_001", "EVD_002"),
        gate_results=("GATE_001", "GATE_002"),
        similarity_refs=("SIM_001",),
        actor="system",
        notes="All criteria met",
    )

    assert decision.decision_id == "SD_001"
    assert decision.factor_id == "F001"
    assert decision.approved
    assert decision.reason == SelectionReason.APPROVED
    assert len(decision.evidence_refs) == 2
    assert len(decision.gate_results) == 2


def test_selection_decision_requires_fields():
    """SelectionDecision must have required fields."""
    with pytest.raises(ValueError, match="decision_id"):
        SelectionDecision(
            decision_id="",
            factor_id="F001",
            approved=True,
            reason=SelectionReason.APPROVED,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="1.0",
            evidence_refs=(),
            gate_results=(),
        )

    with pytest.raises(ValueError, match="factor_id"):
        SelectionDecision(
            decision_id="SD_001",
            factor_id="",
            approved=True,
            reason=SelectionReason.APPROVED,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="1.0",
            evidence_refs=(),
            gate_results=(),
        )

    with pytest.raises(ValueError, match="timestamp"):
        SelectionDecision(
            decision_id="SD_001",
            factor_id="F001",
            approved=True,
            reason=SelectionReason.APPROVED,
            timestamp="",
            policy_version="1.0",
            evidence_refs=(),
            gate_results=(),
        )

    with pytest.raises(ValueError, match="policy_version"):
        SelectionDecision(
            decision_id="SD_001",
            factor_id="F001",
            approved=True,
            reason=SelectionReason.APPROVED,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="",
            evidence_refs=(),
            gate_results=(),
        )


def test_selection_decision_properties():
    """Test SelectionDecision properties."""
    approved_decision = SelectionDecision(
        decision_id="SD_001",
        factor_id="F001",
        approved=True,
        reason=SelectionReason.APPROVED,
        timestamp="2024-01-01T00:00:00Z",
        policy_version="1.0",
        evidence_refs=("EVD_001",),
        gate_results=("GATE_001",),
    )

    assert approved_decision.is_approved
    assert not approved_decision.is_rejected
    assert approved_decision.has_evidence
    assert approved_decision.has_gate_results

    rejected_decision = SelectionDecision(
        decision_id="SD_002",
        factor_id="F002",
        approved=False,
        reason=SelectionReason.REJECTED_GATE_FAILURE,
        timestamp="2024-01-01T00:00:00Z",
        policy_version="1.0",
        evidence_refs=(),
        gate_results=(),
    )

    assert not rejected_decision.is_approved
    assert rejected_decision.is_rejected
    assert not rejected_decision.has_evidence
    assert not rejected_decision.has_gate_results


def test_selection_policy_creation():
    """Test SelectionPolicy creation."""
    policy = SelectionPolicy(
        policy_name="standard_policy",
        policy_version="1.0",
        similarity_threshold=0.7,
        require_evidence=True,
    )

    assert policy.policy_name == "standard_policy"
    assert policy.policy_version == "1.0"


def test_selection_policy_requires_name():
    """SelectionPolicy must have policy_name."""
    with pytest.raises(ValueError, match="policy_name"):
        SelectionPolicy(policy_name="", policy_version="1.0")


def test_selection_policy_requires_version():
    """SelectionPolicy must have policy_version."""
    with pytest.raises(ValueError, match="policy_version"):
        SelectionPolicy(policy_name="test", policy_version="")


def test_selection_policy_similarity_threshold_bounds():
    """Similarity threshold must be in [0, 1]."""
    with pytest.raises(ValueError, match="similarity_threshold must be in"):
        SelectionPolicy(
            policy_name="test",
            policy_version="1.0",
            similarity_threshold=1.5,
        )

    with pytest.raises(ValueError, match="similarity_threshold must be in"):
        SelectionPolicy(
            policy_name="test",
            policy_version="1.0",
            similarity_threshold=-0.1,
        )


def test_selection_policy_approve_all_pass():
    """Test policy approves when all criteria pass."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
        require_evidence=True,
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=("EVD_001",),
    )

    assert decision.approved
    assert decision.reason == SelectionReason.APPROVED


def test_selection_policy_reject_no_evidence():
    """Test policy rejects when evidence is required but missing."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
        require_evidence=True,
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=(),  # No evidence
    )

    assert not decision.approved
    assert decision.reason == SelectionReason.REJECTED_EVIDENCE


def test_selection_policy_allow_no_evidence():
    """Test policy allows missing evidence when not required."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
        require_evidence=False,
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=(),
    )

    assert decision.approved
    assert decision.reason == SelectionReason.APPROVED


def test_selection_policy_reject_gate_failure():
    """Test policy rejects when gates fail."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
    )

    passing_gate = GateEvaluation(
        gate_name="pass_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    failing_gate = GateEvaluation(
        gate_name="fail_gate",
        factor_id="F001",
        result=GateResult.FAIL,
        timestamp="2024-01-01T00:00:00Z",
    )

    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[passing_gate, failing_gate],
        evidence_refs=("EVD_001",),
    )

    assert not decision.approved
    assert decision.reason == SelectionReason.REJECTED_GATE_FAILURE
    assert "fail_gate" in decision.notes


def test_selection_policy_rejects_gate_from_different_factor():
    """A passing gate cannot be replayed to approve another factor."""
    policy = SelectionPolicy(policy_name="test_policy")
    replayed_gate = GateEvaluation(
        gate_name="quality_gate",
        factor_id="DIFFERENT_FACTOR",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    decision = policy.make_decision(
        factor_id="TARGET_FACTOR",
        gate_evaluations=[replayed_gate],
        evidence_refs=("EVD_001",),
    )

    assert not decision.approved
    assert decision.reason == SelectionReason.REJECTED_GATE_FAILURE
    assert "quality_gate" in decision.notes


def test_selection_policy_reject_high_similarity():
    """Test policy rejects when similarity exceeds threshold."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
        similarity_threshold=0.7,
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=("EVD_001",),
        max_similarity=0.85,  # Exceeds 0.7 threshold
    )

    assert not decision.approved
    assert decision.reason == SelectionReason.REJECTED_SIMILARITY


def test_selection_policy_allow_low_similarity():
    """Test policy allows when similarity below threshold."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
        similarity_threshold=0.7,
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=("EVD_001",),
        max_similarity=0.5,  # Below 0.7 threshold
    )

    assert decision.approved
    assert decision.reason == SelectionReason.APPROVED


def test_selection_policy_get_decision():
    """Test retrieving decisions by factor ID."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=("EVD_001",),
    )

    retrieved = policy.get_decision("F001")
    assert retrieved is not None
    assert retrieved.decision_id == decision.decision_id

    not_found = policy.get_decision("F999")
    assert not_found is None


def test_selection_policy_same_timestamp_decisions_have_distinct_ids(monkeypatch):
    """Repeated timestamps must not overwrite decision identity or history."""
    fixed_now = policy_module.datetime(2024, 1, 2, tzinfo=policy_module.timezone.utc)

    class FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(policy_module, "datetime", FixedDateTime)
    policy = SelectionPolicy(policy_name="test_policy")
    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    first = policy.make_decision(
        factor_id="F001", gate_evaluations=[gate_eval], evidence_refs=("EVD_001",)
    )
    second = policy.make_decision(
        factor_id="F001", gate_evaluations=[gate_eval], evidence_refs=("EVD_002",)
    )

    assert first.decision_id.startswith("SD_F001_20240102T000000000000")
    assert second.decision_id.startswith("SD_F001_20240102T000000000000")
    assert first.decision_id != second.decision_id
    assert policy.get_all_decisions("F001") == [first, second]
    assert policy.get_decision("F001") == second


def test_selection_policy_get_most_recent():
    """Test getting most recent decision for a factor."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    # Make multiple decisions for same factor
    decision1 = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=("EVD_001",),
    )

    decision2 = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=("EVD_002",),
    )

    # Should get most recent
    retrieved = policy.get_decision("F001")
    assert retrieved is not None
    assert retrieved.decision_id == decision2.decision_id


def test_selection_policy_get_all_decisions():
    """Test retrieving all decisions."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=("EVD_001",),
    )

    policy.make_decision(
        factor_id="F002",
        gate_evaluations=[gate_eval],
        evidence_refs=("EVD_002",),
    )

    all_decisions = policy.get_all_decisions()
    assert len(all_decisions) == 2

    f001_decisions = policy.get_all_decisions(factor_id="F001")
    assert len(f001_decisions) == 1
    assert f001_decisions[0].factor_id == "F001"


def test_selection_policy_counts():
    """Test counting approved and rejected decisions."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
    )

    passing_gate = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    failing_gate = GateEvaluation(
        gate_name="test_gate",
        factor_id="F002",
        result=GateResult.FAIL,
        timestamp="2024-01-01T00:00:00Z",
    )

    # Approve one
    policy.make_decision(
        factor_id="F001",
        gate_evaluations=[passing_gate],
        evidence_refs=("EVD_001",),
    )

    # Reject one
    policy.make_decision(
        factor_id="F002",
        gate_evaluations=[failing_gate],
        evidence_refs=("EVD_002",),
    )

    assert policy.count_approved() == 1
    assert policy.count_rejected() == 1
    assert policy.count_total() == 2


def test_selection_policy_decision_with_actor():
    """Test decision with actor information."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=("EVD_001",),
        actor="user_123",
        notes="Manual review completed",
    )

    assert decision.actor == "user_123"
    assert decision.notes == "Manual review completed"


def test_selection_reason_enum():
    """Test SelectionReason enum values."""
    assert SelectionReason.APPROVED.value == "APPROVED"
    assert SelectionReason.REJECTED_GATE_FAILURE.value == "REJECTED_GATE_FAILURE"
    assert SelectionReason.REJECTED_SIMILARITY.value == "REJECTED_SIMILARITY"
    assert SelectionReason.REJECTED_EVIDENCE.value == "REJECTED_EVIDENCE"
    assert SelectionReason.REJECTED_QUALITY.value == "REJECTED_QUALITY"
    assert SelectionReason.PENDING_EVALUATION.value == "PENDING_EVALUATION"
    assert SelectionReason.MANUAL_OVERRIDE.value == "MANUAL_OVERRIDE"


def test_selection_decision_metadata_default():
    """Test metadata defaults to empty dict."""
    decision = SelectionDecision(
        decision_id="SD_001",
        factor_id="F001",
        approved=True,
        reason=SelectionReason.APPROVED,
        timestamp="2024-01-01T00:00:00Z",
        policy_version="1.0",
        evidence_refs=(),
        gate_results=(),
    )

    assert decision.metadata == {}


def test_selection_decision_metadata_is_immutable_snapshot():
    metadata = {"source": "gate"}
    decision = SelectionDecision(
        decision_id="SD_001",
        factor_id="F001",
        approved=True,
        reason=SelectionReason.APPROVED,
        timestamp="2024-01-01T00:00:00Z",
        policy_version="1.0",
        evidence_refs=(),
        gate_results=(),
        metadata=metadata,
    )

    metadata["source"] = "caller-mutated"
    assert decision.metadata["source"] == "gate"
    with pytest.raises(TypeError):
        decision.metadata["source"] = "direct-mutation"


def test_selection_policy_priority_order():
    """Test that policy checks are evaluated in correct priority order."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
        similarity_threshold=0.7,
        require_evidence=True,
    )

    gate_eval = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    # Evidence check should come first (even if gates pass and similarity low)
    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[gate_eval],
        evidence_refs=(),  # Missing evidence
        max_similarity=0.5,  # Good similarity
    )

    assert decision.reason == SelectionReason.REJECTED_EVIDENCE


def test_selection_policy_gate_check_before_similarity():
    """Test that gate check comes before similarity check."""
    policy = SelectionPolicy(
        policy_name="test_policy",
        policy_version="1.0",
        similarity_threshold=0.7,
    )

    failing_gate = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.FAIL,
        timestamp="2024-01-01T00:00:00Z",
    )

    # Gate failure should be reported even if similarity is bad
    decision = policy.make_decision(
        factor_id="F001",
        gate_evaluations=[failing_gate],
        evidence_refs=("EVD_001",),
        max_similarity=0.95,  # High similarity
    )

    assert decision.reason == SelectionReason.REJECTED_GATE_FAILURE
