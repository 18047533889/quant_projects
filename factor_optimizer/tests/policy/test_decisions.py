"""Tests for admission decisions policy."""

import pytest
from factor_optimizer.policy.decisions import (
    AdmissionCriteria,
    AdmissionDecision,
    AdmissionPolicy,
    AdmissionVerdict,
    RejectionReason,
)


def test_admission_criteria_creation():
    """Test basic AdmissionCriteria creation."""
    criteria = AdmissionCriteria(
        max_complexity_cost=100.0,
        min_expected_value=0.1,
        max_lookback_periods=60,
    )

    assert criteria.max_complexity_cost == 100.0
    assert criteria.min_expected_value == 0.1
    assert criteria.max_lookback_periods == 60


def test_admission_criteria_defaults():
    """Test AdmissionCriteria default values."""
    criteria = AdmissionCriteria()

    assert criteria.max_complexity_cost == float("inf")
    assert criteria.min_expected_value == 0.0
    assert criteria.max_lookback_periods == 252
    assert criteria.allowed_domains == []
    assert criteria.allowed_sources == []


def test_admission_criteria_serialization():
    """Test AdmissionCriteria serialization round-trip."""
    original = AdmissionCriteria(
        max_complexity_cost=50.0,
        allowed_domains=["equity", "futures"],
        min_parent_quality=0.5,
    )

    serialized = original.to_dict()
    assert serialized["max_complexity_cost"] == 50.0
    assert "equity" in serialized["allowed_domains"]

    restored = AdmissionCriteria.from_dict(serialized)
    assert restored.max_complexity_cost == original.max_complexity_cost
    assert restored.allowed_domains == original.allowed_domains


def test_admission_decision_creation():
    """Test basic AdmissionDecision creation."""
    decision = AdmissionDecision(
        decision_id="dec_001",
        mutation_id="mut_001",
        trial_id="trial_001",
        verdict=AdmissionVerdict.ADMITTED,
    )

    assert decision.decision_id == "dec_001"
    assert decision.mutation_id == "mut_001"
    assert decision.verdict == AdmissionVerdict.ADMITTED


def test_admission_decision_status_checks():
    """Test AdmissionDecision status check methods."""
    admitted = AdmissionDecision(
        decision_id="d1",
        mutation_id="m1",
        trial_id="t1",
        verdict=AdmissionVerdict.ADMITTED,
    )
    assert admitted.is_approved()
    assert not admitted.is_rejected()
    assert not admitted.is_deferred()

    rejected = AdmissionDecision(
        decision_id="d2",
        mutation_id="m2",
        trial_id="t2",
        verdict=AdmissionVerdict.REJECTED,
        rejection_reasons=[RejectionReason.BUDGET_EXCEEDED],
    )
    assert not rejected.is_approved()
    assert rejected.is_rejected()
    assert rejected.primary_rejection_reason() == RejectionReason.BUDGET_EXCEEDED

    deferred = AdmissionDecision(
        decision_id="d3",
        mutation_id="m3",
        trial_id="t3",
        verdict=AdmissionVerdict.DEFERRED,
    )
    assert not deferred.is_approved()
    assert deferred.is_deferred()


def test_admission_decision_conditional_approval():
    """Test conditional admission verdict."""
    decision = AdmissionDecision(
        decision_id="d1",
        mutation_id="m1",
        trial_id="t1",
        verdict=AdmissionVerdict.CONDITIONAL,
        conditions=["Require manual review", "Limited budget allocation"],
    )

    assert decision.is_approved()
    assert len(decision.conditions) == 2


def test_admission_decision_serialization():
    """Test AdmissionDecision serialization round-trip."""
    criteria = AdmissionCriteria(max_complexity_cost=100.0)
    original = AdmissionDecision(
        decision_id="d1",
        mutation_id="m1",
        trial_id="t1",
        verdict=AdmissionVerdict.REJECTED,
        criteria_used=criteria,
        rejection_reasons=[RejectionReason.COMPLEXITY_LIMIT, RejectionReason.LOW_EXPECTED_VALUE],
        complexity_cost=150.0,
        expected_value=0.05,
    )

    serialized = original.to_dict()
    assert serialized["verdict"] == "rejected"
    assert "complexity_limit" in serialized["rejection_reasons"]

    restored = AdmissionDecision.from_dict(serialized)
    assert restored.verdict == original.verdict
    assert restored.rejection_reasons == original.rejection_reasons
    assert restored.complexity_cost == original.complexity_cost


def test_admission_policy_initialization():
    """Test AdmissionPolicy initialization."""
    policy = AdmissionPolicy()
    assert policy.criteria is not None
    assert policy.criteria.max_complexity_cost == float("inf")

    custom_criteria = AdmissionCriteria(max_complexity_cost=50.0)
    policy_with_criteria = AdmissionPolicy(criteria=custom_criteria)
    assert policy_with_criteria.criteria.max_complexity_cost == 50.0


def test_admission_policy_admit_simple():
    """Test simple admission decision."""
    criteria = AdmissionCriteria(
        max_complexity_cost=100.0,
        min_expected_value=0.1,
    )
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=50.0,
        expected_value=0.5,
    )

    assert decision.verdict == AdmissionVerdict.ADMITTED
    assert len(decision.rejection_reasons) == 0


def test_admission_policy_reject_complexity():
    """Test rejection due to complexity limit."""
    criteria = AdmissionCriteria(max_complexity_cost=100.0)
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=150.0,
        expected_value=0.5,
    )

    assert decision.verdict == AdmissionVerdict.REJECTED
    assert RejectionReason.COMPLEXITY_LIMIT in decision.rejection_reasons


def test_admission_policy_reject_low_value():
    """Test rejection due to low expected value."""
    criteria = AdmissionCriteria(min_expected_value=0.3)
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=50.0,
        expected_value=0.1,
    )

    assert decision.verdict == AdmissionVerdict.REJECTED
    assert RejectionReason.LOW_EXPECTED_VALUE in decision.rejection_reasons


def test_admission_policy_reject_lookback():
    """Test rejection due to excessive lookback."""
    criteria = AdmissionCriteria(max_lookback_periods=100)
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=50.0,
        expected_value=0.5,
        lookback_periods=200,
    )

    assert decision.verdict == AdmissionVerdict.REJECTED
    assert RejectionReason.COMPLEXITY_LIMIT in decision.rejection_reasons


def test_admission_policy_reject_domain():
    """Test rejection due to domain violation."""
    criteria = AdmissionCriteria(allowed_domains=["equity"])
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=50.0,
        expected_value=0.5,
        domains=["futures", "options"],
    )

    assert decision.verdict == AdmissionVerdict.REJECTED
    assert RejectionReason.DOMAIN_VIOLATION in decision.rejection_reasons


def test_admission_policy_allow_matching_domain():
    """Test admission with matching domain."""
    criteria = AdmissionCriteria(allowed_domains=["equity", "futures"])
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=50.0,
        expected_value=0.5,
        domains=["equity"],
    )

    assert decision.verdict == AdmissionVerdict.ADMITTED


def test_admission_policy_reject_parent_quality():
    """Test rejection due to low parent quality."""
    criteria = AdmissionCriteria(min_parent_quality=0.7)
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=50.0,
        expected_value=0.5,
        parent_quality=0.4,
    )

    assert decision.verdict == AdmissionVerdict.REJECTED
    assert RejectionReason.PARENT_QUALITY in decision.rejection_reasons


def test_admission_policy_multiple_rejections():
    """Test multiple rejection reasons."""
    criteria = AdmissionCriteria(
        max_complexity_cost=100.0,
        min_expected_value=0.3,
        min_parent_quality=0.6,
    )
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=150.0,
        expected_value=0.1,
        parent_quality=0.3,
    )

    assert decision.verdict == AdmissionVerdict.REJECTED
    assert len(decision.rejection_reasons) == 3


def test_admission_policy_get_decision():
    """Test retrieving decision by ID."""
    policy = AdmissionPolicy()
    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=50.0,
        expected_value=0.5,
    )

    retrieved = policy.get_decision(decision.decision_id)
    assert retrieved is not None
    assert retrieved.mutation_id == "m1"


def test_admission_policy_get_decisions_for_trial():
    """Test retrieving all decisions for a trial."""
    policy = AdmissionPolicy()
    policy.decide(mutation_id="m1", trial_id="t1", complexity_cost=50.0, expected_value=0.5)

    decisions = policy.get_decisions_for_trial("t1")
    assert len(decisions) == 1
    assert decisions[0].trial_id == "t1"


def test_admission_policy_update_criteria():
    """Test updating policy criteria."""
    policy = AdmissionPolicy()
    original_cost = policy.criteria.max_complexity_cost

    new_criteria = AdmissionCriteria(max_complexity_cost=200.0)
    policy.update_criteria(new_criteria)

    assert policy.criteria.max_complexity_cost == 200.0
    assert policy.criteria.max_complexity_cost != original_cost


def test_admission_policy_stats_empty():
    """Test statistics with no decisions."""
    policy = AdmissionPolicy()
    stats = policy.get_admission_stats()

    assert stats["total"] == 0


def test_admission_policy_stats():
    """Test admission statistics."""
    criteria = AdmissionCriteria(max_complexity_cost=100.0)
    policy = AdmissionPolicy(criteria=criteria)

    policy.decide(mutation_id="m1", trial_id="t1", complexity_cost=50.0, expected_value=0.5)
    policy.decide(mutation_id="m2", trial_id="t2", complexity_cost=150.0, expected_value=0.5)
    policy.decide(mutation_id="m3", trial_id="t3", complexity_cost=80.0, expected_value=0.5)

    stats = policy.get_admission_stats()

    assert stats["total"] == 3
    assert stats["admitted"] == 2
    assert stats["rejected"] == 1
    assert stats["admission_rate"] == 2.0 / 3.0
    assert "complexity_limit" in stats["rejection_reasons"]


def test_admission_policy_clear_history():
    """Test clearing decision history."""
    policy = AdmissionPolicy()
    policy.decide(mutation_id="m1", trial_id="t1", complexity_cost=50.0, expected_value=0.5)

    stats = policy.get_admission_stats()
    assert stats["total"] == 1

    policy.clear_history()
    stats = policy.get_admission_stats()
    assert stats["total"] == 0


def test_admission_verdict_enum():
    """Test AdmissionVerdict enum values."""
    assert AdmissionVerdict.ADMITTED.value == "admitted"
    assert AdmissionVerdict.REJECTED.value == "rejected"
    assert AdmissionVerdict.DEFERRED.value == "deferred"
    assert AdmissionVerdict.CONDITIONAL.value == "conditional"


def test_rejection_reason_enum():
    """Test RejectionReason enum values."""
    assert RejectionReason.BUDGET_EXCEEDED.value == "budget_exceeded"
    assert RejectionReason.DUPLICATE.value == "duplicate"
    assert RejectionReason.COMPLEXITY_LIMIT.value == "complexity_limit"


def test_admission_policy_metadata():
    """Test decision metadata storage."""
    policy = AdmissionPolicy()
    metadata = {"campaign_id": "campaign_001", "iteration": 5}

    decision = policy.decide(
        mutation_id="m1",
        trial_id="t1",
        complexity_cost=50.0,
        expected_value=0.5,
        metadata=metadata,
    )

    assert decision.decision_metadata["campaign_id"] == "campaign_001"
    assert decision.decision_metadata["iteration"] == 5
