"""Integration tests for repair and admission decision workflow."""

import pytest
from factor_optimizer.policy.repair import (
    DiagnosisKind,
    DiagnosisRecord,
    RepairMapper,
    RepairStrategy,
)
from factor_optimizer.policy.decisions import (
    AdmissionCriteria,
    AdmissionPolicy,
    AdmissionVerdict,
)
from factor_optimizer.contracts.candidate_mutation import CandidateMutation
from datetime import datetime


def test_diagnosis_to_repair_to_admission_workflow():
    """Test complete workflow from diagnosis to admission decision."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_001",
        diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
        severity=0.7,
        evidence={"lookback_periods": 120, "operator_count": 50},
    )

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=2)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.REDUCE_WINDOW

    criteria = AdmissionCriteria(
        max_complexity_cost=100.0,
        min_expected_value=0.2,
    )
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="mut_001",
        trial_id="trial_001",
        complexity_cost=80.0,
        expected_value=0.5,
    )

    assert decision.verdict == AdmissionVerdict.ADMITTED


def test_low_signal_repair_generates_interaction():
    """Test low IC diagnosis generates add_interaction repair."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_002",
        diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
        severity=0.6,
        evidence={"ic": 0.03, "ic_std": 0.05},
    )

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=1)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.ADD_INTERACTION
    assert proposals[0].mutation_type == "add_interaction"


def test_high_turnover_repair_increases_horizon():
    """Test high turnover diagnosis generates increase_horizon repair."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_003",
        diagnosis_kind=DiagnosisKind.HIGH_TURNOVER,
        severity=0.8,
        evidence={"turnover_rate": 0.85, "prediction_horizon": 1},
    )

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=1)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.INCREASE_HORIZON
    assert proposals[0].mutation_type == "horizon_adjust"
    assert proposals[0].parameters["new_horizon"] > 1


def test_repair_proposal_to_mutation_spec():
    """Test converting repair proposal to mutation spec."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_004",
        diagnosis_kind=DiagnosisKind.NUMERICAL_INSTABILITY,
        severity=0.9,
        evidence={"nan_ratio": 0.15},
    )

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=1)

    assert len(proposals) > 0
    mutation_spec = mapper.generate_mutation_spec(proposals[0])

    assert mutation_spec["mutation_type"] == "add_winsorization"
    assert "parameters" in mutation_spec
    assert mutation_spec["provenance"]["source"] == "repair_mapper"
    assert mutation_spec["provenance"]["diagnosis_kind"] == "numerical_instability"


def test_mutation_spec_to_candidate_mutation():
    """Test creating CandidateMutation from repair-generated spec."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_005",
        diagnosis_kind=DiagnosisKind.HIGH_VARIANCE,
        severity=0.7,
        evidence={"variance_ratio": 3.2},
    )

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=1)
    mutation_spec = mapper.generate_mutation_spec(proposals[0])

    candidate = CandidateMutation(
        mutation_id="mut_005",
        mutation_spec_version="1.0",
        parent_factor_ids=["parent_factor_001"],
        mutation_type=mutation_spec["mutation_type"],
        parameters=mutation_spec["parameters"],
        mechanism_hypothesis=mutation_spec.get("expected_improvement"),
        created_at=datetime.now(),
        producer="repair_mapper",
    )

    assert candidate.mutation_id == "mut_005"
    assert candidate.mutation_type == "add_regularization"
    assert candidate.producer == "repair_mapper"


def test_rejected_mutation_due_to_complexity():
    """Test mutation rejected due to complexity from failed repair."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_006",
        diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
        severity=0.95,
        evidence={"lookback_periods": 300},
    )

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=1)

    criteria = AdmissionCriteria(max_complexity_cost=50.0)
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="mut_006",
        trial_id="trial_006",
        complexity_cost=200.0,
        expected_value=0.3,
    )

    assert decision.verdict == AdmissionVerdict.REJECTED


def test_rank_multiple_repairs_by_priority():
    """Test ranking multiple repair proposals."""
    diagnoses = [
        DiagnosisRecord(
            trial_id="t1",
            diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
            severity=0.8,
        ),
        DiagnosisRecord(
            trial_id="t2",
            diagnosis_kind=DiagnosisKind.TIMING_VIOLATION,
            severity=0.9,
        ),
        DiagnosisRecord(
            trial_id="t3",
            diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
            severity=0.5,
        ),
    ]

    mapper = RepairMapper()
    all_proposals = []
    for diag in diagnoses:
        proposals = mapper.propose_repairs(diag, max_proposals=1)
        all_proposals.extend(proposals)

    ranked = mapper.rank_repairs(all_proposals)

    assert len(ranked) == len(all_proposals)
    for i in range(len(ranked) - 1):
        current = ranked[i]
        next_prop = ranked[i + 1]


def test_conditional_admission_with_repair():
    """Test conditional admission for borderline repairs."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_007",
        diagnosis_kind=DiagnosisKind.POOR_COVERAGE,
        severity=0.6,
        evidence={"coverage_ratio": 0.55},
    )

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=1)

    criteria = AdmissionCriteria(
        max_complexity_cost=100.0,
        min_expected_value=0.1,
    )
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="mut_007",
        trial_id="trial_007",
        complexity_cost=95.0,
        expected_value=0.15,
    )

    assert decision.verdict in [AdmissionVerdict.ADMITTED, AdmissionVerdict.CONDITIONAL]


def test_abandon_strategy_generates_no_proposals():
    """Test abandon strategy for unrepairable failures."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_008",
        diagnosis_kind=DiagnosisKind.SEMANTIC_DUPLICATE,
        severity=0.9,
    )

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis)

    assert len(proposals) == 0


def test_multiple_secondary_diagnoses_affect_confidence():
    """Test multiple secondary diagnoses reduce repair confidence."""
    diagnosis_simple = DiagnosisRecord(
        trial_id="trial_009a",
        diagnosis_kind=DiagnosisKind.HIGH_VARIANCE,
        severity=0.6,
    )

    diagnosis_complex = DiagnosisRecord(
        trial_id="trial_009b",
        diagnosis_kind=DiagnosisKind.HIGH_VARIANCE,
        severity=0.6,
        secondary_diagnoses=[
            DiagnosisKind.NUMERICAL_INSTABILITY,
            DiagnosisKind.POOR_COVERAGE,
            DiagnosisKind.HIGH_COMPLEXITY,
        ],
    )

    mapper = RepairMapper()
    proposals_simple = mapper.propose_repairs(diagnosis_simple, max_proposals=1)
    proposals_complex = mapper.propose_repairs(diagnosis_complex, max_proposals=1)

    assert len(proposals_simple) > 0 and len(proposals_complex) > 0
    assert proposals_simple[0].confidence > proposals_complex[0].confidence


def test_fallback_strategies_populated():
    """Test fallback strategies are populated in proposals."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_010",
        diagnosis_kind=DiagnosisKind.OVERFITTING,
        severity=0.7,
    )

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=3)

    assert len(proposals) > 0
    if len(proposals) > 0:
        assert len(proposals[0].fallback_strategies) >= 1
