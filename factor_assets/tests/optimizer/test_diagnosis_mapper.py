"""Tests for diagnosis mapper."""

import pytest
from factor_assets.optimizer.diagnosis_mapper import (
    DiagnosisKind,
    DiagnosisEvidence,
    RepairCandidate,
    DiagnosisMapper,
    DefaultRepairStrategy,
    create_default_mapper,
)


def test_diagnosis_evidence_creation():
    """Test diagnosis evidence creation."""
    evidence = DiagnosisEvidence(
        kind=DiagnosisKind.POOR_RANK_IC,
        severity=0.8,
        metrics={"rank_ic": 0.01},
        context={"split": "validation"},
        confidence=0.9,
        source="qe_analyzer",
    )

    assert evidence.kind == DiagnosisKind.POOR_RANK_IC
    assert evidence.severity == 0.8
    assert evidence.confidence == 0.9


def test_diagnosis_evidence_validation():
    """Test diagnosis evidence validation."""
    with pytest.raises(ValueError):
        DiagnosisEvidence(
            kind=DiagnosisKind.POOR_RANK_IC,
            severity=1.5,  # Invalid: > 1.0
        )

    with pytest.raises(ValueError):
        DiagnosisEvidence(
            kind=DiagnosisKind.POOR_RANK_IC,
            severity=0.5,
            confidence=-0.1,  # Invalid: < 0.0
        )


def test_repair_candidate_creation():
    """Test repair candidate creation."""
    repair = RepairCandidate(
        mutation_type="adjust_window",
        parameters={"direction": "increase", "scale": 1.5},
        priority=0.7,
        rationale="Increase window to capture more signal",
        expected_improvement={"rank_ic": 0.05},
        risk_score=0.3,
    )

    assert repair.mutation_type == "adjust_window"
    assert repair.priority == 0.7
    assert repair.risk_score == 0.3


def test_repair_candidate_validation():
    """Test repair candidate validation."""
    with pytest.raises(ValueError):
        RepairCandidate(
            mutation_type="adjust_window",
            parameters={},
            priority=1.5,  # Invalid
            rationale="test",
        )


def test_default_repair_strategy_poor_rank_ic():
    """Test default strategy for poor rank IC."""
    strategy = DefaultRepairStrategy()

    diagnosis = DiagnosisEvidence(
        kind=DiagnosisKind.POOR_RANK_IC,
        severity=0.8,
    )

    assert strategy.applicable_to(diagnosis)

    repairs = strategy.generate_repairs(diagnosis, {})
    assert len(repairs) > 0
    assert any(r.mutation_type == "adjust_window" for r in repairs)


def test_default_repair_strategy_high_turnover():
    """Test default strategy for high turnover."""
    strategy = DefaultRepairStrategy()

    diagnosis = DiagnosisEvidence(
        kind=DiagnosisKind.HIGH_TURNOVER,
        severity=0.7,
    )

    repairs = strategy.generate_repairs(diagnosis, {})
    assert len(repairs) > 0
    assert any(r.mutation_type == "add_smoothing" for r in repairs)


def test_diagnosis_mapper():
    """Test diagnosis mapper."""
    mapper = DiagnosisMapper()
    mapper.register_strategy(DefaultRepairStrategy())

    diagnoses = [
        DiagnosisEvidence(
            kind=DiagnosisKind.POOR_RANK_IC,
            severity=0.8,
        ),
        DiagnosisEvidence(
            kind=DiagnosisKind.HIGH_TURNOVER,
            severity=0.6,
        ),
    ]

    repairs = mapper.map_to_repairs(diagnoses, {})
    assert len(repairs) > 0
    assert all(isinstance(r, RepairCandidate) for r in repairs)


def test_diagnosis_mapper_priority_sorting():
    """Test that repairs are sorted by priority."""
    mapper = create_default_mapper()

    diagnoses = [
        DiagnosisEvidence(kind=DiagnosisKind.POOR_RANK_IC, severity=0.5),
        DiagnosisEvidence(kind=DiagnosisKind.HIGH_TURNOVER, severity=0.9),
    ]

    repairs = mapper.map_to_repairs(diagnoses, {})

    # Check that repairs are sorted descending by priority
    for i in range(len(repairs) - 1):
        assert repairs[i].priority >= repairs[i + 1].priority


def test_diagnosis_mapper_max_repairs():
    """Test max_repairs limit."""
    mapper = create_default_mapper()

    diagnoses = [
        DiagnosisEvidence(kind=DiagnosisKind.POOR_RANK_IC, severity=0.8),
        DiagnosisEvidence(kind=DiagnosisKind.HIGH_TURNOVER, severity=0.8),
        DiagnosisEvidence(kind=DiagnosisKind.OVERFITTING, severity=0.8),
    ]

    repairs = mapper.map_to_repairs(diagnoses, {}, max_repairs=2)
    assert len(repairs) <= 2
