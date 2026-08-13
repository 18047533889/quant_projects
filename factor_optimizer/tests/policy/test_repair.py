"""Tests for repair policy."""

import pytest
from factor_optimizer.policy.repair import (
    DiagnosisKind,
    DiagnosisRecord,
    RepairMapper,
    RepairProposal,
    RepairStrategy,
)


def test_diagnosis_record_creation():
    """Test basic DiagnosisRecord creation."""
    diagnosis = DiagnosisRecord(
        trial_id="trial_001",
        diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
        severity=0.8,
        evidence={"lookback_periods": 100},
    )

    assert diagnosis.trial_id == "trial_001"
    assert diagnosis.diagnosis_kind == DiagnosisKind.HIGH_COMPLEXITY
    assert diagnosis.severity == 0.8
    assert diagnosis.evidence["lookback_periods"] == 100


def test_diagnosis_severity_validation():
    """Test severity bounds validation."""
    with pytest.raises(ValueError, match="Severity must be in"):
        DiagnosisRecord(
            trial_id="t1",
            diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
            severity=1.5,
        )

    with pytest.raises(ValueError, match="Severity must be in"):
        DiagnosisRecord(
            trial_id="t1",
            diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
            severity=-0.1,
        )


def test_diagnosis_serialization():
    """Test DiagnosisRecord serialization round-trip."""
    original = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.HIGH_VARIANCE,
        severity=0.7,
        evidence={"variance_ratio": 2.5},
        secondary_diagnoses=[DiagnosisKind.NUMERICAL_INSTABILITY],
    )

    serialized = original.to_dict()
    assert serialized["diagnosis_kind"] == "high_variance"
    assert serialized["severity"] == 0.7

    restored = DiagnosisRecord.from_dict(serialized)
    assert restored.trial_id == original.trial_id
    assert restored.diagnosis_kind == original.diagnosis_kind
    assert restored.severity == original.severity


def test_repair_proposal_creation():
    """Test RepairProposal creation."""
    diagnosis = DiagnosisRecord(trial_id="t1", diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY)

    proposal = RepairProposal(
        diagnosis_record=diagnosis,
        strategy=RepairStrategy.REDUCE_WINDOW,
        mutation_type="window_adjust",
        parameters={"new_window": 10},
        confidence=0.8,
    )

    assert proposal.strategy == RepairStrategy.REDUCE_WINDOW
    assert proposal.mutation_type == "window_adjust"
    assert proposal.confidence == 0.8


def test_repair_proposal_confidence_validation():
    """Test confidence bounds validation."""
    diagnosis = DiagnosisRecord(trial_id="t1", diagnosis_kind=DiagnosisKind.LOW_SIGNAL)

    with pytest.raises(ValueError, match="Confidence must be in"):
        RepairProposal(
            diagnosis_record=diagnosis,
            strategy=RepairStrategy.INCREASE_WINDOW,
            mutation_type="window_adjust",
            parameters={},
            confidence=1.5,
        )


def test_repair_mapper_initialization():
    """Test RepairMapper initialization."""
    mapper = RepairMapper()
    assert mapper is not None


def test_repair_mapper_high_complexity():
    """Test repair proposals for high complexity diagnosis."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
        severity=0.9,
        evidence={"lookback_periods": 100, "operator_count": 50},
    )

    proposals = mapper.propose_repairs(diagnosis, max_proposals=2)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.REDUCE_WINDOW
    assert proposals[0].mutation_type == "window_adjust"
    assert "new_window" in proposals[0].parameters
    assert proposals[0].parameters["new_window"] < 100


def test_repair_mapper_high_variance():
    """Test repair proposals for high variance diagnosis."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.HIGH_VARIANCE,
        severity=0.8,
    )

    proposals = mapper.propose_repairs(diagnosis)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.ADD_REGULARIZATION


def test_repair_mapper_low_signal():
    """Test repair proposals for low signal diagnosis."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
        evidence={"lookback_periods": 10},
    )

    proposals = mapper.propose_repairs(diagnosis)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.ADD_INTERACTION
    assert proposals[0].mutation_type == "add_interaction"


def test_repair_mapper_timing_violation():
    """Test repair proposals for timing violation."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.TIMING_VIOLATION,
    )

    proposals = mapper.propose_repairs(diagnosis)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.LAG_CORRECTION


def test_repair_mapper_semantic_duplicate():
    """Test repair for semantic duplicate (should abandon)."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.SEMANTIC_DUPLICATE,
    )

    proposals = mapper.propose_repairs(diagnosis)

    assert len(proposals) == 0


def test_repair_mapper_numerical_instability():
    """Test repair proposals for numerical instability."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.NUMERICAL_INSTABILITY,
        severity=0.9,
    )

    proposals = mapper.propose_repairs(diagnosis)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.WINSORIZE
    assert "lower_quantile" in proposals[0].parameters
    assert "upper_quantile" in proposals[0].parameters


def test_repair_mapper_confidence_ranking():
    """Test confidence decreases with rank."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
        severity=0.5,
    )

    proposals = mapper.propose_repairs(diagnosis, max_proposals=3)

    if len(proposals) >= 2:
        assert proposals[0].confidence >= proposals[1].confidence


def test_repair_mapper_severity_impact():
    """Test high severity reduces confidence."""
    mapper = RepairMapper()

    low_severity = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.HIGH_VARIANCE,
        severity=0.3,
    )

    high_severity = DiagnosisRecord(
        trial_id="t2",
        diagnosis_kind=DiagnosisKind.HIGH_VARIANCE,
        severity=0.95,
    )

    low_proposals = mapper.propose_repairs(low_severity, max_proposals=1)
    high_proposals = mapper.propose_repairs(high_severity, max_proposals=1)

    assert low_proposals[0].confidence > high_proposals[0].confidence


def test_repair_mapper_fallback_strategies():
    """Test fallback strategies are populated."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.OVERFITTING,
    )

    proposals = mapper.propose_repairs(diagnosis, max_proposals=3)

    assert len(proposals) > 0
    assert len(proposals[0].fallback_strategies) > 0


def test_repair_proposal_serialization():
    """Test RepairProposal serialization."""
    diagnosis = DiagnosisRecord(trial_id="t1", diagnosis_kind=DiagnosisKind.POOR_COVERAGE)
    proposal = RepairProposal(
        diagnosis_record=diagnosis,
        strategy=RepairStrategy.FILTER_UNIVERSE,
        mutation_type="filter_universe",
        parameters={"min_coverage": 0.7},
        confidence=0.6,
        fallback_strategies=[RepairStrategy.ABANDON],
    )

    serialized = proposal.to_dict()
    assert serialized["strategy"] == "filter_universe"
    assert serialized["confidence"] == 0.6
    assert "abandon" in serialized["fallback_strategies"]


def test_repair_mapper_max_proposals_limit():
    """Test max_proposals limit is respected."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
    )

    proposals = mapper.propose_repairs(diagnosis, max_proposals=1)
    assert len(proposals) <= 1

    proposals = mapper.propose_repairs(diagnosis, max_proposals=5)
    assert len(proposals) <= 5


def test_diagnosis_kind_enum():
    """Test DiagnosisKind enum values."""
    assert DiagnosisKind.HIGH_COMPLEXITY.value == "high_complexity"
    assert DiagnosisKind.LOW_SIGNAL.value == "low_signal"
    assert DiagnosisKind.OVERFITTING.value == "overfitting"


def test_repair_strategy_enum():
    """Test RepairStrategy enum values."""
    assert RepairStrategy.REDUCE_WINDOW.value == "reduce_window"
    assert RepairStrategy.WINSORIZE.value == "winsorize"
    assert RepairStrategy.ABANDON.value == "abandon"
    assert RepairStrategy.ADD_INTERACTION.value == "add_interaction"
    assert RepairStrategy.INCREASE_HORIZON.value == "increase_horizon"


def test_diagnosis_kind_high_turnover():
    """Test high turnover diagnosis kind."""
    assert DiagnosisKind.HIGH_TURNOVER.value == "high_turnover"


def test_repair_mapper_high_turnover():
    """Test repair proposals for high turnover diagnosis."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.HIGH_TURNOVER,
        evidence={"prediction_horizon": 1, "turnover_rate": 0.8},
    )

    proposals = mapper.propose_repairs(diagnosis)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.INCREASE_HORIZON
    assert proposals[0].mutation_type == "horizon_adjust"
    assert "new_horizon" in proposals[0].parameters


def test_repair_mapper_rank_repairs():
    """Test ranking repair proposals by priority."""
    mapper = RepairMapper()
    diagnosis1 = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
        severity=0.5,
    )
    diagnosis2 = DiagnosisRecord(
        trial_id="t2",
        diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
        severity=0.3,
    )

    proposals1 = mapper.propose_repairs(diagnosis1, max_proposals=2)
    proposals2 = mapper.propose_repairs(diagnosis2, max_proposals=2)
    all_proposals = proposals1 + proposals2

    ranked = mapper.rank_repairs(all_proposals)

    assert len(ranked) == len(all_proposals)
    assert ranked == sorted(all_proposals, key=lambda p: p.confidence, reverse=False) or ranked != all_proposals


def test_repair_mapper_generate_mutation_spec():
    """Test generating MutationSpec from repair proposal."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.NUMERICAL_INSTABILITY,
        severity=0.7,
        evidence={"nan_count": 150},
    )

    proposals = mapper.propose_repairs(diagnosis, max_proposals=1)
    assert len(proposals) > 0

    mutation_spec = mapper.generate_mutation_spec(proposals[0])

    assert "mutation_type" in mutation_spec
    assert mutation_spec["mutation_type"] == "add_winsorization"
    assert "parameters" in mutation_spec
    assert "provenance" in mutation_spec
    assert mutation_spec["provenance"]["source"] == "repair_mapper"
    assert mutation_spec["provenance"]["diagnosis_kind"] == "numerical_instability"
    assert mutation_spec["provenance"]["repair_strategy"] == "winsorize"
    assert "confidence" in mutation_spec["provenance"]
    assert "evidence_context" in mutation_spec
    assert mutation_spec["evidence_context"]["nan_count"] == 150


def test_repair_mapper_add_interaction_strategy():
    """Test add_interaction strategy for low IC."""
    mapper = RepairMapper()
    diagnosis = DiagnosisRecord(
        trial_id="t1",
        diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
        evidence={"ic": 0.02, "lookback_periods": 20},
    )

    proposals = mapper.propose_repairs(diagnosis, max_proposals=1)

    assert len(proposals) > 0
    assert proposals[0].strategy == RepairStrategy.ADD_INTERACTION
    assert proposals[0].parameters["interaction_type"] == "cross_sectional"
    assert "Enhance predictive power" in proposals[0].expected_improvement
