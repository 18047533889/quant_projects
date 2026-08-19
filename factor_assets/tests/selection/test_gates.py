"""
Tests for evidence threshold gates.
"""

import math
import pytest

from factor_assets.selection import (
    GateResult,
    GateEvaluation,
    ThresholdGate,
    CompositeGate,
)
from factor_assets.selection.gates import (
    MinimumICGate,
    MaximumTurnoverGate,
    MinimumCoverageGate,
    MinimumObservationsGate,
    ParetoDominanceGate,
    MetricBinding,
    MetricDirection,
    MetricEvidence,
)


def test_gate_evaluation_creation():
    """Test GateEvaluation creation."""
    evaluation = GateEvaluation(
        gate_name="sharpe_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
        evidence_id="EVD_001",
        metric_name="sharpe",
        metric_value=1.5,
        threshold=1.0,
        message="sharpe=1.5000 >= 1.0000",
        gate_version="1.0",
    )

    assert evaluation.gate_name == "sharpe_gate"
    assert evaluation.factor_id == "F001"
    assert evaluation.result == GateResult.PASS
    assert evaluation.metric_value == 1.5


def test_gate_evaluation_requires_fields():
    """GateEvaluation must have required fields."""
    with pytest.raises(ValueError, match="gate_name"):
        GateEvaluation(
            gate_name="",
            factor_id="F001",
            result=GateResult.PASS,
            timestamp="2024-01-01T00:00:00Z",
        )

    with pytest.raises(ValueError, match="factor_id"):
        GateEvaluation(
            gate_name="test_gate",
            factor_id="",
            result=GateResult.PASS,
            timestamp="2024-01-01T00:00:00Z",
        )

    with pytest.raises(ValueError, match="timestamp"):
        GateEvaluation(
            gate_name="test_gate",
            factor_id="F001",
            result=GateResult.PASS,
            timestamp="",
        )


def test_gate_evaluation_passed_property():
    """Test passed property."""
    passed = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )
    assert passed.passed

    failed = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.FAIL,
        timestamp="2024-01-01T00:00:00Z",
    )
    assert not failed.passed


def test_gate_evaluation_failed_property():
    """Test failed property."""
    failed = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.FAIL,
        timestamp="2024-01-01T00:00:00Z",
    )
    assert failed.failed

    passed = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )
    assert not passed.failed


def test_threshold_gate_creation():
    """Test ThresholdGate creation."""
    gate = ThresholdGate(
        gate_name="sharpe_gate",
        threshold=1.0,
        higher_is_better=True,
        gate_version="1.0",
    )

    assert gate.gate_name == "sharpe_gate"
    assert gate.threshold == 1.0
    assert gate.gate_version == "1.0"


def test_threshold_gate_requires_name():
    """ThresholdGate must have gate_name."""
    with pytest.raises(ValueError, match="gate_name"):
        ThresholdGate(gate_name="", threshold=1.0)


def test_threshold_gate_higher_is_better_pass():
    """Test threshold gate with higher-is-better metric passing."""
    gate = ThresholdGate(
        gate_name="sharpe_gate",
        threshold=1.0,
        higher_is_better=True,
    )

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="sharpe",
        metric_value=1.5,
    )

    assert evaluation.passed
    assert evaluation.metric_value == 1.5
    assert evaluation.threshold == 1.0
    assert "1.5000 >= 1.0000" in evaluation.message


def test_threshold_gate_higher_is_better_fail():
    """Test threshold gate with higher-is-better metric failing."""
    gate = ThresholdGate(
        gate_name="sharpe_gate",
        threshold=1.0,
        higher_is_better=True,
    )

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="sharpe",
        metric_value=0.5,
    )

    assert evaluation.failed
    assert evaluation.metric_value == 0.5
    assert "0.5000 < 1.0000" in evaluation.message


def test_threshold_gate_lower_is_better_pass():
    """Test threshold gate with lower-is-better metric passing."""
    gate = ThresholdGate(
        gate_name="turnover_gate",
        threshold=0.5,
        higher_is_better=False,
    )

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="turnover",
        metric_value=0.3,
    )

    assert evaluation.passed
    assert evaluation.metric_value == 0.3
    assert "0.3000 <= 0.5000" in evaluation.message


def test_threshold_gate_lower_is_better_fail():
    """Test threshold gate with lower-is-better metric failing."""
    gate = ThresholdGate(
        gate_name="turnover_gate",
        threshold=0.5,
        higher_is_better=False,
    )

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="turnover",
        metric_value=0.8,
    )

    assert evaluation.failed
    assert evaluation.metric_value == 0.8
    assert "0.8000 > 0.5000" in evaluation.message


def test_threshold_gate_at_threshold_higher():
    """Test threshold gate exactly at threshold (higher-is-better)."""
    gate = ThresholdGate(
        gate_name="test_gate",
        threshold=1.0,
        higher_is_better=True,
    )

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="metric",
        metric_value=1.0,
    )

    assert evaluation.passed  # >= threshold


def test_threshold_gate_at_threshold_lower():
    """Test threshold gate exactly at threshold (lower-is-better)."""
    gate = ThresholdGate(
        gate_name="test_gate",
        threshold=1.0,
        higher_is_better=False,
    )

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="metric",
        metric_value=1.0,
    )

    assert evaluation.passed  # <= threshold


def test_composite_gate_creation():
    """Test CompositeGate creation."""
    gate1 = ThresholdGate(gate_name="gate1", threshold=1.0)
    gate2 = ThresholdGate(gate_name="gate2", threshold=0.5)

    composite = CompositeGate(
        gate_name="composite_gate",
        gates=[gate1, gate2],
        require_all=True,
    )

    assert composite.gate_name == "composite_gate"
    assert composite.gate_version == "1.0"


def test_composite_gate_requires_name():
    """CompositeGate must have gate_name."""
    gate1 = ThresholdGate(gate_name="gate1", threshold=1.0)

    with pytest.raises(ValueError, match="gate_name"):
        CompositeGate(gate_name="", gates=[gate1])


def test_composite_gate_rejects_wrong_metric_and_units():
    gate = MaximumTurnoverGate(threshold=0.5)
    composite = CompositeGate(
        "admission", [gate], metric_bindings={
            "maximum_turnover": MetricBinding(
                "turnover", "fraction", MetricDirection.LOWER_IS_BETTER
            )
        }
    )
    result, sub = composite.evaluate_all(
        "F001", "E001", {"other": MetricEvidence(0.1, "fraction", MetricDirection.LOWER_IS_BETTER)}
    )
    assert result.failed
    assert "turnover" in sub[0].message


def test_composite_gate_fails_closed_for_missing_evidence_and_unit_mismatch():
    gate = MinimumCoverageGate(threshold=0.8)
    composite = CompositeGate(
        "admission", [gate], metric_bindings={
            "minimum_coverage": MetricBinding(
                "coverage", "fraction", MetricDirection.HIGHER_IS_BETTER
            )
        }
    )
    missing, _ = composite.evaluate_all("F001", "E001", {})
    assert missing.failed
    wrong_unit, sub = composite.evaluate_all(
        "F001", "E001", {"coverage": MetricEvidence(0.9, "percent", MetricDirection.HIGHER_IS_BETTER)}
    )
    assert wrong_unit.failed
    assert "unit mismatch" in sub[0].message


def test_composite_gate_requires_typed_bindings():
    gate = ThresholdGate("gate", 1.0)
    composite = CompositeGate("admission", [gate], metric_bindings={"gate": "value"})
    result, sub = composite.evaluate_all(
        "F001", "E001", {"value": MetricEvidence(2.0, "count", MetricDirection.HIGHER_IS_BETTER)}
    )
    assert result.failed
    assert "untyped metric binding" in sub[0].message


def test_composite_gate_requires_gates():
    """CompositeGate must have at least one gate."""
    with pytest.raises(ValueError, match="gates list cannot be empty"):
        CompositeGate(gate_name="composite", gates=[])


def test_composite_gate_and_all_pass():
    """Test composite gate with AND logic, all gates pass."""
    gate1 = ThresholdGate(gate_name="sharpe_gate", threshold=1.0, higher_is_better=True)
    gate2 = ThresholdGate(gate_name="ic_gate", threshold=0.02, higher_is_better=True)

    composite = CompositeGate(
        gate_name="composite_gate",
        gates=[gate1, gate2],
        require_all=True,
        metric_bindings={
            "sharpe_gate": MetricBinding("sharpe", "ratio", MetricDirection.HIGHER_IS_BETTER),
            "ic_gate": MetricBinding("ic", "correlation", MetricDirection.HIGHER_IS_BETTER),
        },
    )

    metrics = {
        "sharpe": MetricEvidence(1.5, "ratio", MetricDirection.HIGHER_IS_BETTER),
        "ic": MetricEvidence(0.05, "correlation", MetricDirection.HIGHER_IS_BETTER),
    }

    composite_eval, sub_evals = composite.evaluate_all(
        factor_id="F001",
        evidence_id="EVD_001",
        metrics=metrics,
    )

    assert composite_eval.passed
    assert len(sub_evals) == 2
    assert all(ev.passed for ev in sub_evals)
    assert "2/2 gates passed" in composite_eval.message


def test_composite_gate_and_some_fail():
    """Test composite gate with AND logic, some gates fail."""
    gate1 = ThresholdGate(gate_name="sharpe_gate", threshold=1.0, higher_is_better=True)
    gate2 = ThresholdGate(gate_name="ic_gate", threshold=0.02, higher_is_better=True)

    composite = CompositeGate(
        gate_name="composite_gate",
        gates=[gate1, gate2],
        require_all=True,
        metric_bindings={
            "sharpe_gate": MetricBinding("sharpe", "ratio", MetricDirection.HIGHER_IS_BETTER),
            "ic_gate": MetricBinding("ic", "correlation", MetricDirection.HIGHER_IS_BETTER),
        },
    )

    metrics = {
        "sharpe": MetricEvidence(1.5, "ratio", MetricDirection.HIGHER_IS_BETTER),
        "ic": MetricEvidence(0.01, "correlation", MetricDirection.HIGHER_IS_BETTER),
    }

    composite_eval, sub_evals = composite.evaluate_all(
        factor_id="F001",
        evidence_id="EVD_001",
        metrics=metrics,
    )

    assert composite_eval.failed
    assert len(sub_evals) == 2
    assert "1/2 gates passed" in composite_eval.message


def test_composite_gate_or_any_pass():
    """Test composite gate with OR logic, any gate passes."""
    gate1 = ThresholdGate(gate_name="sharpe_gate", threshold=1.0, higher_is_better=True)
    gate2 = ThresholdGate(gate_name="ic_gate", threshold=0.02, higher_is_better=True)

    composite = CompositeGate(
        gate_name="composite_gate",
        gates=[gate1, gate2],
        require_all=False,  # OR logic
        metric_bindings={
            "sharpe_gate": MetricBinding("sharpe", "ratio", MetricDirection.HIGHER_IS_BETTER),
            "ic_gate": MetricBinding("ic", "correlation", MetricDirection.HIGHER_IS_BETTER),
        },
    )

    metrics = {
        "sharpe": MetricEvidence(0.5, "ratio", MetricDirection.HIGHER_IS_BETTER),
        "ic": MetricEvidence(0.05, "correlation", MetricDirection.HIGHER_IS_BETTER),
    }

    composite_eval, sub_evals = composite.evaluate_all(
        factor_id="F001",
        evidence_id="EVD_001",
        metrics=metrics,
    )

    assert composite_eval.passed
    assert "1/2 gates passed" in composite_eval.message


def test_composite_gate_or_all_fail():
    """Test composite gate with OR logic, all gates fail."""
    gate1 = ThresholdGate(gate_name="sharpe_gate", threshold=1.0, higher_is_better=True)
    gate2 = ThresholdGate(gate_name="ic_gate", threshold=0.02, higher_is_better=True)

    composite = CompositeGate(
        gate_name="composite_gate",
        gates=[gate1, gate2],
        require_all=False,  # OR logic
        metric_bindings={
            "sharpe_gate": MetricBinding("sharpe", "ratio", MetricDirection.HIGHER_IS_BETTER),
            "ic_gate": MetricBinding("ic", "correlation", MetricDirection.HIGHER_IS_BETTER),
        },
    )

    metrics = {
        "sharpe": MetricEvidence(0.5, "ratio", MetricDirection.HIGHER_IS_BETTER),
        "ic": MetricEvidence(0.01, "correlation", MetricDirection.HIGHER_IS_BETTER),
    }

    composite_eval, sub_evals = composite.evaluate_all(
        factor_id="F001",
        evidence_id="EVD_001",
        metrics=metrics,
    )

    assert composite_eval.failed
    assert "0/2 gates passed" in composite_eval.message


def test_composite_gate_no_matching_metrics():
    """Test composite gate when no metrics match."""
    gate1 = ThresholdGate(gate_name="sharpe_gate", threshold=1.0)

    composite = CompositeGate(
        gate_name="composite_gate",
        gates=[gate1],
        metric_bindings={
            "sharpe_gate": MetricBinding("sharpe", "ratio", MetricDirection.HIGHER_IS_BETTER),
        },
    )

    metrics = {}  # No evidence provided

    composite_eval, sub_evals = composite.evaluate_all(
        factor_id="F001",
        evidence_id="EVD_001",
        metrics=metrics,
    )

    assert composite_eval.failed
    assert len(sub_evals) == 1
    assert "Missing evidence for bound metric" in sub_evals[0].message


def test_gate_result_enum():
    """Test GateResult enum values."""
    assert GateResult.PASS.value == "PASS"
    assert GateResult.FAIL.value == "FAIL"
    assert GateResult.SKIP.value == "SKIP"
    assert GateResult.ERROR.value == "ERROR"


def test_threshold_gate_negative_threshold():
    """Test threshold gate with negative threshold."""
    gate = ThresholdGate(
        gate_name="test_gate",
        threshold=-0.5,
        higher_is_better=True,
    )

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="metric",
        metric_value=-0.3,
    )

    assert evaluation.passed  # -0.3 >= -0.5


def test_threshold_gate_zero_threshold():
    """Test threshold gate with zero threshold."""
    gate = ThresholdGate(
        gate_name="test_gate",
        threshold=0.0,
        higher_is_better=True,
    )

    positive_eval = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="metric",
        metric_value=0.1,
    )
    assert positive_eval.passed

    negative_eval = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="metric",
        metric_value=-0.1,
    )
    assert negative_eval.failed


# ============================================================================
# MinimumICGate Tests
# ============================================================================

def test_minimum_ic_gate_creation():
    """Test MinimumICGate creation."""
    gate = MinimumICGate(threshold=0.02)
    assert gate.gate_name == "minimum_ic"
    assert gate.threshold == 0.02


def test_minimum_ic_gate_negative_threshold_rejected():
    """MinimumICGate cannot have negative threshold."""
    with pytest.raises(ValueError, match="IC threshold must be non-negative"):
        MinimumICGate(threshold=-0.01)


def test_minimum_ic_gate_bool_threshold_rejected():
    """MinimumICGate rejects boolean threshold (bool is subclass of int)."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumICGate(threshold=True)
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumICGate(threshold=False)


def test_minimum_ic_gate_nan_threshold_rejected():
    """MinimumICGate rejects NaN threshold."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumICGate(threshold=float("nan"))


def test_minimum_ic_gate_inf_threshold_rejected():
    """MinimumICGate rejects inf threshold."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumICGate(threshold=float("inf"))
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumICGate(threshold=float("-inf"))


def test_minimum_ic_gate_string_threshold_rejected():
    """MinimumICGate rejects string threshold."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumICGate(threshold="0.02")


def test_minimum_ic_gate_positive_ic_pass():
    """Test IC gate with positive IC passing."""
    gate = MinimumICGate(threshold=0.02)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="ic",
        metric_value=0.05,
    )
    assert evaluation.passed
    assert "|ic|=0.0500 >= 0.0200" in evaluation.message
    assert "raw=0.0500" in evaluation.message


def test_minimum_ic_gate_negative_ic_pass():
    """Test IC gate with negative IC but sufficient magnitude passing."""
    gate = MinimumICGate(threshold=0.02)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="ic",
        metric_value=-0.05,
    )
    assert evaluation.passed
    assert "|ic|=0.0500 >= 0.0200" in evaluation.message
    assert "raw=-0.0500" in evaluation.message


def test_minimum_ic_gate_absolute_direction_binding_accepts_negative_ic():
    """CompositeGate preserves absolute IC semantics for negative evidence."""
    gate = MinimumICGate(threshold=0.02)
    composite = CompositeGate(
        gate_name="absolute_ic",
        gates=[gate],
        metric_bindings={
            "minimum_ic": MetricBinding(
                "ic", "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
            )
        },
    )

    result, sub_evaluations = composite.evaluate_all(
        "F001",
        "E001",
        {"ic": MetricEvidence(-0.05, "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER)},
    )

    assert result.passed
    assert sub_evaluations[0].passed


def test_minimum_ic_gate_fail():
    """Test IC gate failing."""
    gate = MinimumICGate(threshold=0.02)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="ic",
        metric_value=0.01,
    )
    assert evaluation.failed
    assert "|ic|=0.0100 < 0.0200" in evaluation.message


def test_minimum_ic_gate_zero_value():
    """Test IC gate with zero IC."""
    gate = MinimumICGate(threshold=0.02)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="ic",
        metric_value=0.0,
    )
    assert evaluation.failed


def test_minimum_ic_gate_at_threshold():
    """Test IC gate exactly at threshold."""
    gate = MinimumICGate(threshold=0.02)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="ic",
        metric_value=0.02,
    )
    assert evaluation.passed


def test_minimum_ic_gate_negative_at_threshold():
    """Test IC gate with negative value at threshold magnitude."""
    gate = MinimumICGate(threshold=0.02)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="ic",
        metric_value=-0.02,
    )
    assert evaluation.passed


def test_minimum_ic_gate_requires_absolute_direction_binding():
    """MinimumICGate admits signed IC evidence only under an absolute contract."""
    gate = MinimumICGate(threshold=0.02)
    composite = CompositeGate(
        "ic_admission",
        [gate],
        metric_bindings={
            "minimum_ic": MetricBinding(
                "ic", "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
            )
        },
    )

    passed, _ = composite.evaluate_all(
        "F001",
        "EVD_001",
        {"ic": MetricEvidence(-0.03, "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER)},
    )
    assert passed.passed

    rejected, sub_evaluations = composite.evaluate_all(
        "F001",
        "EVD_001",
        {"ic": MetricEvidence(-0.03, "correlation", MetricDirection.HIGHER_IS_BETTER)},
    )
    assert rejected.failed
    assert "Metric direction mismatch" in sub_evaluations[0].message


# ============================================================================
# MaximumTurnoverGate Tests
# ============================================================================

def test_maximum_turnover_gate_creation():
    """Test MaximumTurnoverGate creation."""
    gate = MaximumTurnoverGate(threshold=0.5)
    assert gate.gate_name == "maximum_turnover"
    assert gate.threshold == 0.5


def test_maximum_turnover_gate_negative_threshold_rejected():
    """MaximumTurnoverGate cannot have negative threshold."""
    with pytest.raises(ValueError, match="Turnover threshold must be non-negative"):
        MaximumTurnoverGate(threshold=-0.1)


def test_maximum_turnover_gate_bool_threshold_rejected():
    """MaximumTurnoverGate rejects boolean threshold."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MaximumTurnoverGate(threshold=True)
    with pytest.raises(ValueError, match="finite non-boolean"):
        MaximumTurnoverGate(threshold=False)


def test_maximum_turnover_gate_nan_threshold_rejected():
    """MaximumTurnoverGate rejects NaN threshold."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MaximumTurnoverGate(threshold=float("nan"))


def test_maximum_turnover_gate_inf_threshold_rejected():
    """MaximumTurnoverGate rejects inf threshold."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MaximumTurnoverGate(threshold=float("inf"))


def test_maximum_turnover_gate_pass():
    """Test turnover gate passing."""
    gate = MaximumTurnoverGate(threshold=0.5)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="turnover",
        metric_value=0.3,
    )
    assert evaluation.passed
    assert "0.3000 <= 0.5000" in evaluation.message


def test_maximum_turnover_gate_fail():
    """Test turnover gate failing."""
    gate = MaximumTurnoverGate(threshold=0.5)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="turnover",
        metric_value=0.8,
    )
    assert evaluation.failed
    assert "0.8000 > 0.5000" in evaluation.message


def test_maximum_turnover_gate_at_threshold():
    """Test turnover gate exactly at threshold."""
    gate = MaximumTurnoverGate(threshold=0.5)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="turnover",
        metric_value=0.5,
    )
    assert evaluation.passed


def test_maximum_turnover_gate_zero_turnover():
    """Test turnover gate with zero turnover."""
    gate = MaximumTurnoverGate(threshold=0.5)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="turnover",
        metric_value=0.0,
    )
    assert evaluation.passed


# ============================================================================
# MinimumCoverageGate Tests
# ============================================================================

def test_minimum_coverage_gate_creation():
    """Test MinimumCoverageGate creation."""
    gate = MinimumCoverageGate(threshold=0.8)
    assert gate.gate_name == "minimum_coverage"
    assert gate.threshold == 0.8


def test_minimum_coverage_gate_threshold_bounds():
    """MinimumCoverageGate threshold must be in [0, 1]."""
    with pytest.raises(ValueError, match="Coverage threshold must be in"):
        MinimumCoverageGate(threshold=-0.1)

    with pytest.raises(ValueError, match="Coverage threshold must be in"):
        MinimumCoverageGate(threshold=1.5)


def test_minimum_coverage_gate_bool_threshold_rejected():
    """MinimumCoverageGate rejects boolean threshold."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumCoverageGate(threshold=True)
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumCoverageGate(threshold=False)


def test_minimum_coverage_gate_nan_threshold_rejected():
    """MinimumCoverageGate rejects NaN threshold (NaN < 0 is False but NaN > 1 is also False)."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumCoverageGate(threshold=float("nan"))


def test_minimum_coverage_gate_inf_threshold_rejected():
    """MinimumCoverageGate rejects inf threshold."""
    with pytest.raises(ValueError, match="finite non-boolean"):
        MinimumCoverageGate(threshold=float("inf"))


def test_minimum_coverage_gate_pass():
    """Test coverage gate passing."""
    gate = MinimumCoverageGate(threshold=0.8)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="coverage",
        metric_value=0.95,
    )
    assert evaluation.passed
    assert "0.9500 >= 0.8000" in evaluation.message


def test_minimum_coverage_gate_fail():
    """Test coverage gate failing."""
    gate = MinimumCoverageGate(threshold=0.8)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="coverage",
        metric_value=0.7,
    )
    assert evaluation.failed
    assert "0.7000 < 0.8000" in evaluation.message


def test_minimum_coverage_gate_full_coverage():
    """Test coverage gate with 100% coverage."""
    gate = MinimumCoverageGate(threshold=0.8)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="coverage",
        metric_value=1.0,
    )
    assert evaluation.passed


def test_minimum_coverage_gate_zero_coverage():
    """Test coverage gate with 0% coverage."""
    gate = MinimumCoverageGate(threshold=0.8)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="coverage",
        metric_value=0.0,
    )
    assert evaluation.failed


def test_minimum_coverage_gate_boundary_thresholds():
    """Test coverage gate with boundary threshold values."""
    gate_zero = MinimumCoverageGate(threshold=0.0)
    gate_one = MinimumCoverageGate(threshold=1.0)

    assert gate_zero.threshold == 0.0
    assert gate_one.threshold == 1.0


# ============================================================================
# MinimumObservationsGate Tests
# ============================================================================

def test_minimum_observations_gate_creation():
    """Test MinimumObservationsGate creation."""
    gate = MinimumObservationsGate(threshold=100)
    assert gate.gate_name == "minimum_observations"
    assert gate.threshold == 100.0


def test_minimum_observations_gate_threshold_bounds():
    """MinimumObservationsGate threshold must be at least 1."""
    with pytest.raises(ValueError, match="Observations threshold must be at least 1"):
        MinimumObservationsGate(threshold=0)

    with pytest.raises(ValueError, match="Observations threshold must be at least 1"):
        MinimumObservationsGate(threshold=-10)


def test_minimum_observations_gate_pass():
    """Test observations gate passing."""
    gate = MinimumObservationsGate(threshold=100)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="n_obs",
        metric_value=500.0,
    )
    assert evaluation.passed
    assert "n_obs=500 >= 100" in evaluation.message


def test_minimum_observations_gate_fail():
    """Test observations gate failing."""
    gate = MinimumObservationsGate(threshold=100)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="n_obs",
        metric_value=50.0,
    )
    assert evaluation.failed
    assert "n_obs=50 < 100" in evaluation.message


def test_minimum_observations_gate_at_threshold():
    """Test observations gate exactly at threshold."""
    gate = MinimumObservationsGate(threshold=100)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="n_obs",
        metric_value=100.0,
    )
    assert evaluation.passed


def test_minimum_observations_gate_formatting():
    """Test observations gate formats as integers."""
    gate = MinimumObservationsGate(threshold=1000)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="n_obs",
        metric_value=1234.0,
    )
    assert evaluation.passed
    # Check integer formatting (no decimal places)
    assert "1234" in evaluation.message
    assert "1000" in evaluation.message


def test_minimum_observations_gate_minimum_threshold():
    """Test observations gate with minimum threshold."""
    gate = MinimumObservationsGate(threshold=1)
    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="n_obs",
        metric_value=5.0,
    )
    assert evaluation.passed


# ============================================================================
# ParetoDominanceGate Tests
# ============================================================================

def test_pareto_dominance_gate_creation():
    """Test ParetoDominanceGate creation."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe", "ic"],
        lower_is_better_metrics=["turnover"],
    )
    assert gate.gate_name == "pareto_dominance"


def test_pareto_dominance_gate_metric_overlap_rejected():
    """ParetoDominanceGate cannot have metrics in both higher and lower lists."""
    with pytest.raises(ValueError, match="cannot be both higher and lower"):
        ParetoDominanceGate(
            higher_is_better_metrics=["sharpe"],
            lower_is_better_metrics=["sharpe"],
        )


def test_pareto_dominance_gate_not_dominated():
    """Test Pareto gate when factor is not dominated."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe", "ic"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": 1.5, "ic": 0.05, "turnover": 0.3}
    references = [
        ("F002", {"sharpe": 1.2, "ic": 0.04, "turnover": 0.4}),
        ("F003", {"sharpe": 1.3, "ic": 0.03, "turnover": 0.5}),
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.passed
    assert "Not Pareto dominated" in evaluation.message
    assert "checked against 2 factors" in evaluation.message


def test_pareto_dominance_gate_dominated():
    """Test Pareto gate when factor is dominated."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe", "ic"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": 1.0, "ic": 0.03, "turnover": 0.5}
    references = [
        ("F002", {"sharpe": 1.5, "ic": 0.05, "turnover": 0.3}),  # Dominates candidate
        ("F003", {"sharpe": 0.8, "ic": 0.02, "turnover": 0.6}),
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.failed
    assert "dominated by 1 other" in evaluation.message
    assert "F002" in evaluation.message


def test_pareto_dominance_gate_strictly_better_required():
    """Test Pareto gate requires strict improvement on at least one metric."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe", "ic"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": 1.5, "ic": 0.05, "turnover": 0.3}
    # Reference equal on all metrics - should NOT dominate
    references = [
        ("F002", {"sharpe": 1.5, "ic": 0.05, "turnover": 0.3}),
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.passed  # Not dominated (need strict improvement)


def test_pareto_dominance_gate_partial_dominance():
    """Test Pareto gate when reference is better on some but not all metrics."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe", "ic"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": 1.5, "ic": 0.03, "turnover": 0.3}
    # Reference has better sharpe, worse IC - not a complete dominance
    references = [
        ("F002", {"sharpe": 2.0, "ic": 0.02, "turnover": 0.3}),
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.passed  # Not dominated (reference worse on IC)


def test_pareto_dominance_gate_multiple_dominators():
    """Test Pareto gate with multiple dominating factors."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe", "ic"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": 0.8, "ic": 0.02, "turnover": 0.6}
    references = [
        ("F002", {"sharpe": 1.0, "ic": 0.03, "turnover": 0.5}),  # Dominates
        ("F003", {"sharpe": 1.2, "ic": 0.04, "turnover": 0.4}),  # Dominates
        ("F004", {"sharpe": 0.7, "ic": 0.01, "turnover": 0.7}),  # Does not dominate
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.failed
    assert "dominated by 2 other" in evaluation.message


def test_pareto_dominance_gate_many_dominators_truncates():
    """Test Pareto gate truncates message with many dominators."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": 0.5, "turnover": 0.9}
    references = [
        (f"F{i:03d}", {"sharpe": 1.0, "turnover": 0.3})
        for i in range(10)
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.failed
    assert "dominated by 10 other" in evaluation.message
    assert "(+7 more)" in evaluation.message  # Shows first 3, then +7 more


def test_pareto_dominance_gate_no_common_metrics():
    """Test Pareto gate with no common metrics."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": 1.5, "turnover": 0.3}
    # Reference has completely different metrics
    references = [
        ("F002", {"alpha": 0.5, "beta": 1.2}),
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.passed  # Cannot be dominated without common metrics


def test_pareto_dominance_gate_empty_references():
    """Test Pareto gate with no reference factors."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": 1.5, "turnover": 0.3}
    references = []

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.passed
    assert "checked against 0 factors" in evaluation.message


def test_pareto_dominance_gate_unknown_metric_direction():
    """Test Pareto gate with metrics not in either list."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": 1.5, "turnover": 0.3, "unknown_metric": 100}
    # Reference better on known metrics, different on unknown
    references = [
        ("F002", {"sharpe": 2.0, "turnover": 0.2, "unknown_metric": 50}),
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    # Should be dominated based on sharpe and turnover alone
    assert evaluation.failed


# ============================================================================
# CompositeGate Advanced Tests
# ============================================================================

def test_composite_gate_with_specialized_gates_and():
    """Test composite gate combining specialized gates with AND logic."""
    ic_gate = MinimumICGate(threshold=0.02)
    turnover_gate = MaximumTurnoverGate(threshold=0.5)
    coverage_gate = MinimumCoverageGate(threshold=0.8)

    composite = CompositeGate(
        gate_name="production_ready",
        gates=[ic_gate, turnover_gate, coverage_gate],
        require_all=True,
        metric_bindings={
            "minimum_ic": MetricBinding(
                "ic", "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
            ),
            "maximum_turnover": MetricBinding(
                "turnover", "fraction", MetricDirection.LOWER_IS_BETTER
            ),
            "minimum_coverage": MetricBinding(
                "coverage", "fraction", MetricDirection.HIGHER_IS_BETTER
            ),
        },
    )

    # All gates pass
    metrics = {
        "ic": MetricEvidence(
            0.05, "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
        ),
        "turnover": MetricEvidence(
            0.3, "fraction", MetricDirection.LOWER_IS_BETTER
        ),
        "coverage": MetricEvidence(
            0.9, "fraction", MetricDirection.HIGHER_IS_BETTER
        ),
    }

    composite_eval, sub_evals = composite.evaluate_all(
        factor_id="F001",
        evidence_id="EVD_001",
        metrics=metrics,
    )

    assert composite_eval.passed
    assert len(sub_evals) == 3
    assert all(ev.passed for ev in sub_evals)


def test_composite_gate_with_specialized_gates_or():
    """Test composite gate combining specialized gates with OR logic."""
    ic_gate = MinimumICGate(threshold=0.02)
    turnover_gate = MaximumTurnoverGate(threshold=0.5)

    composite = CompositeGate(
        gate_name="acceptable_factor",
        gates=[ic_gate, turnover_gate],
        require_all=False,  # OR logic
        metric_bindings={
            "minimum_ic": MetricBinding(
                "ic", "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
            ),
            "maximum_turnover": MetricBinding(
                "turnover", "fraction", MetricDirection.LOWER_IS_BETTER
            ),
        },
    )

    # IC fails, turnover passes
    metrics = {
        "ic": MetricEvidence(
            0.01, "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
        ),
        "turnover": MetricEvidence(
            0.3, "fraction", MetricDirection.LOWER_IS_BETTER
        ),
    }

    composite_eval, sub_evals = composite.evaluate_all(
        factor_id="F001",
        evidence_id="EVD_001",
        metrics=metrics,
    )

    assert composite_eval.passed  # At least one passed
    assert len(sub_evals) == 2


def test_composite_gate_all_specialized_fail():
    """Test composite gate with all specialized gates failing."""
    ic_gate = MinimumICGate(threshold=0.05)
    coverage_gate = MinimumCoverageGate(threshold=0.9)
    obs_gate = MinimumObservationsGate(threshold=1000)

    composite = CompositeGate(
        gate_name="strict_requirements",
        gates=[ic_gate, coverage_gate, obs_gate],
        require_all=True,
        metric_bindings={
            "minimum_ic": MetricBinding(
                "ic", "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
            ),
            "minimum_coverage": MetricBinding(
                "coverage", "fraction", MetricDirection.HIGHER_IS_BETTER
            ),
            "minimum_observations": MetricBinding(
                "n_obs", "count", MetricDirection.HIGHER_IS_BETTER
            ),
        },
    )

    metrics = {
        "ic": MetricEvidence(
            0.01, "correlation", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
        ),
        "coverage": MetricEvidence(
            0.5, "fraction", MetricDirection.HIGHER_IS_BETTER
        ),
        "n_obs": MetricEvidence(100.0, "count", MetricDirection.HIGHER_IS_BETTER),
    }

    composite_eval, sub_evals = composite.evaluate_all(
        factor_id="F001",
        evidence_id="EVD_001",
        metrics=metrics,
    )

    assert composite_eval.failed
    assert all(not ev.passed for ev in sub_evals)


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================

def test_minimum_ic_gate_very_small_values():
    """Test IC gate with very small values near floating point precision."""
    gate = MinimumICGate(threshold=1e-6)

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="ic",
        metric_value=1e-7,
    )

    assert evaluation.failed


def test_maximum_turnover_gate_large_values():
    """Test turnover gate with very large turnover values."""
    gate = MaximumTurnoverGate(threshold=1.0)

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="turnover",
        metric_value=10.0,
    )

    assert evaluation.failed


def test_minimum_observations_gate_large_threshold():
    """Test observations gate with large threshold."""
    gate = MinimumObservationsGate(threshold=1_000_000)

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        metric_name="n_obs",
        metric_value=500_000.0,
    )

    assert evaluation.failed
    assert "500000 < 1000000" in evaluation.message


def test_pareto_dominance_with_negative_metrics():
    """Test Pareto dominance with negative metric values."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["return"],
        lower_is_better_metrics=["volatility"],
    )

    # Candidate has negative return and high volatility
    candidate = {"return": -0.05, "volatility": 0.3}
    # Reference has positive return and lower volatility - dominates on both
    references = [
        ("F002", {"return": 0.05, "volatility": 0.1}),
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.failed  # Dominated


def test_minimum_ic_gate_inf_fails_closed():
    """An overflowed |IC|=inf is a failed measurement, not skill:
    must FAIL, not pass |inf| >= threshold."""
    gate = MinimumICGate(threshold=0.02)

    for bad in (float("inf"), float("-inf"), float("nan")):
        evaluation = gate.evaluate(
            factor_id="F001",
            evidence_id="EVD_001",
            metric_name="ic",
            metric_value=bad,
        )
        assert evaluation.failed, f"{bad!r} must fail closed"
        assert "non-finite" in evaluation.message


def test_metric_evidence_rejects_nonfinite_value():
    """MetricEvidence must reject NaN/±inf/bool at construction —
    it is the typed channel feeding the gates."""
    for bad in (float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises((ValueError, TypeError)):
            MetricEvidence(bad, "correlation", MetricDirection.HIGHER_IS_BETTER)


def test_pareto_gate_nan_candidate_fails_closed():
    """NaN comparisons are all False, which used to make a partially-NaN
    candidate 'not dominated' (gate PASS) even against a clearly
    dominating reference — comparison undefined must fail closed."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe", "ic"],
        lower_is_better_metrics=["turnover"],
    )

    candidate = {"sharpe": float("nan"), "ic": 0.03, "turnover": 0.9}
    references = [
        ("F002", {"sharpe": 1.5, "ic": 0.05, "turnover": 0.3}),  # dominates all defined
    ]

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EVD_001",
        candidate_metrics=candidate,
        reference_factors=references,
    )

    assert evaluation.failed
    assert "non-finite" in evaluation.message


def test_pareto_gate_nan_reference_not_dominating():
    """A NaN reference metric makes domination unprovable: the reference
    must not count as a dominator (is_dominated returns False on
    non-finite comparison inputs)."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe"],
        lower_is_better_metrics=["turnover"],
    )

    assert not gate.is_dominated(
        {"sharpe": 1.0, "turnover": 0.5},
        {"sharpe": float("nan"), "turnover": 0.3},
    )


def test_pareto_gate_neg_inf_candidate_not_dominated():
    """−inf candidate vs a clearly dominating reference: the comparison is
    undefined (overflow), so domination must NOT be asserted.  The old
    code returned True here (finite check did not exist) — ±inf, not NaN,
    is the distinguishing case for this guard."""
    gate = ParetoDominanceGate(higher_is_better_metrics=["sharpe"])

    assert not gate.is_dominated(
        {"sharpe": float("-inf")},
        {"sharpe": 1.0},
    )
    # Symmetric: +inf reference cannot dominate either
    assert not gate.is_dominated(
        {"sharpe": 1.0},
        {"sharpe": float("inf")},
    )
    # Non-numeric reference value must not raise TypeError from a
    # standalone is_dominated/evaluate call (undefined → not dominated)
    assert not gate.is_dominated({"sharpe": 1.0}, {"sharpe": None})


def test_pareto_gate_untracked_nonfinite_metric_ignored():
    """Only TRACKED metrics (in higher/lower_is_better) participate in the
    fail-closed scan: an untracked NaN/str/None metric must not FAIL the
    gate (it is never compared) and must not raise TypeError."""
    gate = ParetoDominanceGate(
        higher_is_better_metrics=["sharpe"],
        lower_is_better_metrics=["turnover"],
    )

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EV001",
        candidate_metrics={
            "sharpe": 1.2,
            "turnover": 0.4,
            "untracked_nan": float("nan"),   # ignored: never compared
            "untracked_str": "not-a-number",  # ignored: never compared
            "untracked_none": None,           # ignored: never compared
        },
        reference_factors=[("F000", {"sharpe": 0.8, "turnover": 0.6})],
    )

    assert evaluation.passed
    assert "non-finite" not in evaluation.message


def test_pareto_gate_tracked_nonnumeric_metric_fails_closed():
    """A TRACKED metric with a non-numeric value would raise TypeError in
    the comparisons; the gate must fail closed with a clear message
    instead of crashing."""
    gate = ParetoDominanceGate(higher_is_better_metrics=["sharpe"])

    evaluation = gate.evaluate(
        factor_id="F001",
        evidence_id="EV001",
        candidate_metrics={"sharpe": None},
        reference_factors=[],
    )

    assert evaluation.failed
    assert "sharpe" in evaluation.message
    assert "fail closed" in evaluation.message


def test_composite_evaluate_all_unexpected_exception_fails_closed():
    """A sub-gate raising an exception (any type — IndexError here) must
    fail that gate closed, not propagate out of evaluate_all and abort
    the composite run."""
    class ExplodingGate:
        """Minimal gate-like object whose evaluate always raises IndexError."""
        gate_name = "exploding_gate"
        gate_version = "1.0"
        metric_direction = MetricDirection.ABSOLUTE_HIGHER_IS_BETTER

        def evaluate(self, factor_id, evidence_id, metric_name, metric_value):
            raise IndexError("boom — sub-gate crash")

    composite = CompositeGate(
        gate_name="composite",
        gates=[ExplodingGate()],
        metric_bindings={
            "exploding_gate": MetricBinding(
                "sharpe", "ratio", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
            )
        },
    )
    evidence = {
        "sharpe": MetricEvidence(
            1.5, "ratio", MetricDirection.ABSOLUTE_HIGHER_IS_BETTER
        )
    }

    result, sub_evaluations = composite.evaluate_all("F001", "E001", evidence)

    assert result.failed
    assert "IndexError" in result.message or any(
        "IndexError" in ev.message for ev in sub_evaluations
    )
    assert sub_evaluations and sub_evaluations[0].failed


def test_gate_evaluation_immutability():
    """Test that GateEvaluation is immutable."""
    evaluation = GateEvaluation(
        gate_name="test_gate",
        factor_id="F001",
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )

    with pytest.raises(Exception):  # dataclass frozen=True raises FrozenInstanceError or AttributeError
        evaluation.gate_name = "modified_gate"


def test_multiple_gates_same_factor():
    """Test running multiple gate evaluations on same factor sequentially."""
    factor_id = "F001"
    evidence_id = "EVD_001"

    ic_gate = MinimumICGate(threshold=0.02)
    ic_eval = ic_gate.evaluate(factor_id, evidence_id, "ic", 0.05)

    turnover_gate = MaximumTurnoverGate(threshold=0.5)
    turnover_eval = turnover_gate.evaluate(factor_id, evidence_id, "turnover", 0.3)

    coverage_gate = MinimumCoverageGate(threshold=0.8)
    coverage_eval = coverage_gate.evaluate(factor_id, evidence_id, "coverage", 0.9)

    assert ic_eval.passed
    assert turnover_eval.passed
    assert coverage_eval.passed

    # Each evaluation is independent
    assert ic_eval.gate_name != turnover_eval.gate_name
    assert ic_eval.metric_name != turnover_eval.metric_name
