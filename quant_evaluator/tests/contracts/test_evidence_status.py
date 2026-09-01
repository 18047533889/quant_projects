"""
Unit tests for the EvidenceStatus contract.

Pins the single-source-of-truth status enum and the MetricEvidence bundle:
enum values, fail-closed normalization, computed property, serialization
round-trip, and that a non-COMPUTED status never fabricates a 0.0.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_evaluator.contracts.evidence_status import (
    EvidenceStatus,
    EvidenceReasonCode,
    MetricEvidence,
    evidence_for_computed,
    evidence_for_not_computed,
)
from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact


def test_evidence_status_values():
    """The six statuses must exist with their canonical string values."""
    expected = {
        "COMPUTED": "computed",
        "NOT_COMPUTED": "not_computed",
        "UNAVAILABLE": "unavailable",
        "UNSUPPORTED": "unsupported",
        "INSUFFICIENT_DATA": "insufficient_data",
        "FAILED": "failed",
    }
    for name, value in expected.items():
        member = getattr(EvidenceStatus, name)
        assert member.value == value


def test_evidence_status_from_value_accepts_str_and_enum():
    assert EvidenceStatus.from_value("computed") is EvidenceStatus.COMPUTED
    assert (
        EvidenceStatus.from_value(EvidenceStatus.FAILED) is EvidenceStatus.FAILED
    )


def test_evidence_status_from_value_fails_closed_on_unknown():
    with pytest.raises(ValueError):
        EvidenceStatus.from_value("bogus_status")


def test_metric_evidence_computed_property():
    art = ScalarMetricArtifact(metric_id="rank_ic", domain="ic", values=[0.1, 0.2])
    ev = evidence_for_computed(
        artifact=art, observations=42, minimum_required=10
    )
    assert ev.computed is True
    assert ev.status is EvidenceStatus.COMPUTED
    assert ev.reason_code is EvidenceReasonCode.OK
    assert ev.observations == 42
    assert ev.minimum_required == 10


def test_metric_evidence_not_computed_default():
    ev = evidence_for_not_computed()
    assert ev.computed is False
    assert ev.status is EvidenceStatus.NOT_COMPUTED
    assert ev.reason_code is EvidenceReasonCode.NOT_YET_COMPUTED


def test_metric_evidence_accepts_string_reason_code():
    ev = MetricEvidence(
        status="failed",
        reason_code="kernel_failure",
        artifact=ScalarMetricArtifact(metric_id="rank_ic", domain="ic", values=[0.1]),
        reason_detail="boom",
    )
    assert ev.status is EvidenceStatus.FAILED
    assert ev.reason_code is EvidenceReasonCode.KERNEL_FAILURE


def test_metric_evidence_rejects_negative_observations():
    with pytest.raises(ValueError):
        MetricEvidence(status=EvidenceStatus.COMPUTED, observations=-1,
                       artifact=ScalarMetricArtifact(metric_id="rank_ic", domain="ic", values=[0.1]))


def test_metric_evidence_rejects_non_artifact():
    with pytest.raises(TypeError):
        MetricEvidence(status=EvidenceStatus.COMPUTED, artifact="not-an-artifact")


def test_computed_requires_artifact_fail_closed():
    """P0-QE-001: COMPUTED status with artifact=None is a fabricated pass."""
    with pytest.raises(ValueError):
        MetricEvidence(
            status=EvidenceStatus.COMPUTED,
            reason_code=EvidenceReasonCode.OK,
            observations=5,
            artifact=None,
        )


def test_computed_rejects_non_ok_reason():
    """P0-QE-001: COMPUTED + reason != OK is illegal."""
    with pytest.raises(ValueError):
        MetricEvidence(
            status=EvidenceStatus.COMPUTED,
            reason_code=EvidenceReasonCode.KERNEL_FAILURE,
            artifact=ScalarMetricArtifact(metric_id="rank_ic", domain="ic", values=[0.1]),
        )


def test_label_not_mature_rejects_artifact():
    """P0-QE-001: LABEL_NOT_MATURE must never carry a real computed artifact."""
    with pytest.raises(ValueError):
        MetricEvidence(
            status=EvidenceStatus.LABEL_NOT_MATURE,
            reason_code=EvidenceReasonCode.LABEL_NOT_YET_MATURE,
            artifact=ScalarMetricArtifact(metric_id="rank_ic", domain="ic", values=[0.1]),
        )


def test_failed_rejects_ok_reason():
    """P0-QE-001: FAILED + reason=OK is a lie."""
    with pytest.raises(ValueError):
        MetricEvidence(
            status=EvidenceStatus.FAILED,
            reason_code=EvidenceReasonCode.OK,
            artifact=ScalarMetricArtifact(metric_id="rank_ic", domain="ic", values=[0.1]),
        )


def test_metric_evidence_artifact_fields():
    art = ScalarMetricArtifact(metric_id="pearson_ic", domain="ic", values=[0.1, 0.2])
    ev = evidence_for_computed(artifact=art, observations=2)
    assert ev.metric_id == "pearson_ic"
    assert ev.artifact_kind == "scalar"
    assert ev.artifact is art


def test_metric_evidence_to_dict_round_trip():
    art = ScalarMetricArtifact(metric_id="rank_ic", domain="ic", values=[0.3, 0.4])
    ev = evidence_for_computed(artifact=art, observations=2, minimum_required=1)
    payload = ev.to_dict()
    restored = MetricEvidence.from_dict(payload)
    assert restored.status is EvidenceStatus.COMPUTED
    assert restored.reason_code is EvidenceReasonCode.OK
    assert restored.observations == 2
    assert restored.minimum_required == 1
    assert restored.artifact is not None
    assert restored.artifact.metric_id == "rank_ic"
    assert restored.artifact.to_dict() == art.to_dict()


def test_not_computed_status_never_fabricates_zero():
    """A NOT_COMPUTED evidence must never report a fabricated 0.0 value."""
    ev = evidence_for_not_computed(
        reason_code="min_periods_not_met",
        observations=5,
        minimum_required=20,
    )
    assert ev.computed is False
    # There is no numeric payload to read -- absent evidence, not zero.
    assert ev.artifact is None
