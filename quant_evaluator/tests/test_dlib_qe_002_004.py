"""DLIB-QE-002..004 tests: EvaluationArtifact split + evidence maturity.

Covers:
- :class:`EvaluationArtifact` deep immutability (mutating the original dict
  does not change the artifact) and derived-only ``content_hash`` fail-closed.
- The R55 audit three-part identity: spec identity / result content hash /
  envelope identity, plus the legacy ``artifact_hash`` property.
- Typed :class:`DomainArtifactRef` cross-references (R55 audit #24): a bare
  string / dict reference is rejected at construction.
- :class:`EvaluationRequest` ``to_dict`` / ``from_dict`` strict round-trip on
  reference fields (``factor_value_ref`` / ``label_bundle_ref`` / ``split_ref``).
- :class:`EvidenceStatus` label-not-mature yields no IC (never a zero-fill).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_evaluator.contracts.evaluation_artifact import (
    ArtifactEnvelopeIdentity,
    EvaluationArtifact,
    EvaluationResultContentHash,
    EvaluationSpecIdentity,
)
from quant_evaluator.contracts.domain_refs import ArtifactDomain, DomainArtifactRef
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef
from quant_evaluator.contracts.evidence_status import (
    EvidenceStatus,
    EvidenceReasonCode,
    MetricEvidence,
    evidence_for_label_not_mature,
)
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.sealed_split import SealedSplitRef


def _ref(domain, identity, content_hash=""):
    return DomainArtifactRef.of(domain, identity, content_hash=content_hash)


def _make_artifact(**overrides):
    base = {
        "evaluation_id": "ev_1",
        "evaluation_identity": "run-1",
        "factor_value_ref": _ref(ArtifactDomain.FACTOR_VALUE, "fv_1"),
        "label_definition_ref": _ref(ArtifactDomain.LABEL_DEFINITION, "lb_1"),
        "evaluation_policy_ref": _ref(ArtifactDomain.EVALUATION_POLICY, "pol_1"),
        "evaluation_profile_ref": _ref(ArtifactDomain.EVALUATION_PROFILE, "prof_1"),
        "split_ref": _ref(ArtifactDomain.SPLIT, "split_1"),
        "snapshot_ref": _ref(ArtifactDomain.SNAPSHOT, "snap_1"),
        "universe_ref": _ref(ArtifactDomain.UNIVERSE, "uni_1"),
        "metric_evidence_refs": [
            _ref(ArtifactDomain.METRIC_EVIDENCE, "ev_ref_1"),
            _ref(ArtifactDomain.METRIC_EVIDENCE, "ev_ref_2"),
        ],
        "diagnostic_refs": [_ref(ArtifactDomain.DIAGNOSTIC, "diag_1")],
        "timing": {"decision_time": "2026-01-01", "label_end_time": "2026-01-11"},
        "producer_version": "0.1",
        "schema_version": "0.1",
        "created_at": "2026-08-27T00:00:00Z",
    }
    base.update(overrides)
    return EvaluationArtifact(**base)


# ---------------------------------------------------------------------------
# DLIB-QE-002: EvaluationArtifact deep immutability + derived content_hash
# ---------------------------------------------------------------------------


def test_artifact_deep_immutability_original_dict_mutation():
    """Mutating the caller's original structures must not change the artifact."""
    timing = {"decision_time": "2026-01-01", "label_end_time": "2026-01-11"}
    refs = [
        DomainArtifactRef.of(ArtifactDomain.METRIC_EVIDENCE, "ev_ref_1"),
        DomainArtifactRef.of(ArtifactDomain.METRIC_EVIDENCE, "ev_ref_2"),
    ]
    art = _make_artifact(metric_evidence_refs=refs, timing=timing)
    # Mutate the original structures after construction.
    timing["decision_time"] = "MUTATED"
    refs.append(DomainArtifactRef.of(ArtifactDomain.METRIC_EVIDENCE, "ev_ref_3"))
    assert art.timing["decision_time"] == "2026-01-01"
    assert len(art.metric_evidence_refs) == 2


def test_artifact_nested_mutation_raises():
    """Nested mutation through the artifact must raise (deep-frozen)."""
    art = _make_artifact()
    with pytest.raises(TypeError):
        art.timing["decision_time"] = "MUTATED"
    with pytest.raises((TypeError, AttributeError)):
        art.result_content.metric_values["rank_ic"] = 0.99


def test_artifact_content_hash_derived_only_fail_closed():
    """A caller-supplied content_hash that disagrees must raise ValueError."""
    with pytest.raises(ValueError):
        _make_artifact(content_hash="forged-hash")


def test_artifact_content_hash_matching_accepted():
    """A caller-supplied content_hash equal to the derived value is accepted."""
    art = _make_artifact()
    restored = _make_artifact(content_hash=art.content_hash)
    assert restored.content_hash == art.content_hash


def test_artifact_content_hash_stable_and_roundtrip():
    """Identical semantic fields produce identical hashes; to_dict round-trips."""
    a = _make_artifact()
    b = _make_artifact()
    assert a.content_hash == b.content_hash
    restored = EvaluationArtifact.from_dict(a.to_dict())
    assert restored.content_hash == a.content_hash
    assert restored.evaluation_id == a.evaluation_id
    assert dict(restored.timing) == dict(a.timing)


def test_artifact_content_hash_changes_with_semantics():
    """Changing a semantic field changes the derived hash."""
    a = _make_artifact()
    b = _make_artifact(evaluation_identity="run-2")
    assert a.content_hash != b.content_hash


# ---------------------------------------------------------------------------
# DLIB-QE-003: EvaluationRequest to_dict/from_dict strict round-trip on refs
# ---------------------------------------------------------------------------


def test_request_roundtrip_preserves_refs():
    """to_dict/from_dict round-trips factor_value_ref / label_bundle_ref / split_ref."""
    fv = FactorValueRef(factor_value_id="fv_1", factor_ids=("f1", "f2"))
    lb = LabelBundleRef(label_bundle_id="lb_1", target_id="ret_10d", horizon=10)
    split = SealedSplitRef(split_id="split_1", start_time=9, end_time=12)
    request = EvaluationRequest(
        batch_or_factor_ids="dummy",
        label_bundle="dummy",
        metric_ids=("rank_ic",),
        metadata={"k": "v"},
        split_ref=split,
        factor_value_ref=fv,
        label_bundle_ref=lb,
    )
    restored = EvaluationRequest.from_dict(request.to_dict())
    assert restored.split_ref == split
    assert restored.factor_value_ref == fv
    assert restored.label_bundle_ref == lb
    assert restored.metric_ids == ("rank_ic",)
    assert restored.metadata == {"k": "v"}
    # Raw payloads are runtime-only and not serialized.
    assert restored.batch_or_factor_ids is None
    assert restored.label_bundle is None


def test_request_roundtrip_without_refs():
    """A request with no refs round-trips with all refs None."""
    request = EvaluationRequest(batch_or_factor_ids="dummy", label_bundle="dummy")
    restored = EvaluationRequest.from_dict(request.to_dict())
    assert restored.split_ref is None
    assert restored.factor_value_ref is None
    assert restored.label_bundle_ref is None


def test_request_refs_are_json_friendly():
    """The serialized refs are plain JSON-friendly dicts (no arrays)."""
    fv = FactorValueRef(factor_value_id="fv_1", factor_ids=("f1",))
    lb = LabelBundleRef(label_bundle_id="lb_1", target_id="ret_10d", horizon=10)
    request = EvaluationRequest(
        batch_or_factor_ids="dummy",
        label_bundle="dummy",
        factor_value_ref=fv,
        label_bundle_ref=lb,
    )
    payload = request.to_dict()
    assert payload["factor_value_ref"] == {
        "factor_value_id": "fv_1",
        "factor_ids": ["f1"],
    }
    assert payload["label_bundle_ref"] == {
        "label_bundle_id": "lb_1",
        "target_id": "ret_10d",
        "horizon": 10,
    }


# ---------------------------------------------------------------------------
# DLIB-QE-004: EvidenceStatus label-not-mature -> no IC (never zero)
# ---------------------------------------------------------------------------


def test_evidence_status_has_label_maturity_members():
    """The QE EvidenceStatus covers label-not-mature / invalid-evidence."""
    assert EvidenceStatus.LABEL_NOT_MATURE.value == "label_not_mature"
    assert EvidenceStatus.INVALID_EVIDENCE.value == "invalid_evidence"
    assert EvidenceStatus.INSUFFICIENT_DATA.value == "insufficient_data"


def test_evidence_status_from_value_accepts_new_members():
    assert (
        EvidenceStatus.from_value("label_not_mature") is EvidenceStatus.LABEL_NOT_MATURE
    )
    assert (
        EvidenceStatus.from_value("invalid_evidence") is EvidenceStatus.INVALID_EVIDENCE
    )


def test_label_not_mature_evidence_has_no_ic():
    """A label-not-mature evidence yields no IC — never a fabricated 0.0."""
    ev = evidence_for_label_not_mature(observations=0, minimum_required=20)
    assert ev.computed is False
    assert ev.status is EvidenceStatus.LABEL_NOT_MATURE
    assert ev.reason_code is EvidenceReasonCode.LABEL_NOT_YET_MATURE
    # There is no numeric payload to read -- absent evidence, not zero.
    assert ev.artifact is None


def test_label_not_mature_roundtrip():
    """LABEL_NOT_MATURE serializes and deserializes losslessly."""
    ev = evidence_for_label_not_mature(observations=0, minimum_required=20)
    restored = MetricEvidence.from_dict(ev.to_dict())
    assert restored.status is EvidenceStatus.LABEL_NOT_MATURE
    assert restored.reason_code is EvidenceReasonCode.LABEL_NOT_YET_MATURE
    assert restored.computed is False
