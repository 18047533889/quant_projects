"""R55 audit #23/#24 tests: EvaluationArtifact identity split + typed refs.

Pins the R55 platform-audit findings on ``quant_evaluator``:

R55-#23 (one hash conflating three things):
    ``EvaluationArtifact`` previously derived a single ``content_hash`` over
    everything, so a re-run of the same spec produced a different overall hash
    (breaking downstream "same evaluation?" checks) while two artifacts with
    identical results but different specs wrongly deduped.  The artifact now
    exposes three separately-addressable identities:

    - :class:`EvaluationSpecIdentity` (``evaluation_spec_identity``) — the
      SPEC only (what was asked: metric set, universe, window/split, snapshot,
      policy/profile, treatment and subject refs, parameters, version pins);
    - :attr:`evaluation_result_content_hash` — the RESULTS only (metric
      values, series digests, timing);
    - :class:`ArtifactEnvelopeIdentity` (``evaluation_envelope_identity``) —
      the envelope (evaluation id + spec identity + result content hash +
      timestamps), the cross-reference handle.

R55-#24 (loosely-typed refs):
    Every reference field is a validated, frozen :class:`DomainArtifactRef`
    (domain enum + artifact kind + identity + content hash) with
    ``to_dict`` / ``from_dict`` round-trip; plain strings / ad-hoc dicts are
    rejected fail-closed at construction.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_evaluator.contracts._hashutil import stable_hash
from quant_evaluator.contracts.domain_refs import ArtifactDomain, DomainArtifactRef
from quant_evaluator.contracts.evaluation_artifact import (
    ArtifactEnvelopeIdentity,
    EvaluationArtifact,
    EvaluationResultContentHash,
    EvaluationSpecIdentity,
)


def _ref(domain, identity, content_hash=""):
    return DomainArtifactRef.of(domain, identity, content_hash=content_hash)


def _make(**overrides):
    """A fully-referenced artifact; overrides replace whole fields."""
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
        "metric_evidence_refs": [_ref(ArtifactDomain.METRIC_EVIDENCE, "ev_ref_1")],
        "diagnostic_refs": [_ref(ArtifactDomain.DIAGNOSTIC, "diag_1")],
        "result_content": EvaluationResultContentHash(
            metric_values={"rank_ic": 0.03},
            series_digests={"rank_ic": "digest-1"},
            timing={"decision_time": "2026-01-01"},
        ),
        "timing": {"decision_time": "2026-01-01"},
        "created_at": "2026-08-27T00:00:00Z",
    }
    base.update(overrides)
    return EvaluationArtifact(**base)


# ---------------------------------------------------------------------------
# R55-#23 (a): same spec twice with different data
# ---------------------------------------------------------------------------


def test_same_spec_different_data_same_spec_identity_different_result_hash():
    """A re-run of the same spec over moved data: spec stable, result moves."""
    first = _make()
    second = _make(
        result_content=EvaluationResultContentHash(
            metric_values={"rank_ic": 0.041},  # the re-run's data moved
            series_digests={"rank_ic": "digest-2"},
            timing={"decision_time": "2026-02-01"},
        ),
    )
    assert first.evaluation_spec_identity.identity_hash == (
        second.evaluation_spec_identity.identity_hash
    )
    assert (
        first.evaluation_result_content_hash
        != second.evaluation_result_content_hash
    )
    # Envelope moves too (result hash is part of it).
    assert (
        first.evaluation_envelope_identity.identity_hash
        != second.evaluation_envelope_identity.identity_hash
    )


def test_same_spec_survives_rerun_via_from_dict():
    """Round-tripping a rerun payload keeps the spec identity stable."""
    first = _make()
    rerun_payload = _make(
        result_content=EvaluationResultContentHash(
            metric_values={"rank_ic": 0.05},
        ),
    ).to_dict()
    rerun_payload["evaluation_id"] = "ev_1"  # same evaluation re-run
    rerun_payload.pop("content_hash")
    second = EvaluationArtifact.from_dict(rerun_payload)
    assert (
        first.evaluation_spec_identity.identity_hash
        == second.evaluation_spec_identity.identity_hash
    )
    assert (
        first.evaluation_result_content_hash
        != second.evaluation_result_content_hash
    )


# ---------------------------------------------------------------------------
# R55-#23 (b): same results with different spec
# ---------------------------------------------------------------------------


def test_same_results_different_spec_different_spec_identity():
    """Identical results under different specs must NOT dedupe on the result."""
    a = _make()
    b = _make(
        evaluation_policy_ref=_ref(ArtifactDomain.EVALUATION_POLICY, "pol_2"),
        split_ref=_ref(ArtifactDomain.SPLIT, "split_2"),
    )
    assert a.evaluation_result_content_hash == b.evaluation_result_content_hash
    assert (
        a.evaluation_spec_identity.identity_hash
        != b.evaluation_spec_identity.identity_hash
    )
    assert (
        a.evaluation_envelope_identity.identity_hash
        != b.evaluation_envelope_identity.identity_hash
    )


def test_spec_identity_covers_metric_set_and_parameters():
    """Metric set and free parameters are part of the spec, not the result."""
    a = _make()
    spec = a.evaluation_spec_identity
    with_metrics = EvaluationSpecIdentity(
        metric_set=("rank_ic", "coverage"),
        parameters={"min_obs": 20},
    )
    empty = EvaluationSpecIdentity()
    assert with_metrics.identity_hash != empty.identity_hash
    assert spec.metric_set == ()
    # The spec identity object is what the envelope embeds.
    assert a.evaluation_envelope_identity.spec_identity is spec


def test_subject_refs_enter_spec_identity_only():
    """Subject refs (factor value / label) enter the spec as identity-only."""
    a = _make()
    derived = a.evaluation_spec_identity
    assert [r.identity for r in derived.subject_refs] == ["fv_1", "lb_1"]
    # Identity-only: the content hash is stripped so a re-run over re-computed
    # data under the same logical identity keeps the spec stable.
    assert all(r.content_hash == "" for r in derived.subject_refs)


# ---------------------------------------------------------------------------
# R55-#23 (c): envelope identity == combination
# ---------------------------------------------------------------------------


def test_envelope_identity_is_the_combination():
    """The envelope binds id + spec identity + result hash + timestamps."""
    a = _make()
    env = a.evaluation_envelope_identity
    assert env.evaluation_id == a.evaluation_id
    assert env.spec_identity is a.evaluation_spec_identity
    assert env.result_content_hash == a.evaluation_result_content_hash
    assert env.created_at == a.created_at
    # Recomputing the same combination yields the same digest.
    rebuilt = ArtifactEnvelopeIdentity(
        evaluation_id=env.evaluation_id,
        spec_identity=env.spec_identity,
        result_content_hash=env.result_content_hash,
        created_at=env.created_at,
    )
    assert rebuilt.identity_hash == env.identity_hash
    # Any component moving moves the envelope.
    assert (
        ArtifactEnvelopeIdentity(
            evaluation_id="ev_9",
            spec_identity=env.spec_identity,
            result_content_hash=env.result_content_hash,
            created_at=env.created_at,
        ).identity_hash
        != env.identity_hash
    )


def test_envelope_rejects_wrong_types():
    """The envelope is fail-closed on malformed parts."""
    spec = EvaluationSpecIdentity()
    with pytest.raises(ValueError):
        ArtifactEnvelopeIdentity(
            evaluation_id="  ", spec_identity=spec, result_content_hash=""
        )
    with pytest.raises(TypeError):
        ArtifactEnvelopeIdentity(
            evaluation_id="ev", spec_identity="not-a-spec",
            result_content_hash="",
        )
    with pytest.raises(TypeError):
        ArtifactEnvelopeIdentity(
            evaluation_id="ev", spec_identity=spec, result_content_hash=7
        )


def test_result_content_hash_is_derived_only():
    """A forged result content hash is rejected; a matching one is accepted."""
    result = EvaluationResultContentHash(metric_values={"rank_ic": 0.1})
    assert result.content_hash
    with pytest.raises(ValueError):
        EvaluationResultContentHash(
            metric_values={"rank_ic": 0.1}, content_hash="forged"
        )
    ok = EvaluationResultContentHash(
        metric_values={"rank_ic": 0.1}, content_hash=result.content_hash
    )
    assert ok.content_hash == result.content_hash


def test_spec_identity_rejects_bad_metric_set():
    """A metric set of bare non-strings is rejected fail-closed."""
    with pytest.raises(ValueError):
        EvaluationSpecIdentity(metric_set=("rank_ic", 3))
    with pytest.raises(TypeError):
        EvaluationSpecIdentity(metric_set="rank_ic")  # a bare str is not a set


# ---------------------------------------------------------------------------
# R55-#23 (e): legacy property still present and equals the envelope hash
# ---------------------------------------------------------------------------


def test_legacy_artifact_hash_equals_envelope_identity():
    """``artifact_hash`` is retained and is exactly the envelope identity hash."""
    a = _make()
    assert a.artifact_hash == a.evaluation_envelope_identity.identity_hash
    # And the full content hash (legacy DLIB-QE-002 semantics) still derives.
    assert a.content_hash
    restored = EvaluationArtifact.from_dict(a.to_dict())
    assert restored.artifact_hash == a.artifact_hash
    assert restored.content_hash == a.content_hash


def test_three_identities_round_trip():
    """to_dict/from_dict restores all three identities exactly."""
    a = _make()
    payload = a.to_dict()
    assert set(payload) >= {"spec_identity", "result_content", "envelope_identity"}
    restored = EvaluationArtifact.from_dict(payload)
    assert restored.evaluation_spec_identity.identity_hash == (
        a.evaluation_spec_identity.identity_hash
    )
    assert (
        restored.evaluation_result_content_hash
        == a.evaluation_result_content_hash
    )
    assert (
        restored.evaluation_envelope_identity.identity_hash
        == a.evaluation_envelope_identity.identity_hash
    )


# ---------------------------------------------------------------------------
# R55-#24: DomainArtifactRef
# ---------------------------------------------------------------------------


def test_domain_ref_round_trip():
    """A DomainArtifactRef serializes and rebuilds losslessly."""
    ref = DomainArtifactRef.of(
        ArtifactDomain.FACTOR_VALUE, "val:profit:raw",
        content_hash="a" * 64, artifact_kind="factor_value",
    )
    payload = ref.to_dict()
    assert payload == {
        "domain": "factor_value",
        "artifact_kind": "factor_value",
        "identity": "val:profit:raw",
        "content_hash": "a" * 64,
    }
    restored = DomainArtifactRef.from_dict(payload)
    assert restored == ref
    assert restored.ref_hash == ref.ref_hash


def test_domain_ref_round_trip_without_content_hash():
    """An identity-only ref omits the content_hash key and stays stable."""
    ref = DomainArtifactRef.of(ArtifactDomain.SPLIT, "split_1")
    payload = ref.to_dict()
    assert payload == {
        "domain": "split",
        "artifact_kind": "split",
        "identity": "split_1",
    }
    restored = DomainArtifactRef.from_dict(payload)
    assert restored == ref
    assert restored.is_content_addressed is False
    assert ref.identity_only() is ref


def test_domain_ref_rejects_malformed_input():
    """Malformed refs are rejected fail-closed at construction."""
    with pytest.raises(TypeError):
        DomainArtifactRef(domain="factor_value", artifact_kind="k", identity="i")
    with pytest.raises(ValueError):
        DomainArtifactRef(
            domain=ArtifactDomain.FACTOR_VALUE, artifact_kind="", identity="i"
        )
    with pytest.raises(ValueError):
        DomainArtifactRef(
            domain=ArtifactDomain.FACTOR_VALUE, artifact_kind="k", identity="  "
        )
    with pytest.raises(ValueError):
        DomainArtifactRef(
            domain=ArtifactDomain.FACTOR_VALUE, artifact_kind="k", identity="i",
            content_hash="   ",
        )
    with pytest.raises(ValueError):
        DomainArtifactRef.from_dict({"domain": "factor_value"})  # missing keys
    with pytest.raises(ValueError):
        DomainArtifactRef.from_dict(
            {"domain": "not_a_domain", "artifact_kind": "k", "identity": "i"}
        )
    with pytest.raises(TypeError):
        DomainArtifactRef.from_dict("factor_value")  # not a mapping


def test_domain_ref_hashable_and_stable():
    """The ref's own hash is stable across processes (PYTHONHASHSEED-proof)."""
    a = DomainArtifactRef.of(ArtifactDomain.UNIVERSE, "uni_1")
    b = DomainArtifactRef.of(ArtifactDomain.UNIVERSE, "uni_1")
    assert hash(a) == hash(b) == stable_hash(a.ref_hash)
    c = DomainArtifactRef.of(ArtifactDomain.UNIVERSE, "uni_2")
    assert hash(a) != hash(c)
    assert a != c


def test_artifact_rejects_plain_string_refs():
    """Every reference field rejects a bare string (fail-closed)."""
    string_fields = [
        "factor_value_ref",
        "label_definition_ref",
        "evaluation_policy_ref",
        "evaluation_profile_ref",
        "split_ref",
        "snapshot_ref",
        "universe_ref",
    ]
    for field_name in string_fields:
        with pytest.raises(TypeError) as excinfo:
            _make(**{field_name: "loose-string-ref"})
        assert "DomainArtifactRef" in str(excinfo.value)


def test_artifact_rejects_plain_string_ref_tuples():
    """Ref *collections* of bare strings are rejected too."""
    with pytest.raises(TypeError):
        _make(metric_evidence_refs=["ev_ref_1", "ev_ref_2"])
    with pytest.raises(TypeError):
        _make(diagnostic_refs="diag_1")
    with pytest.raises(TypeError):
        _make(metric_evidence_refs=[{"ref": "ev_ref_1"}])


def test_artifact_accepts_typed_refs_everywhere():
    """All reference fields accept DomainArtifactRef and round-trip."""
    a = _make()
    assert a.factor_value_ref.domain is ArtifactDomain.FACTOR_VALUE
    assert a.label_definition_ref.domain is ArtifactDomain.LABEL_DEFINITION
    assert a.evaluation_policy_ref.domain is ArtifactDomain.EVALUATION_POLICY
    assert a.evaluation_profile_ref.domain is ArtifactDomain.EVALUATION_PROFILE
    assert a.split_ref.domain is ArtifactDomain.SPLIT
    assert a.snapshot_ref.domain is ArtifactDomain.SNAPSHOT
    assert a.universe_ref.domain is ArtifactDomain.UNIVERSE
    assert a.metric_evidence_refs[0].domain is ArtifactDomain.METRIC_EVIDENCE
    assert a.diagnostic_refs[0].domain is ArtifactDomain.DIAGNOSTIC
    restored = EvaluationArtifact.from_dict(a.to_dict())
    assert restored.factor_value_ref == a.factor_value_ref
    assert restored.metric_evidence_refs == a.metric_evidence_refs
    assert restored.diagnostic_refs == a.diagnostic_refs


def test_domain_refs_exported_from_contracts_package():
    """The typed refs + identities are on the package's public surface."""
    import quant_evaluator.contracts as contracts

    for name in (
        "ArtifactDomain",
        "DomainArtifactRef",
        "EvaluationSpecIdentity",
        "EvaluationResultContentHash",
        "ArtifactEnvelopeIdentity",
        "EvaluationArtifact",
    ):
        assert hasattr(contracts, name), name