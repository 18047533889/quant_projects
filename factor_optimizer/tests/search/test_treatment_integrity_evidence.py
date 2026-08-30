"""R55 P0-9: TreatmentIntegrityEvidence contract + fail-closed gates.

The historical FO "integrity gates" were fake: they passed unconditionally or
re-derived their verdicts from the very metrics being scored, so a treatment
that was never applied (or applied with different parameters than claimed)
sailed through.  These tests pin the honest behavior:

- the evidence artifact is deep-immutable and content-hashed,
- its digests are MEASURED from the actual data (never self-reported),
- its overall status is DERIVED from the recorded checks (never stored),
- the evidence gate fails closed on missing / stale / tampered / NOT_RUN /
  failing evidence,
- the content hash changes when any check value changes, so evidence cannot
  be forged by editing a single field.
"""

import dataclasses

import numpy as np
import pytest

from factor_optimizer.contracts.treatment_integrity import (
    EVIDENCE_SCHEMA_VERSION,
    RAW_TREATMENT_KIND,
    IntegrityCheckResult,
    TreatmentIntegrityEvidence,
    TreatmentIntegrityStatus,
    build_integrity_evidence,
    describe_integrity_problem,
    digest_value,
    require_integrity_evidence,
)
from factor_optimizer.errors import TreatmentIntegrityError


BEFORE = np.arange(16, dtype=float)
AFTER = BEFORE * 0.5 + 1.0


def _evidence(treatment_id="t1", kind="ewma", **kw):
    return build_integrity_evidence(
        treatment_id, kind, {"span": 5}, BEFORE, AFTER, **kw
    )


# ---------------------------------------------------------------------------
# (e) evidence is frozen (mutation raises)
# ---------------------------------------------------------------------------


def test_evidence_is_frozen():
    ev = _evidence()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.treatment_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.before_digest = "0" * 64
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.integrity_checks = ()


def test_evidence_nested_containers_are_frozen():
    ev = _evidence()
    # MappingProxyType rejects item assignment...
    with pytest.raises(TypeError):
        ev.applied_parameters["span"] = 99
    # ...and the tuple of checks cannot be reassigned (frozen dataclass).
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.integrity_checks = ()
    # The check records are themselves frozen dataclasses: direct field
    # writes raise (only constructing a new check object is possible).
    check = ev.integrity_checks[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(check, "name", "renamed")


def test_evidence_rejects_non_frozen_types():
    # Unsupported container types fail closed instead of silently hashing an
    # unstable object.
    with pytest.raises(TypeError):
        TreatmentIntegrityEvidence(
            treatment_id="t1",
            treatment_kind="ewma",
            applied_parameters={"bad": object()},
            integrity_checks=(
                IntegrityCheckResult("parameters_recorded", True,
                                     "params recorded", 1.0),
            ),
            before_digest="a" * 64,
            after_digest="b" * 64,
        )


# ---------------------------------------------------------------------------
# digest provenance: digests are measured, never self-reported
# ---------------------------------------------------------------------------


def test_digest_value_is_deterministic_and_sensitive():
    assert digest_value(BEFORE) == digest_value(BEFORE.copy())
    assert digest_value(BEFORE) != digest_value(AFTER)
    # Same numbers, different dtype -> different digest (the treated values
    # are not bit-identical, so the evidence must say so).
    assert digest_value(BEFORE.astype(np.float32)) != digest_value(BEFORE)


def test_build_computes_digests_from_actual_data():
    ev = build_integrity_evidence("t1", "ewma", {"span": 5}, BEFORE, AFTER)
    assert ev.before_digest == digest_value(BEFORE)
    assert ev.after_digest == digest_value(AFTER)
    # A caller cannot forge the digests: the evidence stores what it measured.
    forged = build_integrity_evidence("t1", "ewma", {"span": 5}, BEFORE, AFTER)
    assert forged.content_hash == ev.content_hash
    other = build_integrity_evidence("t1", "ewma", {"span": 5}, BEFORE, BEFORE)
    assert other.content_hash != ev.content_hash


def test_untreated_output_fails_the_treatment_ran_check():
    """A treatment whose output equals its input did not run -> FAILED."""
    ev = build_integrity_evidence("t1", "ewma", {"span": 5}, BEFORE, BEFORE)
    assert ev.overall_status is TreatmentIntegrityStatus.FAILED
    assert "treatment_changed_data" in ev.failed_checks
    with pytest.raises(TreatmentIntegrityError, match="treatment_changed_data"):
        require_integrity_evidence("t1", ev)


def test_raw_baseline_must_be_identity():
    """RAW's evidence requires the opposite: before == after."""
    ev = build_integrity_evidence("r1", RAW_TREATMENT_KIND, {}, BEFORE, BEFORE)
    assert ev.overall_status is TreatmentIntegrityStatus.PASSED
    # A "raw" treatment that actually changed the data is a lie.
    ev_bad = build_integrity_evidence(
        "r1", RAW_TREATMENT_KIND, {}, BEFORE, AFTER
    )
    assert ev_bad.overall_status is TreatmentIntegrityStatus.FAILED
    assert "raw_is_identity" in ev_bad.failed_checks


def test_parameters_must_be_recorded_for_non_raw():
    with pytest.raises(TreatmentIntegrityError, match="parameters_recorded"):
        require_integrity_evidence(
            "t1",
            build_integrity_evidence("t1", "ewma", {}, BEFORE, AFTER),
        )


def test_parameters_match_recipe_check():
    ev = build_integrity_evidence(
        "t1", "ewma", {"span": 5}, BEFORE, AFTER,
        expected_parameters={"span": 10},
    )
    assert ev.overall_status is TreatmentIntegrityStatus.FAILED
    assert "parameters_match_recipe" in ev.failed_checks
    ev_ok = build_integrity_evidence(
        "t1", "ewma", {"span": 10}, BEFORE, AFTER,
        expected_parameters={"span": 10},
    )
    assert ev_ok.overall_status is TreatmentIntegrityStatus.PASSED


# ---------------------------------------------------------------------------
# overall_status is derived, not stored
# ---------------------------------------------------------------------------


def test_overall_status_is_derived_not_stored():
    ev = _evidence()
    assert ev.overall_status is TreatmentIntegrityStatus.PASSED
    # NOT_RUN: an evidence with no checks proves nothing.
    empty = TreatmentIntegrityEvidence(
        treatment_id="t1",
        treatment_kind="ewma",
        applied_parameters={"span": 5},
        integrity_checks=(),
        before_digest=ev.before_digest,
        after_digest=ev.after_digest,
    )
    assert empty.overall_status is TreatmentIntegrityStatus.NOT_RUN
    with pytest.raises(TreatmentIntegrityError, match="NOT_RUN|no checks"):
        require_integrity_evidence("t1", empty)


def test_schema_version_and_serialization_roundtrip():
    ev = _evidence()
    assert EVIDENCE_SCHEMA_VERSION == "1.0"
    data = ev.to_dict()
    assert data["overall_status"] == "PASSED"
    restored = TreatmentIntegrityEvidence.from_dict(data)
    assert restored.content_hash == ev.content_hash
    assert restored == ev
    # A self-reported hash that disagrees with the payload fails closed.
    tampered = dict(data)
    tampered["treatment_kind"] = "winsor"
    with pytest.raises(TreatmentIntegrityError, match="content_hash"):
        TreatmentIntegrityEvidence.from_dict(tampered)


# ---------------------------------------------------------------------------
# (a) gate rejects when evidence is missing
# ---------------------------------------------------------------------------


def test_gate_rejects_missing_evidence():
    with pytest.raises(TreatmentIntegrityError, match="missing"):
        require_integrity_evidence("t1", None)


def test_gate_rejects_stale_evidence_bound_to_another_treatment():
    ev = _evidence("t1")
    with pytest.raises(TreatmentIntegrityError, match="bound to treatment"):
        require_integrity_evidence("t9", ev)


def test_gate_rejects_wrong_treatment_kind():
    ev = _evidence("t1", kind="ewma")
    with pytest.raises(TreatmentIntegrityError, match="claims kind"):
        require_integrity_evidence("t1", ev, treatment_kind="kama")


def test_gate_rejects_tampered_evidence():
    ev = _evidence()
    # In-place mutation after construction is detected by verify().
    object.__setattr__(ev, "before_digest", "f" * 64)
    with pytest.raises(TreatmentIntegrityError, match="tampered"):
        require_integrity_evidence("t1", ev)


# ---------------------------------------------------------------------------
# (b) gate rejects when a check failed
# ---------------------------------------------------------------------------


def test_gate_rejects_failed_check():
    good = _evidence()
    failed = TreatmentIntegrityEvidence(
        treatment_id=good.treatment_id,
        treatment_kind=good.treatment_kind,
        applied_parameters=dict(good.applied_parameters),
        integrity_checks=tuple(good.integrity_checks)
        + (
            IntegrityCheckResult(
                name="pit_no_leakage",
                passed=False,
                expected="no future observation may reach the treated value",
                detail="lag check failed",
            ),
        ),
        before_digest=good.before_digest,
        after_digest=good.after_digest,
    )
    assert failed.overall_status is TreatmentIntegrityStatus.FAILED
    with pytest.raises(TreatmentIntegrityError, match="pit_no_leakage"):
        require_integrity_evidence(good.treatment_id, failed)


def test_describe_integrity_problem_reports_reason():
    assert describe_integrity_problem("t1", None) is not None
    assert describe_integrity_problem("t9", _evidence("t1")) is not None
    assert describe_integrity_problem("t1", _evidence("t1")) is None


# ---------------------------------------------------------------------------
# (c) gate accepts when evidence is complete and passing
# ---------------------------------------------------------------------------


def test_gate_accepts_complete_passing_evidence():
    ev = _evidence()
    assert ev.overall_status is TreatmentIntegrityStatus.PASSED
    assert require_integrity_evidence("t1", ev) is ev
    # Extra producer checks (PIT/leakage/NaN) ride along and still pass.
    full = build_integrity_evidence(
        "t1", "ewma", {"span": 5}, BEFORE, AFTER,
        extra_checks=(
            IntegrityCheckResult(
                "pit_no_leakage", True,
                "no future observation may reach the treated value", 0.0,
                "lag verified"),
            IntegrityCheckResult("nan_fraction", True, "no NaN", 0.0),
        ),
    )
    require_integrity_evidence("t1", full)


# ---------------------------------------------------------------------------
# (d) evidence content hash changes when any check value changes
# ---------------------------------------------------------------------------


def test_content_hash_changes_when_a_check_value_changes():
    ev = _evidence()
    variants = []
    for measured in (0.0, 0.25, 0.5):
        variants.append(
            TreatmentIntegrityEvidence(
                treatment_id=ev.treatment_id,
                treatment_kind=ev.treatment_kind,
                applied_parameters=dict(ev.applied_parameters),
                integrity_checks=tuple(ev.integrity_checks)
                + (
                    IntegrityCheckResult(
                        "nan_fraction", True, "no NaN", measured
                    ),
                ),
                before_digest=ev.before_digest,
                after_digest=ev.after_digest,
            )
        )
    hashes = {variant.content_hash for variant in variants}
    # Each distinct measured value yields a distinct hash: editing one field
    # of the evidence cannot be hidden.
    assert len(hashes) == len(variants)
    assert all(h != ev.content_hash for h in hashes)


def test_content_hash_changes_when_a_check_verdict_flips():
    ev = _evidence()
    flipped = TreatmentIntegrityEvidence(
        treatment_id=ev.treatment_id,
        treatment_kind=ev.treatment_kind,
        applied_parameters=dict(ev.applied_parameters),
        integrity_checks=(
            IntegrityCheckResult(
                name="parameters_recorded",
                passed=False,  # flipped verdict
                expected=ev.integrity_checks[0].expected,
                measured_value=ev.integrity_checks[0].measured_value,
            ),
            ev.integrity_checks[1],
        ),
        before_digest=ev.before_digest,
        after_digest=ev.after_digest,
    )
    assert flipped.content_hash != ev.content_hash
    with pytest.raises(TreatmentIntegrityError, match="parameters_recorded"):
        require_integrity_evidence("t1", flipped)


def test_content_hash_changes_when_applied_parameters_change():
    before = _evidence()
    after = build_integrity_evidence(
        "t1", "ewma", {"span": 9}, BEFORE, AFTER
    )
    assert before.content_hash != after.content_hash


def test_check_names_must_be_unique():
    """A duplicated check name would let one verdict stand in for another."""
    with pytest.raises(ValueError, match="unique"):
        TreatmentIntegrityEvidence(
            treatment_id="t1",
            treatment_kind="ewma",
            applied_parameters={"span": 5},
            integrity_checks=(
                IntegrityCheckResult("dup", True),
                IntegrityCheckResult("dup", False),
            ),
            before_digest="a" * 64,
            after_digest="b" * 64,
        )


def test_measured_value_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        IntegrityCheckResult("nan_fraction", True, "no NaN", float("nan"))
    with pytest.raises(TypeError, match="non-boolean"):
        IntegrityCheckResult("nan_fraction", True, "no NaN", True)


def test_digest_algorithm_is_pinned():
    ev = _evidence()
    assert ev.data_digest_algorithm == "sha256"
    with pytest.raises(ValueError, match="sha256"):
        dataclasses.replace(ev, data_digest_algorithm="md5")