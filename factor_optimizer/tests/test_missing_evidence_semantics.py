"""R61-FI-030 / E-TDD-001: missing-evidence semantics — adversarial tests.

Plan §1.5 / §23 / matrix E2.  The historical pipeline defaulted every metric
to a bare ``0.0`` float and treated "no bootstrap series" as full confidence.
These tests FIRST express the required semantics and lock them in:

1. A missing RankIC must NOT be a computed 0.0.
2. A missing coverage must NOT be a computed 0.0.
3. Missing bootstrap evidence must NOT read as FULL uncertainty — a candidate
   with no bootstrap series carries POINT_ESTIMATE_ONLY, and a winner policy
   whose minimum evidence tier is BOOTSTRAP_CONFIDENCE must REJECT it.
4. A candidate lacking a required uncertainty value is REJECTED / DEFERRED by
   a policy that requires uncertainty evidence — never silently admitted.
5. A research/deterministic mode (point-estimates, no bootstrap required) is
   explicit and auditable (declares minimum tier POINT_ESTIMATE_ONLY and
   records the declared tier in its step trace).

These tests pass against the new evidence-bound vocabulary
(:mod:`factor_optimizer.contracts.evidence_value`) that does NOT flatten
missing values to floats; a candidate that carries no bootstrap series has
``evidence_tier=POINT_ESTIMATE_ONLY``, and the decision pipeline's STEP 6
keeps such a candidate only under an explicitly POINT_ESTIMATE_ONLY policy.
"""

import pytest

from factor_optimizer.contracts import (
    CandidateFitnessArtifact,
    EvidenceSeries,
    EvidenceStatus,
    EvidenceTier,
    EvidenceValue,
    FactorFitnessSpec,
)
from factor_optimizer.contracts.evidence_value import status_of


# ---------------------------------------------------------------------------
# 1. Missing RankIC is not a computed 0.0
# ---------------------------------------------------------------------------


def test_missing_rank_ic_is_none_not_zero():
    """A missing RankIC is value=None with a non-COMPUTED status.

    ``EvidenceValue`` refuses to fabricate: a non-COMPUTED status cannot carry
    a numeric payload, and a ``None`` value cannot be marked COMPUTED.
    """
    missing = EvidenceValue(
        metric_id="rank_ic", value=None, status=EvidenceStatus.NOT_COMPUTED
    )
    assert missing.value is None
    assert missing.present is False
    assert missing.status.computed is False
    assert missing.value != 0.0  # None is not 0.0 — it is *missing*
    # A caller may not attach a number to a non-computed status (0.0 is no
    # better than any other fabricated number here).
    with pytest.raises(ValueError, match="numeric value"):
        EvidenceValue(
            metric_id="rank_ic", value=0.0, status=EvidenceStatus.NOT_COMPUTED
        )


def test_missing_rank_ic_serializes_as_none():
    missing = EvidenceValue(
        metric_id="rank_ic", value=None, status=EvidenceStatus.INSUFFICIENT_DATA
    )
    restored = EvidenceValue.from_dict(missing.to_dict())
    assert restored.value is None
    assert restored.status is EvidenceStatus.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# 2. Missing coverage is not a computed 0.0
# ---------------------------------------------------------------------------


def test_missing_coverage_is_none_not_zero():
    missing = EvidenceValue(
        metric_id="coverage", value=None, status=EvidenceStatus.NOT_COMPUTED
    )
    assert missing.value is None
    assert missing.present is False
    # A coverage of 0.0 has a very different meaning (the treatment covered
    # NO observations) from "coverage was never computed".
    computed_zero = EvidenceValue(
        metric_id="coverage", value=0.0, status=EvidenceStatus.COMPUTED
    )
    assert computed_zero.present is True
    assert computed_zero.value == 0.0
    # The two must be distinguishable in the type system: missing is None.
    assert (missing.value is None) != (computed_zero.value is None)


# ---------------------------------------------------------------------------
# 3. Missing bootstrap != FULL_UNCERTAINTY — no-bootstrap candidate is
#    POINT_ESTIMATE_ONLY and a BOOTSTRAP_CONFIDENCE policy rejects it
# ---------------------------------------------------------------------------


def test_missing_bootstrap_is_not_full_uncertainty():
    """An absent bootstrap series is typed (None/not computed), not an empty
    'full confidence' series."""
    absent = EvidenceSeries(samples=(), status=EvidenceStatus.NOT_COMPUTED)
    assert absent.present is False
    with pytest.raises(ValueError, match="at least one"):
        # An empty series must never be labelled computed — an empty computed
        # series would be indistinguishable from "the resampling ran and
        # produced nothing worth showing".
        EvidenceSeries(samples=(), status=EvidenceStatus.COMPUTED)


def test_no_bootstrap_candidate_is_point_estimate_only():
    artifact = CandidateFitnessArtifact(
        trial_id="EWMA",
        evaluation_ref="ev-1",
        health_card_ref="hc-1",
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    assert artifact.evidence_tier is EvidenceTier.POINT_ESTIMATE_ONLY
    assert artifact.meets_evidence_tier(EvidenceTier.POINT_ESTIMATE_ONLY)
    assert not artifact.meets_evidence_tier(EvidenceTier.BOOTSTRAP_CONFIDENCE)
    # And the enum itself knows the ladder ordering.
    assert (
        EvidenceTier.BOOTSTRAP_CONFIDENCE.meets(EvidenceTier.POINT_ESTIMATE_ONLY)
        is True
    )
    assert (
        EvidenceTier.POINT_ESTIMATE_ONLY.meets(EvidenceTier.BOOTSTRAP_CONFIDENCE)
        is False
    )


def test_winner_policy_declaring_bootstrap_minimum_rejects_point_only_candidate():
    """A winner policy whose minimum tier is BOOTSTRAP_CONFIDENCE must not
    select a POINT_ESTIMATE_ONLY candidate — the fitness spec encodes the
    gate."""
    spec = FactorFitnessSpec(
        hard_gate_policy_id="hg-1",
        dimension_floor_policy_id="df-1",
        raw_relative_policy_id="rr-1",
        uncertainty_policy_id="u-1",
        multiplicity_policy_id="m-1",
        complexity_policy_id="c-1",
        winner_policy_id="w-1",
        minimum_evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE,
    )
    assert spec.requires_evidence_tier(EvidenceTier.BOOTSTRAP_CONFIDENCE)
    assert spec.requires_evidence_tier(EvidenceTier.SEALED_TEST_CONFIRMED)
    point_only = CandidateFitnessArtifact(
        trial_id="EWMA",
        evaluation_ref="ev-1",
        health_card_ref="hc-1",
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    assert not spec.requires_evidence_tier(point_only.evidence_tier)
    # A bootstrap-backed candidate clears the gate.
    boot = CandidateFitnessArtifact(
        trial_id="EWMA2",
        evaluation_ref="ev-2",
        health_card_ref="hc-2",
        evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE,
    )
    assert spec.requires_evidence_tier(boot.evidence_tier)


# ---------------------------------------------------------------------------
# 4. Missing uncertainty when policy requires it -> reject/defer, never admit
# ---------------------------------------------------------------------------


def test_candidate_without_uncertainty_not_a_valid_winner_when_required():
    spec = FactorFitnessSpec(
        hard_gate_policy_id="hg-1",
        dimension_floor_policy_id="df-1",
        raw_relative_policy_id="rr-1",
        uncertainty_policy_id="u-uncertainty-required",
        multiplicity_policy_id="m-1",
        complexity_policy_id="c-1",
        winner_policy_id="w-1",
        minimum_evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE,
    )
    # The candidate has a numeric RankIC but its evidence tier is only point
    # estimate (no bootstrap was ever produced).  Under a spec that REQUIRES
    # uncertainty evidence the candidate is not admissible as winner — the
    # contract exposes the rejection predicate instead of silently admitting.
    candidate = CandidateFitnessArtifact(
        trial_id="EWMA",
        evaluation_ref="ev-1",
        health_card_ref="hc-1",
        raw_relative_deltas={"rank_ic": 0.002},
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
        status=EvidenceStatus.COMPUTED,
    )
    assert candidate.status.computed  # numeric value exists...
    assert not spec.requires_evidence_tier(candidate.evidence_tier)
    # ...but the required uncertainty does not, so the candidate cannot win.
    assert candidate.evidence_tier.rank < EvidenceTier.BOOTSTRAP_CONFIDENCE.rank


def test_bad_status_never_implies_a_numeric_value():
    """Every QE status token maps onto the FO enum; none of them compute."""
    tokens = {
        "computed": True,
        "not_computed": False,
        "unavailable": False,
        "unsupported": False,
        "insufficient_data": False,
        "label_not_mature": False,
        "invalid_evidence": False,
        "failed": False,
    }
    for token, computed in tokens.items():
        status = status_of(token)
        assert isinstance(status, EvidenceStatus)
        assert status.computed is computed, token
        assert status.value == token
    # Unknown tokens fail closed — FO never invents a status.
    with pytest.raises(ValueError, match="Unknown EvidenceStatus"):
        status_of("not_a_status")


# ---------------------------------------------------------------------------
# 5. Research deterministic mode is explicit and auditable
# ---------------------------------------------------------------------------


def test_research_point_estimate_mode_is_explicit_and_auditable():
    """A research/deterministic run declares minimum tier POINT_ESTIMATE_ONLY
    (no bootstrap required).  The spec round-trips so the declaration is
    auditable."""
    research = FactorFitnessSpec(
        hard_gate_policy_id="hg-research",
        dimension_floor_policy_id="df-research",
        raw_relative_policy_id="rr-research",
        uncertainty_policy_id="u-none",
        multiplicity_policy_id="m-none",
        complexity_policy_id="c-research",
        winner_policy_id="w-research",
        minimum_evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    assert research.minimum_evidence_tier is EvidenceTier.POINT_ESTIMATE_ONLY
    assert research.requires_evidence_tier(EvidenceTier.POINT_ESTIMATE_ONLY)
    restored = FactorFitnessSpec.from_dict(research.to_dict())
    assert restored.minimum_evidence_tier is EvidenceTier.POINT_ESTIMATE_ONLY
    assert (
        restored.minimum_evidence_tier.value
        == "POINT_ESTIMATE_ONLY"
    )
    # A point-estimate-only candidate is a valid research winner — explicitly.
    candidate = CandidateFitnessArtifact(
        trial_id="RAW",
        evaluation_ref="ev-r",
        health_card_ref="hc-r",
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    assert research.requires_evidence_tier(candidate.evidence_tier)


def test_tier_declaration_is_semantic_and_ordered():
    """The evidence tier ladder is monotone and serializes losslessly."""
    levels = [EvidenceTier.POINT_ESTIMATE_ONLY, EvidenceTier.VALIDATION_SERIES,
              EvidenceTier.BOOTSTRAP_CONFIDENCE, EvidenceTier.MULTIPLE_TESTING_ADJUSTED,
              EvidenceTier.SEALED_TEST_CONFIRMED]
    for i, level in enumerate(levels):
        assert level.rank == i
        for weaker in levels[: i + 1]:
            assert level.meets(weaker)
        for stronger in levels[i + 1:]:
            assert not level.meets(stronger)
    # str-enum equality: serialized values compare to their bare strings.
    assert EvidenceTier.SEALED_TEST_CONFIRMED.value == "SEALED_TEST_CONFIRMED"
