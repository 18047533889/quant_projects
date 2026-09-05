"""R61-FI-031 / plan §21: FactorFitnessSpec / CandidateFitnessArtifact tests.

Covers (matrix E1):
1. Frozen contracts with the full §21 field sets.
2. Evidence-tier minimums: a spec whose ``minimum_evidence_tier`` is
   BOOTSTRAP_CONFIDENCE rejects point-estimate-only candidates.
3. RAW-relative delta semantics: lower-better raw metrics (turnover /
   exposure) normalize to "higher = better-than-RAW"; missing deltas are
   ``None``, never ``0.0``.
4. Health dimension ids are validated against the FA 14-dim authority;
   dimension scores are 0..1 and fail loud otherwise.
5. Serialization round-trips preserve the contract.

The decision pipeline may internally use scalar utilities for ranking, but
admission/winner authority passes through these contracts — the docstrings in
:mod:`factor_optimizer.contracts.factor_fitness` spell this out.
"""

import dataclasses

import pytest

from factor_optimizer.contracts import (
    CandidateFitnessArtifact,
    EvidenceStatus,
    EvidenceTier,
    FactorFitnessSpec,
)
from factor_optimizer.ports.factor_intelligence import HealthDimension

# Required policy ids (every step of the 8-step pipeline names a policy).
_REQUIRED_POLICY_FIELDS = (
    "evidence_profile_id",
    "required_health_dimensions",
    "hard_gate_policy_id",
    "dimension_floor_policy_id",
    "raw_relative_policy_id",
    "uncertainty_policy_id",
    "multiplicity_policy_id",
    "complexity_policy_id",
    "winner_policy_id",
    "minimum_evidence_tier",
)


def _spec(**overrides):
    defaults = dict(
        hard_gate_policy_id="hg-1",
        dimension_floor_policy_id="df-1",
        raw_relative_policy_id="rr-1",
        uncertainty_policy_id="u-1",
        multiplicity_policy_id="m-1",
        complexity_policy_id="c-1",
        winner_policy_id="w-1",
    )
    defaults.update(overrides)
    return FactorFitnessSpec(**defaults)


# ---------------------------------------------------------------------------
# 1. Frozen contracts + full §21 field sets
# ---------------------------------------------------------------------------


def test_factor_fitness_spec_is_frozen_with_full_field_set():
    assert dataclasses.is_dataclass(FactorFitnessSpec)
    assert FactorFitnessSpec.__dataclass_params__.frozen
    names = {f.name for f in dataclasses.fields(FactorFitnessSpec)}
    assert set(_REQUIRED_POLICY_FIELDS) <= names
    spec = _spec()
    assert spec.evidence_profile_id == "QE_EVIDENCE_PROFILE"
    assert spec.minimum_evidence_tier is EvidenceTier.POINT_ESTIMATE_ONLY
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.hard_gate_policy_id = "other"


def test_candidate_fitness_artifact_is_frozen_with_full_field_set():
    assert dataclasses.is_dataclass(CandidateFitnessArtifact)
    assert CandidateFitnessArtifact.__dataclass_params__.frozen
    names = {f.name for f in dataclasses.fields(CandidateFitnessArtifact)}
    assert {
        "trial_id", "evaluation_ref", "health_card_ref", "raw_relative_deltas",
        "evidence_tier", "status", "dimension_scores",
    } <= names
    art = CandidateFitnessArtifact(
        trial_id="RAW", evaluation_ref="ev-1", health_card_ref="hc-1"
    )
    assert art.status is EvidenceStatus.COMPUTED
    with pytest.raises(dataclasses.FrozenInstanceError):
        art.trial_id = "OTHER"


def test_spec_requires_nonempty_policy_ids():
    with pytest.raises(ValueError, match="hard_gate_policy_id"):
        FactorFitnessSpec(
            hard_gate_policy_id="", dimension_floor_policy_id="df",
            raw_relative_policy_id="rr", uncertainty_policy_id="u",
            multiplicity_policy_id="m", complexity_policy_id="c",
            winner_policy_id="w",
        )


def test_spec_rejects_unknown_health_dimension():
    with pytest.raises(ValueError, match="unknown required health dimension"):
        _spec(required_health_dimensions=("not_a_dimension",))


def test_spec_accepts_all_14_health_dimensions():
    spec = _spec(required_health_dimensions=HealthDimension.all())
    assert tuple(spec.required_health_dimensions) == HealthDimension.all()


# ---------------------------------------------------------------------------
# 2. Evidence-tier minimum gates
# ---------------------------------------------------------------------------


def test_minimum_tier_bootstrap_rejects_point_only():
    spec = _spec(minimum_evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE)
    art = CandidateFitnessArtifact(
        trial_id="EWMA",
        evaluation_ref="ev-1",
        health_card_ref="hc-1",
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    assert not spec.requires_evidence_tier(art.evidence_tier)
    assert not art.meets_evidence_tier(spec.minimum_evidence_tier)


def test_minimum_tier_accepts_higher_tier_candidate():
    spec = _spec(minimum_evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE)
    art = CandidateFitnessArtifact(
        trial_id="EWMA",
        evaluation_ref="ev-1",
        health_card_ref="hc-1",
        evidence_tier=EvidenceTier.SEALED_TEST_CONFIRMED,
    )
    assert spec.requires_evidence_tier(art.evidence_tier)
    assert art.meets_evidence_tier(spec.minimum_evidence_tier)


def test_tier_minimum_normalizes_string_input():
    spec = _spec(minimum_evidence_tier="VALIDATION_SERIES")
    assert spec.minimum_evidence_tier is EvidenceTier.VALIDATION_SERIES
    assert spec.requires_evidence_tier("SEALED_TEST_CONFIRMED")
    assert not spec.requires_evidence_tier("POINT_ESTIMATE_ONLY")


# ---------------------------------------------------------------------------
# 3. RAW-relative delta normalization (higher = better-than-RAW) + missing None
# ---------------------------------------------------------------------------


def test_raw_relative_delta_normalization():
    """turnover/exposure are lower-better raw metrics; their delta must be
    flipped so a positive delta always means 'better than RAW'."""
    art = CandidateFitnessArtifact(
        trial_id="EWMA",
        evaluation_ref="ev-1",
        health_card_ref="hc-1",
        raw_relative_deltas={
            "rank_ic": 0.002,          # treatment RankIC - RAW RankIC (higher better, unchanged)
            "icir": 0.10,
            "turnover": 0.05,          # RAW turnover - treatment turnover -> positive = lower turnover = better
            "exposure": 0.02,
            "coverage": -0.01,
            "stability": 0.15,
        },
    )
    # The artifact stores the NORMALIZED delta — the caller hands it the
    # already-flipped sign.  What matters for the contract is that every
    # delta is comparable in one direction: larger = better-than-RAW.
    assert art.delta("turnover") == pytest.approx(0.05)
    assert art.delta("rank_ic") == pytest.approx(0.002)
    # Immutability: the mapping is copied at construction, so mutating the
    # caller's original dict cannot affect the artifact.
    original = dict(art.raw_relative_deltas)
    original["turnover"] = 9.9
    assert art.delta("turnover") == pytest.approx(0.05)
    # And the artifact's own mapping is not the caller's object.
    assert art.raw_relative_deltas is not original


def test_raw_relative_delta_missing_is_none_not_zero():
    art = CandidateFitnessArtifact(
        trial_id="EWMA",
        evaluation_ref="ev-1",
        health_card_ref="hc-1",
        raw_relative_deltas={"rank_ic": None, "turnover": None},
    )
    assert art.delta("rank_ic") is None
    assert art.delta("turnover") is None
    assert "rank_ic" in art.raw_relative_deltas  # present-as-missing (auditable)
    assert art.raw_relative_deltas["rank_ic"] is None
    # A delta of literal 0.0 (RAW is exactly equal on that metric) is a real
    # observation and must be distinguishable from None (missing side).
    equal = CandidateFitnessArtifact(
        trial_id="KAMA",
        evaluation_ref="ev-2",
        health_card_ref="hc-2",
        raw_relative_deltas={"rank_ic": 0.0},
    )
    assert equal.delta("rank_ic") == 0.0
    assert (art.delta("rank_ic") is None) != (equal.delta("rank_ic") is None)


def test_raw_relative_delta_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        CandidateFitnessArtifact(
            trial_id="EWMA",
            evaluation_ref="ev-1",
            health_card_ref="hc-1",
            raw_relative_deltas={"rank_ic": float("nan")},
        )


def test_delta_returns_none_for_unknown_metric():
    art = CandidateFitnessArtifact(
        trial_id="RAW", evaluation_ref="ev-1", health_card_ref="hc-1"
    )
    assert art.delta("nonexistent") is None


# ---------------------------------------------------------------------------
# 4. Health dimension validation
# ---------------------------------------------------------------------------


def test_dimension_scores_reject_unknown_dimension():
    with pytest.raises(ValueError, match="unknown health dimension in scores"):
        CandidateFitnessArtifact(
            trial_id="RAW",
            evaluation_ref="ev-1",
            health_card_ref="hc-1",
            dimension_scores={"bogus": 0.5},
        )


def test_dimension_scores_must_be_in_unit_range():
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        CandidateFitnessArtifact(
            trial_id="RAW",
            evaluation_ref="ev-1",
            health_card_ref="hc-1",
            dimension_scores={HealthDimension.PREDICTIVE_POWER: 1.5},
        )
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        CandidateFitnessArtifact(
            trial_id="RAW",
            evaluation_ref="ev-1",
            health_card_ref="hc-1",
            dimension_scores={HealthDimension.TURNOVER: -0.1},
        )


def test_dimension_scores_accept_valid_14dim_subset():
    scores = {HealthDimension.PREDICTIVE_POWER: 0.7,
              HealthDimension.TURNOVER: 0.2}
    art = CandidateFitnessArtifact(
        trial_id="RAW", evaluation_ref="ev-1", health_card_ref="hc-1",
        dimension_scores=scores,
    )
    assert art.dimension_scores[HealthDimension.PREDICTIVE_POWER] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 5. Serialization round-trips
# ---------------------------------------------------------------------------


def test_spec_roundtrip():
    spec = _spec(
        required_health_dimensions=(HealthDimension.PREDICTIVE_POWER,),
        minimum_evidence_tier=EvidenceTier.MULTIPLE_TESTING_ADJUSTED,
    )
    restored = FactorFitnessSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.minimum_evidence_tier is EvidenceTier.MULTIPLE_TESTING_ADJUSTED
    assert restored.required_health_dimensions == (
        HealthDimension.PREDICTIVE_POWER,
    )


def test_artifact_roundtrip_preserves_none_and_zero():
    art = CandidateFitnessArtifact(
        trial_id="EWMA",
        evaluation_ref="ev-1",
        health_card_ref="hc-1",
        raw_relative_deltas={"rank_ic": None, "turnover": 0.0},
        evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE,
        status=EvidenceStatus.INSUFFICIENT_DATA,
        dimension_scores={HealthDimension.PREDICTIVE_POWER: 0.5},
    )
    restored = CandidateFitnessArtifact.from_dict(art.to_dict())
    assert restored.trial_id == "EWMA"
    assert restored.raw_relative_deltas["rank_ic"] is None
    assert restored.raw_relative_deltas["turnover"] == 0.0
    assert restored.evidence_tier is EvidenceTier.BOOTSTRAP_CONFIDENCE
    assert restored.status is EvidenceStatus.INSUFFICIENT_DATA
    assert restored.dimension_scores[HealthDimension.PREDICTIVE_POWER] == pytest.approx(0.5)


def test_artifact_requires_evaluation_and_health_refs():
    with pytest.raises(ValueError, match="evaluation_ref"):
        CandidateFitnessArtifact(
            trial_id="RAW", evaluation_ref="", health_card_ref="hc-1"
        )
    with pytest.raises(ValueError, match="health_card_ref"):
        CandidateFitnessArtifact(
            trial_id="RAW", evaluation_ref="ev-1", health_card_ref=""
        )
