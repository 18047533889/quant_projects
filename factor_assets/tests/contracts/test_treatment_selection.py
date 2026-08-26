"""Tests for the FA treatment-selection / treatment-policy contracts.

Covers:
  - TreatmentSelectionArtifact immutability + derived content_hash
  - to_dict / from_dict round-trip preserving all fields + hash
  - validation (empty factor_id, non-finite dimension_score, empty ref)
  - TreatmentPolicyRef validation (empty policy_id)
"""

from __future__ import annotations

import math

import pytest

from factor_assets.contracts.treatment_policy import TreatmentPolicyRef
from factor_assets.contracts.treatment_selection import TreatmentSelectionArtifact


def _make_artifact(**overrides):
    base = dict(
        factor_id="F01",
        factor_version="v1",
        raw_baseline_evidence_ref="ref:raw_baseline",
        factor_profile_ref="ref:profile",
        eligibility_policy_ref="ref:eligibility",
        search_space_ref="ref:search_space",
        all_trial_refs=("ref:trial_a", "ref:trial_b"),
        pareto_candidate_refs=("ref:p_1", "ref:p_2"),
        winner_recipe={"preprocess": "zscore", "winsorize": 0.01},
        winner_policy_identity="policy:auto-treat/v3",
        absolute_metric_refs={"sharpe": "ref:sharpe", "turnover": "ref:turnover"},
        delta_metric_refs={"delta_sharpe": "ref:delta_sharpe"},
        dimension_scores={"return": 0.8, "robustness": 0.65},
        hard_gate_results={"min_obs": "PASS", "non_nan": "PASS"},
        soft_floor_results={"min_sharpe": 0.2},
        robustness_evidence="ref:robustness",
        complexity_score=0.42,
        snapshot_ref="ref:snapshot",
        universe_ref="ref:universe",
        split_ref="ref:split",
        created_at="2026-08-25T00:00:00+00:00",
    )
    base.update(overrides)
    return TreatmentSelectionArtifact(**base)


# --- immutability + content_hash --------------------------------------------


def test_artifact_immutable_and_content_hash():
    a1 = _make_artifact()
    a2 = _make_artifact()
    assert a1.content_hash
    # identical fields -> identical hash
    assert a1.content_hash == a2.content_hash

    changed = _make_artifact(dimension_scores={"return": 0.99, "robustness": 0.65})
    assert changed.content_hash != a1.content_hash

    changed_recipe = _make_artifact(winner_recipe={"preprocess": "clip"})
    assert changed_recipe.content_hash != a1.content_hash

    # frozen dataclass -> immutable
    with pytest.raises(AttributeError):
        a1.factor_id = "F02"  # type: ignore[misc]


def test_caller_supplied_wrong_content_hash_rejected():
    a = _make_artifact()
    with pytest.raises(ValueError):
        _make_artifact(content_hash="deadbeef")


def test_caller_supplied_correct_content_hash_accepted():
    a = _make_artifact()
    roundtripped = _make_artifact(content_hash=a.content_hash)
    assert roundtripped.content_hash == a.content_hash


# --- serialization round-trip ------------------------------------------------


def test_serialization_roundtrip():
    a = _make_artifact()
    data = a.to_dict()
    restored = TreatmentSelectionArtifact.from_dict(data)
    assert restored == a
    assert restored.content_hash == a.content_hash
    # all semantic fields preserved
    for field in ("factor_id", "factor_version", "raw_baseline_evidence_ref",
                  "factor_profile_ref", "eligibility_policy_ref", "search_space_ref",
                  "winner_recipe", "winner_policy_identity", "snapshot_ref",
                  "universe_ref", "split_ref"):
        assert getattr(restored, field) == getattr(a, field)
    assert restored.all_trial_refs == a.all_trial_refs
    assert restored.pareto_candidate_refs == a.pareto_candidate_refs
    assert restored.absolute_metric_refs == a.absolute_metric_refs
    assert restored.delta_metric_refs == a.delta_metric_refs
    assert restored.dimension_scores == a.dimension_scores
    assert restored.hard_gate_results == a.hard_gate_results
    assert restored.soft_floor_results == a.soft_floor_results
    assert restored.complexity_score == a.complexity_score


# --- validation --------------------------------------------------------------


def test_validation_empty_factor_id_raises():
    with pytest.raises(ValueError):
        _make_artifact(factor_id="")


def test_validation_non_finite_dimension_raises():
    with pytest.raises(ValueError):
        _make_artifact(dimension_scores={"return": float("inf")})
    with pytest.raises(ValueError):
        _make_artifact(dimension_scores={"return": float("nan")})


def test_validation_bool_dimension_raises():
    with pytest.raises(TypeError):
        _make_artifact(dimension_scores={"return": True})


def test_validation_empty_trial_ref_raises():
    with pytest.raises(ValueError):
        _make_artifact(all_trial_refs=("ref:a", ""))
    with pytest.raises(ValueError):
        _make_artifact(pareto_candidate_refs=("", "ref:b"))


def test_validation_empty_gate_key_raises():
    with pytest.raises(ValueError):
        _make_artifact(hard_gate_results={"": "PASS"})


# --- treatment policy ref ----------------------------------------------------


def test_treatment_policy_ref_validation():
    ref = TreatmentPolicyRef(
        policy_id="pp1",
        policy_version="v3",
        implementation_hash="abc123",
    )
    assert ref.policy_id == "pp1"
    assert ref.treatment_selection_ref is None


def test_treatment_policy_ref_empty_id_raises():
    with pytest.raises(ValueError):
        TreatmentPolicyRef(
            policy_id="",
            policy_version="v1",
            implementation_hash="h",
        )


def test_treatment_policy_ref_empty_version_raises():
    with pytest.raises(ValueError):
        TreatmentPolicyRef(
            policy_id="pp1",
            policy_version="",
            implementation_hash="h",
        )
