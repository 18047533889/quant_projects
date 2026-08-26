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


# --- recursive deep-freeze ---------------------------------------------------


def test_nested_recipe_is_deep_frozen_after_construction():
    nested = {"winsor": {"q": [0.01, 0.99], "name": "clip"}, "outer": ["a", 1]}
    a = _make_artifact(winner_recipe=nested)

    # mutate the original nested structures after construction
    nested["winsor"]["q"].append(0.5)
    nested["winsor"]["name"] = "MUTATED"
    nested["outer"].append("b")
    nested["added"] = "x"

    stored = a.winner_recipe
    # frozen top level
    assert stored["winsor"]["q"] == (0.01, 0.99)
    assert stored["winsor"]["name"] == "clip"
    assert stored["outer"] == ("a", 1)
    assert "added" not in stored
    # nested inner dict itself is immutable too
    with pytest.raises(TypeError):
        stored["winsor"]["name"] = "boom"  # type: ignore[index]


def test_nested_mutation_does_not_change_content_hash():
    nested = {"winsor": {"q": [0.01, 0.99]}}
    a = _make_artifact(winner_recipe=nested)
    recomputed = _make_artifact(
        winner_recipe=nested,
        content_hash=a.content_hash,
    )
    assert recomputed.content_hash == a.content_hash
    # recompute against the exact current stored hash still matches
    from factor_assets.contracts.treatment_selection import _content_hash

    assert a.content_hash == _content_hash(
        a.factor_id,
        a.factor_version,
        a.raw_baseline_evidence_ref,
        a.factor_profile_ref,
        a.eligibility_policy_ref,
        a.search_space_ref,
        a.all_trial_refs,
        a.pareto_candidate_refs,
        a.winner_recipe,
        a.winner_policy_identity,
        a.absolute_metric_refs,
        a.delta_metric_refs,
        a.dimension_scores,
        a.hard_gate_results,
        a.soft_floor_results,
        a.robustness_evidence,
        a.complexity_score,
        a.snapshot_ref,
        a.universe_ref,
        a.split_ref,
    )


# --- canonical structural hashing --------------------------------------------


def test_structural_equality_same_hash_regardless_of_insertion_order():
    recipe_a = {"winsor": {"q": [0.01, 0.99], "name": "clip"}, "pre": "zscore"}
    recipe_b = {"pre": "zscore", "winsor": {"name": "clip", "q": [0.01, 0.99]}}
    a = _make_artifact(winner_recipe=recipe_a)
    b = _make_artifact(winner_recipe=recipe_b)
    assert a.content_hash == b.content_hash


def test_distinct_types_produce_different_hashes():
    int_a = _make_artifact(winner_recipe={"winsorize": 1})
    float_a = _make_artifact(winner_recipe={"winsorize": 1.0})
    str_a = _make_artifact(winner_recipe={"winsorize": "1"})
    assert int_a.content_hash != float_a.content_hash
    assert int_a.content_hash != str_a.content_hash
    assert float_a.content_hash != str_a.content_hash


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
