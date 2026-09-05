"""
R61-FI-042 preset semantic-order certification (plan §27 F-TDD-001).

Every production preset must map to the certified semantic order:

    A Missingness -> B Outlier -> C Temporal Stabilization
    -> D Neutralization -> E Representation/Scaling

Rules locked here:
1. A production preset's ordered transforms (mapped through registry metadata
   to canonical Stages) must be a certified template OR a canonical-order
   subsequence — arbitrary permutations (E before D, D before B, ...) FAIL.
2. ``production_full`` previously placed ``ols_neutralize`` AFTER
   ``cs_rank``/``cs_zscore`` (D after E) — this file asserts the corrected
   Option-A order: missingness -> outlier -> temporal (ewma) -> risk scaling
   (volatility_scale, C segment) -> neutralization -> representation.
3. A single representation output is applied last.  ``cs_zscore`` after
   ``cs_rank`` in the same axis would be a double-standardization the lineage
   duplicate guard (plan §26 F4, R61-FI-043) rejects, so ``production_full``
   ends with the certified single representation ``cs_rank``.
4. Certified exceptions are EXPLICIT named recipes only
   (``CERTIFIED_PRODUCTION_RECIPES``).  Arbitrary permutations must not be
   certified even when each individual stage is legal.
"""
from __future__ import annotations

import pytest

from factor_preprocess.grammar.search_grammar import (
    Stage,
    STAGE_ORDER,
    CERTIFIED_TEMPLATES,
    CERTIFIED_PRODUCTION_RECIPES,
    stage_of_transform_name,
    certify_production_presets,
    validate_stage_order,
    is_certified_template,
)
from factor_preprocess.registry.policies import (
    PolicyLevel,
    get_default_policy_registry,
)
from factor_preprocess.registry.transforms import get_default_registry

_CANONICAL_ORDER = tuple(STAGE_ORDER)


def _stages_of_preset(preset, registry=None):
    if registry is None:
        registry = get_default_registry()
    return tuple(
        stage_of_transform_name(step.name, registry=registry)
        for step in preset.steps
    )


def _collapsed(stages):
    out = []
    for s in stages:
        if out and out[-1] == s:
            continue
        out.append(s)
    return tuple(out)


# ---------------------------------------------------------------------------
# F-TDD-001: every production preset maps to a certified semantic order
# ---------------------------------------------------------------------------

def test_all_production_presets_certify():
    policy_registry = get_default_policy_registry()
    production = [
        p for p in policy_registry.all_policies()
        if p.level == PolicyLevel.PRODUCTION
    ]
    assert production, "expected at least one production preset"
    results = certify_production_presets(policy_registry)
    assert results, "certify_production_presets returned nothing"
    for preset_name, valid, detail in results:
        assert valid, (
            f"production preset {preset_name!r} does NOT map to a certified "
            f"semantic order: {detail}"
        )


def test_production_full_is_canonical_order_after_fix():
    """F-TDD-001 regression: production_full must be canonical A->E now."""
    preset = get_default_policy_registry().get("production_full")
    assert preset is not None
    stages = _stages_of_preset(preset)
    collapsed = _collapsed(stages)
    # missingness(A) -> outlier(B) -> temporal C(ewma, vol-scale) ->
    # neutralization(D) -> representation(E)
    assert collapsed[0] == Stage.MISSINGNESS
    assert Stage.OUTLIER in collapsed
    assert Stage.NEUTRALIZATION in collapsed
    assert collapsed[-1] == Stage.REPRESENTATION
    # neutralization must come BEFORE representation (no D-after-E)
    assert collapsed.index(Stage.NEUTRALIZATION) < collapsed.index(Stage.REPRESENTATION)
    valid, errors = validate_stage_order(collapsed)
    assert valid, errors


def test_production_full_step_order():
    preset = get_default_policy_registry().get("production_full")
    names = [step.name for step in preset.steps]
    assert names == [
        "forward_fill",
        "missing_indicator",
        "cs_winsor",
        "ewma",
        "volatility_scale",
        "ols_neutralize",
        "cs_rank",
    ]


def test_production_full_representation_is_single_rank_not_double_standardized():
    """cs_rank and cs_zscore are both E; the preset must apply ONE of them."""
    preset = get_default_policy_registry().get("production_full")
    rep_names = [
        step.name for step in preset.steps
        if stage_of_transform_name(step.name) == Stage.REPRESENTATION
    ]
    assert len(rep_names) == 1, f"expected single representation step, got {rep_names}"


# ---------------------------------------------------------------------------
# Arbitrary permutations are rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_order", [
    # D (neutralization) before B (outlier) — B must come before D
    (Stage.NEUTRALIZATION, Stage.OUTLIER),
    # E (representation) before D (neutralization) — D must precede E
    (Stage.REPRESENTATION, Stage.NEUTRALIZATION),
    # E before B
    (Stage.REPRESENTATION, Stage.OUTLIER),
    # C before A
    (Stage.TEMPORAL, Stage.MISSINGNESS),
    # full reverse
    (Stage.REPRESENTATION, Stage.NEUTRALIZATION, Stage.TEMPORAL,
     Stage.OUTLIER, Stage.MISSINGNESS),
])
def test_arbitrary_permutations_fail(bad_order):
    assert not is_certified_template(bad_order)
    valid, errors = validate_stage_order(bad_order)
    assert not valid
    assert errors


@pytest.mark.parametrize("recipe_stages", [
    (Stage.TEMPORAL, Stage.NEUTRALIZATION, Stage.REPRESENTATION),   # SMOOTH_NEUTRALIZE_RANK
    (Stage.NEUTRALIZATION, Stage.TEMPORAL, Stage.REPRESENTATION),   # NEUTRALIZE_SMOOTH_RANK
    (Stage.OUTLIER, Stage.NEUTRALIZATION, Stage.REPRESENTATION),    # WINSOR_NEUTRALIZE_ZSCORE
    (),
])
def test_certified_templates_still_pass(recipe_stages):
    assert is_certified_template(recipe_stages)
    valid, _ = validate_stage_order(recipe_stages)
    assert valid


def test_certified_exception_allow_list_is_explicit():
    """Only the five named recipes are certified; no wildcard permutations."""
    expected = {
        "SMOOTH_NEUTRALIZE_RANK",
        "NEUTRALIZE_SMOOTH_RANK",
        "WINSOR_NEUTRALIZE_ZSCORE",
        "RANK_THEN_NEUTRALIZE",
        "RAW",
    }
    assert set(CERTIFIED_PRODUCTION_RECIPES) == expected
    # every certified template is inside the allow-list
    for t in CERTIFIED_TEMPLATES:
        assert t.name in expected


# ---------------------------------------------------------------------------
# Canonical order helper / stage mapping
# ---------------------------------------------------------------------------

def test_stage_mapping_scaling_is_temporal_not_representation():
    """volatility_scale is risk scaling (C), never representation (E)."""
    assert stage_of_transform_name("volatility_scale") == Stage.TEMPORAL
    assert stage_of_transform_name("volatility_scale_returns") == Stage.TEMPORAL
    assert stage_of_transform_name("cs_rank") == Stage.REPRESENTATION
    assert stage_of_transform_name("cs_zscore") == Stage.REPRESENTATION
    assert stage_of_transform_name("ols_neutralize") == Stage.NEUTRALIZATION


def test_stage_of_unknown_transform_fails_closed():
    with pytest.raises(ValueError):
        stage_of_transform_name("no_such_transform")


# ---------------------------------------------------------------------------
# Every certified recipe name is reachable from the exception allow-list
# ---------------------------------------------------------------------------

def test_certified_recipes_names_cover_grammar_templates():
    names = {t.name for t in CERTIFIED_TEMPLATES}
    # RAW and the three non-default certified recipes are the allow-list set.
    assert names.issubset(set(CERTIFIED_PRODUCTION_RECIPES))
