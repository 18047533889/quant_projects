# -*- coding: utf-8 -*-
"""DLIB-FP production conventions locked in as tests.

Covers:
- DLIB-FP-002: single-transform-authority — every eligibility proposal resolves
  in the canonical TransformRegistry; no dangling transform id.
- DLIB-FP-003 / DLIB-FP-026: deep immutability (hash-safe) of lineage steps,
  recipe, metadata enrichments, policy presets, search spaces.
- DLIB-FP-004: lineage with an unknown semantic ids -> UNKNOWN status (fail-safe).
- DLIB-FP-006: rich profile descriptive fields participate in content hash.
- DLIB-FP-009: canonical top-level public API.
- DLIB-FP-014: every advertised treatment id is executable in the registry.
- DLIB-FP-015: registry is self-describing (semantic surface on metadata).
- DLIB-FP-017: recipe content hash changes when step ordering changes.
- DLIB-FP-020/021: neutralization has a deep-immutable audit artifact; the
  canonical kernel is ols_neutralize and industry/size/dual aliases bind to it.
- DLIB-FP-023: family rule registry is the single authority for allowed
  treatment families.
- DLIB-FP-024: RAW always present.
- DLIB-FP-025: event_decay and freshness_aware_fill are production-causal
  (one-sided) and registered.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import pytest

import factor_preprocess
from factor_preprocess.registry.transforms import (
    get_default_registry,
    ALL_FAMILY_TAGS,
)
from factor_preprocess.registry.policies import TransformStep, PolicyPreset, PolicyLevel
from factor_preprocess.contracts.treatment_lineage import (
    TransformStage,
    TransformSemanticID,
    TransformStep as LineageStep,
    TransformLineage,
    ExistingTreatmentSignature,
    ExistingTreatmentStatus,
    build_signature_from_lineage,
)
from factor_preprocess.contracts.treatment_recipe import (
    TreatmentRecipe,
    RecipeStep,
    FitBoundary,
)
from factor_preprocess.contracts.factor_profile import FactorProfileArtifact
from factor_preprocess.neutralization.diagnostics_artifact import NeutralizationDiagnostics
from factor_preprocess.neutralization.spec import NeutralizationSpec
from factor_preprocess.eligibility.engine import (
    TreatmentEligibilityEngine,
    PRICE_VOLUME,
    FUNDAMENTAL,
    EVENT,
    BINARY,
    RAW_SEMANTIC_ID,
)
from factor_preprocess.eligibility.rules import (
    TreatmentFamily,
    FactorFamily,
    FamilyRule,
    EligibilityRuleRegistry,
    create_default_eligibility_rules,
    get_default_eligibility_rules,
)
from factor_preprocess.transforms.treatment_variants import freshness_aware_fill
from factor_preprocess.transforms.event_decay import event_decay


# ---------------------------------------------------------------------------
# DLIB-FP-009: canonical top-level public API
# ---------------------------------------------------------------------------
def test_canonical_top_level_api_exposes_the_library_contracts():
    # The primary public API is the registry-based, deep-immutable contracts.
    for name in [
        "TreatmentRecipe", "RecipeStep", "FitBoundary", "RecipeSchemaVersion",
        "TransformStage", "TransformSemanticID", "TransformLineage",
        "ExistingTreatmentSignature", "ExistingTreatmentStatus",
        "build_signature_from_lineage",
        "PolicyPreset", "PolicyRegistry", "PolicyLevel", "TransformStep",
        "create_default_policies", "get_default_policy_registry",
        "TransformRegistry", "TransformMetadata", "TransformCategory",
        "create_default_registry", "get_default_registry",
        "FactorProfileArtifact",
        "NeutralizationDiagnostics", "NeutralizationSpec",
        "FeatureBundle", "FittedState",
    ]:
        assert hasattr(factor_preprocess, name), f"{name} must be in top-level API"

    # Deprecated compatibility view is still importable by name.
    assert hasattr(factor_preprocess, "PreprocessingPolicy")
    assert hasattr(factor_preprocess, "TransformSpec")


# ---------------------------------------------------------------------------
# DLIB-FP-002 / FP-014: single-transform-authority, no dangling proposals
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def engine():
    return TreatmentEligibilityEngine()


def _profile(family, **tb):
    return FactorProfileArtifact(
        factor_id="f_probe",
        factor_version="1.0.0",
        semantic_family=family,
        source_type="price",
        update_frequency="daily",
        natural_horizon=20,
        time_behavior=tb,
    )


def test_every_eligibility_proposal_resolves_in_canonical_registry(engine):
    registry = get_default_registry()
    for family in (PRICE_VOLUME, FUNDAMENTAL, EVENT, BINARY):
        for tb in ({}, {"raw_turnover": 0.03}):
            space = engine.build_search_space(_profile(family, **tb))
            for name in space.transform_names():
                if name == RAW_SEMANTIC_ID:
                    continue
                meta = registry.get(name)
                assert meta is not None, (
                    f"eligibility proposed dangling transform {name!r} "
                    f"(family={family})"
                )
                assert meta.semantic_id, (
                    f"resolved transform {name!r} lacks semantic_id; "
                    f"the eligibility layer must query self-describing metadata"
                )


def test_raw_always_present(engine):
    for family in (PRICE_VOLUME, FUNDAMENTAL, EVENT, BINARY, "PRICE_VOLUME"):
        space = engine.build_search_space(_profile(family))
        assert RAW_SEMANTIC_ID in space.allowed_transform_ids


# ---------------------------------------------------------------------------
# DLIB-FP-023: family rule registry is the single semantic authority
# ---------------------------------------------------------------------------
def test_family_rule_registry_single_authority():
    rules = get_default_eligibility_rules()
    # Fundamental does NOT allow short temporal smoothing.
    allowed = set(rules.allowed_families(FactorFamily.FUNDAMENTAL.value))
    assert TreatmentFamily.SMOOTHING not in allowed
    assert TreatmentFamily.FRESHNESS_FILL in allowed

    pv = set(rules.allowed_families(FactorFamily.PRICE_VOLUME.value))
    assert TreatmentFamily.SMOOTHING in pv

    ev = set(rules.allowed_families(FactorFamily.EVENT.value))
    assert TreatmentFamily.EVENT_DECAY in ev
    assert TreatmentFamily.SMOOTHING not in ev


def test_rule_registry_returns_isolated_tuples():
    rules = create_default_eligibility_rules()
    a = rules.allowed_families("PRICE_VOLUME")
    assert isinstance(a, tuple)
    # Rule can be mutated via a new registration; constants remain stable.
    assert rules.policy_version_for("PRICE_VOLUME") == "1.0.0"


# ---------------------------------------------------------------------------
# DLIB-FP-004: lineage with unknown semantic ids -> UNKNOWN / fail-safe
# ---------------------------------------------------------------------------
def build_lineage_with_unknown():
    known = LineageStep(semantic_id="CS_RANK:pct", stage=TransformStage.POST_NEUTRALIZATION, name="cs_rank")
    unknown = LineageStep(semantic_id="ALIEN:jedi", stage=TransformStage.PRE_NEUTRALIZATION, name="mystery_op")
    return TransformLineage((known, unknown))


def test_unknown_semantic_in_lineage_is_fail_safe():
    lineage = build_lineage_with_unknown()
    sig = build_signature_from_lineage(lineage)
    assert sig.is_unknown_or_incomplete is True
    assert sig.status == ExistingTreatmentStatus.UNKNOWN
    # The known part is still detected correctly.
    assert sig.cs_rank is True


def test_known_lineage_yields_known_signature():
    known = LineageStep(semantic_id="CS_RANK:pct", stage=TransformStage.POST_NEUTRALIZATION, name="cs_rank")
    lineage = TransformLineage((known,))
    sig = build_signature_from_lineage(lineage)
    assert sig.status == ExistingTreatmentStatus.KNOWN
    assert not sig.is_unknown_or_incomplete


# ---------------------------------------------------------------------------
# DLIB-FP-003 / FP-026: deep immutability contracts
# ---------------------------------------------------------------------------
def test_lineage_step_parameters_deep_frozen():
    step = LineageStep(
        semantic_id="WINSOR:cs", stage=TransformStage.OUTLIER, name="cs_winsor",
        parameters={"lower": 0.01, "nested": {"a": [1, 2]}},
    )
    with pytest.raises(TypeError):
        step.parameters["lower"] = 0.5  # FrozenDict / Mapping is immutable
    # Deeply immutable and hash-safe (frozen dataclass computes __hash__).
    assert hash(step)
    # content hashing is deterministic (dataclass hash equality)
    second = LineageStep(
        semantic_id="WINSOR:cs", stage=TransformStage.OUTLIER, name="cs_winsor",
        parameters={"lower": 0.01, "nested": {"a": [1, 2]}},
    )
    assert hash(step) == hash(second)


def test_treatment_recipe_is_deep_immutable_and_content_addressed():
    step = RecipeStep(
        step_id="s1", semantic_transform_id="WINSOR:cs",
        implementation_ref="cs_winsor@v1", stage="outlier",
    )
    recipe = TreatmentRecipe(
        recipe_id="r1",
        source_factor_definition_ref="fac.def.1",
        source_factor_value_ref="fac.val.1",
        ordered_steps=(step,),
    )
    assert recipe.content_hash
    with pytest.raises(TypeError):
        recipe.ordered_steps[0].parameters["x"] = 1
    assert hash(recipe)
    # strict fail-closed on caller-supplied hash mismatch:
    import factor_preprocess.errors as errs
    with pytest.raises(errs.InvalidContractError):
        TreatmentRecipe(
            recipe_id="r2",
            source_factor_definition_ref="a",
            source_factor_value_ref="b",
            ordered_steps=(RecipeStep(
                step_id="s", semantic_transform_id="RANK:pct",
                implementation_ref="cs_rank", stage="representation",
            ),),
            content_hash="0" * 64,
        )


def test_recipe_reordering_changes_identity():
    a = RecipeStep(step_id="s1", semantic_transform_id="WINSOR:cs", implementation_ref="cs_winsor", stage="outlier")
    b = RecipeStep(step_id="s2", semantic_transform_id="NEUTRAL:ols", implementation_ref="ols_neutralize", stage="neutralization")
    r1 = TreatmentRecipe(recipe_id="r", source_factor_definition_ref="d", source_factor_value_ref="v", ordered_steps=(a, b))
    r2 = TreatmentRecipe(recipe_id="r", source_factor_definition_ref="d", source_factor_value_ref="v", ordered_steps=(b, a))
    assert r1.content_hash != r2.content_hash


def test_policy_preset_steps_deep_frozen():
    step = TransformStep(name="cs_winsor", parameters={"lower": 0.01})
    with pytest.raises(TypeError):
        step.parameters["lower"] = 0.9
    preset = PolicyPreset(name="p", description="d", level=PolicyLevel.RESEARCH, steps=[step], causal_safe=True)
    assert hash(step)


def test_search_space_deep_frozen():
    from factor_preprocess.eligibility.engine import TreatmentSearchSpace
    space = TreatmentSearchSpace(
        factor_id="f", allowed_transform_ids={"cs_rank": {"pct": (1.0, 1.0)}},
    )
    with pytest.raises(TypeError):
        space.allowed_transform_ids["cs_rank"]["pct"] = (0.0, 0.5)
    assert space.allows("cs_rank")


# ---------------------------------------------------------------------------
# DLIB-FP-006: descriptive profile fields are content-hashed
# ---------------------------------------------------------------------------
def test_rich_profile_hash_and_descriptive_load():
    base = _profile(PRICE_VOLUME)
    assert base.content_hash
    rich = FactorProfileArtifact(
        factor_id="f_probe", factor_version="1.0.0", semantic_family=PRICE_VOLUME,
        source_type="price", update_frequency="daily", natural_horizon=20,
        raw_turnover=0.42, half_life=7.0, sparsity=0.3, missingness=0.05,
        event_semantics="earnings",
    )
    assert rich.content_hash
    assert rich.half_life == 7.0
    assert rich.event_semantics == "earnings"
    # descriptive field change must change content hash
    rich2 = FactorProfileArtifact(
        factor_id="f_probe", factor_version="1.0.0", semantic_family=PRICE_VOLUME,
        source_type="price", update_frequency="daily", natural_horizon=20,
        raw_turnover=0.43, half_life=7.0,
    )
    assert rich.content_hash != rich2.content_hash


# ---------------------------------------------------------------------------
# DLIB-FP-020 / FP-021: neutralization audit artifact + canonical kernel
# ---------------------------------------------------------------------------
def test_neutralization_diagnostics_artifact_deep_immutable():
    diag = NeutralizationDiagnostics(
        neutralization_ref="neutral.1",
        effective_n=100, rank=3, condition_number=4.2,
        r_squared=0.15, residual_variance=0.9,
        missing_exposure_count=2, industry_coverage=0.95,
        solver="lstsq", rank_deficient_resolution=None,
    )
    assert diag.content_hash
    with pytest.raises(Exception):
        diag.r_squared = 0.0  # frozen
    # content mismatch fail-closed
    with pytest.raises(Exception):
        NeutralizationDiagnostics(neutralization_ref="n2", r_squared=0.1, content_hash="0" * 64)


def test_neutralization_spec_is_stable_and_production_admissible():
    spec = NeutralizationSpec(method="ols", exposure_set="industry", industry_schema="SW_L1")
    assert spec.is_production_admissible
    assert spec.kernel_name() == "ols_neutralize"


def test_industry_size_dual_aliases_bind_to_ols_kernel():
    registry = get_default_registry()
    for name in ["industry_neutral", "size_neutral", "dual_neutral"]:
        meta = registry.get(name)
        assert meta.func is registry.get("ols_neutralize").func
        assert meta.semantic_id


# ---------------------------------------------------------------------------
# DLIB-FP-014: event_decay / freshness_aware_fill are registered + executable
# ---------------------------------------------------------------------------
def test_treatment_variants_registered_and_executable():
    registry = get_default_registry()
    for name in ["event_decay", "freshness_aware_fill"]:
        meta = registry.get(name)
        assert meta is not None
        assert meta.causal_safe is True
        assert meta.semantic_id

    # Executable end-to-end.
    n = 20
    df = pd.DataFrame({
        "asset_id": ["a"] * n,
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "value": np.concatenate([np.random.RandomState(0).randn(10), np.full(10, np.nan)]),
    })
    out = event_decay(df, halflife=2.0)
    assert out.shape == (n,)
    out2 = freshness_aware_fill(df, max_lag=5)
    assert out2.shape == (n,)


def test_all_family_tags_constant():
    for f in ("PRICE_VOLUME", "HIGH_TURNOVER", "FUNDAMENTAL", "SPARSE_UPDATE", "EVENT", "BINARY", "DISCRETE"):
        assert f in ALL_FAMILY_TAGS