"""
A股 PIT / unit cross-package contract tests (P0-FP #103).

These tests express the A股 (A-share) treatment contract entirely inside
factor_preprocess — they MUST NOT import factor_optimizer / quant_evaluator /
factor_assets / quant_platform. Other packages (H task) will later round-trip
against the same contract surface.

The contract surface locked here:

1. TreatmentSpecIdentity — derived from the treatment definition ONLY
   (recipe semantic hash: transform names + ordered params + neutralization
   spec identity). Identical spec -> identical spec identity regardless of
   the data snapshot / universe / split it is materialized against.

2. TreatmentMaterializationIdentity — derived from the spec identity PLUS the
   materialization context (data_snapshot_ref / universe_ref / split_ref /
   asof_timestamp / price_basis) and the A股 neutralization binding. A
   materialization identity can never be confused with a spec identity
   (different typed classes).

3. FULL_SAMPLE_RESEARCH runtime reject — a full-sample-research recipe
   materialized against an evaluation-valid split (validation / test /
   production) raises TreatmentMaterializationError instead of silently
   allowing full-sample leakage.

4. Neutralization binding identity — the A股 neutralization spec (industry
   schema sw_l1, size, unit/basis) participates in the materialization
   identity: two materializations differing only in the neutralization binding
   produce different materialization identities.

5. A股 PIT / unit contract — the neutralization binding names the A股
   conventions (SW_L1 industry schema, log_mktcap size proxy) and the
   unit/basis of materialized values is explicit and carried (decimal vs
   basis-point, price basis adj/unadj).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import pytest
from datetime import datetime

import factor_preprocess
from factor_preprocess.contracts.treatment_recipe import (
    TreatmentRecipe,
    RecipeStep,
    FitBoundary,
)
from factor_preprocess.contracts.treatment_spec import (
    PriceBasis,
    ValueUnit,
    MaterializationSplit,
    TreatmentSpecIdentity,
    TreatmentMaterializationIdentity,
    neutralization_spec_identity,
    neutralization_binding,
    ensure_materialization_split_valid,
    materialize_identity,
    spec_to_materialization_identity,
    ASHARE_INDUSTRY_SCHEMA,
    ASHARE_SIZE_DEFINITION,
)
from factor_preprocess.neutralization.spec import (
    NeutralizationSpec,
    NeutralizationMethod,
    ExposureSet,
)
from factor_preprocess.errors import (
    InvalidContractError,
    TreatmentMaterializationError,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _recipe(recipe_id="r1", fit_boundary=FitBoundary.EXPANDING, neutralization_spec=None):
    """A minimal winsor -> rank recipe with optional neutralization spec."""
    return TreatmentRecipe(
        recipe_id=recipe_id,
        source_factor_definition_ref="fac.def.1",
        source_factor_value_ref="fac.val.1",
        ordered_steps=(
            RecipeStep(
                step_id="s1",
                semantic_transform_id="WINSOR:cs",
                implementation_ref="cs_winsor@v1",
                stage="outlier",
                parameters={"lower": 0.01, "upper": 0.99},
            ),
            RecipeStep(
                step_id="s2",
                semantic_transform_id="CS_RANK:pct",
                implementation_ref="cs_rank@v1",
                stage="representation",
                parameters={"pct": True},
            ),
        ),
        fit_boundary=fit_boundary,
        neutralization_spec=neutralization_spec,
    )


def _ashare_spec():
    """Canonical A股 neutralization spec: sw_l1 industry + log_mktcap size."""
    return NeutralizationSpec(
        method="ols",
        exposure_set="industry_size",
        industry_schema="SW_L1",
        size_definition="log_mktcap",
    )


def _materialize(recipe, **overrides):
    """Materialize a recipe with a full A股 context (defaults overridable)."""
    kwargs = dict(
        materialization_ref="mat.1",
        data_snapshot_ref="snap.2026Q3",
        universe_ref="ashare_500",
        split_ref="split.2026-09",
        split=MaterializationSplit.TRAIN,
        neutralization_spec=_ashare_spec(),
        asof_timestamp=datetime(2026, 9, 1),
        price_basis=PriceBasis.ADJ,
        unit=ValueUnit.DECIMAL,
    )
    kwargs.update(overrides)
    return materialize_identity(recipe, **kwargs)


# ===========================================================================
# 1. SPEC vs MATERIALIZATION split — typed classes, no confusion
# ===========================================================================


def test_spec_and_materialization_identity_are_different_types():
    spec = TreatmentSpecIdentity(spec_id="s", recipe_ref="r", semantic_transform_ids=(("a", "b"),))
    mat = TreatmentMaterializationIdentity(
        spec_identity=spec,
        materialization_ref="m",
        data_snapshot_ref="d",
        universe_ref="u",
        split_ref="sp",
        split=MaterializationSplit.TRAIN,
    )
    # Never the same class, never equal by type.
    assert not isinstance(mat, TreatmentSpecIdentity)
    assert not isinstance(spec, TreatmentMaterializationIdentity)
    assert spec.identity != mat.identity


def test_spec_identity_data_independent_same_spec_same_identity():
    """Identical spec -> identical spec identity regardless of data context."""
    recipe = _recipe()
    spec_a = TreatmentSpecIdentity.from_recipe(recipe, _ashare_spec())
    # Same recipe/spec constructed again.
    spec_b = TreatmentSpecIdentity.from_recipe(_recipe(), _ashare_spec())
    assert spec_a.identity == spec_b.identity


def test_materialization_identity_changes_when_data_snapshot_changes():
    """Same spec, different data snapshot -> different materialization identity."""
    recipe = _recipe()
    a = _materialize(recipe, data_snapshot_ref="snap.Q1")
    b = _materialize(recipe, data_snapshot_ref="snap.Q2")
    assert a.identity != b.identity


def test_materialization_identity_changes_when_split_changes():
    recipe = _recipe()
    a = _materialize(recipe, split=MaterializationSplit.TRAIN)
    b = _materialize(recipe, split=MaterializationSplit.FULL_SAMPLE_RESEARCH)
    assert a.identity != b.identity


# ===========================================================================
# 2. FULL_SAMPLE_RESEARCH runtime reject
# ===========================================================================


def test_full_sample_research_recipe_rejected_on_validation_split():
    recipe = _recipe(fit_boundary=FitBoundary.FULL_SAMPLE_RESEARCH)
    with pytest.raises(TreatmentMaterializationError):
        ensure_materialization_split_valid(recipe, MaterializationSplit.VALIDATION)


@pytest.mark.parametrize("split", [
    MaterializationSplit.VALIDATION,
    MaterializationSplit.TEST,
    MaterializationSplit.PRODUCTION,
])
def test_full_sample_research_recipe_rejected_on_evaluation_splits(split):
    recipe = _recipe(fit_boundary=FitBoundary.FULL_SAMPLE_RESEARCH)
    with pytest.raises(TreatmentMaterializationError):
        _materialize(recipe, split=split)


def test_full_sample_research_recipe_allowed_on_train_split():
    recipe = _recipe(fit_boundary=FitBoundary.FULL_SAMPLE_RESEARCH)
    ensure_materialization_split_valid(recipe, MaterializationSplit.TRAIN)
    # Materialization itself is allowed on train / research splits.
    mat = _materialize(recipe, split=MaterializationSplit.TRAIN)
    assert mat.identity


def test_research_only_neutralization_rejected_on_evaluation_split():
    """Research-only neutralization methods cannot leak into validation."""
    recipe = _recipe(neutralization_spec={
        "method": "pca",
        "exposure_set": "industry",
        "industry_schema": "SW_L1",
    })
    with pytest.raises(TreatmentMaterializationError):
        ensure_materialization_split_valid(
            recipe, MaterializationSplit.TEST,
            neutralization_spec=NeutralizationSpec(
                method="pca", exposure_set="industry", industry_schema="SW_L1"
            ),
        )


def test_production_admissible_neutralization_allowed_on_validation():
    recipe = _recipe(neutralization_spec=_ashare_spec())
    mat = _materialize(recipe, split=MaterializationSplit.VALIDATION)
    assert mat.identity


# ===========================================================================
# 3. Neutralization binding participates in the materialization identity
# ===========================================================================


def test_neutralization_binding_changes_materialization_identity():
    """SW_L1 vs SW_L2 binding -> different materialization identities."""
    recipe = _recipe()
    a = _materialize(recipe, industry_schema="SW_L1")
    b = _materialize(recipe, industry_schema="SW_L2")
    assert a.identity != b.identity
    assert a.industry_schema == "SW_L1"
    assert b.industry_schema == "SW_L2"


def test_size_definition_change_changes_materialization_identity():
    recipe = _recipe()
    a = _materialize(recipe, size_definition="log_mktcap")
    b = _materialize(recipe, size_definition="mktcap")
    assert a.identity != b.identity


def test_unit_change_changes_materialization_identity():
    """decimal vs basis_point materialized values are distinct identities."""
    recipe = _recipe()
    a = _materialize(recipe, unit=ValueUnit.DECIMAL)
    b = _materialize(recipe, unit=ValueUnit.BASIS_POINT)
    assert a.identity != b.identity


def test_pit_identity_change_changes_materialization_identity():
    recipe = _recipe()
    a = _materialize(recipe, pit_identity="pit")
    b = _materialize(recipe, pit_identity="as_of")
    assert a.identity != b.identity


def test_identity_without_neutralization_binding_uses_ashare_defaults():
    """Bare ols neutralization still names the A股 conventions."""
    recipe = _recipe()
    mat = _materialize(recipe, neutralization_spec=None)
    assert mat.industry_schema == ASHARE_INDUSTRY_SCHEMA == "SW_L1"
    assert mat.size_definition == ASHARE_SIZE_DEFINITION == "log_mktcap"
    assert mat.unit == ValueUnit.DECIMAL
    assert mat.price_basis == PriceBasis.ADJ
    assert mat.pit_identity.value == "pit"


# ===========================================================================
# 4. A股 PIT / unit contract surface (pure FP)
# ===========================================================================


def test_ashare_convention_constants():
    assert ASHARE_INDUSTRY_SCHEMA == "SW_L1"
    assert ASHARE_SIZE_DEFINITION == "log_mktcap"


def test_neutralization_binding_names_ashare_conventions():
    spec = _ashare_spec()
    binding = neutralization_binding(spec)
    assert binding["industry_schema"] == "SW_L1"
    assert binding["size_definition"] == "log_mktcap"
    assert binding["pit_identity"].value == "pit"


def test_neutralization_spec_identity_changes_with_binding():
    a = neutralization_spec_identity(_ashare_spec())
    b = neutralization_spec_identity(
        NeutralizationSpec(
            method="ols",
            exposure_set="industry_size",
            industry_schema="SW_L1",
            size_definition="mktcap",
        )
    )
    assert a != b


def test_recipe_spec_identity_matches_from_recipe():
    recipe = _recipe(neutralization_spec=_ashare_spec())
    direct = TreatmentSpecIdentity.from_recipe(recipe, _ashare_spec())
    assert recipe.spec_identity.identity == direct.identity
    # recipe_ref ties the spec identity to the content-addressed recipe.
    assert recipe.spec_identity.recipe_ref == recipe.content_hash


def test_price_basis_and_unit_are_carried_on_materialization():
    mat = _materialize(_recipe(), price_basis="unadj", unit="basis_point")
    assert mat.price_basis == PriceBasis.UNADJ
    assert mat.price_basis.value == "unadj"
    assert mat.unit == ValueUnit.BASIS_POINT


def test_materialization_identity_fail_closed_on_mismatch():
    spec = TreatmentSpecIdentity(spec_id="s", recipe_ref="r")
    with pytest.raises(InvalidContractError):
        TreatmentMaterializationIdentity(
            spec_identity=spec,
            materialization_ref="m",
            data_snapshot_ref="d",
            universe_ref="u",
            split_ref="sp",
            split=MaterializationSplit.TRAIN,
            identity="0" * 64,
        )


def test_spec_identity_requires_spec_fields():
    with pytest.raises(InvalidContractError):
        TreatmentSpecIdentity(spec_id="", recipe_ref="r")


def test_materialization_requires_spec_identity_typed():
    with pytest.raises(InvalidContractError):
        TreatmentMaterializationIdentity(
            spec_identity="not-a-spec-identity",  # type confusion fails closed
            materialization_ref="m",
            data_snapshot_ref="d",
            universe_ref="u",
            split_ref="sp",
            split=MaterializationSplit.TRAIN,
        )


def test_unknown_split_string_fails_closed():
    with pytest.raises(InvalidContractError):
        ensure_materialization_split_valid(_recipe(), "not_a_split")


# ===========================================================================
# 5. The public contract is reachable without importing other packages
# ===========================================================================


def test_recipe_materialize_method_returns_typed_materialization_identity():
    """Recipe.materialize() is the one-step materialization entry point."""
    recipe = _recipe()
    mat = recipe.materialize(
        materialization_ref="mat.1",
        data_snapshot_ref="snap.2026Q3",
        universe_ref="ashare_500",
        split_ref="split.2026-09",
        split=MaterializationSplit.TRAIN,
        neutralization_spec=_ashare_spec(),
        asof_timestamp=datetime(2026, 9, 1),
        price_basis=PriceBasis.ADJ,
        unit=ValueUnit.DECIMAL,
    )
    assert isinstance(mat, TreatmentMaterializationIdentity)
    # Same context -> same identity; different context -> different identity.
    again = recipe.materialize(
        materialization_ref="mat.1",
        data_snapshot_ref="snap.2026Q3",
        universe_ref="ashare_500",
        split_ref="split.2026-09",
        split=MaterializationSplit.TRAIN,
        neutralization_spec=_ashare_spec(),
        asof_timestamp=datetime(2026, 9, 1),
        price_basis=PriceBasis.ADJ,
        unit=ValueUnit.DECIMAL,
    )
    assert mat.identity == again.identity
    # Materialization identity can never be confused with a spec identity.
    assert not isinstance(mat, TreatmentSpecIdentity)


def test_recipe_materialize_rejects_full_sample_research_on_validation():
    recipe = _recipe(fit_boundary=FitBoundary.FULL_SAMPLE_RESEARCH)
    with pytest.raises(TreatmentMaterializationError):
        recipe.materialize(
            materialization_ref="m",
            data_snapshot_ref="d",
            universe_ref="u",
            split_ref="sp",
            split=MaterializationSplit.VALIDATION,
        )


def test_no_forbidden_package_imports():
    """The contract module must not import optimizer/evaluator/assets/platform."""
    import sys
    import factor_preprocess.contracts.treatment_spec as ts
    imported = {m.split(".")[0] for m in sys.modules}
    forbidden = {
        "factor_optimizer", "quant_evaluator", "factor_assets", "quant_platform",
    }
    assert not (imported & forbidden), imported & forbidden


def test_top_level_public_api_exposes_identity_contracts():
    for name in [
        "PriceBasis", "ValueUnit", "MaterializationSplit",
        "TreatmentSpecIdentity", "TreatmentMaterializationIdentity",
        "neutralization_spec_identity", "neutralization_binding",
        "ensure_materialization_split_valid", "materialize_identity",
        "spec_to_materialization_identity",
        "ASHARE_INDUSTRY_SCHEMA", "ASHARE_SIZE_DEFINITION",
        "TreatmentMaterializationError",
    ]:
        assert hasattr(factor_preprocess, name), f"{name} must be in top-level API"
