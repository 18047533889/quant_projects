import pandas as pd
import numpy as np
import pytest

from factor_preprocess.adapters.fe_operator import _long_to_wide
from factor_preprocess.contracts.treatment_lineage import (
    TransformLineage, TransformSemanticID, TransformStage, TransformStep,
    derive_output_properties,
)
from factor_preprocess.registry.policies import PolicyLevel, PolicyPreset
from factor_preprocess.registry.transforms import TransformCategory, TransformRegistry, _source_of
from factor_preprocess.registry.transforms import get_default_registry
from factor_preprocess.contracts.feature_bundle import AxisRef, FeatureBundle
from factor_preprocess.contracts.treatment_recipe import TreatmentRecipe, RecipeStep
from factor_preprocess.grammar.search_grammar import certify_production_presets


def test_integer_asset_identity_survives_long_to_wide():
    frame = pd.DataFrame({"date": [1, 1], "asset_id": [1, 2], "value": [3.0, 4.0]})
    result = _long_to_wide(frame, "value", "date", "asset_id")
    assert list(result.columns) == [1, 2]
    assert result.notna().sum().sum() == 2


def test_conflicting_duplicate_primary_key_fails_in_any_order():
    frame = pd.DataFrame({"date": [1, 1], "asset_id": [1, 1], "value": [1.0, 9.0]})
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _long_to_wide(frame, "value", "date", "asset_id")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _long_to_wide(frame.iloc[::-1], "value", "date", "asset_id")


def test_public_execution_binds_forward_fill_max_lag_to_effective_fe_parameter():
    frame = pd.DataFrame({
        "date": [1, 2, 3, 4], "asset_id": [1, 1, 1, 1],
        "value": [1.0, float("nan"), float("nan"), 4.0],
    })
    short = get_default_registry().get_execution("forward_fill")
    long = get_default_registry().get_execution("forward_fill")
    assert short(values=frame, max_lag=1).isna().iloc[2]
    assert long(values=frame, max_lag=2).iloc[2] == 1.0
    assert short.effective_parameters == {"max_periods": 1}


def test_auxiliary_channels_preserve_primary_signal_and_original_mask():
    primary = np.array([[[10.0], [20.0]], [[30.0], [40.0]]])
    validity = np.array([[[True], [False]], [[True], [True]]])
    freshness = np.array([[[1.0], [0.0]], [[0.5], [1.0]]])
    bundle = FeatureBundle.from_primary_with_auxiliary(
        bundle_id="gold26",
        primary_values=primary,
        feature_ids=("alpha",),
        time_axis=AxisRef("time", (1, 2), "int64"),
        asset_axis=AxisRef("asset", ("A", "B"), "str"),
        original_validity_mask=validity,
        freshness_values=freshness,
    )
    np.testing.assert_array_equal(bundle.get_primary_values(), primary)
    np.testing.assert_array_equal(bundle.get_original_validity_mask(), validity)
    np.testing.assert_array_equal(bundle.get_channel_values("freshness"), freshness)
    assert bundle.values.shape == (2, 2, 3)


def test_auxiliary_channel_shapes_and_mask_dtype_fail_closed():
    kwargs = dict(
        bundle_id="bad", primary_values=np.ones((1, 1, 1)), feature_ids=("x",),
        time_axis=AxisRef("time", (1,), "int64"),
        asset_axis=AxisRef("asset", ("A",), "str"),
    )
    with pytest.raises(Exception, match="boolean dtype"):
        FeatureBundle.from_primary_with_auxiliary(original_validity_mask=np.ones((1, 1, 1)), **kwargs)
    with pytest.raises(Exception, match="shape"):
        FeatureBundle.from_primary_with_auxiliary(
            original_validity_mask=np.ones((1, 1, 1), dtype=bool),
            freshness_values=np.ones((2, 1, 1)), **kwargs,
        )


def test_lineage_dedupe_only_removes_adjacent_exact_duplicate():
    z = TransformStep("ZSCORE:cs", TransformStage.REPRESENTATION, "zscore", {"axis": "cs"})
    nonlinear = TransformStep("ABS", TransformStage.REPRESENTATION, "abs", {})
    assert len(TransformLineage((z, z)).dedupe()) == 1
    assert len(TransformLineage((z, nonlinear, z)).dedupe()) == 3


def test_semantic_identity_and_policy_steps_are_immutable():
    semantic = TransformSemanticID("ZSCORE:cs")
    with pytest.raises(Exception):
        semantic.value = "OTHER"


def test_output_properties_track_root_not_historical_occurrence():
    neutral = TransformStep(
        "NEUTRAL:ols", TransformStage.NEUTRALIZATION, "ols_neutralize",
        {"exposure_ids": ("industry", "size")},
    )
    rank = TransformStep("CS_RANK:pct", TransformStage.REPRESENTATION, "rank", {"axis": "cs"})
    assert derive_output_properties(TransformLineage((neutral,))).orthogonal_to == ("industry", "size")
    after_rank = derive_output_properties(TransformLineage((neutral, rank)))
    assert after_rank.ranked is True
    assert after_rank.orthogonal_to == ()


def test_affine_scaling_preserves_neutrality_only_with_same_mask_and_weights():
    neutral = TransformStep("NEUTRAL:ols", TransformStage.NEUTRALIZATION, "ols", {"exposure_ids": ("size",)})
    safe = TransformStep("ZSCORE:cs", TransformStage.REPRESENTATION, "zscore", {"preserves_mask_and_weights": True})
    unsafe = TransformStep("ZSCORE:cs", TransformStage.REPRESENTATION, "zscore", {})
    assert derive_output_properties((neutral, safe)).orthogonal_to == ("size",)
    assert derive_output_properties((neutral, unsafe)).orthogonal_to == ()


def test_raw_tag_cannot_certify_nonempty_illegal_recipe():
    registry = TransformRegistry()
    def op(values): return values
    registry.register("rank", op, TransformCategory.REPRESENTATION, admission="PRODUCTION", causal_safe=True, stage="representation")
    registry.register("neutral", op, TransformCategory.NEUTRALIZATION, admission="PRODUCTION", causal_safe=True, stage="neutralization")
    from factor_preprocess.registry.policies import TransformStep as PolicyStep
    preset = PolicyPreset("bad", "bad", PolicyLevel.PRODUCTION, (PolicyStep("rank"), PolicyStep("neutral")), tags=("RAW",))
    class Policies:
        def all_policies(self): return (preset,)
    assert certify_production_presets(Policies(), registry)[0][1] is False


def test_dynamic_function_fingerprint_is_content_based_and_mutable_closure_rejected():
    assert _source_of(lambda x: x + 1) != _source_of(lambda x: x + 2)
    mutable = []
    def closure(x): return x + len(mutable)
    with pytest.raises(ValueError, match="mutable closure"):
        _source_of(closure)


def test_new_registry_entry_defaults_research_and_runtime_domain_is_enforced():
    registry = TransformRegistry()
    def transform(values, halflife=1): return values
    registry.register(
        "new", transform, TransformCategory.TEMPORAL,
        parameter_domain={"halflife": (1, 5)},
    )
    assert registry.get("new").admission == "RESEARCH_ONLY"
    with pytest.raises(ValueError, match="below"):
        registry.get_execution("new")(values=1, halflife=0)
    assert registry.get_execution("new")(values=1, halflife=2) == 1


def test_recipe_canonical_roundtrip_and_compile_fail_closed():
    recipe = TreatmentRecipe(
        recipe_id="r", source_factor_definition_ref="d", source_factor_value_ref="v",
        ordered_steps=(RecipeStep("s", "FILL:forward", "forward_fill", "missingness", parameters={"max_lag": 2}),),
    )
    rebuilt = TreatmentRecipe.from_canonical_dict(recipe.to_canonical_dict())
    assert rebuilt.content_hash == recipe.content_hash
    assert rebuilt.ordered_steps[0].parameters["max_lag"] == 2
    assert len(rebuilt.compile(get_default_registry())) == 1
    bad = TreatmentRecipe(
        recipe_id="bad", source_factor_definition_ref="d", source_factor_value_ref="v",
        ordered_steps=(RecipeStep("s", "UNKNOWN", "not_registered", "missingness"),),
    )
    with pytest.raises(Exception, match="unknown recipe implementation"):
        bad.compile(get_default_registry())


def test_fitted_recipe_requires_bound_state_ref_at_compile():
    registry = TransformRegistry()
    registry.register("fitop", lambda values: values, TransformCategory.TEMPORAL)
    recipe = TreatmentRecipe(
        recipe_id="fit", source_factor_definition_ref="d", source_factor_value_ref="v",
        ordered_steps=(RecipeStep("s", "FIT", "fitop", "temporal", requires_fit=True, state_ref="state:1"),),
    )
    with pytest.raises(Exception, match="missing fitted state"):
        recipe.compile(registry)
    assert len(recipe.compile(registry, fitted_state_refs=("state:1",))) == 1


def test_public_recipe_execution_uses_single_panel_boundary():
    frame = pd.DataFrame({
        "date": [1, 1, 2, 2], "asset_id": ["A", "B", "A", "B"],
        "value": [1.0, 2.0, np.nan, 4.0],
    })
    recipe = TreatmentRecipe(
        recipe_id="batch", source_factor_definition_ref="d", source_factor_value_ref="v",
        ordered_steps=(
            RecipeStep("fill", "FILL:forward", "forward_fill", "missingness", parameters={"max_lag": 1}),
            RecipeStep("rank", "CS_RANK:pct", "cs_rank", "representation", parameters={"pct": True}),
        ),
    )
    executor = get_default_registry().get_recipe_execution(recipe, allow_research=True)
    result = executor(frame)
    assert result.notna().all()
    assert executor.runtime_stats["recipe_input_load_count"] == 1
    assert executor.runtime_stats["recipe_operator_count"] == 2
