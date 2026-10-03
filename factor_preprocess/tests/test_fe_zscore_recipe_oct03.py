"""End-to-end contract tests for opt-in FE z-score TreatmentRecipe plans."""
from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest


def _decimal_oracle_axis_0_2(values, *, ddof, constant_value):
    """Independent Decimal reference for a 3-D recipe reduction over axes 0,2."""
    source = np.asarray(values, dtype=np.float64)
    result = np.full(source.shape, np.nan, dtype=np.float64)
    with localcontext() as context:
        context.prec = 1200
        for group in range(source.shape[1]):
            block = source[:, group, :]
            output = result[:, group, :]
            non_nan = ~np.isnan(block)
            finite = block[np.isfinite(block)]
            if np.isinf(block).any():
                output[non_nan] = constant_value
                continue
            if (len(finite) < 2 or len(finite) <= ddof
                    or np.min(finite) == np.max(finite)):
                output[non_nan] = constant_value
                continue
            exact = [Decimal.from_float(float(value)) for value in finite]
            mean = sum(exact, Decimal(0)) / Decimal(len(exact))
            squared_sum = sum(((value - mean) ** 2 for value in exact), Decimal(0))
            variance = squared_sum / (Decimal(len(exact)) - Decimal(str(ddof)))
            deviation = variance.sqrt()
            output[np.isfinite(block)] = [
                float((Decimal.from_float(float(value)) - mean) / deviation)
                for value in finite
            ]
    return result


def _default_registry():
    if not hasattr(_default_registry, "value"):
        from factor_preprocess.registry.transforms import create_default_registry
        _default_registry.value = create_default_registry()
    return _default_registry.value


def _recipe(*, parameters=None, steps=None):
    from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe

    registry = _default_registry()
    metadata = registry.get("cs_zscore")
    if steps is None:
        steps = (RecipeStep(
            step_id="normalize",
            semantic_transform_id=metadata.semantic_id,
            implementation_ref="cs_zscore",
            stage=metadata.stage,
            parameters=parameters or {},
        ),)
    recipe = TreatmentRecipe(
        recipe_id="fe-zscore-candidate-test",
        source_factor_definition_ref="test-factor",
        source_factor_value_ref="test-values",
        ordered_steps=steps,
    )
    return recipe, registry


def test_compiled_recipe_executes_fe_grouped_long_and_preserves_spec_identity(monkeypatch):
    import factor_engine.backend.long_stable_zscore as fe_long
    import factor_preprocess.transforms.cross_sectional as fp_transforms
    from factor_preprocess.adapters.fe_zscore_recipe import compile_fe_native_zscore_recipe

    recipe, registry = _recipe(parameters={
        "axis": (0, 2), "ddof": 0.5, "constant_value": -3.0,
    })
    seen = []
    real_fe = fe_long.finite_anchor_centered_zscore_long

    def observed_fe(frame, **kwargs):
        seen.append((frame.height, dict(kwargs)))
        return real_fe(frame, **kwargs)

    def forbidden_fp(*args, **kwargs):
        raise AssertionError("FP-native cs_zscore fallback must not run")

    monkeypatch.setattr(fe_long, "finite_anchor_centered_zscore_long", observed_fe)
    monkeypatch.setattr(fp_transforms, "finite_anchor_centered_zscore", forbidden_fp)
    plan = compile_fe_native_zscore_recipe(
        recipe, registry, max_chunk_cells=8, max_result_bytes=4096
    )

    values = np.array([
        [[1.0, 2.0, np.nan], [3.0, 4.0, 5.0], [5.0, 5.0, 5.0]],
        [[4.0, 8.0, 16.0], [6.0, 7.0, np.inf], [5.0, 5.0, 5.0]],
    ], dtype=np.float64)
    expected = _decimal_oracle_axis_0_2(values, ddof=0.5, constant_value=-3.0)
    values.setflags(write=False)
    original = values.copy()
    result = plan.run(values)
    np.testing.assert_allclose(result, expected, rtol=2e-15, atol=2e-15, equal_nan=True)
    np.testing.assert_array_equal(values, original)
    assert result.shape == values.shape
    assert result.dtype == np.float64
    assert all(kwargs["ddof"] == 0.5 and kwargs["constant_value"] == -3.0 for _, kwargs in seen)
    assert registry.resolve_origin("cs_zscore") == "FP_NATIVE"
    assert seen and all(height <= 8 for height, _ in seen)
    assert plan.spec_identity == recipe.spec_identity
    identity = plan.execution_identity
    assert identity["identity_kind"] == "recipe_execution_candidate"
    assert identity["status"] == "opt_in_candidate_not_production_admitted"
    assert identity["spec_identity"]["identity"] == recipe.spec_identity.identity
    assert identity["configured_bounds"] == {
        "max_chunk_cells": 8, "max_result_bytes": 4096,
    }
    assert identity["fallback"] == "disabled"
    assert identity["identity_scope"].startswith("bounded on-disk source only")
    assert identity["candidate_module_identity"]["sha256"]
    assert identity["steps"][0]["parameters"] == {
        "axis": [0, 2], "ddof": 0.5, "constant_value": -3.0,
        "numeric_policy": "finite_anchor_centered_v2",
    }
    assert identity["digest"]
    assert identity["digest"] != recipe.spec_identity.identity
    identity["configured_bounds"]["max_chunk_cells"] = 99
    assert plan.execution_identity["configured_bounds"]["max_chunk_cells"] == 8


def test_compile_rejects_other_steps_and_unsupported_numeric_policy():
    from factor_preprocess.adapters.fe_zscore_recipe import compile_fe_native_zscore_recipe
    from factor_preprocess.contracts.treatment_recipe import RecipeStep
    from factor_preprocess.errors import InvalidContractError

    recipe, registry = _recipe()
    first = recipe.ordered_steps[0]
    extra = RecipeStep(
        step_id="rank", semantic_transform_id="CS_RANK:pct",
        implementation_ref="cs_rank", stage="representation", parameters={"pct": True},
    )
    other, other_registry = _recipe(steps=(extra,))
    with pytest.raises(InvalidContractError, match="only implementation_ref='cs_zscore'"):
        compile_fe_native_zscore_recipe(other, other_registry)

    legacy, legacy_registry = _recipe(parameters={"numeric_policy": "legacy_numpy_v1"})
    with pytest.raises(ValueError, match="numeric_policy"):
        compile_fe_native_zscore_recipe(legacy, legacy_registry)


def test_compile_rejects_invalid_bounds_and_preserves_default_bound_parameters():
    from factor_preprocess.adapters.fe_zscore_recipe import compile_fe_native_zscore_recipe

    recipe, registry = _recipe()
    with pytest.raises(ValueError, match="max_chunk_cells"):
        compile_fe_native_zscore_recipe(recipe, registry, max_chunk_cells=0)

    plan = compile_fe_native_zscore_recipe(recipe, registry)
    assert dict(plan.steps[0].parameters) == {
        "axis": -1, "ddof": 1, "constant_value": 0.0,
        "numeric_policy": "finite_anchor_centered_v2",
    }
    values = np.array([[1.0, 2.0, 3.0]])
    result = plan.apply(values)
    np.testing.assert_allclose(result, [[-1.0, 0.0, 1.0]])


def test_candidate_result_budget_fails_without_any_numeric_fallback():
    from factor_preprocess.adapters.fe_zscore_recipe import compile_fe_native_zscore_recipe

    recipe, registry = _recipe()
    plan = compile_fe_native_zscore_recipe(
        recipe, registry, max_chunk_cells=3, max_result_bytes=8
    )
    with pytest.raises(MemoryError, match="max_result_bytes"):
        plan.run(np.arange(3, dtype=np.float64))



def test_invalid_semantic_stage_and_fit_bindings_fail_closed():
    from factor_preprocess.adapters.fe_zscore_recipe import compile_fe_native_zscore_recipe
    from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe
    from factor_preprocess.errors import InvalidContractError

    base, registry = _recipe()
    for semantic_id, stage in (
        ("CS_RANK:pct", "representation"),
        ("CROSS_SECTIONAL_ZSCORE:cs", "outlier"),
    ):
        wrong = RecipeStep(
            step_id="normalize", semantic_transform_id=semantic_id,
            implementation_ref="cs_zscore", stage=stage, parameters={},
        )
        recipe = TreatmentRecipe(
            recipe_id="wrong-binding", source_factor_definition_ref="test-factor",
            source_factor_value_ref="test-values", ordered_steps=(wrong,),
        )
        with pytest.raises(InvalidContractError):
            compile_fe_native_zscore_recipe(recipe, registry)

    stateful = RecipeStep(
        step_id="normalize", semantic_transform_id=base.ordered_steps[0].semantic_transform_id,
        implementation_ref="cs_zscore", stage="representation", state_ref="state",
    )
    state_recipe = TreatmentRecipe(
        recipe_id="state-binding", source_factor_definition_ref="test-factor",
        source_factor_value_ref="test-values", ordered_steps=(stateful,),
    )
    with pytest.raises(InvalidContractError, match="fitted state"):
        compile_fe_native_zscore_recipe(state_recipe, registry)

    fitted = RecipeStep(
        step_id="normalize", semantic_transform_id=base.ordered_steps[0].semantic_transform_id,
        implementation_ref="cs_zscore", stage="representation", requires_fit=True,
        state_ref="state",
    )
    fitted_recipe = TreatmentRecipe(
        recipe_id="fit-binding", source_factor_definition_ref="test-factor",
        source_factor_value_ref="test-values", ordered_steps=(fitted,),
    )
    with pytest.raises(InvalidContractError, match="fit requirement disagrees"):
        compile_fe_native_zscore_recipe(fitted_recipe, registry)


def test_invalid_numeric_axis_bounds_and_reduction_group_fail_closed():
    from factor_preprocess.adapters.fe_zscore_recipe import compile_fe_native_zscore_recipe
    from factor_preprocess.errors import InvalidContractError

    recipe, registry = _recipe()
    data_argument, data_argument_registry = _recipe(parameters={"values": True})
    with pytest.raises(InvalidContractError, match="unsupported cs_zscore recipe parameters.*values"):
        compile_fe_native_zscore_recipe(data_argument, data_argument_registry)

    invalid_ddof, invalid_ddof_registry = _recipe(parameters={"ddof": 1.5})
    with pytest.raises(ValueError, match="ddof"):
        compile_fe_native_zscore_recipe(invalid_ddof, invalid_ddof_registry)
    for kwargs in ({"max_chunk_cells": True}, {"max_result_bytes": False}):
        with pytest.raises(ValueError, match="max_chunk_cells|max_result_bytes"):
            compile_fe_native_zscore_recipe(recipe, registry, **kwargs)

    boolean_axis, boolean_registry = _recipe(parameters={"axis": True})
    with pytest.raises(InvalidContractError, match="axis"):
        compile_fe_native_zscore_recipe(boolean_axis, boolean_registry)

    duplicate_axis, duplicate_registry = _recipe(parameters={"axis": (0, 0)})
    with pytest.raises(InvalidContractError, match="duplicate values"):
        compile_fe_native_zscore_recipe(duplicate_axis, duplicate_registry)

    large_group, large_registry = _recipe(parameters={"axis": (0, 1)})
    bounded = compile_fe_native_zscore_recipe(
        large_group, large_registry, max_chunk_cells=5, max_result_bytes=4096
    )
    with pytest.raises(ValueError, match="complete reduction group"):
        bounded.run(np.arange(6.0).reshape(2, 3))


def test_candidate_module_source_identity_is_bounded_and_participates_in_digest(tmp_path):
    from factor_preprocess.adapters.fe_zscore_recipe import (
        _identity_digest, _module_source_identity,
    )

    source = tmp_path / "candidate.py"
    source.write_bytes(b"candidate-v1")
    first = _module_source_identity(source, max_bytes=64)
    source.write_bytes(b"candidate-v2")
    second = _module_source_identity(source, max_bytes=64)
    assert first["sha256"] != second["sha256"]
    assert _identity_digest({"candidate_module_identity": first}) != _identity_digest(
        {"candidate_module_identity": second}
    )
    source.write_bytes(b"x" * 65)
    with pytest.raises(ValueError, match="read limit"):
        _module_source_identity(source, max_bytes=64)
