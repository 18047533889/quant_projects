"""Regression coverage for runtime data arguments vs recipe parameters."""
from __future__ import annotations
import numpy as np

import pytest

from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe
from factor_preprocess.errors import InvalidContractError
from factor_preprocess.registry.transforms import (
    TransformCategory,
    TransformRegistry,
    create_default_registry,
)


def _input_x(x, axis=-1):
    return x


def _input_data(data, lag=1):
    return data


def _input_values_data(values, data=None):
    return values


def _input_values_posonly(values, /, lag=1):
    return values


def _input_varargs_kwargs(*args, **kwargs):
    return args[0] if args else kwargs.get("values")


def _input_kwargs_only(**kwargs):
    return kwargs.get("values")


def _registered_recipe(registry, *, name, func, parameters, suffix="one"):
    semantic_id = "CS_RANK:pct"
    registry.register(
        name,
        func,
        TransformCategory.CROSS_SECTIONAL,
        semantic_id=semantic_id,
        stage="representation",
        parameter_domain={"lag": (1, 3)},
        admission="PRODUCTION",
        causal_safe=True,
    )
    return TreatmentRecipe(
        recipe_id=f"runtime-input-{name}-{suffix}",
        source_factor_definition_ref="test-factor",
        source_factor_value_ref="test-values",
        ordered_steps=(RecipeStep(
            step_id="step",
            semantic_transform_id=semantic_id,
            implementation_ref=name,
            stage="representation",
            parameters=parameters,
        ),),
    )


def test_real_recipe_compile_rejects_default_registry_values_input_binding():
    registry = create_default_registry()
    metadata = registry.get("cs_zscore")
    recipe = TreatmentRecipe(
        recipe_id="runtime-input-values-regression",
        source_factor_definition_ref="test-factor",
        source_factor_value_ref="test-values",
        ordered_steps=(RecipeStep(
            step_id="zscore",
            semantic_transform_id=metadata.semantic_id,
            implementation_ref="cs_zscore",
            stage=metadata.stage,
            parameters={"values": True},
        ),),
    )
    with pytest.raises(InvalidContractError, match="runtime input parameter 'values'"):
        recipe.compile(registry)


@pytest.mark.parametrize(
    "name,func,input_name",
    [
        ("runtime_input_x", _input_x, "x"),
        ("runtime_input_data", _input_data, "data"),
        ("runtime_input_values", _input_values_posonly, "values"),
    ],
)
def test_compile_rejects_first_input_aliases_including_positional_only(name, func, input_name):
    registry = create_default_registry()
    recipe = _registered_recipe(
        registry, name=name, func=func, parameters={input_name: "not-data"}
    )
    with pytest.raises(InvalidContractError, match=f"runtime input parameter '{input_name}'"):
        recipe.compile(registry)


def test_first_runtime_input_wins_over_later_alias_named_configuration():
    registry = create_default_registry()
    recipe = _registered_recipe(
        registry,
        name="runtime_input_values_with_data_option",
        func=_input_values_data,
        parameters={"data": "legitimate-option"},
    )
    compiled = recipe.compile(registry)
    assert len(compiled) == 1
    assert callable(compiled[0])
    values = np.array([1.0, 2.0, 3.0])
    values.setflags(write=False)
    result = compiled[0](values, data="legitimate-option")
    np.testing.assert_array_equal(result, values)
    assert not values.flags.writeable


def test_variadic_keyword_signatures_reserve_only_known_input_aliases():
    registry = create_default_registry()
    bad = _registered_recipe(
        registry,
        name="runtime_kwargs_reserved",
        func=_input_kwargs_only,
        parameters={"values": "not-data"},
    )
    with pytest.raises(InvalidContractError, match="reserved runtime input alias"):
        bad.compile(registry)

    kwargs_registry = create_default_registry()
    kwargs_recipe = _registered_recipe(
        kwargs_registry, name="runtime_kwargs_named_data", func=_input_kwargs_only,
        parameters={"lag": 2},
    )
    kwargs_executor = kwargs_recipe.compile(kwargs_registry)[0]
    keyword_values = np.array([4.0, 5.0])
    np.testing.assert_array_equal(
        kwargs_executor(values=keyword_values, lag=2), keyword_values
    )

    good_registry = create_default_registry()
    good = _registered_recipe(
        good_registry,
        name="runtime_varargs_kwargs_config",
        func=_input_varargs_kwargs,
        parameters={"lag": 2},
    )
    compiled = good.compile(good_registry)
    assert len(compiled) == 1
    assert callable(compiled[0])
    positional_values = np.array([6.0, 7.0])
    positional_values.setflags(write=False)
    np.testing.assert_array_equal(compiled[0](positional_values, lag=2), positional_values)
    assert not positional_values.flags.writeable


def test_legal_axis_lag_and_parameter_domain_checks_remain_active():
    registry = create_default_registry()
    metadata = registry.get("cs_zscore")
    recipe = TreatmentRecipe(
        recipe_id="runtime-input-legal-zscore-params",
        source_factor_definition_ref="test-factor",
        source_factor_value_ref="test-values",
        ordered_steps=(RecipeStep(
            step_id="zscore", semantic_transform_id=metadata.semantic_id,
            implementation_ref="cs_zscore", stage=metadata.stage,
            parameters={"axis": 0, "ddof": 0.5},
        ),),
    )
    assert len(recipe.compile(registry)) == 1

    custom_registry = create_default_registry()
    out_of_domain = _registered_recipe(
        custom_registry,
        name="runtime_input_data_bad_lag",
        func=_input_data,
        parameters={"lag": 9},
        suffix="invalid-domain",
    )
    with pytest.raises(ValueError, match="parameter 'lag'"):
        out_of_domain.compile(custom_registry)
