"""Regression coverage for runtime data arguments vs recipe parameters."""
from __future__ import annotations
import numpy as np

import pytest

from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe
from factor_preprocess.errors import InvalidContractError
from factor_preprocess.contracts.recipe_parameters import validate_recipe_runtime_input_binding
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


def _input_varargs_axis(*args, axis=-1):
    return args[0], axis


def _input_varargs_data(*args, data=None):
    return args[0], data


def _input_varargs_only(*args):
    return args[0]


def _input_varargs_data_kwargs(*args, data=None, **kwargs):
    return args[0], data


class _UninspectableCallable:
    @property
    def __signature__(self):
        raise ValueError("opaque callable signature")

    def __call__(self, *args, **kwargs):
        return args[0] if args else None


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

    kwargs_only_registry = create_default_registry()
    kwargs_only = _registered_recipe(
        kwargs_only_registry, name="runtime_kwargs_keyword_input",
        func=_input_kwargs_only, parameters={},
    )
    kwargs_executor = kwargs_only.compile(kwargs_only_registry)[0]
    keyword_values = np.array([8.0, 9.0])
    keyword_values.setflags(write=False)
    np.testing.assert_array_equal(kwargs_executor(values=keyword_values), keyword_values)
    assert not keyword_values.flags.writeable


@pytest.mark.parametrize(
    "name,func,parameter,value",
    [("runtime_varargs_axis_option", _input_varargs_axis, "axis", 0),
     ("runtime_varargs_data_option", _input_varargs_data, "data", "legitimate-option")],
 )
def test_varargs_runtime_input_is_separate_from_keyword_configuration(name, func, parameter, value):
    registry = create_default_registry()
    recipe = _registered_recipe(
        registry, name=name, func=func, parameters={parameter: value}
    )
    executor = recipe.compile(registry)[0]
    values = np.array([10.0, 11.0])
    values.setflags(write=False)
    result, configured = executor(values, **{parameter: value})
    np.testing.assert_array_equal(result, values)
    assert configured == value
    assert not values.flags.writeable


def test_varargs_kwargs_allows_explicit_data_option_and_rejects_undeclared_alias():
    registry = create_default_registry()
    recipe = _registered_recipe(
        registry,
        name="runtime_varargs_data_option_with_kwargs",
        func=_input_varargs_data_kwargs,
        parameters={"data": "legitimate-option"},
    )
    executor = recipe.compile(registry)[0]
    values = np.array([12.0, 13.0])
    values.setflags(write=False)
    result, configured = executor(values, data="legitimate-option")
    np.testing.assert_array_equal(result, values)
    assert configured == "legitimate-option"
    assert not values.flags.writeable

    bad_registry = create_default_registry()
    bad = _registered_recipe(
        bad_registry,
        name="runtime_varargs_kwargs_values_reserved",
        func=_input_varargs_data_kwargs,
        parameters={"values": "not-data"},
    )
    with pytest.raises(InvalidContractError, match="reserved runtime input alias"):
        bad.compile(bad_registry)


def test_varargs_only_configuration_is_rejected_by_signature_binder():
    registry = create_default_registry()
    recipe = _registered_recipe(
        registry,
        name="runtime_varargs_only_invalid_keyword",
        func=_input_varargs_only,
        parameters={"lag": 2},
    )
    # The runtime-input guard accepts variadic positional signatures, but the
    # registered signature binder still rejects keywords that the transform
    # itself cannot receive.
    with pytest.raises(ValueError, match="invalid parameters"):
        recipe.compile(registry)


def test_uninspectable_runtime_input_signature_fails_closed():
    with pytest.raises(InvalidContractError, match="no inspectable runtime input signature"):
        validate_recipe_runtime_input_binding(
            _UninspectableCallable(), {}, transform_name="opaque_transform"
        )


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
