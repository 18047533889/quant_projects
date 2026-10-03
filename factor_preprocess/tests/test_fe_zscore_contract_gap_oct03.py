"""Independent FP contract probes for FE cross-sectional z-score reuse.

These tests characterize public FP behavior and the fractional-ddof gap. They
do not change FE admission or assert that a name/version label establishes
numerical equivalence.
"""
from __future__ import annotations

import numpy as np
import pytest
from decimal import Decimal, localcontext


def _legacy_inf_oracle(values, *, axis, ddof, constant_value):
    """FP v2 deliberately preserves v1 NumPy moments for Inf-containing slices."""
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mean = np.nanmean(values, axis=axis, keepdims=True)
        std = np.nanstd(values, axis=axis, keepdims=True, ddof=ddof)
        result = np.where(std > 0.0, (values - mean) / std, constant_value)
    return np.where(np.isnan(values), np.nan, result)


def _decimal_finite_oracle(values, *, axis, ddof, constant_value):
    """Exact represented-float centered moments, independent of FP's kernel."""
    values = np.asarray(values, dtype=np.float64)
    if axis is None:
        axes = tuple(range(values.ndim))
    elif isinstance(axis, tuple):
        axes = tuple(a % values.ndim for a in axis)
    else:
        axes = (axis % values.ndim,)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    remaining = tuple(i for i in range(values.ndim) if i not in axes)
    remaining_shape = tuple(values.shape[i] for i in remaining)
    selectors = np.ndindex(remaining_shape) if remaining else [()]
    with localcontext() as ctx:
        ctx.prec = 1200
        for coords in selectors:
            index = [slice(None)] * values.ndim
            for dim, coord in zip(remaining, coords):
                index[dim] = coord
            selected = values[tuple(index)]
            if np.isinf(selected).any():
                raise AssertionError("Decimal oracle finite policy does not accept Inf")
            flat = selected.reshape(-1)
            present = ~np.isnan(flat)
            finite = flat[present]
            selected_result = np.full(flat.shape, np.nan, dtype=np.float64)
            if not finite.size:
                out[tuple(index)] = selected_result.reshape(selected.shape)
                continue
            dx = [Decimal.from_float(float(value)) for value in finite]
            if len(dx) <= ddof or len(dx) < 2 or min(dx) == max(dx):
                selected_result[present] = constant_value
                out[tuple(index)] = selected_result.reshape(selected.shape)
                continue
            mean = sum(dx, Decimal(0)) / Decimal(len(dx))
            ss = sum(((value - mean) ** 2 for value in dx), Decimal(0))
            std = (ss / (Decimal(len(dx)) - Decimal(str(ddof)))).sqrt()
            selected_result[present] = np.asarray(
                [float((value - mean) / std) for value in dx], dtype=np.float64
            )
            out[tuple(index)] = selected_result.reshape(selected.shape)
    return out


@pytest.mark.parametrize(
    "axis,ddof",
    [(-1, 0.25), (0, 0.5), ((0, 2), 0.75), (None, 0.25)],
)
def test_fractional_ddof_and_numpy_axis_forms_are_fp_contract(axis, ddof):
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    values = np.array(
        [
            [[1.0, 4.0, np.nan], [2.0, 7.0, 9.0]],
            [[3.0, 5.0, 8.0], [6.0, np.nan, 10.0]],
        ],
        dtype=np.float64,
    )
    expected = _decimal_finite_oracle(
        values, axis=axis, ddof=ddof, constant_value=-2.5
    )
    actual = cs_zscore(values, axis=axis, ddof=ddof, constant_value=-2.5)
    assert actual.shape == values.shape
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15, equal_nan=True)


def test_inf_and_nan_follow_fp_missing_and_legacy_inf_moment_convention():
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    values = np.array(
        [[1.0, np.nan, 3.0, 7.0], [np.inf, 2.0, -np.inf, np.nan]],
        dtype=np.float64,
    )
    expected = _legacy_inf_oracle(
        values, axis=1, ddof=0.5, constant_value=4.25
    )
    actual = cs_zscore(values, axis=1, ddof=0.5, constant_value=4.25)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-15, equal_nan=True)
    assert np.isnan(actual[0, 1])
    assert np.isnan(actual[1, 3])


@pytest.mark.parametrize("ddof", [0.25, 0.5])
@pytest.mark.parametrize("values", [
    np.array([1e16, 1e16 + 2.0, 1e16 + 4.0, 1e16 + 6.0]),
    np.array([-np.finfo(np.float64).max, 0.0, np.finfo(np.float64).max]),
])
def test_fractional_ddof_stable_finite_extremes_against_decimal(values, ddof):
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    actual = cs_zscore(values, ddof=ddof, constant_value=-1.75)
    expected = _decimal_finite_oracle(
        values, axis=-1, ddof=ddof, constant_value=-1.75
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-15)


def test_recipe_convention_and_registry_admission_do_not_imply_fe_equivalence():
    from factor_preprocess.registry.policies import get_default_policy_registry
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe

    metadata = get_default_registry().get("cs_zscore")
    assert metadata.numeric_policy == "finite_anchor_centered_v2"
    assert metadata.fe_equivalent_semantics is None
    # The public registry range includes non-integral values; FE's current
    # integer-only candidate therefore cannot cover the full FP domain.
    low, high = metadata.parameter_domain["ddof"]
    assert low <= 0.5 <= high

    policies = get_default_policy_registry()
    for policy_name in ("cs_only", "causal_basic"):
        steps = policies.get(policy_name).steps
        step = next(step for step in steps if step.name == "cs_zscore")
        assert step.parameters["ddof"] == 1

    recipe = TreatmentRecipe(
        recipe_id="cs-zscore-contract-oct03",
        source_factor_definition_ref="contract-test",
        source_factor_value_ref="contract-values",
        ordered_steps=(RecipeStep(
            step_id="z", semantic_transform_id=metadata.semantic_id,
            implementation_ref="cs_zscore", stage=metadata.stage,
            parameters={"ddof": 0.5},
        ),),
    )
    compiled = recipe.compile(get_default_registry())
    panel = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = compiled[0](values=panel, axis=(0, 2), ddof=0.5)
    assert result.shape == panel.shape
    np.testing.assert_allclose(
        result, _decimal_finite_oracle(panel, axis=(0, 2), ddof=0.5,
                                       constant_value=0.0),
        rtol=2e-15, atol=2e-15,
    )
