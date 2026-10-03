"""Static contract tests for the explicit FP-to-FE z-score adapter."""
from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from factor_preprocess.transforms.zscore_numeric import FINITE_ANCHOR_CENTERED_V2


def _decimal_oracle(values, *, axis, ddof, constant):
    """Independent represented-Float64 oracle for each selected axis slice."""
    source = np.asarray(values, dtype=np.float64)
    if axis is None:
        axes = tuple(range(source.ndim))
    elif source.ndim == 0 and not isinstance(axis, tuple) and int(axis) in (0, -1):
        axes = ()
    elif isinstance(axis, tuple):
        axes = tuple(int(value) % source.ndim for value in axis)
    else:
        axes = (int(axis) % source.ndim,)
    if len(set(axes)) != len(axes):
        raise ValueError("duplicate axis")
    remaining = tuple(index for index in range(source.ndim) if index not in axes)
    permutation = remaining + axes
    ordered = source.transpose(permutation)
    group_count = int(np.prod(ordered.shape[:len(remaining)], dtype=np.int64))
    reduction_size = int(np.prod(ordered.shape[len(remaining):], dtype=np.int64))
    rows = ordered.reshape(group_count, reduction_size)
    result = np.full(rows.shape, np.nan, dtype=np.float64)
    with localcontext() as ctx:
        ctx.prec = 1200
        for index, row in enumerate(rows):
            missing = np.isnan(row)
            finite = row[np.isfinite(row)]
            if np.isinf(row).any():
                result[index, ~missing] = constant
                continue
            if len(finite) < 2 or len(finite) <= ddof or np.min(finite) == np.max(finite):
                result[index, ~missing] = constant
                continue
            exact = [Decimal.from_float(float(value)) for value in finite]
            mean = sum(exact, Decimal(0)) / Decimal(len(exact))
            squared_sum = sum(((value - mean) ** 2 for value in exact), Decimal(0))
            variance = squared_sum / (Decimal(len(exact)) - Decimal(str(ddof)))
            standard_deviation = variance.sqrt()
            result[index, np.isfinite(row)] = [
                float((Decimal.from_float(float(value)) - mean) / standard_deviation)
                for value in finite
            ]
    restored = result.reshape(ordered.shape)
    if not permutation:
        return restored.reshape(source.shape)
    return restored.transpose(tuple(int(value) for value in np.argsort(permutation)))


@pytest.fixture
def adapter():
    from factor_preprocess.adapters.fe_zscore import cs_zscore_finite_anchor_fe_native

    return cs_zscore_finite_anchor_fe_native


@pytest.mark.parametrize(
    "axis,ddof",
    [(-1, 0.25), (0, 0.5), ((0, 2), 0.75), (None, 0.25), ((), 0.5)],
)
def test_adapter_matches_decimal_for_axis_forms_and_fractional_ddof(adapter, axis, ddof):
    values = np.array(
        [
            [[-3.5, 0.25, np.nan], [2.0, 7.0, 9.0]],
            [[3.0, 5.0, 8.0], [6.0, np.nan, 10.0]],
        ],
        dtype=np.float64,
    )
    expected = _decimal_oracle(values, axis=axis, ddof=ddof, constant=-2.75)
    actual = adapter(values, axis=axis, ddof=ddof, constant_value=-2.75)
    assert actual.shape == values.shape
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=8e-12, atol=8e-12, equal_nan=True)


@pytest.mark.parametrize("ddof", [0.25, 0.5, 0.75])
def test_offset_extreme_inf_missing_and_degenerate_slices_use_fe_math(adapter, ddof):
    maximum = np.finfo(np.float64).max
    offset = 1e16
    values = np.array(
        [
            [-3.5, 0.25, 4.0, np.nan],
            [offset, offset + 2.0, offset + 6.0, offset + 8.0],
            [-maximum, 0.0, maximum, np.nan],
            [1.0, np.inf, 3.0, np.nan],
            [5.0, 5.0, 5.0, np.nan],
            [np.nan, 7.0, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    expected = _decimal_oracle(values, axis=-1, ddof=ddof, constant=-1.5)
    actual = adapter(values, ddof=ddof, constant_value=-1.5)
    np.testing.assert_allclose(actual, expected, rtol=8e-12, atol=8e-12, equal_nan=True)


def test_recipe_style_kwargs_call_reaches_fe_without_mutating_readonly_view(
    adapter, monkeypatch
):
    import factor_engine.backend.long_stable_zscore as fe_long
    import factor_preprocess.transforms.cross_sectional as fp_transform
    import factor_preprocess.transforms.zscore_numeric as fp_numeric

    original_fe = fe_long.finite_anchor_centered_zscore_long
    calls = []

    def observed_fe(*args, **kwargs):
        calls.append((args, kwargs))
        return original_fe(*args, **kwargs)

    def forbidden_fp(*args, **kwargs):
        raise AssertionError("adapter must not fall back to FP numerical execution")

    monkeypatch.setattr(fe_long, "finite_anchor_centered_zscore_long", observed_fe)
    monkeypatch.setattr(fp_transform, "cs_zscore", forbidden_fp)
    monkeypatch.setattr(fp_numeric, "finite_anchor_centered_zscore", forbidden_fp)
    backing = np.arange(48, dtype=np.float64).reshape(4, 6, 2)
    view = backing[:, ::2, :]
    view.setflags(write=False)
    before = backing.copy()
    expected = _decimal_oracle(view, axis=(0, 2), ddof=0.5, constant=3.0)

    result = adapter(
        values=view,
        axis=(0, 2),
        ddof=0.5,
        constant_value=3.0,
        numeric_policy=FINITE_ANCHOR_CENTERED_V2,
        max_chunk_cells=16,
        max_result_bytes=1_000_000,
    )

    assert calls
    assert calls[0][1]["ddof"] == 0.5
    assert result.shape == view.shape
    np.testing.assert_allclose(result, expected, rtol=8e-12, atol=8e-12)
    np.testing.assert_array_equal(backing, before)
    assert not view.flags.writeable


@pytest.mark.parametrize("axis", [(0, 0), (4,), "last", 5, True, False, np.bool_(True)])
def test_adapter_rejects_invalid_axes(adapter, axis):
    with pytest.raises((TypeError, ValueError)):
        adapter(np.ones((2, 3), dtype=np.float64), axis=axis)


@pytest.mark.parametrize("axis", [None, (), 0, -1])
@pytest.mark.parametrize("value", [3.0, np.nan, np.inf])
@pytest.mark.parametrize("ddof,constant", [(0.25, -2.0), (0.75, 4.5)])
def test_scalar_ndarray_axis_semantics_match_fp_for_finite_nan_and_inf(
    adapter, axis, value, ddof, constant
):
    scalar = np.array(value, dtype=np.float64)
    expected = _decimal_oracle(scalar, axis=axis, ddof=ddof, constant=constant)
    actual = adapter(scalar, axis=axis, ddof=ddof, constant_value=constant)
    assert actual.shape == ()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_scalar_tuple_axis_is_rejected_like_fp(adapter):
    with pytest.raises((TypeError, ValueError)):
        adapter(np.array(3.0), axis=(0,))


@pytest.mark.parametrize("ddof", [-0.1, 1.01, float("nan"), float("inf"), True])
def test_adapter_fails_closed_outside_admitted_fp_ddof_domain(adapter, ddof):
    with pytest.raises(ValueError):
        adapter(np.ones((2, 3), dtype=np.float64), ddof=ddof)


def test_legacy_numeric_policy_is_not_silently_relabelled_as_fe(adapter):
    from factor_preprocess.adapters.fe_zscore import execution_identity

    with pytest.raises(ValueError, match="supports only numeric_policy"):
        adapter(np.ones((2, 3)), numeric_policy="legacy_numpy_v1")
    identity = execution_identity()
    assert identity["status"] == "opt_in_candidate"
    assert identity["fe_function"].endswith("finite_anchor_centered_zscore_long")
    assert identity["numeric_policy"] == FINITE_ANCHOR_CENTERED_V2
    assert identity["production_admitted"] is False


def test_complex_inputs_fail_closed_instead_of_discarding_imaginary_values(adapter):
    with pytest.raises(TypeError, match="real numeric NumPy dtype"):
        adapter(np.array([[1.0 + 2.0j, 3.0 + 4.0j]]))


def test_execution_identity_tracks_both_on_disk_modules_and_runtime_versions():
    import polars as pl
    import factor_preprocess.adapters.fe_zscore as adapter_module
    from factor_engine.backend.long_stable_zscore import (
        finite_anchor_centered_zscore_long,
    )
    from factor_preprocess.adapters.fe_zscore import (
        _module_source_identity,
        execution_identity,
    )

    identity = execution_identity()
    adapter_source = _module_source_identity(adapter_module.__file__)
    fe_source = _module_source_identity(finite_anchor_centered_zscore_long.__code__.co_filename)
    assert identity["modules"]["adapter"] == adapter_source
    assert identity["modules"]["fe_helper"] == fe_source
    assert identity["runtime_versions"] == {"numpy": np.__version__, "polars": pl.__version__}
    assert identity["default_memory_bounds"] == {
        "max_chunk_cells": 1_000_000,
        "max_result_bytes": 256 * 1024 * 1024,
    }
    assert "not a loaded-code closure" in identity["identity_scope"]
    assert identity["digest"] == execution_identity()["digest"]


def test_module_source_identity_changes_with_source_and_enforces_read_bound(tmp_path):
    from factor_preprocess.adapters.fe_zscore import _module_source_identity

    source = tmp_path / "module.py"
    source.write_bytes(b"helper-one")
    first = _module_source_identity(source, max_bytes=32)
    source.write_bytes(b"helper-two")
    second = _module_source_identity(source, max_bytes=32)
    assert first["sha256"] != second["sha256"]
    assert first["bytes"] == second["bytes"] == 10

    source.write_bytes(b"0123456789")
    with pytest.raises(ValueError, match="read limit"):
        _module_source_identity(source, max_bytes=9)


def test_large_universe_noncontiguous_axis_uses_bounded_complete_group_tiles(
    adapter, monkeypatch
):
    import factor_engine.backend.long_stable_zscore as fe_long
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    original_fe = fe_long.finite_anchor_centered_zscore_long
    heights = []

    def observed_fe(frame, **kwargs):
        heights.append(frame.height)
        return original_fe(frame, **kwargs)

    monkeypatch.setattr(fe_long, "finite_anchor_centered_zscore_long", observed_fe)
    rng = np.random.default_rng(20261003)
    values = rng.normal(size=(3, 5461, 4))
    values[0, 17, 2] = np.nan
    before = values.copy()
    cap = 8192

    actual = adapter(
        values,
        axis=(0, 2),
        ddof=0.5,
        constant_value=-2.0,
        max_chunk_cells=cap,
        max_result_bytes=2_000_000,
    )
    expected = cs_zscore(
        values, axis=(0, 2), ddof=0.5, constant_value=-2.0
    )

    assert len(heights) > 1
    assert max(heights) <= cap
    assert all(height % 12 == 0 for height in heights)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=8e-12, atol=8e-12, equal_nan=True)
    for asset_index in (0, 2730, 5460):
        expected_decimal = _decimal_oracle(
            values[:, asset_index, :], axis=None, ddof=0.5, constant=-2.0
        )
        np.testing.assert_allclose(
            actual[:, asset_index, :], expected_decimal, rtol=8e-12, atol=8e-12
        )
    np.testing.assert_array_equal(values, before)


def test_chunk_and_output_bounds_reject_without_splitting_reduction_groups(
    adapter, monkeypatch
):
    import factor_engine.backend.long_stable_zscore as fe_long

    def forbidden_fe(*args, **kwargs):
        raise AssertionError("FE must not run after a memory-bound rejection")

    monkeypatch.setattr(fe_long, "finite_anchor_centered_zscore_long", forbidden_fe)
    with pytest.raises(ValueError, match="complete reduction group"):
        adapter(np.ones((2, 5)), axis=-1, max_chunk_cells=4)
    with pytest.raises(MemoryError, match="max_result_bytes"):
        adapter(np.ones((2, 3)), axis=-1, max_result_bytes=8)


@pytest.mark.parametrize("max_chunk_cells", [0, -1, True, 1.5])
def test_invalid_chunk_bounds_fail_closed(adapter, max_chunk_cells):
    with pytest.raises(ValueError, match="max_chunk_cells"):
        adapter(np.ones((2, 3)), max_chunk_cells=max_chunk_cells)


@pytest.mark.parametrize("max_result_bytes", [0, -1, True, 1.5])
def test_invalid_result_bounds_fail_closed(adapter, max_result_bytes):
    with pytest.raises(ValueError, match="max_result_bytes"):
        adapter(np.ones((2, 3)), max_result_bytes=max_result_bytes)
