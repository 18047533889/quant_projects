"""Contract tests for the opt-in FP → FE native Polars rank candidate."""
from __future__ import annotations

import math

import numpy as np
import pytest


@pytest.fixture
def adapter():
    from factor_preprocess.adapters.fe_rank import cs_rank_fe_native

    return cs_rank_fe_native


def _python_rank_oracle(values):
    """Independent average-rank oracle based on Python sorting and tie runs."""
    source = np.asarray(values)
    result = np.full(source.shape, np.nan, dtype=np.float64)
    if source.ndim == 0:
        raise ValueError("oracle expects at least one dimension")
    width = source.shape[-1]
    if width == 0:
        return result
    group_shape = source.shape[:-1]
    groups = np.ndindex(group_shape) if group_shape else [()]
    for group in groups:
        row = source[group]
        finite_positions = [i for i, value in enumerate(row) if math.isfinite(float(value))]
        ordered = sorted(finite_positions, key=lambda i: float(row[i]))
        start = 0
        while start < len(ordered):
            stop = start + 1
            value = float(row[ordered[start]])
            while stop < len(ordered) and float(row[ordered[stop]]) == value:
                stop += 1
            average_rank = ((start + 1) + stop) / 2.0
            percentile = 0.5 if len(ordered) == 1 else (average_rank - 1.0) / (len(ordered) - 1.0)
            for position in ordered[start:stop]:
                result[group + (position,)] = percentile
            start = stop
    return result


def test_candidate_matches_independent_oracle_and_real_fp_contract(adapter):
    from factor_preprocess.transforms.cross_sectional import cs_rank

    values = np.array(
        [
            [2.0, 2.0, 5.0, np.nan, np.inf, -np.inf],
            [7.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            [3.0, 3.0, 3.0, np.nan, np.inf, -np.inf],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )
    expected = _python_rank_oracle(values)
    fp_actual = cs_rank(values, method="average", pct=True, axis=-1)
    actual = adapter(values, method="average", pct=True, axis=-1)

    assert actual.shape == values.shape
    assert actual.dtype == np.float64
    np.testing.assert_allclose(fp_actual, expected, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "values,expected",
    [
        (
            np.array([[2**53, 2**53 + 1, 2**53 + 2]], dtype=np.int64),
            np.array([[0.0, 0.5, 1.0]]),
        ),
        (
            np.array([[False, True, True]], dtype=np.bool_),
            np.array([[0.0, 0.75, 0.75]]),
        ),
    ],
)
def test_integer_precision_and_boolean_transport_preserve_fp_ordering(
    adapter, values, expected
):
    from factor_preprocess.transforms.cross_sectional import cs_rank

    actual = adapter(values)
    fp_actual = cs_rank(values, method="average", pct=True, axis=-1)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fp_actual, expected)


def test_float_dtype_wider_than_fe_float64_fails_closed(adapter):
    if np.dtype(np.longdouble).itemsize <= np.dtype(np.float64).itemsize:
        pytest.skip("platform longdouble is not wider than Float64")
    with pytest.raises(TypeError, match="wider than Float64"):
        adapter(np.array([[1.0, 2.0]], dtype=np.longdouble))


def test_candidate_executes_real_fe_polars_helper_and_never_fp_fallback(
    adapter, monkeypatch
):
    import factor_engine.backend.rank_spec as rank_spec
    import factor_preprocess.transforms.cross_sectional as fp_cross_sectional

    original = rank_spec.polars_cs_rank_expr
    calls = []

    def observed_helper(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    def forbidden_fp(*args, **kwargs):
        raise AssertionError("FE candidate must not invoke FP rank fallback")

    monkeypatch.setattr(rank_spec, "polars_cs_rank_expr", observed_helper)
    monkeypatch.setattr(fp_cross_sectional, "cs_rank", forbidden_fp)
    backing = np.array([[3.0, 1.0, 1.0], [np.nan, 4.0, 2.0]])
    values = backing[:, ::-1]
    values.setflags(write=False)
    before = backing.copy()

    result = adapter(values)

    assert len(calls) == 1
    assert calls[0][1]["partition_cols"] == ("__fp_fe_group",)
    assert calls[0][1]["canon"] == "rank"
    np.testing.assert_allclose(result, _python_rank_oracle(values), equal_nan=True)
    np.testing.assert_array_equal(backing, before)
    assert not values.flags.writeable


def test_noncontiguous_view_is_processed_in_multiple_complete_group_chunks(
    adapter, monkeypatch
):
    import factor_engine.backend.rank_spec as rank_spec
    from factor_preprocess.transforms.cross_sectional import cs_rank

    original = rank_spec.polars_cs_rank_expr
    calls = []

    def observed_helper(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(rank_spec, "polars_cs_rank_expr", observed_helper)
    backing = np.arange(5 * 2 * 12, dtype=np.float64).reshape(5, 2, 12)
    values = backing[..., ::2]
    values.setflags(write=False)
    before = backing.copy()
    cap = 12  # two full six-asset cross-sections per FE carrier

    actual = adapter(values, max_chunk_cells=cap)
    expected = cs_rank(values, method="average", pct=True, axis=-1)

    assert len(calls) == 5
    assert all(call[1]["partition_cols"] == ("__fp_fe_group",) for call in calls)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(backing, before)
    assert not values.flags.writeable


def test_empty_and_zero_dimensional_inputs_match_fp_boundary(adapter):
    from factor_preprocess.transforms.cross_sectional import cs_rank

    empty = np.empty((2, 0), dtype=np.float32)
    actual = adapter(empty, axis=0)
    expected = cs_rank(empty, axis=0, pct=True)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)

    with pytest.raises(np.exceptions.AxisError):
        cs_rank(np.array(3.0), pct=True, axis=-1)
    with pytest.raises(np.exceptions.AxisError):
        adapter(np.array(3.0), axis=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "min"},
        {"method": "ordinal"},
        {"pct": False},
        {"pct": np.bool_(True)},
        {"axis": 0},
        {"axis": (1,)},
    ],
)
def test_unsupported_rank_semantics_fail_closed(adapter, kwargs):
    with pytest.raises((TypeError, ValueError)):
        adapter(np.ones((2, 3)), **kwargs)


@pytest.mark.parametrize("budget", [0, -1, True, False, 1.5, np.nan])
def test_invalid_chunk_budget_types_fail_closed(adapter, budget):
    with pytest.raises((TypeError, ValueError)):
        adapter(np.ones((2, 3)), max_chunk_cells=budget)


@pytest.mark.parametrize("budget", [0, -1, True, False, 1.5, np.nan])
def test_invalid_result_budget_types_fail_closed(adapter, budget):
    with pytest.raises((TypeError, ValueError)):
        adapter(np.ones((2, 3)), max_result_bytes=budget)


def test_group_width_and_output_budget_reject_before_real_fe_execution(
    adapter, monkeypatch
):
    import factor_engine.backend.rank_spec as rank_spec

    def forbidden_helper(*args, **kwargs):
        raise AssertionError("FE rank helper must not run after budget rejection")

    monkeypatch.setattr(rank_spec, "polars_cs_rank_expr", forbidden_helper)
    with pytest.raises(ValueError, match="complete cross-section"):
        adapter(np.ones((2, 5)), max_chunk_cells=4)
    with pytest.raises(MemoryError, match="max_result_bytes"):
        adapter(np.ones((2, 3)), max_result_bytes=8)


@pytest.mark.parametrize(
    "values",
    [
        np.array([[1.0 + 2.0j, 3.0 + 4.0j]]),
        np.array([["a", "b"]], dtype=object),
    ],
)
def test_non_real_numeric_inputs_fail_closed(adapter, values):
    with pytest.raises(TypeError, match="real numeric NumPy dtype"):
        adapter(values)


def test_execution_identity_is_explicit_candidate_and_tracks_fe_helper():
    import polars as pl

    import factor_preprocess.adapters.fe_rank as adapter_module
    from factor_preprocess.adapters.fe_rank import _module_source_identity, execution_identity
    from factor_engine.backend.rank_spec import polars_cs_rank_expr

    identity = execution_identity()
    assert identity["status"] == "opt_in_candidate"
    assert identity["production_admitted"] is False
    assert identity["fe_function"].endswith("rank_spec.polars_cs_rank_expr")
    assert identity["modules"]["adapter"] == _module_source_identity(adapter_module.__file__)
    assert identity["modules"]["fe_helper"] == _module_source_identity(
        polars_cs_rank_expr.__code__.co_filename
    )
    assert identity["runtime_versions"] == {"numpy": np.__version__, "polars": pl.__version__}
    assert identity["default_memory_bounds"] == {
        "max_chunk_cells": 1_000_000,
        "max_result_bytes": 256 * 1024 * 1024,
    }
    assert identity["digest"] == execution_identity()["digest"]


def test_input_must_be_ndarray(adapter):
    with pytest.raises(TypeError, match="NumPy ndarray"):
        adapter([[1.0, 2.0, 3.0]])
