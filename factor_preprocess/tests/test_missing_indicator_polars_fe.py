"""Parity and identity checks for the native FE Polars missing predicate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")
import polars as pl

_TEST_REPLACEMENT_CALLS = []

from factor_preprocess.adapters.fe_native_elementwise import (
    get_fe_native_elementwise_executor,
)


def _values(values, *, dtype=None, index=None):
    n = len(values)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "asset_id": [f"A{i}" for i in range(n)],
            "value": pd.Series(values, dtype=dtype, index=index),
        },
        index=index,
    )


def test_native_float_preserves_row_order_duplicate_index_and_inf_semantics():
    executor = get_fe_native_elementwise_executor("is_null")
    values = _values(
        [np.nan, np.inf, -np.inf, 3.0, np.nan, 4.0],
        dtype="float64", index=[81, 7, 45, 22, 7, 13],
    )
    actual = executor(values)
    expected = pd.Series([1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                         index=values.index, name="value")
    pd.testing.assert_series_equal(actual, expected)
    identity = executor.execution_identity
    assert identity["backend"] == "polars"
    assert identity["runtime_versions"]["polars"]
    assert identity["fe_canonical_id"] == "is_null"
    assert identity["adapter_implementation_hash"]


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        ("Float64", [1.0, pd.NA, np.inf], [0.0, 1.0, 0.0]),
        ("Int64", [1, pd.NA, 0], [0.0, 1.0, 0.0]),
    ],
)
def test_nullable_numeric_uses_native_predicate(dtype, values, expected):
    executor = get_fe_native_elementwise_executor("is_null")
    frame = _values(values, dtype=dtype, index=[4, 2, 9])
    actual = executor(frame)
    pd.testing.assert_series_equal(
        actual, pd.Series(expected, index=frame.index, name="value")
    )
    assert executor.execution_identity["backend"] == "polars"


@pytest.mark.parametrize(
    ("dtype", "values"),
    [(object, ["x", None, np.nan]), ("datetime64[ns]", [pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-01-03")])],
)
def test_unsupported_dtypes_use_truthful_fe_pandas_route(dtype, values):
    executor = get_fe_native_elementwise_executor("is_null")
    frame = _values(values, dtype=dtype, index=[5, 3, 8])
    actual = executor(frame)
    expected = pd.Series(pd.isna(frame["value"]).astype(float),
                         index=frame.index, name="value")
    pd.testing.assert_series_equal(actual, expected)
    identity = executor.execution_identity
    assert identity["backend"] == "pandas_numpy"
    assert identity["backend_source"]


def test_empty_float_input_returns_empty_native_result():
    executor = get_fe_native_elementwise_executor("is_null")
    frame = pd.DataFrame({
        "date": pd.Series([], dtype="datetime64[ns]"),
        "asset_id": pd.Series([], dtype=object),
        "value": pd.Series([], dtype="float64"),
    }, index=pd.Index([], dtype="int64"))
    actual = executor(frame)
    assert actual.empty
    assert actual.name == "value"
    assert executor.execution_identity["backend"] == "polars"


@pytest.mark.parametrize("bad_identity", ["null", "duplicate"])
def test_null_or_duplicate_identity_is_rejected(bad_identity):
    executor = get_fe_native_elementwise_executor("is_null")
    frame = _values([np.nan, 1.0], dtype="float64")
    if bad_identity == "null":
        frame.loc[0, "asset_id"] = None
    else:
        frame.loc[1, "asset_id"] = frame.loc[0, "asset_id"]
        frame.loc[1, "date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError, match="null|duplicate"):
        executor(frame)

def test_warmed_executor_refreshes_native_operator_and_catalog_as_one_binding(monkeypatch):

    executor = get_fe_native_elementwise_executor("is_null")
    frame = _values([None, float("nan"), float("inf")], dtype="float64")
    executor(frame)
    old_identity = executor.execution_identity

    registry = executor._registry
    operators, aliases, catalog = registry._read_state()
    old_op = operators["is_null"]["polars"]
    _TEST_REPLACEMENT_CALLS.clear()

    class Replacement(type(old_op)):
        _physical_spec = old_op._physical_spec

        def _calculate_series(self, x, **kwargs):
            _TEST_REPLACEMENT_CALLS.append(True)
            cols = [c for c in x.columns if c != "__fe_time__"]
            return x.with_columns([
                (pl.col(c).is_null() | pl.col(c).is_nan()).cast(pl.Float64).alias(c)
                for c in cols
            ])

    replacement = Replacement()
    next_operators = {name: dict(backends) for name, backends in operators.items()}
    next_operators["is_null"]["polars"] = replacement
    next_catalog = dict(catalog)
    entry = dict(catalog["is_null"])
    backend_meta = dict(entry["backend_meta"])
    backend_meta["polars"] = dict(backend_meta["polars"], source="native_is_null_test_replacement")
    entry["backend_meta"] = backend_meta
    entry["semantic_version"] = "test-replacement-v2"
    next_catalog["is_null"] = entry

    monkeypatch.setattr(
        registry, "_read_state",
        classmethod(lambda cls: (next_operators, aliases, next_catalog)),
    )
    result = executor(frame)
    new_identity = executor.execution_identity
    assert result.tolist() == [1.0, 1.0, 0.0]
    assert _TEST_REPLACEMENT_CALLS == [True]
    assert old_identity["backend_source"] != new_identity["backend_source"]
    assert new_identity["backend_source"] == "native_is_null_test_replacement"
    assert new_identity["binding_state"] == "last_execution"
    assert new_identity["semantic_version"] == "test-replacement-v2"
    assert old_identity["fe_implementation_hash"] != new_identity["fe_implementation_hash"]


@pytest.mark.skipif(np.dtype(np.longdouble).itemsize <= 8, reason="no extended float")
def test_extended_float_public_registry_uses_fe_without_narrowing():
    from factor_preprocess.registry.transforms import get_default_registry

    values = np.array([np.nan, np.inf, -np.inf, np.finfo(np.longdouble).max,
                       np.finfo(np.longdouble).tiny, 1], dtype=np.longdouble)
    frame = _values(values, index=[81, 7, 45, 22, 7, 13])
    before = frame.copy(deep=True)
    executor = get_default_registry().get_execution("missing_indicator")
    actual = executor(frame)
    expected = pd.Series([1., 0., 0., 0., 0., 0.], index=frame.index, name="value")
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, before)
    assert executor.execution_identity["backend"] == "pandas_numpy"


@pytest.mark.parametrize("dtype", ["uint64", "UInt64"])
def test_uint64_max_stays_native(dtype):
    values = [0, np.iinfo(np.uint64).max, 2]
    if dtype == "UInt64":
        values[2] = pd.NA
    frame = _values(values, dtype=dtype, index=[5, 2, 5])
    executor = get_fe_native_elementwise_executor("is_null")
    actual = executor(frame)
    expected = pd.Series(pd.isna(frame["value"]).astype(float), index=frame.index, name="value")
    pd.testing.assert_series_equal(actual, expected)
    assert executor.execution_identity["backend"] == "polars"


