import numpy as np
import pandas as pd
import polars as pl
import pytest

import factor_engine.cleaned_operators.common.data_cleaning  # noqa: F401
import factor_engine.cleaned_operators.common.polars_auto  # noqa: F401
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract


def _op(backend):
    return OperatorRegistry.get("fillna", backend=backend)


def _panel(backend):
    frame = pd.DataFrame({"A": [1., np.nan, np.nan, 4.], "B": [np.nan, 3., 5., np.nan]})
    return pl.from_pandas(frame) if backend == "polars" else frame


def _pandas(result):
    return result.to_pandas() if isinstance(result, pl.DataFrame) else result


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("method", ["zero", 2.5, "mean", "median"])
def test_default_registry_fillna_matches_independent_reference(backend, method):
    source = pd.DataFrame({"A": [1., np.nan, np.nan, 4.], "B": [np.nan, 3., 5., np.nan]})
    if method == "zero":
        expected = source.fillna(0.0)
    elif isinstance(method, float):
        expected = source.fillna(method)
    elif method == "mean":
        expected = source.T.fillna(source.mean(axis=1)).T
    else:
        expected = source.T.fillna(source.median(axis=1)).T
    result = _op(backend).calculate(_panel(backend), method)
    pd.testing.assert_frame_equal(_pandas(result), expected)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_forward_fill_policy_is_declared_and_enforced(backend):
    op = _op(backend)
    blocked = _pandas(op.calculate(x=_panel(backend), method="ffill", forward_fill_allowed=False))
    pd.testing.assert_frame_equal(blocked, _pandas(_panel(backend)))
    limited = _pandas(op.calculate(x=_panel(backend), method="pad", max_ffill_gap=1))
    expected = _pandas(_panel(backend)).ffill(limit=1)
    pd.testing.assert_frame_equal(limited, expected)
    with pytest.raises((TypeError, ValueError)):
        op.calculate(_panel(backend), "bfill")
    for bad in (True, np.inf, "typo"):
        with pytest.raises((TypeError, ValueError)):
            op.calculate(_panel(backend), bad)


def test_fillna_union_schema_is_verified():
    for backend in ("pandas_numpy", "polars"):
        op = _op(backend)
        schema, _, verified = _parameter_contract(op, ("x",))
        assert verified
        assert schema["properties"]["method"]["anyOf"] == [
            {"x-factor-engine-role": None, "x-searchable": True, "type": "string"},
            {"x-factor-engine-role": None, "x-searchable": True, "type": "number"},
        ]
        assert schema["properties"]["forward_fill_allowed"]["type"] == "boolean"
        assert schema["properties"]["max_ffill_gap"]["minimum"] == 0
