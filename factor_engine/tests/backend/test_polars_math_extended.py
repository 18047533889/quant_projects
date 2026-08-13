# -*- coding: utf-8
"""Polars 数学扩展算子与 pandas 对齐测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


def _panel() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {"A": np.linspace(0.1, 2.0, 8), "B": np.linspace(2.0, 0.2, 8)},
        index=idx,
    )


def _assert_close(got: pl.DataFrame, exp: pd.DataFrame):
    cols = list(exp.columns)
    np.testing.assert_allclose(
        got.select(cols).to_numpy(),
        exp[cols].to_numpy(),
        equal_nan=True,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "name",
    ["asin", "acos", "atan", "tan", "cbrt", "ceil", "floor", "inv", "inverse", "reverse"],
)
def test_unary_math_polars_matches_pandas(name: str):
    op_p = OperatorRegistry.get(name, backend="polars")
    op_n = OperatorRegistry.get(name, backend="pandas_numpy")
    x = _panel().clip(lower=0.11)
    pl_x = pl.from_pandas(x.reset_index(names=["date"]))
    _assert_close(op_p.calculate(pl_x), op_n.calculate(x))


def test_round_polars_matches_pandas():
    op_p = OperatorRegistry.get("round", backend="polars")
    op_n = OperatorRegistry.get("round", backend="pandas_numpy")
    x = _panel()
    pl_x = pl.from_pandas(x.reset_index(names=["date"]))
    _assert_close(op_p.calculate(pl_x, decimals=2), op_n.calculate(x, 2))



    """Near-zero, NaN, and Inf operands must produce null, not Inf/NaN."""
    op = OperatorRegistry.get("safe_div_null", backend="polars")
    x = pl.DataFrame({
        "date": [1, 2, 3, 4],
        "A": [4.0, 4.0, np.inf, 4.0],
        "B": [2.0, 4.0, 4.0, np.nan],
    })
    y = pl.DataFrame({
        "date": [1, 2, 3, 4],
        "A": [2.0, 1e-13, 2.0, np.nan],
        "B": [0.0, 2.0, np.inf, 2.0],
    })
    out = op.calculate(x, y)
    assert out["A"].to_list() == [2.0, None, None, None]
    assert out["B"].to_list() == [None, 2.0, None, None]


def test_safe_div_null_rejects_invalid_epsilon():
    op = OperatorRegistry.get("safe_div_null", backend="polars")
    x = pl.DataFrame({"A": [1.0]})
    y = pl.DataFrame({"A": [1.0]})
    with pytest.raises(ValueError):
        op.calculate(x, y, epsilon=-1.0)


def test_atan2_lerp_polars_matches_pandas():
    x = _panel()
    y = _panel() * 0.5
    pl_x = pl.from_pandas(x.reset_index(names=["date"]))
    pl_y = pl.from_pandas(y.reset_index(names=["date"]))
    for name, args in (("atan2", (pl_y, pl_x)), ("lerp", (pl_x, pl_y, 0.3))):
        op_p = OperatorRegistry.get(name, backend="polars")
        op_n = OperatorRegistry.get(name, backend="pandas_numpy")
        if op_p is None or op_n is None:
            continue
        pargs = args
        nargs = (y, x) if name == "atan2" else (x, y, 0.3)
        _assert_close(op_p.calculate(*pargs), op_n.calculate(*nargs))


def test_removed_convenience_names_are_not_daily():
    from cleaned_operators.operator_surface import DAILY_CANONICALS

    for name in ("avg2", "at_imax", "at_imin", "digital_count", "blom_transform"):
        assert name not in DAILY_CANONICALS
