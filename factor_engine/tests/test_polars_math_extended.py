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


def test_atan2_lerp_polars_matches_pandas():
    x = _panel()
    y = _panel() * 0.5
    pl_x = pl.from_pandas(x.reset_index(names=["date"]))
    pl_y = pl.from_pandas(y.reset_index(names=["date"]))
    for name, args in (("atan2", (pl_y, pl_x)), ("lerp", (pl_x, pl_y, 0.3))):
        op_p = OperatorRegistry.get(name, backend="polars")
        op_n = OperatorRegistry.get(name, backend="pandas_numpy")
        pargs = args
        nargs = (y, x) if name == "atan2" else (x, y, 0.3)
        _assert_close(op_p.calculate(*pargs), op_n.calculate(*nargs))


def test_avg2_at_imax_digital_count_have_polars():
    for name in ("avg2", "at_imax", "at_imin", "digital_count", "blom_transform"):
        assert "polars" in OperatorRegistry.backends_for(name), name
