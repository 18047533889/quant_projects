# -*- coding: utf-8 -*-
"""Polars flex_max/flex_min 与扩展均值算子对齐测试。"""
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
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    return pd.DataFrame({"A": [1.0, 5.0, 3.0, 2.0, 4.0, 6.0], "B": [2.0, 1.0, 4.0, 3.0, 5.0, 0.0]}, index=idx)


def _assert_close(got: pl.DataFrame, exp: pd.DataFrame):
    cols = list(exp.columns)
    np.testing.assert_allclose(
        got.select(cols).to_numpy(),
        exp[cols].to_numpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize("name", ["flex_max", "flex_min"])
def test_flex_elementwise_matches_pandas(name: str):
    op_p = OperatorRegistry.get(name, backend="polars")
    op_n = OperatorRegistry.get(name, backend="pandas_numpy")
    x = _panel()
    y = _panel() * 0.5 + 1.0
    pl_x = pl.from_pandas(x.reset_index(names=["date"]))
    pl_y = pl.from_pandas(y.reset_index(names=["date"]))
    _assert_close(op_p.calculate(pl_x, pl_y), op_n.calculate(x, y))


@pytest.mark.parametrize("name", ["flex_max", "flex_min"])
def test_flex_rolling_matches_pandas(name: str):
    op_p = OperatorRegistry.get(name, backend="polars")
    op_n = OperatorRegistry.get(name, backend="pandas_numpy")
    x = _panel()
    pl_x = pl.from_pandas(x.reset_index(names=["date"]))
    _assert_close(op_p.calculate(pl_x, 3), op_n.calculate(x, 3))


@pytest.mark.parametrize("name", ["geometric_mean", "harmonic_mean", "first_not_null"])
def test_expanding_ops_match_pandas(name: str):
    op_p = OperatorRegistry.get(name, backend="polars")
    op_n = OperatorRegistry.get(name, backend="pandas_numpy")
    if op_p is None or op_n is None:
        pytest.skip(f"{name} is not an active dual-backend primitive")
    x = _panel()
    pl_x = pl.from_pandas(x.reset_index(names=["date"]))
    _assert_close(op_p.calculate(pl_x), op_n.calculate(x))
