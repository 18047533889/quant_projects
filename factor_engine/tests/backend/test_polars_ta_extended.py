# -*- coding: utf-8 -*-
"""Polars TA / 统计算子与 pandas 对齐。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _panel(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "A": np.cumsum(rng.normal(0, 1, n)) + 100,
            "B": np.cumsum(rng.normal(0, 0.5, n)) + 50,
        },
        index=idx,
    )


def _to_polars(panel: pd.DataFrame):
    import polars as pl

    return pl.DataFrame({c: panel[c].to_numpy() for c in panel.columns})


@pytest.fixture(scope="module", autouse=True)
def _load():
    load_all()


def test_adx_polars_matches_pandas():
    panel = _panel(40)
    high = panel * 1.01
    low = panel * 0.99
    close = panel
    pd_op = OperatorRegistry.get("ADX", backend="pandas_numpy")
    pl_op = OperatorRegistry.get("ADX", backend="polars")
    pd_out = pd_op.calculate(high, low, close, window=14)
    pl_out = pl_op.calculate(_to_polars(high), _to_polars(low), _to_polars(close), window=14)
    for c in panel.columns:
        np.testing.assert_allclose(
            pd_out[c].values,
            pl_out[c].to_numpy(),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )


def test_aroon_polars_matches_pandas():
    panel = _panel(40)
    pd_op = OperatorRegistry.get("AROON", backend="pandas_numpy")
    pl_op = OperatorRegistry.get("AROON", backend="polars")
    if pd_op is None or pl_op is None:
        pytest.skip("AROON is not an active dual-backend primitive")
    pd_out = pd_op.calculate(panel, window=25)
    pl_out = pl_op.calculate(_to_polars(panel), window=25)
    for c in panel.columns:
        np.testing.assert_allclose(
            pd_out[c].values,
            pl_out[c].to_numpy(),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )


def test_slope_polars_matches_pandas():
    panel = _panel(40)
    pd_op = OperatorRegistry.get("Slope", backend="pandas_numpy")
    pl_op = OperatorRegistry.get("Slope", backend="polars")
    if pd_op is None or pl_op is None:
        pytest.skip("Slope is not an active dual-backend primitive")
    pd_out = pd_op.calculate(panel, window=10)
    pl_out = pl_op.calculate(_to_polars(panel), window=10)
    for c in panel.columns:
        np.testing.assert_allclose(
            pd_out[c].values,
            pl_out[c].to_numpy(),
            rtol=1e-4,
            atol=1e-4,
            equal_nan=True,
        )
