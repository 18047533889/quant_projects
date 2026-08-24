# -*- coding: utf-8 -*-
"""截面 rank 前缀不变性：截断未来日期后历史 rank 不变。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


def _panel(rows: int = 5, cols: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    data = rng.normal(size=(rows, cols))
    return pd.DataFrame(data, index=idx, columns=[f"S{i}" for i in range(cols)])


def test_rank_prefix_invariant_pandas():
    x = _panel()
    op = OperatorRegistry.get("rank", backend="pandas_numpy")
    full = op.calculate(x)
    truncated = op.calculate(x.iloc[:3])
    pd.testing.assert_frame_equal(full.iloc[:3], truncated)


def test_rank_prefix_invariant_polars():
    x = _panel()
    op_p = OperatorRegistry.get("rank", backend="polars")
    op_n = OperatorRegistry.get("rank", backend="pandas_numpy")
    pl_full = pl.from_pandas(x.reset_index(names=["date"]))
    pl_trunc = pl.from_pandas(x.iloc[:3].reset_index(names=["date"]))
    cols = list(x.columns)
    got_full = op_p.calculate(pl_full).select(cols).to_pandas().values
    got_trunc = op_p.calculate(pl_trunc).select(cols).to_pandas().values
    exp_trunc = op_n.calculate(x.iloc[:3])[cols].to_numpy()
    np.testing.assert_allclose(got_trunc, exp_trunc, equal_nan=True)
    np.testing.assert_allclose(got_full[:3], exp_trunc, equal_nan=True)


def test_zscore_prefix_invariant_pandas():
    x = _panel()
    op = OperatorRegistry.get("zscore", backend="pandas_numpy")
    full = op.calculate(x)
    truncated = op.calculate(x.iloc[:3])
    pd.testing.assert_frame_equal(full.iloc[:3], truncated)
