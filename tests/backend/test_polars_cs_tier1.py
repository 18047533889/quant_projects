# -*- coding: utf-8 -*-
"""Polars normalize / quantile 与 staging 行级删除测试。"""

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


def _panel(values) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"A": values, "B": [v * 2 for v in values]}, index=idx)


def test_normalize_has_certified_native_polars_evidence():
    from factor_engine.backend.fastpath_evidence import polars_executed_parity_canonicals

    assert "normalize" in polars_executed_parity_canonicals()


def test_ambiguous_quantile_name_is_not_daily():
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS

    assert "quantile" not in DAILY_CANONICALS


def test_normalize_is_daily_but_quantile_is_not():
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS

    assert "normalize" in DAILY_CANONICALS
    assert "quantile" not in DAILY_CANONICALS


def test_rank_polars_matches_pandas():
    op_p = OperatorRegistry.get("rank", backend="polars")
    op_n = OperatorRegistry.get("rank", backend="pandas_numpy")
    x = _panel([1.0, 2.0, 3.0, 4.0, 5.0])
    got = op_p.calculate(pl.from_pandas(x.reset_index(names=["date"])))
    exp = op_n.calculate(x)
    cols = list(x.columns)
    np.testing.assert_allclose(
        got.select(cols).to_numpy(),
        exp[cols].to_numpy(),
        equal_nan=True,
    )


def test_zscore_polars_matches_pandas():
    op_p = OperatorRegistry.get("zscore", backend="polars")
    op_n = OperatorRegistry.get("zscore", backend="pandas_numpy")
    x = _panel([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    got = op_p.calculate(pl.from_pandas(x.reset_index(names=["date"])))
    exp = op_n.calculate(x)
    cols = list(x.columns)
    np.testing.assert_allclose(
        got.select(cols).to_numpy(),
        exp[cols].to_numpy(),
        equal_nan=True,
        rtol=1e-10,
        atol=1e-10,
    )
