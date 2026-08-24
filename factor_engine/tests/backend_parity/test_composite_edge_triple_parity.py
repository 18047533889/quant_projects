# -*- coding: utf-8
"""Composite edge-case parity：Pandas vs PolarsLong（与 evidence/composite_verified.json 同步）。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.backend_parity.composite_edge_helpers import COMPOSITE_EDGE_CASES
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def edge_source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
            (pd.Timestamp("2024-01-05"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 20.0, 19.0, 21.0, 22.0], index=idx)
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series([100.0, 110.0, 0.0, 120.0, 200.0, 210.0, 190.0, 220.0], index=idx)
    return InMemorySeriesSource(
        data={
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "operating_income": pd.Series([100.0, 110.0, 0.0, 120.0, 200.0, 180.0, 210.0, 220.0], index=idx),
            "revenue": pd.Series([200.0, 220.0, 110.0, 240.0, 400.0, 0.0, 420.0, 440.0], index=idx),
            "current_assets": pd.Series([50.0, 55.0, 52.0, 60.0, 80.0, 78.0, 82.0, 85.0], index=idx),
            "current_liabilities": pd.Series([25.0, 0.0, 26.0, 30.0, 40.0, 41.0, 0.0, 42.0], index=idx),
            "inventory": pd.Series([5.0, 5.5, 5.2, 6.0, 8.0, 7.8, 8.2, 8.5], index=idx),
            "total_debt": pd.Series([30.0, 31.0, 32.0, 33.0, 60.0, 61.0, 62.0, 63.0], index=idx),
            "total_equity": pd.Series([70.0, 0.0, 72.0, 73.0, 140.0, 141.0, 142.0, 143.0], index=idx),
            "float_shares": pd.Series([1e6, 1e6, 0.0, 1e6, 2e6, 2e6, 2e6, 2e6], index=idx),
        }
    )


def _expr_for_case(case):
    cols = [col(c) for c in case.columns]
    kwargs = dict(case.calc_kwargs)
    if case.canon in {"BollingerUpper", "BollingerLower", "BollingerBands"}:
        return F(case.canon)(*cols, case.window, **kwargs)
    if case.window is not None and case.canon not in {"OBV", "operating_margin", "current_ratio", "quick_ratio", "debt_to_equity", "real_turnover_rate", "micro_spread"}:
        return F(case.canon)(*cols, case.window, **kwargs)
    return F(case.canon)(*cols, **kwargs)


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


@pytest.mark.parametrize("case", COMPOSITE_EDGE_CASES, ids=lambda c: c.canon)
def test_composite_edge_polars_long_matches_pandas(edge_source, case):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    if case.canon not in OperatorRegistry._operators:
        pytest.skip("migrated to recipe or research layer")
    expr = _expr_for_case(case)
    pd_out = _run(edge_source, expr, "pandas")["result"].sort_index()
    long_out = _run(edge_source, expr, "polars_long")["result"].sort_index()
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-5, atol=1e-5)
