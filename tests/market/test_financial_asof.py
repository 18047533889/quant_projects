# -*- coding: utf-8 -*-
"""FinancialPeriodAdapter P0-1: as-of joins must not cross stocks.

The asof merge MUST carry the instrument column (``by``): without it a stock's
signal row can legally match another company's filing when the dates line up.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Symbol": ["AAA", "AAA", "BBB", "BBB"],
            "filing_date": pd.to_datetime(
                ["2024-04-20", "2024-05-01", "2024-04-21", "2024-05-02"]
            ),
            "value": [1.0, 2.0, 999.0, 1000.0],
        }
    )


def test_asof_requires_instrument_column_when_mixed() -> None:
    from factor_engine.fields.providers import FinancialPeriodAdapter

    with pytest.raises(ValueError):
        FinancialPeriodAdapter.asof(_events(), pd.Series(["2024-04-30"]), market="us")


def test_asof_by_instrument_never_crosses_stocks() -> None:
    from factor_engine.fields.providers import FinancialPeriodAdapter

    signals = pd.DataFrame(
        {
            "Symbol": ["AAA", "BBB", "AAA", "BBB"],
            "__signal_date__": pd.to_datetime(
                ["2024-04-25", "2024-04-25", "2024-05-05", "2024-05-05"]
            ),
        }
    )
    merged = FinancialPeriodAdapter.asof(_events(), signals, market="us", by="Symbol")
    got = {
        (row["Symbol"], pd.Timestamp(row["__signal_date__"]).strftime("%Y-%m-%d")): row["value"]
        for _, row in merged.iterrows()
    }
    # AAA at 2024-04-25 -> AAA's 2024-04-20 filing (1.0), NEVER BBB's 999.
    assert got[("AAA", "2024-04-25")] == pytest.approx(1.0)
    # BBB at 2024-04-25 -> BBB's 2024-04-21 filing (999), never AAA's 1.0.
    assert got[("BBB", "2024-04-25")] == pytest.approx(999.0)
    # later dates pick up each instrument's own newer filing.
    assert got[("AAA", "2024-05-05")] == pytest.approx(2.0)
    assert got[("BBB", "2024-05-05")] == pytest.approx(1000.0)


def test_derived_provider_executes_real_multiplication() -> None:
    import numpy as np

    from factor_engine.fields.providers import PROVIDER_REGISTRY, apply_binding_transform

    # ADJ_FIELD_MIGRATION: the A-share continuous_close binding now reads the
    # precomputed AdjClose on StockDailyBarAdj (identity) — no runtime multiply.
    b = PROVIDER_REGISTRY.require_binding("continuous_close", "ashare")
    out = apply_binding_transform(
        b,
        None,
        fields={
            "StockDailyBarAdj.AdjClose": np.array([10.0, 20.0]),
        },
    )
    np.testing.assert_allclose(out, [10.0, 20.0])


def test_financial_bindings_point_at_statement_datasets() -> None:
    from factor_engine.fields.providers import PROVIDER_REGISTRY

    assert PROVIDER_REGISTRY.require_binding("operating_revenue", "ashare").dataset == "ashare_stock_income"
    assert PROVIDER_REGISTRY.require_binding("total_assets", "ashare").dataset == "ashare_stock_balance"
    assert PROVIDER_REGISTRY.require_binding("operating_cash_flow", "ashare").dataset == "ashare_stock_cashflow"
    assert PROVIDER_REGISTRY.require_binding("operating_revenue", "us").dataset == "us_stock_income"
    assert PROVIDER_REGISTRY.require_binding("equity", "us").dataset == "us_stock_balance"
