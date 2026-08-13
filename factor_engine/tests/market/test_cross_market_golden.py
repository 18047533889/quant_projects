# -*- coding: utf-8 -*-
"""Cross-market golden tests (spec §103-§106).

The market difference must be FULLY encapsulated in the field adapter: once A
(Return in bp) and US (Ret in decimal) both flow through their binding
transforms, they are numerically identical, and shared operators produce
identical output.  Every cross-market misuse must FAIL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fields.providers import apply_binding_transform, require_binding


@pytest.fixture(scope="module", autouse=True)
def _load():
    from cleaned_operators import load_all

    load_all()


def _synthetic_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A-side Return in bp and US-side Ret in decimal for the SAME returns.

    Example: A Return=-200.5 bp == US Ret=-0.02005 decimal.
    """
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=60)
    base = rng.normal(0.0, 0.02, (60, 2))  # decimal returns
    bp = base * 10000.0                     # A: basis points
    return pd.DataFrame(bp, index=idx, columns=["A", "B"]), pd.DataFrame(
        base, index=idx, columns=["A", "B"]
    )


# ---------------------------------------------------------------------------
# Unit golden tests (spec §105 Unit).
# ---------------------------------------------------------------------------
def test_golden_return_unit_identity() -> None:
    a_bp, us_dec = _synthetic_panels()
    a_binding = require_binding("return_decimal", "ashare")
    us_binding = require_binding("return_decimal", "us")
    a_dec = apply_binding_transform(a_binding, a_bp)
    us_out = apply_binding_transform(us_binding, us_dec)
    # A Return/10000 == US Ret identity
    assert np.allclose(a_dec, us_dec, atol=1e-12)
    assert np.allclose(us_out, us_dec, atol=1e-12)
    # a single known value: -200.5 bp -> -0.02005 decimal
    assert float(a_binding.transform(np.array([-200.5]))[0]) == pytest.approx(-0.02005)
    assert float(us_binding.transform(np.array([-0.02005]))[0]) == pytest.approx(-0.02005)


def test_golden_ratio_unit_identity() -> None:
    # A TurnoverRatio/Roe/DividendRatio are percent; /100 -> decimal.
    a_turn = require_binding("turnover_ratio_decimal", "ashare")
    assert float(a_turn.transform(np.array([1.6647]))[0]) == pytest.approx(0.016647)
    # US ROE already decimal.
    from fields.providers import binding

    us_roe = binding("roe_decimal", "us")
    assert us_roe is not None
    assert us_roe.quality.value == "SPARSE"


# ---------------------------------------------------------------------------
# Adjustment golden tests (spec §105 Adjustment).
# ---------------------------------------------------------------------------
def test_golden_adjustment_backward_multiplier() -> None:
    close = np.array([10.0, 20.0, 30.0])
    factor = np.array([1.0, 1.1, 1.32])
    # A: continuous_close == Close * Factor (verified 2026-08-08)
    a = require_binding("continuous_close", "ashare")
    # US: continuous_close == Close * AdjFactor
    us = require_binding("continuous_close", "us")
    assert "Close * Factor" in a.transform_description
    assert "Close * AdjFactor" in us.transform_description
    # both multiply raw by the backward factor
    assert np.allclose(close * factor, close * factor)


# ---------------------------------------------------------------------------
# Cross-market identity through shared operators (spec §104).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "op",
    ["ts_mean", "ts_std", "ts_rank"],
)
def test_golden_shared_operator_identity(op: str) -> None:
    from cleaned_operators import OperatorRegistry

    a_bp, us_dec = _synthetic_panels()
    a_dec = apply_binding_transform(require_binding("return_decimal", "ashare"), a_bp)
    # After the adapters both panels are the same decimal-return panel.
    assert np.allclose(a_dec, us_dec, atol=1e-12)

    impl = OperatorRegistry.get(op, "pandas_numpy")
    out_a = impl.calculate(a_dec, window=5)
    out_us = impl.calculate(us_dec, window=5)
    assert out_a.index.equals(out_us.index)
    assert out_a.columns.equals(out_us.columns)
    # NaN positions are identical; equal_nan allows NaN==NaN.
    assert np.allclose(out_a, out_us, atol=1e-12, equal_nan=True), (
        f"{op} diverges cross-market (maxdiff "
        f"{float((out_a - out_us).abs().max().max())})"
    )


# ---------------------------------------------------------------------------
# Cross-market FAIL tests (spec §106): these must all fail.
# ---------------------------------------------------------------------------
def test_fail_a_return_without_10000() -> None:
    # Using A bp as if it were decimal is a 10000x error.
    a_bp, us_dec = _synthetic_panels()
    assert not np.allclose(a_bp, us_dec)  # A raw bp != US decimal
    assert np.max(np.abs(a_bp / 10000.0 - us_dec) < 1e-9  # correct scaling matches


def test_fail_us_ret_divided_by_10000() -> None:
    _, us_dec = _synthetic_panels()
    wrong = us_dec / 10000.0
    assert not np.allclose(wrong, us_dec)


def test_fail_cross_currency_market_cap() -> None:
    from fields.units_v2 import CNY, USD

    with pytest.raises(ValueError):
        CNY.assert_compatible_with(USD)


def test_fail_us_industry_without_provider() -> None:
    from market.capability_resolver import explain_expression_support

    result = explain_expression_support(
        "industry_neutralize(ret, industry_code)", "us", production=True
    )
    assert not result["supported"]


def test_fail_us_price_limit() -> None:
    from market.capability_resolver import explain_expression_support

    result = explain_expression_support(
        "ashare_limit_up_touch(close, high_limit)", "us", production=True
    )
    assert not result["supported"]


def test_fail_us_top_holder_without_provider() -> None:
    from market.capability_resolver import explain_expression_support

    result = explain_expression_support(
        "holder_concentration(share_ratio)", "us", production=True
    )
    assert not result["supported"]


def test_fail_strict_pit_ashare_dividend() -> None:
    from fields.providers import explain_field_support

    support = explain_field_support("cash_dividend_per_share", "ashare")
    from market import MarketStatus

    assert support.status == MarketStatus.PIT_BLOCKED


# ---------------------------------------------------------------------------
# PIT semantics (spec §105 PIT).
# ---------------------------------------------------------------------------
def test_pit_ashare_uses_pubdate_us_uses_filing_date() -> None:
    from fields import MULTI_MARKET_FIELD_REGISTRY

    a_table = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare").resolve_table("StockIncome")
    us_table = MULTI_MARKET_FIELD_REGISTRY.registry_for("us").resolve_table("StockIncome")
    assert a_table.knowledge_time_column == "PubDate"
    assert us_table.knowledge_time_column == "filing_date"
    assert us_table.required_parameters == ("timeframe",)


def test_table_collision_isolation() -> None:
    from fields import MULTI_MARKET_FIELD_REGISTRY

    # StockValuationDaily / StockIndicator exist in BOTH markets and must not
    # share a contract.  R17-019: US StockCapitalDaily is split into
    # USStockCapitalSplitEvent + USTickerSharesPITEvent (no single US
    # StockCapitalDaily table), so only the surviving shared names are checked
    # for cross-market collision.
    for market in ("ashare", "us"):
        reg = MULTI_MARKET_FIELD_REGISTRY.registry_for(market)
        for table in ("StockValuationDaily", "StockIndicator"):
            spec = reg.resolve_table(table)
            assert spec is not None, f"{market}.{table} should exist"
            assert spec.dataset.startswith("us_") if market == "us" else spec.dataset.startswith("ashare_"), (
                f"{market}.{table} leaked into wrong dataset {spec.dataset}"
            )
    # R17-019: the US capital split tables are US-only; A-share keeps
    # StockCapitalDaily (S1 state).
    us_split = MULTI_MARKET_FIELD_REGISTRY.registry_for("us").resolve_table("USStockCapitalSplitEvent")
    assert us_split is not None and us_split.dataset == "us_stock_capital_split"
    us_pit = MULTI_MARKET_FIELD_REGISTRY.registry_for("us").resolve_table("USTickerSharesPITEvent")
    assert us_pit is not None and us_pit.dataset == "us_stock_capital_shares"
    a_cap = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare").resolve_table("StockCapitalDaily")
    assert a_cap is not None and a_cap.dataset == "ashare_stock_capital_daily"


# ---------------------------------------------------------------------------
# Session / timezone golden (spec §107).
# ---------------------------------------------------------------------------
def test_session_contracts() -> None:
    from market import ASHARE_CONTEXT, US_CONTEXT

    assert ASHARE_CONTEXT.session_id == "ASHARE_CONTINUOUS"
    assert ASHARE_CONTEXT.timezone == "Asia/Shanghai"
    assert US_CONTEXT.session_id == "US_REGULAR"
    assert US_CONTEXT.timezone == "America/New_York"
    assert US_CONTEXT.annualization_factor == pytest.approx(252.0)
    assert ASHARE_CONTEXT.annualization_factor == pytest.approx(252.0)
