# -*- coding: utf-8 -*-
"""Capability resolver — operator / expression level market support."""
from __future__ import annotations

import pytest

from market import MarketStatus
from market.capability_resolver import (
    explain_expression_support,
    explain_operator_support,
    operator_support,
)


@pytest.fixture(scope="module", autouse=True)
def _load():
    from cleaned_operators import load_all

    load_all()


def test_generic_operators_both_markets() -> None:
    # Generic math/TS/CS operators are INPUT_DEPENDENT (not "CERTIFIED_NATIVE"):
    # they are supported in both markets, but their actual eligibility is decided
    # by the input field providers, not by the operator alone (P1-9).
    for op in ("ts_mean", "ts_std", "ts_rank", "cs_rank_gaussian", "group_neutralize"):
        a = operator_support(op, "ashare", production=False)
        u = operator_support(op, "us", production=False)
        assert a.status == MarketStatus.INPUT_DEPENDENT, op
        assert u.status == MarketStatus.INPUT_DEPENDENT, op
        assert a.depends_on_inputs and u.depends_on_inputs


def test_price_limit_operators_ashare_only() -> None:
    for op in (
        "ashare_limit_up_touch", "ashare_limit_down_touch",
        "ashare_limit_one_price", "limit_up_close", "limit_down_close",
    ):
        a = operator_support(op, "ashare", production=False)
        u = operator_support(op, "us", production=False)
        assert a.status == MarketStatus.CERTIFIED_NATIVE, op
        assert u.status == MarketStatus.UNSUPPORTED_MARKET_MECHANISM, op


def test_industry_operators_us_provider_required() -> None:
    u = operator_support("industry_size_neutralize", "us", production=False)
    assert u.status == MarketStatus.PROVIDER_REQUIRED
    assert "INDUSTRY_CLASSIFICATION" in u.missing_capabilities
    a = operator_support("industry_size_neutralize", "ashare", production=False)
    assert a.status == MarketStatus.CERTIFIED_NATIVE


def test_holder_and_index_weight_us_provider_required() -> None:
    for op in ("holder_concentration", "holder_pledge_ratio", "index_weight"):
        u = operator_support(op, "us", production=False)
        assert u.status == MarketStatus.PROVIDER_REQUIRED, op
        a = operator_support(op, "ashare", production=False)
        assert a.status == MarketStatus.CERTIFIED_NATIVE, op


def test_unknown_operator() -> None:
    assert (
        operator_support("this_operator_does_not_exist", "us").status
        == MarketStatus.UNKNOWN
    )


def test_production_gate_research_only() -> None:
    # An operator NOT on the production allowlist fails closed in production.
    from cleaned_operators.registry import OperatorRegistry
    from market.capability_resolver import _production_allowlist

    prod = _production_allowlist()
    if not prod:
        pytest.skip("production allowlist unavailable")
    # Pick ANY registered canonical that is not production-certified (robust to
    # the allowlist changing as the certification layer evolves).
    non_prod = next(
        c for c in OperatorRegistry.list_canonical() if c not in prod
    )
    support = operator_support(non_prod, "us", production=True)
    assert support.status == MarketStatus.RESEARCH_ONLY


def test_explain_operator_support_suggests_providers() -> None:
    result = explain_operator_support("industry_size_neutralize", "us", production=False)
    assert result["status"] == "PROVIDER_REQUIRED"
    assert any("GICS" in s for s in result.get("suggested_providers", []))


def test_expression_supported_cross_market() -> None:
    expr = "rank(ts_mean(ret, 20))"
    # These assert the ADAPTER semantics (decimal-return cross-market identity),
    # not production certification — so they run in research mode where the
    # production allowlist gate does not apply.
    assert explain_expression_support(expr, "us", production=False)["supported"]
    assert explain_expression_support(expr, "ashare", production=False)["supported"]


def test_expression_mechanism_rejected_compile_time() -> None:
    expr = "ashare_limit_up_touch(close, high_limit)"
    result = explain_expression_support(expr, "us", production=False)
    assert not result["supported"]
    reasons = {f["reason"] for f in result["failed_nodes"]}
    assert "UNSUPPORTED_MARKET_MECHANISM" in reasons
    # same expression is fine in ashare (market-mechanism gate is mode-independent)
    assert explain_expression_support(expr, "ashare", production=False)["supported"]


def test_expression_field_gate() -> None:
    result = explain_expression_support(
        "industry_neutralize(ret, industry_code)", "us", production=False
    )
    assert not result["supported"]
    # industry_code maps to the industry_group concept whose US provider is
    # unavailable -> the field node must be a failed node.
    assert any(
        f["reason"] in ("UNKNOWN_FIELD", "UNSUPPORTED_MARKET_MECHANISM", "NO_PROVIDER")
        for f in result["failed_nodes"]
    )
    # A-share resolves industry_code via its own registry
    assert explain_expression_support(
        "industry_neutralize(ret, industry_code)", "ashare", production=False
    )["supported"]


def test_expression_market_cap_concept_level() -> None:
    # A-side physical StockValuationDaily.MarketCap must resolve in US via the
    # market_cap_local concept (Close * weighted_shares provider).
    expr = "rank(ts_mean(market_cap, 5))"
    assert explain_expression_support(expr, "us", production=False)["supported"]
