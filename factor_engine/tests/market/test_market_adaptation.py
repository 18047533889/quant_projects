# -*- coding: utf-8 -*-
"""US expressions must never gain capabilities the US market does not have.

P1-013: the market capability matrix must be the TRUTH.  A US expression that
references a minute / turnover / chip / free-float / price-limit / industry /
index-weight / top-holder operator must fail the compile-time capability gate —
the search grammar must not let AlphaMiner spend budget on operators US cannot
execute, and the resolver must not claim a provider exists when it does not.
"""
from __future__ import annotations

import pytest

from factor_engine.market.capability_resolver import explain_expression_support

_US_BLOCKED_EXPRESSIONS = [
    # minute-microstructure family (FULL_MINUTE_OHLCV is A-share only)
    "intra_amihud(close, volume)",
    "intraday_activity_duration_curvature(volume, 5)",
    "micro_vpin(close, volume)",
    "session_event_recovery_score(close)",
    # A-share lunch-break mechanism
    "intra_lunch_gap_return(close)",
    # daily turnover family (DAILY_TURNOVER is A-share only)
    "average_turnover(volume)",
    "turnover_volatility(turnover, 20)",
    # turnover-survival chip family
    "ts_turnover_reference_price(turnover, 20)",
    # free-float family
    "free_float_turnover(close, 20)",
    # A-share price-limit mechanism
    "ashare_limit_up_touch(close, high_limit)",
    # industry (US INDUSTRY_CLASSIFICATION is empty)
    "industry_size_neutralize(ret, industry_code)",
    # index weight (US components have no Weight)
    "index_weight(index_code)",
    # top-holder (US has no isomorphic holder feed)
    "holder_concentration(holder_id)",
]


@pytest.fixture(scope="module", autouse=True)
def _load():
    from factor_engine.cleaned_operators import load_all

    load_all()


@pytest.mark.parametrize("expr", _US_BLOCKED_EXPRESSIONS)
def test_us_expression_blocked_at_compile_time(expr) -> None:
    result = explain_expression_support(expr, "us", production=True)
    assert not result["supported"], f"US expression should be blocked: {expr}"
    reasons = {f["reason"] for f in result["failed_nodes"]}
    assert (
        "UNSUPPORTED_MARKET_MECHANISM" in reasons
        or "PROVIDER_REQUIRED" in reasons
        or "PARSE_ERROR" in reasons  # not even in the daily grammar
    ), (expr, reasons)


def test_ashare_same_expression_not_blocked_by_operator_gate() -> None:
    # On A-share the operator gate must not reject the mechanism ops (the field
    # provider / value availability is a separate data question).  Research mode
    # isolates the mechanism/capability gate from the production allowlist.
    for expr in ("intra_amihud(close, volume)", "average_turnover(volume)",
                 "ashare_limit_up_touch(close, high_limit)"):
        result = explain_expression_support(expr, "ashare", production=False)
        reasons = {f["reason"] for f in result["failed_nodes"]}
        assert "UNSUPPORTED_MARKET_MECHANISM" not in reasons, (expr, reasons)
        assert "PROVIDER_REQUIRED" not in reasons, (expr, reasons)


def test_us_generic_math_still_supported() -> None:
    # The gate must not be over-aggressive: plain math/TS over common fields stays
    # (research mode: mechanism/capability gates only, not production allowlist).
    for expr in ("ts_mean(ret, 20)", "rank(ts_std(close, 10))", "log(volume)"):
        assert explain_expression_support(expr, "us", production=False)["supported"], expr
