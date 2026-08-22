# -*- coding: utf-8 -*-
"""R23-010..015 / R23-241/242: date-only knowledge time -> next-session policy.

A date-only PubDate/filing_date is NOT usable at the same-day open.  Every
financial source certificate must carry an explicit knowledge-time resolution
policy, and the A-share/US financial statements must declare
``DATE_ONLY_NEXT_SESSION`` (strict PIT uses the value only from the next trading
session after the announcement date).
"""
from __future__ import annotations

from fields.catalog import ASHARE_TABLE_SPECS
from fields.catalog_us import US_TABLE_SPECS

_FIN_TABLES_A = {"StockIncome", "StockBalance", "StockCashFlow", "StockIndicator"}
_FIN_TABLES_US = {"StockIncome", "StockBalance", "StockCashFlow"}


def _find(specs, name):
    for t in specs:
        if t.name == name:
            return t
    return None


def test_ashare_financials_declare_date_only_next_session():
    for name in _FIN_TABLES_A:
        t = _find(ASHARE_TABLE_SPECS, name)
        assert t.metadata.get("knowledge_time_resolution") == "DATE_ONLY_NEXT_SESSION", name


def test_us_financials_declare_date_only_next_session():
    # R23-013: filing_date is a date-only placeholder (00:00:00) unless the SEC
    # submission timestamp is proven -> next-session conservative policy.
    for name in _FIN_TABLES_US:
        t = _find(US_TABLE_SPECS, name)
        assert t.metadata.get("knowledge_time_resolution") == "DATE_ONLY_NEXT_SESSION", name


def test_every_pit_financial_source_has_resolution_policy():
    # R23-015: any fundamental PIT source must carry the policy.  Current-
    # snapshot-only sources (X0 sparse StockIndicator, strict_pit_allowed=False)
    # are explicitly NOT historical PIT and are excluded.
    all_tables = list(ASHARE_TABLE_SPECS) + list(US_TABLE_SPECS)
    for t in all_tables:
        if t.domain == "fundamental" and t.strict_pit_allowed:
            assert t.metadata.get("knowledge_time_resolution"), t.name
            assert t.metadata.get("knowledge_time_resolution") != "UNPROVEN", t.name


def test_us_financials_require_timeframe_filter():
    # R23-048: a US financial read without an explicit timeframe is a production
    # hard fail — the tables must declare timeframe as required.
    for name in _FIN_TABLES_US:
        t = _find(US_TABLE_SPECS, name)
        assert "timeframe" in t.required_parameters, name
        assert t.metadata.get("timeframe_required") is True, name
