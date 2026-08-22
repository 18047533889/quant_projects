# -*- coding: utf-8 -*-
"""R23-005/006/021/022/023: ReportPeriod != KnowledgeTime, and UpdateTime is
NEVER a revision clock.

The A-share and US financial tables must carry SEPARATE knowledge-time and
period-id columns (never collapse them), and their catalog certificates must
record that announcement PIT and revision-vintage PIT are distinct.
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


def test_ashare_report_period_is_not_knowledge_time():
    for name in _FIN_TABLES_A:
        t = _find(ASHARE_TABLE_SPECS, name)
        assert t is not None, name
        assert t.knowledge_time_column == "PubDate", name
        assert t.period_id_column == "ReportPeriodEndDate", name
        assert t.knowledge_time_column != t.period_id_column, name


def test_us_report_period_is_not_knowledge_time():
    for name in _FIN_TABLES_US:
        t = _find(US_TABLE_SPECS, name)
        assert t is not None, name
        assert t.knowledge_time_column == "filing_date", name
        assert t.period_id_column == "period_end", name
        assert t.knowledge_time_column != t.period_id_column, name


def test_update_time_is_declared_not_a_revision_clock():
    # R23-021/022/023: the A-share revision column is UpdateTime (a physical
    # vendor/pipeline write timestamp) and the certificate must mark it as
    # NOT a revision-available-at clock.
    for name in _FIN_TABLES_A:
        t = _find(ASHARE_TABLE_SPECS, name)
        assert t.revision_column == "UpdateTime", name
        assert t.metadata.get("revision_vintage_pit_certified") is False, name
        assert "revision_clock_note" in t.metadata, name


def test_ashare_announcement_pit_certified_with_vintage_split():
    for name in _FIN_TABLES_A:
        t = _find(ASHARE_TABLE_SPECS, name)
        assert t.metadata.get("announcement_pit_certified") is True, name
        assert t.metadata.get("revision_vintage_pit_certified") is False, name
        assert t.metadata.get("restatement_risk") is True, name


def test_us_financial_announcement_vs_vintage_split():
    for name in _FIN_TABLES_US:
        t = _find(US_TABLE_SPECS, name)
        assert t.metadata.get("announcement_pit_certified") is True, name
        assert t.metadata.get("revision_vintage_pit_certified") is False, name
        assert t.metadata.get("restatement_risk") is True, name
