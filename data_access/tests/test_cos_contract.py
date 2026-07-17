# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from data_access.core.exceptions import ValidationError
from data_access.cos_contract import normalize_return_values, validate_panel_request
from data_access.cos_runtime import install_cos_contract_methods


def test_panel_contracts_fail_closed():
    with pytest.raises(ValidationError, match="E1"):
        validate_panel_request("ashare_stock_balance")
    with pytest.raises(ValidationError, match="X0"):
        validate_panel_request("us_stock_valuation_daily")
    with pytest.raises(ValidationError, match="EMPTY"):
        validate_panel_request("us_stock_industry")
    with pytest.raises(ValidationError, match="IndustrySource"):
        validate_panel_request("ashare_stock_industry")
    assert validate_panel_request(
        "ashare_stock_industry", semantic_filters={"IndustrySource": "sw_l1"}
    ).time_model == "D1"


def test_return_units_are_market_specific():
    ashare = normalize_return_values(pa.array([100.0, -250.0]), "ashare_stock_daily")
    us = normalize_return_values(pa.array([0.01, -0.025]), "us_stock_daily")
    assert ashare.to_pylist() == pytest.approx([0.01, -0.025])
    assert us.to_pylist() == pytest.approx([0.01, -0.025])


class _FakeStore:
    def __init__(self):
        self.sql_call = None

    def sql(self, query, **kwargs):
        self.sql_call = (query, kwargs)
        return pa.table({
            "TradeDate": [pd.Timestamp("2024-01-02")],
            "Symbol": ["000001.SZ"],
            "IndustrySource": ["sw_l1"],
        })

    def read_arrow(self, dataset, **kwargs):
        if dataset == "ashare_stock_daily":
            return pa.table({"Return": [100.0]})
        return pa.table({"Ret": [0.01]})

    def read_frame(self, dataset, **kwargs):
        return pd.DataFrame({
            "Symbol": ["000001.SZ", "000001.SZ"],
            "PubDate": ["2024-04-30", "2024-08-30"],
            "ReportPeriodEndDate": ["2024-03-31", "2024-06-30"],
            "UpdateTime": ["2024-04-30T08:00:00Z", "2024-08-30T08:00:00Z"],
            "TotalAssets": [100.0, 120.0],
        })


def test_runtime_enforces_industry_filter():
    store = install_cos_contract_methods(_FakeStore())
    industry = store.read_cos_panel(
        "ashare_stock_industry",
        columns=["TradeDate", "Symbol", "IndustrySource"],
        semantic_filters={"IndustrySource": "sw_l1"},
    )
    assert industry.num_rows == 1
    query, kwargs = store.sql_call
    assert '"IndustrySource" = ?' in query
    assert kwargs["params"] == ["sw_l1"]


def test_runtime_enforces_filter_and_normalizes_return():
    """Compatibility node retained for older CI revisions."""
    test_runtime_enforces_industry_filter()


def test_runtime_normalizes_ashare_return():
    store = install_cos_contract_methods(_FakeStore())
    daily = store.read_cos_panel(
        "ashare_stock_daily", columns=["Return"], normalize_returns=True
    )
    assert daily["return_decimal"].to_pylist() == pytest.approx([0.01])


def test_event_asof_uses_publication_time_and_staleness():
    store = install_cos_contract_methods(_FakeStore())
    decisions = pd.DataFrame({
        "instrument": ["000001.SZ", "000001.SZ"],
        "decision_timestamp": ["2024-05-02", "2024-09-02"],
    })
    result = store.read_cos_events_asof(
        "ashare_stock_balance", decisions, columns=["TotalAssets"], max_age_days=180
    ).sort_values("decision_timestamp")
    assert result["TotalAssets"].tolist() == [100.0, 120.0]
    assert (result["fundamental_staleness_days"] >= 0).all()
