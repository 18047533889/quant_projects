from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pyarrow as pa
import pytest

from data_access.core.exceptions import ValidationError
from data_access.cos_contract import (
    COS_DATASET_CONTRACTS,
    normalize_return_values,
    resolve_event_clock,
    semantic_contract_fingerprint,
    validate_panel_request,
)
from data_access.cos_runtime import _read_cos_panel


def test_contract_surface_and_market_units():
    assert len(COS_DATASET_CONTRACTS) >= 43
    assert semantic_contract_fingerprint()
    a = normalize_return_values(pa.array([100.0, -250.0]), "ashare_stock_daily")
    u = normalize_return_values(pa.array([0.01, -0.025]), "us_stock_daily")
    assert a.to_pylist() == pytest.approx([0.01, -0.025])
    assert u.to_pylist() == pytest.approx([0.01, -0.025])


def test_semantic_contract_fingerprint_preserves_date_type(monkeypatch):
    dataset, original = next(iter(COS_DATASET_CONTRACTS.items()))
    date_contract = replace(
        original,
        allowed_filter_values=(("probe", (date(2024, 1, 2),)),),
    )
    string_contract = replace(
        original,
        allowed_filter_values=(("probe", ("2024-01-02",)),),
    )

    monkeypatch.setitem(COS_DATASET_CONTRACTS, dataset, date_contract)
    date_digest = semantic_contract_fingerprint()
    monkeypatch.setitem(COS_DATASET_CONTRACTS, dataset, string_contract)
    string_digest = semantic_contract_fingerprint()

    assert len(date_digest) == 64
    assert len(string_digest) == 64
    assert date_digest != string_digest


def test_panel_contracts_fail_closed():
    for dataset, label in [
        ("ashare_stock_balance", "E1"),
        ("us_stock_valuation_daily", "X0"),
        ("us_stock_industry", "EMPTY"),
    ]:
        with pytest.raises(ValidationError, match=label):
            validate_panel_request(dataset)
    with pytest.raises(ValidationError, match="IndustrySource"):
        validate_panel_request("ashare_stock_industry")
    contract = validate_panel_request(
        "ashare_stock_industry",
        semantic_filters={"IndustrySource": "sw_l2"},
    )
    assert contract.temporal_model == "D1"
    with pytest.raises(ValidationError, match="允许值"):
        validate_panel_request(
            "ashare_stock_industry",
            semantic_filters={"IndustrySource": "unknown"},
        )


def test_effective_date_requires_opt_in():
    # us_stock_dividend 已升级为 strict-PIT（declaration_date），不再走
    # effective-only；这里用仍为 effective_time_only 的 split 验证。
    with pytest.raises(ValidationError, match="可靠公告"):
        resolve_event_clock("us_stock_capital_split")
    contract, clock = resolve_event_clock(
        "us_stock_capital_split", allow_effective_time=True
    )
    assert contract.pit_policy == "effective_time_only"
    assert clock == "execution_date"


class FakePanelStore:
    def __init__(self):
        self.calls = []

    def sql(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return pa.table({
            "TradeDate": [pd.Timestamp("2024-01-02")],
            "Symbol": ["000001.SZ"],
            "IndustrySource": ["sw_l2"],
        })

    def read_arrow(self, dataset, **kwargs):
        return pa.table({"Return": [100.0]})


def test_panel_filter_and_return_normalization():
    store = FakePanelStore()
    table = _read_cos_panel(
        store,
        "ashare_stock_industry",
        columns=["TradeDate", "Symbol", "IndustrySource"],
        semantic_filters={"IndustrySource": "sw_l2"},
    )
    assert table.num_rows == 1
    query, kwargs = store.calls[-1]
    assert '"IndustrySource" = ?' in query
    assert kwargs["params"] == ["sw_l2"]
    daily = _read_cos_panel(
        store,
        "ashare_stock_daily",
        columns=["Return"],
        normalize_returns=True,
    )
    assert daily["return_decimal"].to_pylist() == pytest.approx([0.01])
