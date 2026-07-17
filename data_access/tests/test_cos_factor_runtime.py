# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from data_access.core.exceptions import ValidationError
from data_access.cos_factor_runtime import _guarded_load_columns, _load_factor_columns
from data_access.cos_runtime import install_cos_contract_methods


def test_load_columns_cannot_bypass_event_sparse_and_unit_contracts():
    original = lambda dataset, **kwargs: {"dataset": dataset, "columns": kwargs.get("columns")}
    with pytest.raises(ValidationError, match="false daily panel"):
        _guarded_load_columns(original, "ashare_stock_balance", {"columns": ["TotalAssets"]})
    with pytest.raises(ValidationError, match="X0"):
        _guarded_load_columns(original, "us_stock_valuation_daily", {"columns": ["PE"]})
    with pytest.raises(ValidationError, match="semantic filters"):
        _guarded_load_columns(original, "ashare_stock_industry", {"columns": ["IndustryCode"]})
    with pytest.raises(ValidationError, match="not decimal return"):
        _guarded_load_columns(original, "ashare_stock_daily", {"columns": ["Return"]})
    assert _guarded_load_columns(original, "ashare_stock_daily", {"columns": ["Close"]})["columns"] == ["Close"]


class _Registry:
    def get(self, name):
        assert name == "ashare_stock_daily"
        return SimpleNamespace(time_column="TradeDate", instrument_column="Symbol")


class _FactorStore:
    def __init__(self):
        self._registry = _Registry()

    def read_cos_panel(self, dataset, **kwargs):
        assert dataset == "ashare_stock_daily"
        table = pa.table({
            "TradeDate": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "Symbol": ["000001.SZ", "000001.SZ"],
            "Return": [100.0, -250.0],
        })
        if kwargs.get("normalize_returns"):
            table = table.append_column("__return_decimal", pa.array([0.01, -0.025]))
        return table


def test_load_factor_columns_normalizes_ashare_return_before_panelization():
    result = _load_factor_columns(
        _FactorStore(),
        "ashare_stock_daily",
        columns=["Return"],
        normalize_returns=True,
    )
    assert list(result) == ["Return"]
    assert result["Return"].tolist() == pytest.approx([0.01, -0.025])
