from __future__ import annotations

from types import SimpleNamespace
import pandas as pd
import pyarrow as pa
import pytest

from data_access.core.exceptions import ValidationError
from data_access.cos_factor_runtime import install_cos_factor_runtime
from data_access.cos_runtime import _guarded_load_columns, _load_factor_columns


def test_legacy_load_columns_cannot_bypass_contracts():
    original = lambda dataset, **kwargs: {"columns": kwargs.get("columns")}
    cases = [
        ("ashare_stock_balance", ["TotalAssets"], "伪日频面板"),
        ("us_stock_valuation_daily", ["price_to_earnings"], "X0"),
        ("ashare_stock_industry", ["IndustryCode"], "语义过滤"),
        ("ashare_stock_daily", ["Return"], "不是小数收益"),
    ]
    for dataset, columns, message in cases:
        with pytest.raises(ValidationError, match=message):
            _guarded_load_columns(original, dataset, {"columns": columns})
    assert _guarded_load_columns(
        original, "ashare_stock_daily", {"columns": ["Close"]}
    )["columns"] == ["Close"]


class Registry:
    def get(self, name):
        return SimpleNamespace(time_column="TradeDate", instrument_column="Symbol")


class FactorStore:
    def __init__(self):
        self._registry = Registry()

    def read_arrow(self, dataset, **kwargs):
        return pa.table({
            "TradeDate": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "Symbol": ["000001.SZ", "000001.SZ"],
            "Return": [100.0, -250.0],
        })


def test_factor_columns_normalize_ashare_return():
    result = _load_factor_columns(
        FactorStore(), "ashare_stock_daily", columns=["Return"]
    )
    assert list(result) == ["Return"]
    assert result["Return"].tolist() == pytest.approx([0.01, -0.025])


def test_lightweight_store_installs_raw_helpers_without_registry():
    class Lightweight:
        def sql(self, *args, **kwargs):
            return pa.table({"Return": [100.0]})

        def read_arrow(self, *args, **kwargs):
            return pa.table({"Return": [100.0]})

    store = install_cos_factor_runtime(Lightweight())
    assert callable(store.read_cos_panel)
    assert callable(store.read_cos_events)
    assert not hasattr(store, "load_factor_columns")
