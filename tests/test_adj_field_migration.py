# -*- coding: utf-8 -*-
"""ADJ_FIELD_MIGRATION（2026-08-28）验证：A 股行情权威口径 = 后复权表。

data_access.config.semantic_fields.yaml 把 A 股 close/open/high/low/vwap/
amount/pre_close/high_limit/low_limit/return 解析到 StockDailyBarAdj.Adj*，
分钟 minute_* 解析到 StockMinuteBarAdj.Adj*；FactorEngine fields/catalog.py +
api/columns.py 的 anchor 偏好同步指向复权表。未复权表仅保留 Factor/Volume 用途，
上游 LQTP gRPC DataTable（StockDailyBar/StockMinuteBar）保持不变。
"""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "factor_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")


# --------------------------------------------------------------------------- data_access
def test_semantic_catalog_ashare_bare_fields_resolve_to_adj():
    from data_access.read.semantic_catalog import SemanticFieldCatalog

    sc = SemanticFieldCatalog.from_yaml(
        os.path.join(_ROOT, "data_access/config/semantic_fields.yaml")
    )
    expected = {
        "close": ("ashare_stock_daily_adj", "AdjClose"),
        "open": ("ashare_stock_daily_adj", "AdjOpen"),
        "high": ("ashare_stock_daily_adj", "AdjHigh"),
        "low": ("ashare_stock_daily_adj", "AdjLow"),
        "vwap": ("ashare_stock_daily_adj", "AdjVwap"),
        "amount": ("ashare_stock_daily_adj", "AdjAmount"),
        "pre_close": ("ashare_stock_daily_adj", "AdjPreClose"),
        "high_limit": ("ashare_stock_daily_adj", "AdjHighLimit"),
        "low_limit": ("ashare_stock_daily_adj", "AdjLowLimit"),
        "ret": ("ashare_stock_daily_adj", "Return"),
        "return_bp": ("ashare_stock_daily_adj", "Return"),
        "volume": ("ashare_stock_daily_adj", "Volume"),
    }
    for name, (ds, phys) in expected.items():
        f = sc.resolve_one(name, market="ashare")
        assert f is not None, f"{name} should resolve"
        assert (f.dataset, f.physical_name) == (ds, phys), f"{name}: {f}\n want {(ds, phys)}"


def test_semantic_catalog_minute_fields_resolve_to_adj():
    from data_access.read.semantic_catalog import SemanticFieldCatalog

    sc = SemanticFieldCatalog.from_yaml(
        os.path.join(_ROOT, "data_access/config/semantic_fields.yaml")
    )
    for name, phys in (
        ("minute_vwap", "AdjVwap"),
        ("minute_close", "AdjClose"),
        ("minute_open", "AdjOpen"),
        ("minute_amount", "AdjAmount"),
        ("m_open", "AdjOpen"),
        ("m_vwap", "AdjVwap"),
    ):
        f = sc.resolve_one(name, market="ashare")
        assert f is not None and f.dataset == "ashare_stock_minute_adj", name
        assert f.physical_name == phys, (name, f.physical_name)


def test_registry_has_adj_datasets():
    from data_access.registry.loader import load_registry

    reg = load_registry(os.path.join(_ROOT, "data_access/config/datasets.yaml"))
    assert "ashare_stock_daily_adj" in reg.names()
    assert "ashare_stock_minute_adj" in reg.names()
    s = reg.get("ashare_stock_minute_adj").schema
    assert "AdjVwap" in s and "AdjClose" in s and "Volume" in s and "AdjOpen" in s


def test_cos_contract_ashare_has_adj_contracts():
    from data_access.cos_contract_ashare import ASHARE_COS_CONTRACTS
    for n in ("ashare_stock_daily_adj", "ashare_stock_minute_adj"):
        c = ASHARE_COS_CONTRACTS[n]
        assert c.adjustment_convention == "backward_vendor_factor" if n.endswith("_daily_adj") else True


# --------------------------------------------------------------------------- factor_engine
def test_fe_catalog_adj_tables_registered():
    from factor_engine.fields import FIELD_REGISTRY

    for t in ("StockDailyBarAdj", "StockMinuteBarAdj"):
        assert FIELD_REGISTRY.resolve_table(t) is not None, t
    assert len(FIELD_REGISTRY.fields(table="StockDailyBarAdj")) >= 14


def test_fe_api_field_resolves_bare_ashare_to_adj():
    from factor_engine.api.columns import field

    for name, source in (
        ("close", "AdjClose"),
        ("open", "AdjOpen"),
        ("high", "AdjHigh"),
        ("low", "AdjLow"),
        ("vwap", "AdjVwap"),
        ("amount", "AdjAmount"),
        ("pre_close", "AdjPreClose"),
        ("high_limit", "AdjHighLimit"),
        ("low_limit", "AdjLowLimit"),
        ("volume", "Volume"),
    ):
        f = field(name, strict=False, for_mining=True)
        assert f is not None and getattr(f, "table", None) == "StockDailyBarAdj", name
        assert f.source_name == source, (name, f.source_name)


def test_fe_api_field_minute_adj_vwap():
    from factor_engine.api.columns import field

    f = field("minute_vwap", table="StockMinuteBarAdj", strict=False, for_mining=True)
    assert f.table == "StockMinuteBarAdj" and f.source_name == "AdjVwap"
    f2 = field("minute_close", table="StockMinuteBarAdj", strict=False, for_mining=True)
    assert f2.table == "StockMinuteBarAdj" and f2.source_name == "AdjClose"


def test_fe_adj_aliases():
    from factor_engine.api.columns import field

    assert field("adj_close", strict=False).source_name == "AdjClose"
    assert field("adj_vwap", strict=False).source_name == "AdjVwap"
    assert field("ret", strict=False).source_name == "Return"
    assert field("return", strict=False).table == "StockDailyBarAdj"


def test_fe_market_registry_resolution():
    from factor_engine.fields.resolver import resolve_market_field
    from factor_engine.market.context import ASHARE_CONTEXT

    spec = resolve_market_field("close", ASHARE_CONTEXT, table="StockDailyBarAdj", strict=False)
    assert spec is not None and spec.spec.source_name == "AdjClose"


# --------------------------------------------------------------------------- store E2E (synthetic adj parquet; no COS)
def test_store_resolve_and_read_synthetic_adj(tmp_path, monkeypatch):
    import pandas as pd

    import data_access.registry.loader as loader
    from data_access.store import DataAccessStore, get_store, reset_store

    adj_root = tmp_path / "StockDailyBarAdj"
    adj_root.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "Symbol": ["000001.SZ", "000001.SZ"],
            "AdjOpen": [10.0, 10.5],
            "AdjHigh": [11.0, 11.2],
            "AdjLow": [9.8, 10.1],
            "AdjClose": [10.8, 11.0],
            "AdjPreClose": [10.0, 10.8],
            "Volume": [1000, 1100],
            "AdjAmount": [10000.0, 11000.0],
            "AdjHighLimit": [11.0, 11.3],
            "AdjLowLimit": [9.0, 9.1],
            "Return": [0.0, 0.008],
            "Factor": [2.0, 2.0],
            "AdjVwap": [10.5, 10.9],
            "IsSuspend": [False, False],
            "UpdateTime": pd.to_datetime(
                ["2026-01-05T02:00:00Z", "2026-01-06T02:00:00Z"]
            ),
        }
    )
    df.to_parquet(adj_root / "2026-01-05.parquet", index=False)

    monkeypatch.setenv("ASHARE_PARQUET_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    reset_store()

    reg = loader.load_registry(os.path.join(_ROOT, "data_access/config/datasets.yaml"))
    from data_access.store import get_shared_engine

    store = DataAccessStore(registry=reg, engine=get_shared_engine())
    try:
        resolved = store.resolve_fields(["close", "vwap", "ret"], dataset="ashare_stock_daily_adj")
        assert resolved[0].dataset == "ashare_stock_daily_adj"
        assert resolved[0].physical_name == "AdjClose"
        handle = store.read(
            "ashare_stock_daily_adj",
            columns=["TradeDate", "Symbol", "AdjClose", "AdjVwap"],
            time_range=("2026-01-05", "2026-01-06"),
        )
        tbl = handle.to_arrow()
        assert tbl.num_rows == 2
        assert "AdjClose" in tbl.column_names and "AdjVwap" in tbl.column_names
        assert list(tbl.column("AdjClose").to_pylist()) == [10.8, 11.0]
    finally:
        reset_store()