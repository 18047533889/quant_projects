# -*- coding: utf-8 -*-
"""真实 parquet 路径可选集成测试（nightly / 本地有数据时跑）。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from factor_engine.workspace_paths import quant_projects_root


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _ashare_daily_available() -> bool:
    root = os.environ.get(
        "ASHARE_PARQUET_ROOT",
        str(quant_projects_root() / "data/a_share/lqtp_data"),
    )
    bar = _expand(root) / "StockDailyBar"
    return bar.is_dir() and any(bar.glob("**/*.parquet"))


def _us_daily_available() -> bool:
    root = os.environ.get(
        "US_MASSIVE_ROOT",
        str(quant_projects_root() / "data/us_stock/massive_data"),
    )
    bar = _expand(root) / "StockDailyBar"
    return bar.is_dir() and any(bar.glob("**/*.parquet"))


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not _ashare_daily_available(), reason="A 股 StockDailyBar parquet 不可用")
def test_real_ashare_stock_daily_loads_close():
    from factor_engine.storage.factory import build_data_source

    src = build_data_source(
        {
            "type": "data_access",
            "dataset": "ashare_stock_daily_adj",
            "fields": {"close": "AdjClose"},
            "start_date": "2024-01-02",
            "end_date": "2024-01-05",
        }
    )
    close = src.load_column("close")
    assert len(close) > 0
    assert close.index.names == ["timestamp", "instrument"]


@pytest.mark.skipif(not _us_daily_available(), reason="美股 StockDailyBar parquet 不可用")
def test_real_us_stock_daily_loads_close():
    from factor_engine.storage.factory import build_data_source

    src = build_data_source(
        {
            "type": "data_access",
            "dataset": "us_stock_daily",
            "fields": {"close": "Close"},
            "start_date": "2024-01-02",
            "end_date": "2024-01-05",
        }
    )
    close = src.load_column("close")
    assert len(close) > 0


@pytest.mark.skipif(not _ashare_daily_available(), reason="A 股 parquet 不可用")
def test_real_ashare_rank_smoke():
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.factory import build_data_source

    src = build_data_source(
        {
            "type": "data_access",
            "dataset": "ashare_stock_daily_adj",
            "fields": {"close": "AdjClose"},
            "start_date": "2024-01-02",
            "end_date": "2024-01-10",
        }
    )
    factor = Factor(name="real_rank", expr=col("close"))
    engine = FactorEngine(backend=PandasBackend(), data_source=src)
    out = engine.run(factor)
    assert out is not None
    assert len(out) > 0
