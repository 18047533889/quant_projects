# -*- coding: utf-8
"""Mining composite preset 契约与 CompositeSource 构建。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.delenv("DATA_ACCESS_CONFIG", raising=False)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    yield
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass


def _write_registry(tmp_path: Path) -> Path:
    content = f"""
_daily: &daily
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true

daily_market_summary:
  <<: *daily
  root: {tmp_path / "polygon"}
  glob: "*.parquet"
  time_column: trade_date
  instrument_column: T
  schema:
    T: string
    trade_date: string
    o: double
    c: double
    h: double
    l: double
    v: double
    vw: double

stocks_floats:
  <<: *daily
  root: {tmp_path / "floats"}
  glob: "*.parquet"
  time_column: effective_date
  instrument_column: ticker
  schema:
    ticker: string
    effective_date: string
    free_float: int
    free_float_percent: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed(tmp_path: Path) -> None:
    poly = tmp_path / "polygon"
    poly.mkdir()
    pd.DataFrame(
        {
            "T": ["AAPL", "AAPL"],
            "trade_date": ["2024-01-02", "2024-01-03"],
            "o": [1.0, 2.0],
            "c": [1.1, 2.1],
            "h": [1.2, 2.2],
            "l": [0.9, 1.9],
            "v": [100.0, 200.0],
            "vw": [1.05, 2.05],
        }
    ).to_parquet(poly / "daily_market_summary_2024.parquet")

    fl = tmp_path / "floats"
    fl.mkdir()
    pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "effective_date": ["2024-01-01", "2024-01-02"],
            "free_float": [1000, 1000],
            "free_float_percent": [0.8, 0.8],
        }
    ).to_parquet(fl / "stocks_floats_all.parquet")


def test_us_polygon_floats_composite_builds_and_loads(tmp_path, monkeypatch):
    from api.mining_integration import default_us_polygon_floats_composite_config
    from storage.factory import build_data_source

    reg = _write_registry(tmp_path)
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(reg))
    _seed(tmp_path)

    cfg = default_us_polygon_floats_composite_config()
    src = build_data_source(cfg)
    close = src.load_column("close")
    ff = src.load_column("free_float_percent")
    assert len(close) > 0
    assert len(ff) > 0


def test_polygon_floats_preset_join_is_asof_backward():
    from api.mining_integration import audit_composite_join_policies, default_us_polygon_floats_composite_config

    cfg = default_us_polygon_floats_composite_config()
    assert audit_composite_join_policies(cfg) == []
    assert cfg["aliases"]["free_float_percent"] == "floats.free_float_percent"


def test_us_polygon_floats_in_pit_audit_presets():
    from api.mining_integration import audit_default_data_source_configs

    report = audit_default_data_source_configs()
    assert report["us_polygon_floats"] == []


def test_us_sip_balance_sheet_composite_builds_and_loads(tmp_path, monkeypatch):
    from api.mining_integration import default_us_sip_balance_sheet_composite_config
    from storage.factory import build_data_source

    content = f"""
_daily: &daily
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true

us_stocks_sip_day_aggs:
  <<: *daily
  root: {tmp_path / "day"}
  glob: "*.parquet"
  time_column: window_start
  instrument_column: ticker
  schema:
    ticker: string
    window_start: int
    close: double

fundamentals_balance_sheet:
  <<: *daily
  root: {tmp_path / "bs"}
  glob: "*.parquet"
  time_column: period_end
  instrument_column: tickers
  schema:
    tickers: string
    period_end: string
    total_assets: double
    total_equity: double
    total_liabilities: double
"""
    reg = tmp_path / "datasets.yaml"
    reg.write_text(content.strip() + "\n", encoding="utf-8")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(reg))

    day_dir = tmp_path / "day"
    day_dir.mkdir()
    pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "window_start": [1704153600000000000, 1704240000000000000],
            "close": [100.0, 101.0],
        }
    ).to_parquet(day_dir / "day.parquet")

    bs_dir = tmp_path / "bs"
    bs_dir.mkdir()
    pd.DataFrame(
        {
            "tickers": ["AAPL", "AAPL"],
            "period_end": ["2024-01-01", "2024-01-02"],
            "total_assets": [1000.0, 1100.0],
            "total_equity": [600.0, 650.0],
            "total_liabilities": [400.0, 450.0],
        }
    ).to_parquet(bs_dir / "bs.parquet")

    cfg = default_us_sip_balance_sheet_composite_config()
    src = build_data_source(cfg)
    assets = src.load_column("total_assets")
    assert len(assets) > 0


def test_default_mining_presets_single_registry():
    from api.datasets_contract import audit_mining_dataset_contract
    from api.mining_integration import default_mining_data_source_presets

    presets = default_mining_data_source_presets()
    assert "us_sip_balance_sheet" in presets
    assert presets["us_sip_cash_flow"]["joins"]["cash_flow"] == "asof_backward"
    report = audit_mining_dataset_contract()
    assert report["ok"], report.get("violations")
    assert "fundamentals_balance_sheet" in report["datasets_checked"]
