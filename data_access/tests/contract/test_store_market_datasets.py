"""contract：A 股 / 美股登记数据集经 store 读取（本地 parquet，不访问 COS）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_access import reset_store
from data_access.registry import load_registry
from data_access.store import DataAccessStore
from data_access.engine import DuckDBEngine


@pytest.fixture(autouse=True)
def _skip_cos_and_reset_store(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    reset_store()


def _write_config(tmp_path: Path, *, ashare_root: Path, us_root: Path, us_clean: Path) -> Path:
  content = f"""
_ashare_defaults: &ashare_defaults
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true

ashare_stock_daily:
  <<: *ashare_defaults
  root: {ashare_root}/StockDailyBar
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol

_us_defaults: &us_defaults
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true

us_stock_daily:
  <<: *us_defaults
  root: {us_root}/StockDailyBar
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Ticker

_us_hive: &us_hive
  kind: static
  access_mode: published
  layout: hive
  hive_partitioning: true
  union_by_name: true

us_adj_factor:
  <<: *us_hive
  root: {us_clean}/adj_factor
  glob: "date=*/data.parquet"
  time_column: date
  instrument_column: ticker
"""
  path = tmp_path / "datasets.yaml"
  path.write_text(content.strip() + "\n", encoding="utf-8")
  return path


def _seed_market_parquet(ashare_root: Path, us_root: Path, us_clean: Path) -> None:
    ashare_dir = ashare_root / "StockDailyBar"
    ashare_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "Symbol": ["000001.SZ", "000002.SZ"],
            "Close": [10.0, 20.0],
        }
    ).to_parquet(ashare_dir / "2024-01-02.parquet")

    us_dir = us_root / "StockDailyBar"
    us_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "Ticker": ["AAPL", "MSFT"],
            "Close": [190.0, 380.0],
        }
    ).to_parquet(us_dir / "2024-01-02.parquet")

    adj_dir = us_clean / "adj_factor" / "date=2024-01-02"
    adj_dir.mkdir(parents=True)
    pd.DataFrame({"ticker": ["AAPL", "MSFT"], "adj_factor": [1.0, 1.1]}).to_parquet(
        adj_dir / "data.parquet"
    )


def test_store_reads_ashare_and_us_daily(tmp_path):
    ashare_root = tmp_path / "ashare"
    us_root = tmp_path / "us"
    us_clean = tmp_path / "clean"
    _seed_market_parquet(ashare_root, us_root, us_clean)
    config = _write_config(
        tmp_path,
        ashare_root=ashare_root,
        us_root=us_root,
        us_clean=us_clean,
    )
    store = DataAccessStore(load_registry(config), DuckDBEngine())

    ashare = store.load_columns(
        "ashare_stock_daily",
        columns=["Close"],
        output_names={"Close": "close"},
        time_range=("2024-01-02", "2024-01-02"),
    )
    us = store.load_columns(
        "us_stock_daily",
        columns=["Close"],
        output_names={"Close": "close"},
        time_range=("2024-01-02", "2024-01-02"),
    )
    assert len(ashare["close"]) == 2
    assert len(us["close"]) == 2
    assert ashare["close"].index.names == ["timestamp", "instrument"]
    assert us["close"].index.names == ["timestamp", "instrument"]


def test_store_reads_us_hive_adj_factor(tmp_path):
    ashare_root = tmp_path / "ashare"
    us_root = tmp_path / "us"
    us_clean = tmp_path / "clean"
    _seed_market_parquet(ashare_root, us_root, us_clean)
    config = _write_config(
        tmp_path,
        ashare_root=ashare_root,
        us_root=us_root,
        us_clean=us_clean,
    )
    store = DataAccessStore(load_registry(config), DuckDBEngine())

    table = store.read_arrow(
        "us_adj_factor",
        columns=["ticker", "adj_factor"],
        time_range=("2024-01-02", "2024-01-02"),
    )
    assert table.num_rows == 2
    assert set(table.column_names) == {"ticker", "adj_factor"}
