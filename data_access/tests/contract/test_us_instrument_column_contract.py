"""contract：美股 instrument_column（Ticker / ticker）与 parquet 列名契约。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_access import reset_store
from data_access.exceptions import ValidationError
from data_access.registry import load_registry
from data_access.store import DataAccessStore
from data_access.engine import DuckDBEngine


@pytest.fixture(autouse=True)
def _skip_cos_and_reset_store(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    reset_store()


def _write_config(tmp_path: Path, *, us_root: Path, us_clean: Path) -> Path:
    content = f"""
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
  schema:
    TradeDate: date
    Ticker: string
    Close: double

us_stock_valuation_daily:
  <<: *us_defaults
  root: {us_root}/StockValuationDaily
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Ticker
  schema:
    TradeDate: date
    Ticker: string
    PeRatio: double

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
  schema:
    date: date
    ticker: string
    adj_factor: double

us_universe_daily:
  <<: *us_hive
  root: {us_clean}/universe_daily
  glob: "year=*/data.parquet"
  time_column: trade_date
  instrument_column: ticker
  schema:
    trade_date: date
    ticker: string

us_ticker_map:
  <<: *us_defaults
  root: {us_root}/TickerMap
  glob: "*.parquet"
  time_column: effective_date
  instrument_column: ticker
  schema:
    effective_date: date
    ticker: string
    composite_ticker: string
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_us_parquet(us_root: Path, us_clean: Path) -> None:
    daily = us_root / "StockDailyBar"
    daily.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "Ticker": ["AAPL", "MSFT", "AAPL"],
            "Close": [190.0, 380.0, 191.0],
        }
    ).to_parquet(daily / "2024-01-02.parquet")

    val = us_root / "StockValuationDaily"
    val.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "Ticker": ["AAPL", "MSFT"],
            "PeRatio": [25.0, 30.0],
        }
    ).to_parquet(val / "2024-01-02.parquet")

    adj = us_clean / "adj_factor" / "date=2024-01-02"
    adj.mkdir(parents=True)
    pd.DataFrame(
        {"ticker": ["AAPL", "MSFT"], "adj_factor": [1.0, 1.1]}
    ).to_parquet(adj / "data.parquet")

    uni = us_clean / "universe_daily" / "year=2024"
    uni.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]).date,
            "ticker": ["AAPL", "GOOG"],
        }
    ).to_parquet(uni / "data.parquet")

    tmap = us_root / "TickerMap"
    tmap.mkdir(parents=True)
    pd.DataFrame(
        {
            "effective_date": pd.to_datetime(["2024-01-02"]).date,
            "ticker": ["AAPL"],
            "composite_ticker": ["AAPL.US"],
        }
    ).to_parquet(tmap / "map.parquet")


@pytest.mark.parametrize(
    "dataset,instrument_col,filter_tickers,expected_rows",
    [
        ("us_stock_daily", "Ticker", ["AAPL"], 2),
        ("us_stock_valuation_daily", "Ticker", ["MSFT"], 1),
        ("us_adj_factor", "ticker", ["AAPL"], 1),
        ("us_universe_daily", "ticker", ["GOOG"], 1),
        ("us_ticker_map", "ticker", ["AAPL"], 1),
    ],
)
def test_us_instrument_filter_respects_declared_column(
    tmp_path,
    dataset,
    instrument_col,
    filter_tickers,
    expected_rows,
):
    us_root = tmp_path / "us"
    us_clean = tmp_path / "clean"
    _seed_us_parquet(us_root, us_clean)
    config = _write_config(tmp_path, us_root=us_root, us_clean=us_clean)
    store = DataAccessStore(load_registry(config), DuckDBEngine())

    table = store.read_arrow(
        dataset,
        columns=[instrument_col],
        instrument_filter=filter_tickers,
    )
    assert table.num_rows == expected_rows


def test_scan_polars_us_ticker_instrument_filter(tmp_path):
    us_root = tmp_path / "us"
    us_clean = tmp_path / "clean"
    _seed_us_parquet(us_root, us_clean)
    config = _write_config(tmp_path, us_root=us_root, us_clean=us_clean)
    store = DataAccessStore(load_registry(config), DuckDBEngine())

    lf = store.scan_polars(
        "us_stock_daily",
        columns=["Close"],
        instrument_filter=["MSFT"],
    )
    df = lf.collect().to_pandas()
    assert len(df) == 1
    assert df["Close"].iloc[0] == pytest.approx(380.0)


def test_instrument_filter_rejected_when_column_missing_from_schema(tmp_path):
    us_root = tmp_path / "us"
    us_clean = tmp_path / "clean"
    _seed_us_parquet(us_root, us_clean)
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"""
bad_dataset:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {us_root}/StockDailyBar
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: symbol
  schema:
    TradeDate: date
    Ticker: string
    Close: double
""".strip()
        + "\n",
        encoding="utf-8",
    )
    store = DataAccessStore(load_registry(path), DuckDBEngine())
    with pytest.raises(ValidationError, match="instrument_column='symbol'"):
        store.read_arrow("bad_dataset", instrument_filter=["AAPL"])
