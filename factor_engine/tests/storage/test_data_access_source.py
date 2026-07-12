"""factor_engine DataAccessSource 双市场（A 股 / 美股）契约测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from storage.factory import build_data_source


@pytest.fixture(autouse=True)
def _reset_store_between_tests(monkeypatch):
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


def _write_registry(tmp_path: Path, *, ashare_root: Path, us_root: Path) -> Path:
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
  schema:
    TradeDate: date
    Symbol: string
    Close: double

ashare_stock_valuation_daily:
  <<: *ashare_defaults
  root: {ashare_root}/StockValuationDaily
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    PeRatio: double

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
    TradeDate: timestamp
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

ashare_stock_status:
  <<: *ashare_defaults
  root: {ashare_root}/StockStatus
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    ListedState: string

ashare_index_constituent:
  <<: *ashare_defaults
  root: {ashare_root}/IndexConstituent
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    IndexSymbol: string
    Weight: double

_us_hive: &us_hive
  kind: static
  access_mode: published
  layout: hive
  hive_partitioning: true
  union_by_name: true

us_universe_daily:
  <<: *us_hive
  root: {us_root}/universe_daily
  glob: "year=*/data.parquet"
  time_column: trade_date
  instrument_column: ticker
  schema:
    trade_date: date
    ticker: string
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_market_data(ashare_root: Path, us_root: Path) -> None:
    ash_dir = ashare_root / "StockDailyBar"
    ash_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-02"]).date,
            "Symbol": ["000001.SZ", "000002.SZ"],
            "Close": [10.0, 20.0],
        }
    ).to_parquet(ash_dir / "2024-01-02.parquet")

    ash_val = ashare_root / "StockValuationDaily"
    ash_val.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-01", "2024-01-01"]).date,
            "Symbol": ["000001.SZ", "000002.SZ"],
            "PeRatio": [8.0, 12.0],
        }
    ).to_parquet(ash_val / "2024-01-01.parquet")

    us_dir = us_root / "StockDailyBar"
    us_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "Ticker": ["AAPL", "MSFT"],
            "Close": [190.0, 380.0],
        }
    ).to_parquet(us_dir / "2024-01-02.parquet")

    us_val = us_root / "StockValuationDaily"
    us_val.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-01", "2024-01-01"]).date,
            "Ticker": ["AAPL", "MSFT"],
            "PeRatio": [25.0, 30.0],
        }
    ).to_parquet(us_val / "2024-01-01.parquet")

    ash_status = ashare_root / "StockStatus"
    ash_status.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-01", "2024-01-01"]).date,
            "Symbol": ["000001.SZ", "000002.SZ"],
            "ListedState": ["Listed", "Listed"],
        }
    ).to_parquet(ash_status / "2024-01-01.parquet")

    ash_idx = ashare_root / "IndexConstituent"
    ash_idx.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-01", "2024-01-01"]).date,
            "Symbol": ["000001.SZ", "000002.SZ"],
            "IndexSymbol": ["000300.SH", "000300.SH"],
            "Weight": [0.6, 0.4],
        }
    ).to_parquet(ash_idx / "2024-01-01.parquet")

    us_uni = us_root / "universe_daily" / "year=2024"
    us_uni.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]).date,
            "ticker": ["AAPL", "MSFT"],
        }
    ).to_parquet(us_uni / "data.parquet")


@pytest.fixture
def market_env(tmp_path, monkeypatch):
    ashare_root = tmp_path / "ashare"
    us_root = tmp_path / "us"
    _seed_market_data(ashare_root, us_root)
    cfg = _write_registry(tmp_path, ashare_root=ashare_root, us_root=us_root)
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(cfg))
    from data_access import reset_store

    reset_store()
    return ashare_root, us_root


def test_data_access_source_ashare_daily(market_env):
    from api.mining_integration import default_ashare_pv_data_source_config

    source = build_data_source(default_ashare_pv_data_source_config())
    close = source.load_column("close")
    assert close.index.names == ["timestamp", "instrument"]
    assert len(close) == 2
    ts = close.index.get_level_values(0)
    assert ts[0].hour == 0 and ts[0].minute == 0


def test_data_access_source_us_daily(market_env):
    from api.mining_integration import default_us_pv_data_source_config

    source = build_data_source(default_us_pv_data_source_config())
    close = source.load_column("close")
    assert close.index.names == ["timestamp", "instrument"]
    assert len(close) == 2
    assert set(close.index.get_level_values(1)) == {"AAPL", "MSFT"}


def test_composite_ashare_valuation_asof(market_env):
    from api.mining_integration import default_ashare_pv_valuation_data_source_config

    source = build_data_source(default_ashare_pv_valuation_data_source_config())
    pe = source.load_column("valuation.pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "000001.SZ")] == pytest.approx(8.0)


def test_composite_ashare_valuation_pe_alias(market_env):
    from api.mining_integration import default_ashare_pv_valuation_data_source_config

    source = build_data_source(default_ashare_pv_valuation_data_source_config())
    pe = source.load_column("pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "000001.SZ")] == pytest.approx(8.0)


def test_composite_us_valuation_asof(market_env):
    from api.mining_integration import default_us_pv_valuation_data_source_config

    source = build_data_source(default_us_pv_valuation_data_source_config())
    pe = source.load_column("valuation.pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAPL")] == pytest.approx(25.0)


def test_composite_us_valuation_pe_alias(market_env):
    from api.mining_integration import default_us_pv_valuation_data_source_config

    source = build_data_source(default_us_pv_valuation_data_source_config())
    pe = source.load_column("pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAPL")] == pytest.approx(25.0)


def test_composite_ashare_universe_asof(market_env):
    from api.mining_integration import default_ashare_pv_universe_data_source_config

    source = build_data_source(default_ashare_pv_universe_data_source_config())
    weight = source.load_column("constituent.weight")
    assert weight.loc[(pd.Timestamp("2024-01-02"), "000001.SZ")] == pytest.approx(0.6)
    listed = source.load_column("status.listed_state")
    assert listed.loc[(pd.Timestamp("2024-01-02"), "000001.SZ")] == "Listed"


def test_composite_us_universe_exact(market_env):
    from api.mining_integration import default_us_pv_universe_data_source_config

    source = build_data_source(default_us_pv_universe_data_source_config())
    uni = source.load_column("universe.universe_ticker")
    assert uni.loc[(pd.Timestamp("2024-01-02"), "AAPL")] == "AAPL"
    assert uni.loc[(pd.Timestamp("2024-01-02"), "MSFT")] == "MSFT"


def test_local_ashare_parquet_smoke_if_present():
    """本地有 A 股 COS 镜像时，走真实 datasets.yaml 读一行。"""
    root = Path("/home/shw/quant_projects/data/a_share/lqtp_data/StockDailyBar")
    if not root.exists():
        pytest.skip("local ashare parquet not present")
    from api.mining_integration import default_ashare_pv_data_source_config

    cfg = default_ashare_pv_data_source_config(
        start_date="2016-01-04",
        end_date="2016-01-04",
    )
    source = build_data_source(cfg)
    close = source.load_column("close")
    assert len(close) > 0


def test_local_us_parquet_smoke_if_present():
    root = Path("/home/shw/quant_projects/data/us_stock/massive_data/StockDailyBar")
    if not root.exists():
        pytest.skip("local us parquet not present")
    from api.mining_integration import default_us_pv_data_source_config

    cfg = default_us_pv_data_source_config(
        start_date="2003-09-10",
        end_date="2003-09-10",
    )
    source = build_data_source(cfg)
    close = source.load_column("close")
    assert len(close) > 0


def test_data_access_source_forwards_params_to_store(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeDataset:
        time_column = "ts"
        instrument_column = "inst"

    class _FakeSnapshot:
        snapshot_id = "snap_test"

    class _FakeReadResult:
        table = None
        snapshot = _FakeSnapshot()

    class _FakeStore:
        def get_dataset(self, _name):
            return _FakeDataset()

        def describe_dataset(self, _dataset, **kwargs):
            return _FakeSnapshot()

        def read_result(self, dataset, *, columns, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs
            import pyarrow as pa

            _FakeReadResult.table = pa.table(
                {
                    "ts": [pd.Timestamp("2024-01-02")],
                    "inst": ["AAPL"],
                    "price": [100.0],
                }
            )
            return _FakeReadResult()

    monkeypatch.setattr(
        "storage.data_access_source._get_store",
        lambda: _FakeStore(),
    )
    from storage.data_access_source import DataAccessSource

    src = DataAccessSource(
        dataset="massive_ticks",
        fields={"price": "price"},
        params={"kind": "trades_v1"},
        normalize_timestamp=True,
        timestamp_unit="ns",
    )
    series = src.load_column("price")
    assert series.iloc[0] == pytest.approx(100.0)
    assert captured["dataset"] == "massive_ticks"
    assert captured["kwargs"]["kind"] == "trades_v1"


def test_factory_data_access_kind_shorthand():
    from storage.data_access_source import DataAccessSource

    src = build_data_source(
        {
            "type": "data_access",
            "dataset": "massive_ticks",
            "kind": "quotes_v1",
            "fields": {"bid_price": "bid_price"},
        }
    )
    assert isinstance(src, DataAccessSource)
    assert src.dataset == "massive_ticks"
    assert src.params == {"kind": "quotes_v1"}
