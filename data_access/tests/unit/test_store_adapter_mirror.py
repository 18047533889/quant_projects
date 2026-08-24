"""store 双市场 adapter 与 COS 镜像覆盖测试。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_access import reset_store
from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.store import DataAccessStore, adapter_options_for_dataset


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    reset_store()


def test_adapter_options_normalize_for_date_and_timestamp(tmp_path):
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        """
ashare:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: /tmp/a
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string

us:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: /tmp/u
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Ticker
  schema:
    TradeDate: timestamp
    Ticker: string
""".strip()
        + "\n",
        encoding="utf-8",
    )
    reg = load_registry(cfg)
    # date 时间列：归一化到日（A 股日频）
    assert adapter_options_for_dataset(reg.get("ashare"))["normalize_timestamp"] is True
    # timestamp 时间列：保留完整精度（分钟/逐笔等多条同日内记录），
    # 不设置 normalize_timestamp（否则会折叠成每 (日期, 标的) 一行）
    assert "normalize_timestamp" not in adapter_options_for_dataset(reg.get("us"))


def test_scan_polars_triggers_cos_mirror(monkeypatch, tmp_path):
    root = tmp_path / "us" / "StockDailyBar"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02"]),
            "Ticker": ["AAPL"],
            "Close": [1.0],
        }
    ).to_parquet(root / "2024-01-02.parquet")

    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
us_stock_daily:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {tmp_path / "us" / "StockDailyBar"}
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Ticker
""".strip()
        + "\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def _fake_mirror(ds, *, time_range=None):
        calls.append(ds.name)

    monkeypatch.setattr(
        "data_access.cos.mirror.ensure_local_mirror_for_dataset",
        _fake_mirror,
    )

    pytest.importorskip("polars")
    store = DataAccessStore(load_registry(cfg), DuckDBEngine())
    lf = store.scan_polars("us_stock_daily", columns=["Close"])
    # scan_polars 必须触发 COS 镜像。mirror 会被主路径解析调用一次，并可能被
    # prepare_read 内 _estimate_read_memory 的成本估算路径（estimate_scan_cost
    # 内部 load_manifest / dataset_read_stats 各自 _resolve_raw_paths）再次触发
    # ——这是幂等重入（文件已就绪即 no-op），不是缺失。验证目标是「被触发」，
    # 而不是计数恰好为 1。
    assert calls and all(c == "us_stock_daily" for c in calls)
    df = lf.collect()
    assert df.height == 1
