"""
并发读测试：多线程同时读同一个 store，结果必须稳定、不报错、数据不串。

WHY：AI 方案里提到「线程/连接模型要实测」，这就是实测。本测试锁定我们选的
方案（进程单 conn + per-call cursor）在典型并发量下可用。
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from data_access import reset_store
from data_access.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture
def concurrent_store(tmp_path, monkeypatch):
    """搭一个有足量数据的 store（让并发跑得有点意义）。"""
    data_root = tmp_path / "data"
    # 造 100 个 ticker × 30 天 = 3000 行
    tickers = [f"SYM{i:03d}" for i in range(100)]
    rows = []
    for day in range(1, 31):
        for t in tickers:
            rows.append({
                "align_time": pd.Timestamp(f"2024-01-{day:02d} 05:00:00"),
                "ticker": t,
                "close": 100.0 + day + hash(t) % 100 * 0.01,
                "volume": 1000000 + hash(t) % 10000,
            })
    data_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(data_root / "data.parquet")

    monkeypatch.setenv("TEST_ROOT", str(data_root))
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(dedent("""
        concurrent_data:
          kind: static
          access_mode: published
          layout: plain
          root: ${TEST_ROOT}
          time_column: align_time
          instrument_column: ticker
    """).strip() + "\n", encoding="utf-8")

    reset_store()
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=4)
    store = DataAccessStore(registry=reg, engine=engine)
    yield store
    engine.close()
    reset_store()


def test_parallel_read_same_dataset(concurrent_store):
    """16 个线程同时读，行数必须稳定，不报错。"""
    def read_all():
        tbl = concurrent_store.read_arrow("concurrent_data", columns=["ticker", "close"])
        return tbl.num_rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(read_all) for _ in range(32)]
        results = [f.result(timeout=30) for f in futures]

    # 每次都是完整 3000 行
    assert all(r == 3000 for r in results), f"行数不稳定: {set(results)}"


def test_parallel_read_with_different_filters(concurrent_store):
    """不同线程用不同 instrument_filter，结果必须各自独立（不互相污染）。"""
    def read_for(symbols):
        tbl = concurrent_store.read_arrow(
            "concurrent_data",
            columns=["ticker", "close"],
            instrument_filter=symbols,
        )
        tickers_in_result = set(tbl.column("ticker").to_pylist())
        return symbols, tickers_in_result, tbl.num_rows

    tasks = [
        ["SYM000", "SYM001"],
        ["SYM050"],
        ["SYM099", "SYM001"],
        ["SYM010", "SYM020", "SYM030"],
    ] * 4  # 跑 16 个

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(read_for, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures, timeout=30):
            symbols, got_tickers, n_rows = f.result()
            # 结果里只能有请求的 tickers
            assert got_tickers.issubset(set(symbols)), (
                f"过滤失效：请求 {symbols}，结果含 {got_tickers}"
            )
            assert n_rows == len(symbols) * 30  # 每个 symbol 30 天


def test_parallel_load_columns(concurrent_store):
    """load_columns 是 PR1 主改造点，也要确保并发安全。"""
    def call():
        cols = concurrent_store.load_columns(
            "concurrent_data",
            columns=["close", "volume"],
            instrument_filter=["SYM001", "SYM002"],
        )
        return len(cols["close"]), cols["close"].name

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result(timeout=30) for f in [pool.submit(call) for _ in range(16)]]

    lengths = {r[0] for r in results}
    assert len(lengths) == 1, f"长度不稳定: {lengths}"
    assert all(r[1] == "close" for r in results)
