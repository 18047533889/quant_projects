#!/usr/bin/env python3
"""
data_access 读路径基准（Benchmark Gate 骨架）。

Gate：DataAccess 不允许明显慢于该 workload 的最佳基础 engine。

用法
    python benchmarks/bench_read.py --rows 5_000_000 --cols 8 --repeat 3
    python benchmarks/bench_read.py --rows 50_000_000 --cols 32 --repeat 2

覆盖引擎
    pd.read_parquet / pyarrow / polars / duckdb / data_access.read_arrow

指标
    wall time / rows/sec / GB/sec；输出 JSON 与对比表。
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def gen_data(rows: int, cols: int, tmp: Path) -> Path:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=rows, freq="s"),
            "symbol": rng.choice(["AAPL", "MSFT", "GOOG", "TSLA", "NVDA"], rows),
        }
    )
    for i in range(cols):
        df[f"col{i}"] = rng.standard_normal(rows)
    path = tmp / "bench.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))
    return path


def bench(fn, repeat: int) -> tuple[float, float, float]:
    """返回 (best_wall_ms, rows_per_sec, gb_per_sec)。"""
    best = float("inf")
    rows = cols = 0.0
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        dt = (time.perf_counter() - t0) * 1000
        best = min(best, dt)
        if isinstance(result, pa.Table):
            rows = result.num_rows
            cols = result.nbytes
        elif result is not None:
            rows = len(result)
    return best, (rows / (best / 1000.0) if best > 0 else 0.0), (cols / (best / 1000.0) / 1e9 if best > 0 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5_000_000)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp())
    path = gen_data(args.rows, args.cols, tmp)
    # read_uri 走 dev 白名单：把临时目录登记进去
    import os

    os.environ["DATA_ACCESS_READ_URI_ROOTS"] = str(tmp)

    results: dict[str, dict] = {}
    import pyarrow.parquet as pq

    def _pd():
        return pd.read_parquet(path)

    def _pa():
        return pq.read_table(path)

    def _polars():
        import polars as pl

        return pl.scan_parquet(path).collect().to_arrow()

    def _duckdb():
        import duckdb

        return duckdb.connect(":memory:").execute(f"SELECT * FROM read_parquet('{path}')").fetch_arrow_table()

    def _data_access():
        from data_access import get_store

        # 不做 reset：schema 校验/缓存生效后的真实生产读路径（warm read）
        return get_store().read_uri(path).to_arrow()

    for name, fn in (
        ("pd.read_parquet", _pd),
        ("pyarrow", _pa),
        ("polars", _polars),
        ("duckdb", _duckdb),
        ("data_access", _data_access),
    ):
        try:
            wall, rps, gbps = bench(fn, args.repeat)
            results[name] = {"best_ms": round(wall, 2), "rows_per_sec": int(rps), "gb_per_sec": round(gbps, 2)}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": str(exc)}

    # Gate：data_access 默认引擎是 DuckDB，应对比 duckdb（而非最快的 pyarrow）。
    # 治理层开销不允许让 data_access 明显慢于它所用的底层引擎。
    best = min(v["best_ms"] for v in results.values() if "best_ms" in v)
    gate_base = results.get("duckdb", {}).get("best_ms", best)
    da = results.get("data_access", {}).get("best_ms")
    print(json.dumps(results, indent=2))
    if da is not None:
        ratio = da / gate_base if gate_base else float("inf")
        print(
            f"\nGate ratio data_access/duckdb = {ratio:.2f} "
            f"(data_access={da:.0f}ms, duckdb={gate_base:.0f}ms, fastest={best:.0f}ms)"
        )
        if ratio > 1.5:
            print("⚠️  data_access 慢于 DuckDB 1.5x，治理层开销需优化")
        else:
            print("✅  Gate 通过（相对默认引擎 ≤1.5x）")


if __name__ == "__main__":
    main()
