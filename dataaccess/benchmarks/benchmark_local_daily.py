#!/usr/bin/env python3
"""R30-P0-001 —— B01 固定 workload：10y daily panel 全量 + 裁剪读。

固定可复现：fixtures 用确定性 seed 生成 3000-5000 只 × 10 年日频
(OHLCV / Amount / VWAP)。测全量读 + 裁剪（time_range + instrument_filter），
并对比直接 DuckDB 后端（governance 开销 gate：warm throughput >= 90%）。

用法
    python -m data_access.benchmarks.benchmark_local_daily --scale small --repeat 3 --out /tmp/b01
    python -m data_access.benchmarks.benchmark_local_daily --scale full  --repeat 3 --out /tmp/b01
"""
from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path
from typing import Any

from data_access.benchmarks.fixtures import (
    ScanProbe,
    _DAILY_COLUMNS,
    build_fixture_store,
    time_ms,
)
from data_access.benchmarks.report import BenchmarkReport

WORKLOAD = "B01 10y daily panel: full + pruned"


def _direct_backend(paths, columns=None, best_of: int = 3):
    """直接 DuckDB 读同一组 parquet（无 data_access 治理层）。

    best-of-N + 1 次 warmup，消除冷连接/footer 解析方差。
    """
    import duckdb

    files = sorted(Path(paths["ashare_stock_daily"]).glob("part-*.parquet"))
    if not files:
        return None
    sql_files = ", ".join(f"'{f}'" for f in files)
    conn = duckdb.connect(":memory:")
    sql = f"SELECT * FROM read_parquet([{sql_files}])"

    def _run():
        rel = conn.execute(sql)
        fetch = getattr(rel, "to_arrow_table", None) or getattr(rel, "fetch_arrow_table")
        return fetch()

    _run()  # warmup
    best = float("inf")
    tbl = None
    for _ in range(max(1, best_of)):
        ms, t = time_ms(_run)
        if ms < best:
            best, tbl = ms, t
    conn.close()
    return best, tbl


def run_workload(store, paths, *, scale="small", repeat=1, out_dir=None) -> BenchmarkReport:
    report = BenchmarkReport("B01", WORKLOAD, scale)
    ds = "ashare_stock_daily"

    # ---- 直接后端基线（governance gate 参照）----
    direct = _direct_backend(paths)
    if direct is not None:
        direct_ms, direct_tbl = direct
        report.record_extra("direct_backend_ms", round(direct_ms, 2))
        report.record_extra(
            "direct_backend_rows_per_sec",
            round(direct_tbl.num_rows / (direct_ms / 1000.0), 1) if direct_ms > 0 else None,
        )

    # ---- 全量读（所有 OHLCV/Amount/VWAP 列）----
    prepare_ms, prepared = time_ms(
        lambda: store.prepare_read(ds, columns=list(_DAILY_COLUMNS))
    )
    report.record_metric("resolution_ms", round(prepare_ms, 2))
    report.record_metric("physical_object_count", len(prepared.physical_scope))
    report.record_extra("column_count", len(_DAILY_COLUMNS))

    best_ms = float("inf")
    best_table = None
    with ScanProbe(store) as probe:
        for i in range(max(1, repeat)):
            if i == 0:
                report.mark("content")
            ms, result = time_ms(lambda: store.execute_prepared_read(prepared))
            if i == 0:
                report.mark("data")
            if ms < best_ms:
                best_ms = ms
                best_table = result.table
        probe_snap = probe.snapshot()

    report.record_metric("wall_ms", best_ms)
    report.record_metric("duckdb_execute_ms", round(best_ms, 2))
    report.compute_throughput(best_table.num_rows, best_table.nbytes)
    for k, v in probe_snap.items():
        report.record_metric(k, v)
    report.record_extra("rows_read_full", best_table.num_rows)
    report.record_extra("cols_read_full", best_table.num_columns)

    # ---- warm throughput + gate vs direct backend（best-of-3，冷热一致）----
    warm_ms, warm_result = _best_of_n(
        lambda: store.execute_prepared_read(prepared), n=3
    )
    report.record_metric(
        "warm_throughput",
        round(warm_result.table.num_rows / (warm_ms / 1000.0), 1) if warm_ms > 0 else None,
    )
    if direct is not None and warm_ms > 0 and direct_ms > 0:
        da_rate = warm_result.table.num_rows / (warm_ms / 1000.0)
        direct_rate = direct_tbl.num_rows / (direct_ms / 1000.0)
        ratio = da_rate / direct_rate if direct_rate > 0 else 0.0
        report.record_extra("warm_direct_ratio", round(ratio, 4))
        # tiny 只是冒烟测试：固定治理开销在微秒级读上占比大，gate 只在真实规模判定。
        if scale != "tiny":
            report.add_gate(
                "warm_throughput",
                "warm throughput >= 90% of direct backend",
                "PASS" if ratio >= 0.9 else "FAIL",
                round(ratio - 1.0, 4),
            )

    # ---- 裁剪读：最后一年 × 前 200 只标的 ----
    dmax = _last_date(store, ds)
    tr = (dt.date(dmax.year, 1, 1), dt.date(dmax.year + 1, 1, 1))
    instruments = _first_symbols(store, 200)
    prepared_p = store.prepare_read(ds, time_range=tr, instrument_filter=instruments)
    report.record_extra("pruned_object_count", len(prepared_p.physical_scope))
    pruned_ms, result_p = time_ms(lambda: store.execute_prepared_read(prepared_p))
    report.record_extra("pruned_ms", round(pruned_ms, 2))
    report.record_extra("pruned_rows", result_p.table.num_rows)
    report.record_extra("pruned_bytes", result_p.table.nbytes)

    report.finish()
    if out_dir:
        report.write_evidence(out_dir)
    return report


def _best_of_n(fn, n: int = 3):
    """执行 fn n 次，返回 (best_ms, best_result)。"""
    best_ms = float("inf")
    best_result = None
    for _ in range(max(1, n)):
        ms, result = time_ms(fn)
        if ms < best_ms:
            best_ms, best_result = ms, result
    return best_ms, best_result


def _last_date(store, ds: str) -> dt.date:
    """读 TradeDate 列的最大值（本地 benchmark，量级可接受）。"""
    t = store.read(ds, columns=["TradeDate"]).to_arrow()
    vals = t.column("TradeDate").to_pylist()
    if not vals:
        return dt.date(2024, 12, 31)
    return max(vals)


def _first_symbols(store, n: int):
    import pyarrow.parquet as pq

    ds = store._registry.get("ashare_stock_daily")
    root = Path(ds.root) if hasattr(ds, "root") else None
    if root is None:
        return None
    for f in sorted(root.glob("part-*.parquet")):
        tbl = pq.read_table(f, columns=["Symbol"])
        syms = sorted(set(tbl.column("Symbol").to_pylist()))
        if len(syms) >= n:
            return syms[:n]
    return None


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=WORKLOAD)
    p.add_argument("--scale", choices=["tiny", "small", "full"], default="small")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--tmp", type=str, default=None)
    return p


def main(argv=None) -> int:
    args: Any = _arg_parser().parse_args(argv)
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="b01_"))
    store, paths = build_fixture_store(tmp, scale=args.scale, freq="daily")
    report = run_workload(store, paths, scale=args.scale, repeat=args.repeat, out_dir=args.out)
    print(
        f"B01 verdict={report.gate_verdict()} wall_ms={report.metrics.get('wall_ms')} "
        f"rows/s={report.metrics.get('rows_per_sec')} "
        f"warm/direct_ratio={report.extra.get('warm_direct_ratio')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
