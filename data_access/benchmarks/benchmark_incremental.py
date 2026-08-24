#!/usr/bin/env python3
"""R30-P0-001 —— B09 固定 workload：单日增量更新。

写入新一天 → build manifest → 增量读（只读新文件）。记录
incremental_write_ms / incremental_read_ms / rows_added / 物理对象数。

用法
    python -m data_access.benchmarks.benchmark_incremental --scale tiny --out /tmp/b09
"""
from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from data_access.benchmarks.fixtures import ScanProbe, build_fixture_store, time_ms
from data_access.benchmarks.report import BenchmarkReport

WORKLOAD = "B09 single-day incremental update + incremental read"
_DAY = dt.timedelta(days=1)


def _next_trading_day(d: dt.date) -> dt.date:
    while d.weekday() >= 5:  # 跳过周末
        d += _DAY
    return d


def _new_day_table(symbols, day: dt.date, seed: int = 20260811):
    rng = np.random.default_rng(seed)
    n = len(symbols)
    close = rng.lognormal(mean=4.2, sigma=0.7, size=n)
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n))
    hi = np.maximum(open_, close) * (1.0 + rng.uniform(0.002, 0.018, size=n))
    lo = np.minimum(open_, close) * (1.0 - rng.uniform(0.002, 0.018, size=n))
    vol = rng.integers(50_000, 5_000_000, size=n)
    return pa.table(
        {
            "TradeDate": pa.array(np.array([day] * n, dtype="datetime64[D]"), type=pa.date32()),
            "Symbol": pa.array(symbols),
            "Open": pa.array(open_),
            "High": pa.array(hi),
            "Low": pa.array(lo),
            "Close": pa.array(close),
            "Volume": pa.array(vol),
            "Amount": pa.array(vol * close * 100.0),
            "VWAP": pa.array(close * (1.0 + rng.normal(0.0, 0.001, size=n))),
        }
    )


def _all_symbols(store, ds: str):
    import pyarrow.parquet as pq

    d = store._registry.get(ds)
    root = Path(d.root) if hasattr(d, "root") else None
    if root is None:
        return None
    for f in sorted(root.glob("part-*.parquet")):
        tbl = pq.read_table(f, columns=["Symbol"])
        syms = sorted(set(tbl.column("Symbol").to_pylist()))
        if syms:
            return syms
    return None


def _last_date(store, ds: str) -> dt.date:
    t = store.read(ds, columns=["TradeDate"]).to_arrow()
    vals = t.column("TradeDate").to_pylist()
    return max(vals) if vals else dt.date(2024, 12, 31)


def run_workload(store, paths, *, scale="small", repeat=1, out_dir=None) -> BenchmarkReport:
    report = BenchmarkReport("B09", WORKLOAD, scale)
    ds = "ashare_stock_daily"

    # 基线读：最后一年
    d = _last_date(store, ds)
    tr_base = (dt.date(d.year - 1, 1, 1), dt.date(d.year + 1, 1, 1))
    with ScanProbe(store) as probe:
        base_ms, t_base = time_ms(lambda: store.read(ds, time_range=tr_base).to_arrow())
        base_snap = probe.snapshot()
    report.record_extra("baseline_ms", round(base_ms, 2))
    report.record_extra("baseline_rows", t_base.num_rows)

    # 增量写：新一天
    new_day = _next_trading_day(d + _DAY)
    symbols = _all_symbols(store, ds) or [f"{600000 + i:06d}" for i in range(8)]
    new_table = _new_day_table(symbols, new_day)
    write_ms, _res = time_ms(
        lambda: store.write_arrow(ds, new_table, mode="append")
    )
    manifest_ms, _mv = time_ms(lambda: store.build_dataset_manifest(ds))
    report.record_extra("incremental_write_ms", round(write_ms, 2))
    report.record_extra("manifest_build_ms", round(manifest_ms, 2))
    report.record_extra("rows_added", new_table.num_rows)
    report.record_extra("new_day", str(new_day))

    # 增量读：只读新一天
    with ScanProbe(store) as probe2:
        inc_ms, t_inc = time_ms(
            lambda: store.read(ds, time_range=(new_day, new_day + _DAY)).to_arrow()
        )
        inc_snap = probe2.snapshot()
    report.record_metric("wall_ms", round(inc_ms, 2))
    report.record_metric("duckdb_execute_ms", round(inc_ms, 2))
    report.record_metric("physical_scan_count", inc_snap["physical_scan_count"])
    report.record_metric("bytes_scanned", inc_snap["bytes_scanned"])
    report.record_metric("physical_object_count", len(set(_paths_read(store, ds, new_day))))
    report.record_extra("incremental_read_ms", round(inc_ms, 2))
    report.record_extra("incremental_rows", t_inc.num_rows)
    report.record_extra("incremental_bytes", t_inc.nbytes)
    mv = store.manifest_version(ds)
    report.record_extra("manifest_fresh", bool(mv.get("fresh")))
    report.record_extra("manifest_file_count", mv.get("file_count"))

    report.compute_throughput(t_inc.num_rows, t_inc.nbytes)
    report.finish()
    if out_dir:
        report.write_evidence(out_dir)
    return report


def _paths_read(store, ds: str, day: dt.date):
    """增量读实际命中的物理文件（通过 manifest 裁剪路径）。"""
    d = store._registry.get(ds)
    paths = store._prepare_dataset_read(
        d, time_range=(day, day + _DAY), params={}
    )
    return paths


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=WORKLOAD)
    p.add_argument("--scale", choices=["tiny", "small", "full"], default="tiny")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--tmp", type=str, default=None)
    return p


def main(argv=None) -> int:
    args: Any = _arg_parser().parse_args(argv)
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="b09_"))
    store, paths = build_fixture_store(tmp, scale=args.scale, freq="daily")
    report = run_workload(store, paths, scale=args.scale, repeat=args.repeat, out_dir=args.out)
    print(f"B09 verdict={report.gate_verdict()} incremental_ms={report.extra.get('incremental_read_ms')} "
          f"rows={report.extra.get('incremental_rows')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
