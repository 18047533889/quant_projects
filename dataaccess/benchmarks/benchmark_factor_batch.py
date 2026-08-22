#!/usr/bin/env python3
"""R30-P0-001 —— B04/B05/B06 固定 workload：因子批量读（shared/mixed source）。

构造 N 个因子请求共享少量源列（ashare_stock_daily 列子集），通过
``DataReadSession`` 批量读——resolution 缓存命中后跳过重复 glob/stat。
记录 factor_count / physical_scan_count / source_group_count / session_reuse。

    B04 = 100 因子     B05 = 1000 因子     B06 = 10000 因子（full scale）
    small scale 自动降档；tiny 仅供测试。

用法
    python -m data_access.benchmarks.benchmark_factor_batch --scale small --out /tmp/b04b05b06
"""
from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path
from typing import Any

from data_access.benchmarks.fixtures import ScanProbe, build_fixture_store, time_ms
from data_access.benchmarks.report import BenchmarkReport

WORKLOAD = "B04/B05/B06 factor batch reads (shared source columns)"

_COL_POOL = (
    ("Open", "High", "Low", "Close"),
    ("Close", "Volume"),
    ("Volume", "Amount", "VWAP"),
    ("Open", "Close", "Volume"),
    ("High", "Low"),
    ("Close", "VWAP"),
)

# 每 17 个因子换一个 universe（instrument_filter 不同 → 单独 resolution 项）
_UNIVERSE_STRIDE = 17

_FACTOR_COUNTS = {
    "tiny": {"B04": 20, "B05": 30, "B06": 40},
    "small": {"B04": 100, "B05": 300, "B06": 600},
    "full": {"B04": 100, "B05": 1000, "B06": 10000},
}


def _last_year_range(store, ds: str) -> tuple[dt.date, dt.date]:
    t = store.read(ds, columns=["TradeDate"], limit=1).to_arrow()
    vals = t.column("TradeDate").to_pylist()
    d = vals[0] if vals else dt.date(2024, 12, 31)
    return (dt.date(d.year - 1, 1, 1), dt.date(d.year + 1, 1, 1))


def _all_symbols(store, ds: str, cap: int = 50):
    import pyarrow.parquet as pq

    d = store._registry.get(ds)
    root = Path(d.root) if hasattr(d, "root") else None
    if root is None:
        return None
    for f in sorted(root.glob("part-*.parquet")):
        tbl = pq.read_table(f, columns=["Symbol"])
        syms = sorted(set(tbl.column("Symbol").to_pylist()))
        if syms:
            return syms[:cap]
    return None


def _source_groups(factor_count: int) -> list[tuple[str, ...]]:
    return [_COL_POOL[i % len(_COL_POOL)] for i in range(factor_count)]


def run_factor_batch(
    store,
    paths,
    *,
    factor_count: int,
    scale: str = "small",
    label: str = "B04",
    out_dir=None,
) -> BenchmarkReport:
    """跑一个因子批量 workload（B04/B05/B06 之一）。"""
    report = BenchmarkReport(label, f"{label} {factor_count} factors shared source", scale)
    ds = "ashare_stock_daily"
    tr = _last_year_range(store, ds)
    subset = _all_symbols(store, ds, cap=50)
    groups = _source_groups(factor_count)
    source_group_count = len({frozenset(g) for g in groups})

    from data_access.read.read_session import DataReadSession

    with ScanProbe(store) as probe, DataReadSession(store, request_id=f"{label}-bench") as session:
        t0 = _now_ms()
        for i, cols in enumerate(groups):
            kw: dict[str, Any] = dict(dataset=ds, columns=list(cols), time_range=tr)
            if i % _UNIVERSE_STRIDE == 0 and subset is not None:
                kw["instrument_filter"] = subset
            session.read(**kw).to_arrow()
        batch_ms = _now_ms() - t0
        cache_entries = len(session._resolution_cache)
        probe_snap = probe.snapshot()

    report.record_metric("wall_ms", round(batch_ms, 2))
    report.record_metric("physical_scan_count", probe_snap["physical_scan_count"])
    report.record_metric("bytes_scanned", probe_snap["bytes_scanned"])
    report.record_metric("physical_object_count", None)
    report.record_metric("duckdb_execute_ms", None)
    report.record_extra("factor_count", factor_count)
    report.record_extra("source_group_count", source_group_count)
    report.record_extra("resolution_cache_entries", cache_entries)
    reuse = 0.0
    if factor_count > 0:
        reuse = round(1.0 - cache_entries / factor_count, 4)
    report.record_metric("session_reuse", reuse)
    report.record_metric("rows_per_sec", None)
    report.record_metric("gb_per_sec", None)

    report.finish()
    if out_dir:
        report.write_evidence(out_dir)
    return report


def _now_ms() -> float:
    import time

    return time.perf_counter() * 1000.0


def run_workload(store, paths, *, scale="small", repeat=1, out_dir=None):
    """B04/B05/B06 三个 workload；返回 {label: report}。"""
    counts = _FACTOR_COUNTS.get(scale, _FACTOR_COUNTS["small"])
    reports = {}
    base_out = Path(out_dir) if out_dir else None
    for label, n in counts.items():
        sub = (base_out / label) if base_out else None
        reports[label] = run_factor_batch(
            store, paths, factor_count=n, scale=scale, label=label, out_dir=sub
        )
    return reports


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=WORKLOAD)
    p.add_argument("--scale", choices=["tiny", "small", "full"], default="small")
    p.add_argument("--factors", type=int, default=None, help="覆盖因子数（默认按 scale）")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--tmp", type=str, default=None)
    return p


def main(argv=None) -> int:
    args: Any = _arg_parser().parse_args(argv)
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="b04_"))
    store, paths = build_fixture_store(tmp, scale=args.scale, freq="daily")
    if args.factors is not None:
        report = run_factor_batch(
            store, paths, factor_count=args.factors, scale=args.scale,
            label="B04", out_dir=args.out,
        )
        reports = {"B04": report}
    else:
        reports = run_workload(store, paths, scale=args.scale, out_dir=args.out)
    for label, r in reports.items():
        print(f"{label} verdict={r.gate_verdict()} factors={r.extra.get('factor_count')} "
              f"scan_count={r.metrics.get('physical_scan_count')} reuse={r.metrics.get('session_reuse')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
