#!/usr/bin/env python3
"""R30-P0-001 —— B02 固定 workload：分钟数据读（单日/单月/一年）+ 分钟→日聚合。

固定可复现：fixtures 确定性生成 A 股分钟数据（UTC 时间戳，
market='ashare' 聚合转北京时段）。时间窗用半开区间 [start, end)。

用法
    python -m data_access.benchmarks.benchmark_local_minute --scale small --repeat 3 --out /tmp/b02
"""
from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path
from typing import Any

from data_access.benchmarks.fixtures import ScanProbe, build_fixture_store, time_ms
from data_access.benchmarks.report import BenchmarkReport

WORKLOAD = "B02 minute read (single-day / month / year) + minute→daily aggregate"

_DAY = dt.timedelta(days=1)


def _minute_bounds(store, ds: str) -> tuple[dt.date, dt.date]:
    """从 parquet footer 拿 datetime 列 min/max（metadata-only，不读数据）。"""
    import pyarrow.parquet as pq

    d = store._registry.get(ds)
    root = Path(d.root) if hasattr(d, "root") else None
    if root is None:
        t = store.read(ds, columns=["datetime"]).to_arrow()
        vals = t.column("datetime").to_pylist()
        return (min(vals).date(), max(vals).date())
    lo = hi = None
    for f in sorted(root.glob("part-*.parquet")):
        meta = pq.read_metadata(f)
        for rg in range(meta.num_row_groups):
            for col in range(meta.num_columns):
                cname = meta.row_group(rg).column(col).path_in_schema
                if cname == "datetime":
                    st = meta.row_group(rg).column(col).statistics
                    if st is not None and st.has_min_max:
                        lo = st.min if lo is None or st.min < lo else lo
                        hi = st.max if hi is None or st.max > hi else hi
    if lo is None:
        return (dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    return (lo.date(), hi.date())


def _first_symbol(store, ds: str):
    import pyarrow.parquet as pq

    d = store._registry.get(ds)
    root = Path(d.root) if hasattr(d, "root") else None
    if root is None:
        return None
    for f in sorted(root.glob("part-*.parquet")):
        tbl = pq.read_table(f, columns=["Symbol"])
        syms = sorted(set(tbl.column("Symbol").to_pylist()))
        if syms:
            return syms[0]
    return None


def run_workload(store, paths, *, scale="small", repeat=1, out_dir=None) -> BenchmarkReport:
    report = BenchmarkReport("B02", WORKLOAD, scale)
    ds = "ashare_stock_minute"
    sym = _first_symbol(store, ds)
    dmin, dmax = _minute_bounds(store, ds)

    # ---- 单日 ----
    tr_day = (dmax, dmax + _DAY)
    best = float("inf")
    rows = bytes_ = 0
    with ScanProbe(store) as probe:
        for i in range(max(1, repeat)):
            if i == 0:
                report.mark("content")
            ms, t = time_ms(lambda: store.read(ds, time_range=tr_day, instrument_filter=[sym]).to_arrow())
            if i == 0:
                report.mark("data")
            if ms < best:
                best, rows, bytes_ = ms, t.num_rows, t.nbytes
    report.record_metric("wall_ms", best)
    report.record_metric("resolution_ms", None)
    report.record_metric("duckdb_execute_ms", round(best, 2))
    report.compute_throughput(rows, bytes_)
    for k, v in probe.snapshot().items():
        report.record_metric(k, v)
    report.record_extra("window", f"single_day:{tr_day[0]}")
    report.record_extra("day_rows", rows)

    # ---- 单月（覆盖 dmax 所在月）----
    month_start = dt.date(dmax.year, dmax.month, 1)
    next_month = dt.date(dmax.year + (1 if dmax.month == 12 else 0), (dmax.month % 12) + 1, 1)
    ms, t = time_ms(lambda: store.read(ds, time_range=(month_start, next_month)).to_arrow())
    report.record_extra("month_ms", round(ms, 2))
    report.record_extra("month_rows", t.num_rows)

    # ---- 一年（覆盖 dmax 所在年）----
    year_start = dt.date(dmax.year, 1, 1)
    year_end = dt.date(dmax.year + 1, 1, 1)
    ms, t = time_ms(lambda: store.read(ds, time_range=(year_start, year_end)).to_arrow())
    report.record_extra("year_ms", round(ms, 2))
    report.record_extra("year_rows", t.num_rows)

    # ---- 分钟→日聚合 ----
    from data_access.read.aggregation import AggregationSpec, aggregate_minute_to_daily

    spec = AggregationSpec(
        aggregation="minute_range", start="09:30", end="11:30", metric="sum", market="ashare"
    )
    try:
        agg_ms, handle = time_ms(
            lambda: aggregate_minute_to_daily(
                store, ds, "volume", spec,
                time_range=(year_start, year_end),
                instrument_filter=[sym] if sym else None,
            )
        )
        t_agg = handle.to_arrow()
        report.record_extra("agg_ms", round(agg_ms, 2))
        report.record_extra("agg_rows", t_agg.num_rows)
        report.record_extra("agg_sample", t_agg.to_pylist()[:1])
    except Exception as exc:  # noqa: BLE001 —— 聚合失败不假通过，但记录详情
        report.record_extra("agg_error", str(exc))

    report.finish()
    if out_dir:
        report.write_evidence(out_dir)
    return report


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=WORKLOAD)
    p.add_argument("--scale", choices=["tiny", "small", "full"], default="small")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--tmp", type=str, default=None)
    return p


def main(argv=None) -> int:
    args: Any = _arg_parser().parse_args(argv)
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="b02_"))
    store, paths = build_fixture_store(tmp, scale=args.scale, freq="minute")
    report = run_workload(store, paths, scale=args.scale, repeat=args.repeat, out_dir=args.out)
    print(f"B02 verdict={report.gate_verdict()} wall_ms={report.metrics.get('wall_ms')} "
          f"rows/s={report.metrics.get('rows_per_sec')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
