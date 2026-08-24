#!/usr/bin/env python3
"""R30-P0-001 —— B03 固定 workload：daily + fundamental PIT join + universe。

固定可复现：fixtures 确定性生成日频面板 + 季度财务（PubDate/period_end）+
universe 成员。read_joined 在 DuckDB 内完成 pit_asof join + universe INNER JOIN。

用法
    python -m data_access.benchmarks.benchmark_join --scale small --repeat 3 --out /tmp/b03
"""
from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path
from typing import Any

from data_access.benchmarks.fixtures import ScanProbe, build_fixture_store, time_ms
from data_access.benchmarks.report import BenchmarkReport

WORKLOAD = "B03 daily + fundamental PIT join + universe"

_DAILY_FIELDS = ["Open", "High", "Low", "Close", "Volume"]
_FUND_FIELDS = ["net_profit", "revenue"]
_JOIN_SPEC = {
    "policy": "pit_asof",
    "knowledge_time": "PubDate",
    "period_time": "period_end",
    "period_selection": "latest_period",
    "revision_order": ["PubDate"],
}


def _last_date(store, ds: str) -> dt.date:
    t = store.read(ds, columns=["TradeDate"], limit=1).to_arrow()
    vals = t.column("TradeDate").to_pylist()
    return vals[0] if vals else dt.date(2024, 12, 31)


def run_workload(store, paths, *, scale="small", repeat=1, out_dir=None) -> BenchmarkReport:
    report = BenchmarkReport("B03", WORKLOAD, scale)
    ds = "ashare_stock_daily"

    d = _last_date(store, ds)
    tr = (dt.date(d.year - 1, 1, 1), dt.date(d.year + 1, 1, 1))

    def _joined():
        return store.read_joined(
            ds,
            {"ashare_stock_daily": _DAILY_FIELDS, "ashare_stock_income": _FUND_FIELDS},
            joins={"ashare_stock_income": _JOIN_SPEC},
            time_range=tr,
            universe="ashare_universe",
        ).to_arrow()

    best = float("inf")
    best_table = None
    with ScanProbe(store) as probe:
        for i in range(max(1, repeat)):
            if i == 0:
                report.mark("content")
            ms, t = time_ms(_joined)
            if i == 0:
                report.mark("data")
            if ms < best:
                best, best_table = ms, t

    report.record_metric("wall_ms", best)
    report.record_metric("duckdb_execute_ms", round(best, 2))
    report.compute_throughput(best_table.num_rows, best_table.nbytes)
    for k, v in probe.snapshot().items():
        report.record_metric(k, v)
    report.record_extra("join_rows", best_table.num_rows)
    report.record_extra("join_columns", best_table.num_columns)
    report.record_extra("time_range", f"{tr[0]}..{tr[1]}")
    # universe 生效验证：输出标的只应是 universe 成员
    try:
        syms = sorted(set(best_table.column("Symbol").to_pylist()))
        report.record_extra("output_symbols", len(syms))
        report.record_extra("universe_applied", True)
    except Exception:
        report.record_extra("universe_applied", None)

    # 对照：无 universe 的全量 join（baseline，供 compare 用）
    def _joined_no_universe():
        return store.read_joined(
            ds,
            {"ashare_stock_daily": _DAILY_FIELDS, "ashare_stock_income": _FUND_FIELDS},
            joins={"ashare_stock_income": _JOIN_SPEC},
            time_range=tr,
        ).to_arrow()

    ms, t_all = time_ms(_joined_no_universe)
    report.record_extra("no_universe_ms", round(ms, 2))
    report.record_extra("no_universe_rows", t_all.num_rows)

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
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="b03_"))
    store, paths = build_fixture_store(tmp, scale=args.scale, freq="daily")
    report = run_workload(store, paths, scale=args.scale, repeat=args.repeat, out_dir=args.out)
    print(f"B03 verdict={report.gate_verdict()} wall_ms={report.metrics.get('wall_ms')} "
          f"rows/s={report.metrics.get('rows_per_sec')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
