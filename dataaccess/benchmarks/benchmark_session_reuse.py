#!/usr/bin/env python3
"""R30-P0-001 —— 固定 workload：同一 DataReadSession 内重复读同一 dataset。

断言第二次读不再走全量 glob/stat（resolution 缓存命中）：monkeypatch 计数
``store._prepare_dataset_read``，两次读只应触发一次解析。报告 session_reuse。

用法
    python -m data_access.benchmarks.benchmark_session_reuse --scale tiny --out /tmp/bsession
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from data_access.benchmarks.fixtures import build_fixture_store, time_ms
from data_access.benchmarks.report import BenchmarkReport

WORKLOAD = "session reuse: same dataset read twice in one DataReadSession"


def run_workload(store, paths, *, scale="small", repeat=1, out_dir=None) -> BenchmarkReport:
    report = BenchmarkReport("B-session", WORKLOAD, scale)
    ds = "ashare_stock_daily"
    calls = {"n": 0}
    orig = store._prepare_dataset_read

    def _counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    from data_access.read.read_session import DataReadSession

    store._prepare_dataset_read = _counting
    try:
        with DataReadSession(store, request_id="reuse-bench") as session:
            ms1, t1 = time_ms(lambda: session.read(ds, columns=["Close"]).to_arrow())
            ms2, t2 = time_ms(lambda: session.read(ds, columns=["Close"]).to_arrow())
            cache_entries = len(session._resolution_cache)
        inside_resolves = calls["n"]
        # 会话外再读一次：ContextVar cache 已复位 → 重新解析
        ms3, t3 = time_ms(lambda: store.read(ds, columns=["Close"]).to_arrow())
        total_resolves = calls["n"]
    finally:
        store._prepare_dataset_read = orig

    report.record_metric("wall_ms", round(ms1 + ms2, 2))
    report.record_metric("resolution_ms", round(ms1, 2))
    report.record_metric("duckdb_execute_ms", round(ms2, 2))
    reuse = 0.0
    if ms1 > 0:
        reuse = round(max(0.0, 1.0 - inside_resolves / 2.0), 4)  # 2 读 / 1 解析 → 0.5
    report.record_metric("session_reuse", reuse)
    report.record_extra("read1_ms", round(ms1, 2))
    report.record_extra("read2_ms", round(ms2, 2))
    report.record_extra("read3_outside_ms", round(ms3, 2))
    report.record_extra("resolution_calls_inside_session", inside_resolves)
    report.record_extra("resolution_calls_total", total_resolves)
    report.record_extra("cache_entries", cache_entries)
    report.record_extra("reuse_hit_second_read", inside_resolves <= 1 and ms2 <= ms1 + 1e-6)
    report.record_extra("rows_each", t1.num_rows)
    # gate：第二次读不应重新解析
    report.add_gate(
        "resolution_calls_inside_session",
        "2 reads in session resolve exactly once",
        "PASS" if inside_resolves <= 1 else "FAIL",
        None,
    )

    report.finish()
    if out_dir:
        report.write_evidence(out_dir)
    return report


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=WORKLOAD)
    p.add_argument("--scale", choices=["tiny", "small", "full"], default="tiny")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--tmp", type=str, default=None)
    return p


def main(argv=None) -> int:
    args: Any = _arg_parser().parse_args(argv)
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="bsess_"))
    store, paths = build_fixture_store(tmp, scale=args.scale, freq="daily")
    report = run_workload(store, paths, scale=args.scale, repeat=args.repeat, out_dir=args.out)
    print(f"session verdict={report.gate_verdict()} reuse={report.metrics.get('session_reuse')} "
          f"inside_resolves={report.extra.get('resolution_calls_inside_session')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
