#!/usr/bin/env python3
"""Factor 值存储布局 benchmark —— 布局 A（禁止）vs 布局 B（生产）。

对应平台总规范 §21（COS 数据湖布局）、§54（性能与规模测试）、§55（"每个日期
一个 factor 文件 → object explosion → bucket/partition"）。

对比两种布局在 10k / 50k / 100k 因子规模下的对象数 / PUT / LIST / 读延迟 /
字节扫描 / DuckDB 规划延迟 / compaction 成本：

    (A) per-factor-per-day 小对象布局（**禁止**）：每因子 × 每天一个小 parquet。
        100k × 250 天 = 25M 对象/年（评估再翻倍）→ 必然超对象数预算。
    (B) 日频分片 delta（factor_bucket 1024 分片 + year/month 分片 + compaction）：
        对象数被 1024 × 12 × 2 ≈ 24.6k/年 封顶，与因子数无关。

用 data_access.read.object_store.LocalObjectStore 提供真实对象数语义；用
**采样行数 / 采样天数** 让 benchmark 在合理时间内跑完，再按采样比例把指标
外推到全规模（scale the metrics, don't wait for 25M real objects）。

用法
    python -m data_access.benchmarks.benchmark_factor_storage_layout \
        --scale small --out /tmp/fe_bench/ev
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Any

from data_access.benchmarks.factor_storage_layout import (
    DEFAULT_BUCKET_COUNT,
    check_object_count_budget,
    compaction_decision,
    layout_a_object_count,
    layout_b_object_count,
    partition_key,
    shard_key,
    target_parquet_size,
)
from data_access.benchmarks.report import BenchmarkReport
from data_access.read.object_store import LocalObjectStore

WORKLOAD = "Factor storage layout: (A) per-factor-per-day vs (B) sharded delta+compaction"

# 采样规模：真实行数 ~5000 标的/日、~250 天/年。采样让 benchmark 可跑。
_SAMPLE_INSTRUMENTS = 500  # 采样 500 标的/日（真实 5000 的 1/10）
_SAMPLE_DAYS = 25  # 采样 25 天（真实 250 的 1/10）
_SAMPLE_FACTORS = 2000  # 采样 2000 因子（真实 100k 的 1/50）

# 外推比例：采样 → 全规模
_INSTRUMENT_SCALE = 5000 / _SAMPLE_INSTRUMENTS  # 10
_DAY_SCALE = 250 / _SAMPLE_DAYS  # 10
_FACTOR_SCALE = 100_000 / _SAMPLE_FACTORS  # 50


def _put_all(store: LocalObjectStore, keys: list[str], data: bytes) -> float:
    """批量 PUT，返回总耗时 ms。"""
    t0 = time.perf_counter()
    for k in keys:
        store.put_object(k, data)
    return (time.perf_counter() - t0) * 1000.0


def _list_all(store: LocalObjectStore, prefix: str) -> tuple[float, int]:
    """LIST 一个 prefix，返回 (耗时 ms, 对象数)。"""
    t0 = time.perf_counter()
    objs = store.list_objects(prefix)
    return (time.perf_counter() - t0) * 1000.0, len(objs)


def _read_all(store: LocalObjectStore, keys: list[str]) -> tuple[float, int]:
    """顺序读全部对象（模拟全量回读），返回 (耗时 ms, 字节数)。"""
    t0 = time.perf_counter()
    total = 0
    for k in keys:
        r = store.open_reader(k)
        if r is not None:
            total += len(r.read())
            r.close()
    return (time.perf_counter() - t0) * 1000.0, total


def _duckdb_plan_ms(keys: list[str]) -> float:
    """DuckDB 对 N 个 parquet 的规划延迟（list + 打开 footer）。"""
    try:
        import duckdb
    except Exception:
        return 0.0
    t0 = time.perf_counter()
    try:
        con = duckdb.connect()
        con.execute("SELECT count(*) FROM read_parquet(?)", [keys])
        con.close()
    except Exception:
        pass
    return (time.perf_counter() - t0) * 1000.0


def _layout_a_keys(factor_count: int, days: int) -> list[str]:
    """布局 A：每因子 × 每天一个 key。"""
    keys: list[str] = []
    for f in range(factor_count):
        fid = f"F_{f:06d}"
        for d in range(days):
            keys.append(f"factor_values/{fid}/day={d:03d}/part.parquet")
    return keys


def _layout_b_keys(factor_count: int, days: int, bucket_count: int) -> list[str]:
    """布局 B：每 (shard, month) 一个 delta key（compaction 后每 (shard, month) 一个）。"""
    keys: list[str] = []
    for f in range(factor_count):
        shard = shard_key(f"F_{f:06d}", bucket_count)
        for d in range(days):
            month = d // 22 + 1  # 采样 25 天 → 2 个月
            keys.append(
                f"factor_values/factor_bucket={shard:04d}/"
                f"{partition_key('ashare', 'cn', '1d', 2026, month)}/delta.parquet"
            )
    return sorted(set(keys))


def _run_layout(
    store: LocalObjectStore,
    label: str,
    keys: list[str],
    data: bytes,
    *,
    factor_count: int,
    scale: str,
) -> BenchmarkReport:
    report = BenchmarkReport(label, WORKLOAD, scale)
    report.record_metric("physical_object_count", len(keys))
    put_ms = _put_all(store, keys, data)
    report.record_metric("put_ms", round(put_ms, 2))
    list_ms, n = _list_all(store, "factor_values/")
    report.record_metric("list_ms", round(list_ms, 2))
    report.record_metric("list_object_count", n)
    read_ms, bytes_ = _read_all(store, keys)
    report.record_metric("read_ms", round(read_ms, 2))
    report.record_metric("bytes_scanned", bytes_)
    report.record_metric("duckdb_plan_ms", round(_duckdb_plan_ms(keys), 2))
    report.record_metric("factor_count", factor_count)
    report.finish()
    return report


def _extrapolate(report: BenchmarkReport, factor_scale: float) -> dict[str, Any]:
    """把采样指标外推到全规模（scale the metrics）。

    对象数用**真实全规模**（不靠采样外推）：布局 A = factor_count × 250 天；
    布局 B = bucket_count × 12 月 × 2（delta+compacted）。延迟/字节按采样比例
    外推（factor_scale × day_scale），因为真实 25M 对象无法在 CI 内物化。
    """
    fc = int(report.metrics.get("factor_count") or report.extra.get("factor_count") or 0)
    layout = "A" if report.name.startswith("layoutA") else "B"
    if layout == "A":
        full_obj = layout_a_object_count(fc)  # fc × 250
        scale = factor_scale * _DAY_SCALE
    else:
        full_obj = layout_b_object_count(fc)  # bucket × 12 × 2
        scale = factor_scale * (_DAY_SCALE / 10)  # 采样 2 月 vs 12 月 → 6×
    return {
        "extrapolated_object_count": int(full_obj),
        "extrapolated_put_ms": round((report.metrics.get("put_ms") or 0) * scale, 1),
        "extrapolated_list_ms": round((report.metrics.get("list_ms") or 0) * scale, 1),
        "extrapolated_read_ms": round((report.metrics.get("read_ms") or 0) * scale, 1),
        "extrapolated_bytes_scanned": int((report.metrics.get("bytes_scanned") or 0) * scale),
    }


def run_workload(store, paths, *, scale="small", repeat=1, out_dir=None):
    """返回 [A, B] 报告（每个 scale 一个）。"""
    reports: list[BenchmarkReport] = []
    data = b"\x00" * 1024  # 1KB 采样对象（真实 parquet 更大，但对象数语义一致）
    for factor_count in (10_000, 50_000, 100_000):
        # 采样因子数（外推用）
        sample_factors = min(_SAMPLE_FACTORS, factor_count)
        factor_scale = factor_count / sample_factors
        # 布局 A：采样 days × 采样 factors
        a_keys = _layout_a_keys(sample_factors, _SAMPLE_DAYS)
        # 布局 B：采样 days × 采样 factors（分片去重）
        b_keys = _layout_b_keys(sample_factors, _SAMPLE_DAYS, DEFAULT_BUCKET_COUNT)

        ra = _run_layout(
            store, f"layoutA_{factor_count}", a_keys, data,
            factor_count=factor_count, scale=scale,
        )
        rb = _run_layout(
            store, f"layoutB_{factor_count}", b_keys, data,
            factor_count=factor_count, scale=scale,
        )
        # 外推 + 预算判定
        ra.record_extra("extrapolated", _extrapolate(ra, factor_scale))
        rb.record_extra("extrapolated", _extrapolate(rb, factor_scale))
        budget_a = check_object_count_budget(factor_count, "A")
        budget_b = check_object_count_budget(factor_count, "B")
        ra.record_extra("budget_check", budget_a)
        rb.record_extra("budget_check", budget_b)
        # gate：布局 A 在 100k 必须 OVER_BUDGET；布局 B 必须 UNDER_BUDGET
        if factor_count == 100_000:
            ra.add_gate(
                "object_count_budget", "layout A @100k must be OVER_BUDGET",
                "PASS" if budget_a["verdict"] == "OVER_BUDGET" else "FAIL",
                None,
            )
            rb.add_gate(
                "object_count_budget", "layout B @100k must be UNDER_BUDGET",
                "PASS" if budget_b["verdict"] == "UNDER_BUDGET" else "FAIL",
                None,
            )
        reports.append(ra)
        reports.append(rb)
    if out_dir:
        base = Path(out_dir)
        for r in reports:
            r.write_evidence(base / r.name)
    return reports


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=WORKLOAD)
    p.add_argument("--scale", choices=["tiny", "small", "full"], default="small")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    return p


def main(argv=None) -> int:
    args: Any = _arg_parser().parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="fstor_"))
    store = LocalObjectStore(tmp / "objects")
    reports = run_workload(store, {}, scale=args.scale, repeat=args.repeat, out_dir=args.out)
    def _m(r, k):
        return r.metrics.get(k) if r.metrics.get(k) is not None else r.extra.get(k)

    print(f"{'layout':<14}{'factors':>8}{'objects':>10}{'put_ms':>10}{'list_ms':>9}"
          f"{'read_ms':>9}{'budget':>12}")
    for r in reports:
        name = r.name
        fc = _m(r, "factor_count")
        obj = _m(r, "physical_object_count")
        bc = r.extra.get("budget_check", {})
        print(f"{name:<14}{fc:>8}{obj:>10}{_m(r,'put_ms'):>10}"
              f"{_m(r,'list_ms'):>9}{_m(r,'read_ms'):>9}"
              f"{bc.get('verdict',''):>12}")
    # 打印外推 + 预算汇总
    print("\n--- extrapolated (full scale) + budget ---")
    for r in reports:
        ex = r.extra.get("extrapolated", {})
        bc = r.extra.get("budget_check", {})
        print(f"{r.name:<14} extrap_obj={ex.get('extrapolated_object_count'):>12} "
              f"verdict={bc.get('verdict'):<12} gate={r.gate_verdict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
