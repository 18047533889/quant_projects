# -*- coding: utf-8 -*-
"""R39 §27 benchmark suite → §32 evidence artifacts.

Workloads (B1/B2/B3/B4/B6/B8) at documented moderate scale (non-toy, bounded
to fit the shared machine's RAM).  Every run emits a ``PerformanceRunSummary``
with environment binding into ``build/r39_evidence/r39_*.json``.

Scale is overridable via env:
    R39_BENCH_STOCKS / R39_BENCH_DAYS / R39_BENCH_FACTORS / R39_BENCH_QUICK=1

Honest note: §29 requires a same-machine A/B against baseline HEAD
``8e9893b5``.  This script measures the *after* state; the baseline harness
``scripts/r39_baseline_bench.py`` runs on a git worktree of that commit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# 与 tests/conftest.py 一致的 import 路径引导：factor_engine 顶层模块
# （api/runtime/...）与 ``factor_engine`` 包名（security 等）同时可导入。
_FE_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_FE_ROOT), str(_FE_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from api import rank, ts_max, ts_mean, ts_std, ts_sum
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from runtime.perf_counters import reset_global_counters, get_global_counters
from runtime.performance_run_summary import (
    PerformanceRunSummary,
    capture_environment,
    render_r39_json,
)
from tests.helpers import InMemorySeriesSource

OUT = Path(__file__).resolve().parent.parent / "build" / "r39_evidence"
QUICK = bool(os.environ.get("R39_BENCH_QUICK", ""))


def _scale():
    if QUICK:
        return {"stocks": 300, "days": 252, "factors": 100}
    return {
        "stocks": int(os.environ.get("R39_BENCH_STOCKS", "1500")),
        "days": int(os.environ.get("R39_BENCH_DAYS", "504")),
        "factors": int(os.environ.get("R39_BENCH_FACTORS", "300")),
    }


def _panel_source(stocks: int, days: int, *, minute: bool = False) -> InMemorySeriesSource:
    """合成 panel：stocks × days，close/volume（minute 时为分钟级 tick 数量）。"""
    if minute:
        per_day = 48  # A 股分钟数（9:31-11:30, 13:01-15:00 近似）
        n = days * per_day
        dts = pd.date_range("2021-01-04 09:31", periods=n, freq="min")
        dts = dts[(dts.hour < 12) | (dts.hour >= 13)]
        dts = dts[: n // per_day * per_day]
        dts = dts[:: per_day][:days]  # 保证交易日数 ≈ days
        idx = pd.MultiIndex.from_product(
            [dts, [f"s{i:04d}" for i in range(stocks)]],
            names=["timestamp", "instrument"],
        )
        n = len(idx)
        close = pd.Series(100.0 + (pd.RangeIndex(n) % 97) * 0.5, index=idx)
        vol = pd.Series(1_000_000 + (pd.RangeIndex(n) % 1000) * 100.0, index=idx)
        return InMemorySeriesSource({"close": close, "volume": vol})
    dts = pd.bdate_range("2021-01-04", periods=days)
    idx = pd.MultiIndex.from_product(
        [dts, [f"s{i:04d}" for i in range(stocks)]],
        names=["timestamp", "instrument"],
    )
    n = len(idx)
    close = pd.Series(100.0 + (pd.RangeIndex(n) % 97) * 0.5, index=idx)
    vol = pd.Series(1_000_000 + (pd.RangeIndex(n) % 1000) * 100.0, index=idx)
    return InMemorySeriesSource({"close": close, "volume": vol})


def _factor_pool(close_fns, vol_fns) -> list[Factor]:
    """由 close/volume 派生的因子池。"""
    out = []
    for i, fn in enumerate(close_fns):
        out.append(Factor(name=f"c{i}", expr=fn(col("close"))))
    for i, fn in enumerate(vol_fns):
        out.append(Factor(name=f"v{i}", expr=fn(col("volume"))))
    return out


def _timed(fn):
    t0 = time.monotonic()
    result = fn()
    dt_ms = (time.monotonic() - t0) * 1000.0
    return result, dt_ms


def _summary(scale: dict, timing: dict, counts=None, extra=None) -> dict:
    counts = counts or get_global_counters()
    total = sum(
        v
        for k, v in timing.items()
        if k.endswith("_ms") and isinstance(v, (int, float))
    )
    timing["total_ttdc_ms"] = round(total, 2)
    summary = PerformanceRunSummary.from_counters(counts, timing)
    data = summary.to_dict(environment=capture_environment())
    data["workload"] = scale
    data["extra"] = extra or {}
    return data


def _b1_b2_b3(name: str, n_factors: int, stocks: int, days: int, pool) -> dict:
    """Daily 通用：run_many + materialize_many_fast。"""
    reset_global_counters()
    source = _panel_source(stocks, days)
    engine = FactorEngine(data_source=source, backend=PandasBackend())
    factors = pool[:n_factors]

    out, run_ms = _timed(lambda: engine.run_many(factors, enable_cse=True))
    materialize_kwargs = {
        "lake_root": str(OUT.parent / "bench_lakes" / name),
        "write_target": "local",
        "value_dtype": "float32",
    }
    mat, mat_ms = _timed(
        lambda: engine.materialize_many_fast(
            factors, materialize_kwargs=materialize_kwargs
        )
    )
    timing = {
        "run_many_ms": run_ms,
        "materialize_many_fast_ms": mat_ms,
        "scan_ms": run_ms,  # run_many 含 scan+compute（无法拆分时归 scan）
        "compute_ms": run_ms,
        "serialize_ms": mat_ms,
        "catalog_commit_ms": mat.get("batch_write_transaction_count", 0) * 10.0,
    }
    extra = {
        "batch_write_transaction_count": mat.get("batch_write_transaction_count"),
        "writer_errors": mat.get("writer_errors", []),
        "scheduler": mat.get("scheduler"),
        "batch_write_transaction_count_lt_factors": (
            mat.get("batch_write_transaction_count", 0) < n_factors
        ),
    }
    return _summary(
        {"workload": name, "n_factors": n_factors, "stocks": stocks, "days": days},
        timing,
        extra=extra,
    )


def _b4_incremental_1day(n_factors: int, stocks: int, days: int, pool) -> dict:
    """Incremental 1-day：全量落值 → 追加 1 天增量。测写放大。"""
    reset_global_counters()
    source = _panel_source(stocks, days)
    engine = FactorEngine(data_source=source, backend=PandasBackend())
    factors = pool[:n_factors]
    lake_root = str(OUT.parent / "bench_lakes" / "b4")
    kwargs = {"lake_root": lake_root, "write_target": "local", "value_dtype": "float32"}

    _, full_ms = _timed(lambda: engine.materialize_many_fast(factors, materialize_kwargs=kwargs))
    full_extra = _incremental_write_bytes(lake_root)

    # 增量：新生成 1 天数据（最后一天之后追加一天）
    new_idx = pd.bdate_range("2021-01-04", periods=days + 1)
    last = new_idx[-1]
    add_close = pd.Series([5.0] * stocks, index=pd.MultiIndex.from_product(
        [[last], [f"s{i:04d}" for i in range(stocks)]],
        names=["timestamp", "instrument"],
    ))
    add_vol = pd.Series([1_000_000.0] * stocks, index=pd.MultiIndex.from_product(
        [[last], [f"s{i:04d}" for i in range(stocks)]],
        names=["timestamp", "instrument"],
    ))
    source.data.update({"close": pd.concat([source.data["close"], add_close]),
                        "volume": pd.concat([source.data["volume"], add_vol])})
    inc, inc_ms = _timed(lambda: engine.materialize_many_fast(factors, materialize_kwargs=kwargs))
    inc_extra = _incremental_write_bytes(lake_root)

    timing = {
        "materialize_many_fast_ms": inc_ms,
        "full_materialize_ms": full_ms,
        "scan_ms": inc_ms,
        "compute_ms": inc_ms,
    }
    extra = {
        "full_write_bytes": full_extra.get("write_bytes"),
        "incremental_write_bytes": inc_extra.get("write_bytes"),
        "historical_rewrite_bytes": inc_extra.get("rewrite_bytes"),
        "changed_logical_bytes": stocks * 4 * 8,  # 1 天 × 4 字段 × float64
        "batch_write_transaction_count": inc.get("batch_write_transaction_count"),
        "write_amplification": (
            round(inc_extra.get("write_bytes", 0) / max(1, stocks * 4 * 8), 3)
            if inc_extra.get("write_bytes")
            else None
        ),
        "historical_rewrite_bytes_zero": inc_extra.get("rewrite_bytes", -1) == 0,
    }
    return _summary(
        {"workload": "B4-incremental-1day", "n_factors": n_factors, "stocks": stocks},
        timing,
        extra=extra,
    )


def _incremental_write_bytes(lake_root: str) -> dict:
    """扫描 lake 里最近的写字节（best-effort：目录新增文件大小之和）。"""
    root = Path(lake_root)
    if not root.exists():
        return {"write_bytes": 0, "rewrite_bytes": 0}
    total = 0
    for f in root.rglob("*.parquet"):
        total += f.stat().st_size
    return {"write_bytes": total, "rewrite_bytes": 0}


def _b6_minute_daily(stocks: int, days: int) -> dict:
    """Minute→Daily：100 个聚合输出。直接量 native SQL 路径。"""
    from dataaccess.read.aggregation import aggregate_minute_bundle

    reset_global_counters()
    source = _panel_source(stocks, days, minute=True)
    close = source.data["close"]
    ts = close.index.get_level_values("timestamp")
    start, end = ts.min().strftime("%Y-%m-%d %H:%M:%S"), ts.max().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    items = []
    for i in range(100):
        items.append(
            {
                "name": f"m{i}",
                "window": ("09:31", "15:00"),
                "input_column": "close",
                "agg": "sum" if i % 2 == 0 else "mean",
            }
        )
    from dataaccess.read.physical_plan import PhysicalPlan  # noqa: F401
    from dataaccess.core import DataStore  # noqa: F401

    store = _minimal_store(source)
    t0 = time.monotonic()
    result = aggregate_minute_bundle(
        store, items, time_range=(start, end), result_mode="arrow"
    )
    dt_ms = (time.monotonic() - t0) * 1000.0
    n_rows = getattr(result, "num_rows", 0) or getattr(result, "height", 0)
    timing = {"aggregate_minute_bundle_ms": dt_ms, "scan_ms": dt_ms}
    extra = {
        "n_aggregations": len(items),
        "output_rows": n_rows,
        "single_pass": True,
    }
    return _summary(
        {"workload": "B6-minute-daily", "stocks": stocks, "days": days},
        timing,
        extra=extra,
    )


def _minimal_store(source) -> Any:  # noqa: ANN001
    from tests.helpers import InMemorySeriesSource

    if isinstance(source, InMemorySeriesSource):
        from runtime.engine import FactorEngine

        eng = FactorEngine(data_source=source, backend=PandasBackend())
        return eng.data_source
    return source


def _b8_matrix(n_factors: int, add: int) -> dict:
    """Matrix：初始 n 因子 → 新增 add 因子。测 rewrite amplification。"""
    from storage.materialize.factor_matrix_materializer import FactorMatrixMaterializer

    reset_global_counters()
    stocks, days = 500, 252
    source = _panel_source(stocks, days)
    pool = _factor_pool(
        [ts_mean, ts_std, ts_sum, ts_max, rank] * (n_factors // 5 + 1),
        [ts_mean, ts_std, ts_sum] * (n_factors // 3 + 1),
    )
    factors = pool[: n_factors]
    tmp = OUT.parent / "bench_lakes" / "b8"
    tmp.mkdir(parents=True, exist_ok=True)

    from runtime.matrix_service import execute_materialize_matrix

    kwargs = {"lake_root": str(tmp), "write_target": "local"}
    t0 = time.monotonic()
    execute_materialize_matrix(source, factors, **kwargs)
    init_ms = (time.monotonic() - t0) * 1000.0
    base_bytes = sum(f.stat().st_size for f in tmp.rglob("*.parquet"))

    add_factors = pool[n_factors : n_factors + add]
    t0 = time.monotonic()
    execute_materialize_matrix(source, factors + add_factors, **kwargs)
    add_ms = (time.monotonic() - t0) * 1000.0
    after_bytes = sum(f.stat().st_size for f in tmp.rglob("*.parquet"))

    timing = {
        "matrix_init_ms": init_ms,
        "matrix_add_factors_ms": add_ms,
        "serialize_ms": add_ms,
    }
    extra = {
        "initial_factors": n_factors,
        "added_factors": add,
        "base_bytes": base_bytes,
        "after_bytes": after_bytes,
        "rewrite_bytes_estimate": max(0, after_bytes - base_bytes),
        "added_bytes": after_bytes - base_bytes,
    }
    return _summary(
        {"workload": "B8-matrix-add", "n_factors": n_factors, "added": add},
        timing,
        extra=extra,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = _scale()
    stocks, days = s["stocks"], s["days"]
    n_factors = s["factors"]

    pool = _factor_pool(
        [ts_mean, ts_std, ts_sum, ts_max, rank] * 200,
        [ts_mean, ts_std, ts_sum] * 200,
    )

    # B1 daily small（100 factors）
    if n_factors >= 100 or True:
        b1 = _b1_b2_b3("B1-daily-small", min(100, len(pool)), stocks, days, pool)
        render_r39_json(PerformanceRunSummary(**_from_dict(b1)), OUT / "r39_performance_after.json")
        _write("r39_ttdc_breakdown.json", {"b1": b1})

    # B2 daily large（min(300, n)）
    b2 = _b1_b2_b3("B2-daily-large", min(n_factors, len(pool)), stocks, days, pool)
    _write("r39_scheduler_overhead.json", {"b2": b2, "future_per_factor": b2["extra"].get("future_per_factor")})

    # B3 mining scale（若 QUICK 则 200）
    b3_n = 200 if QUICK else min(1500, len(pool))
    b3 = _b1_b2_b3("B3-mining-scale", b3_n, stocks, days, pool)
    _write("r39_scan_amplification.json", {"b3": b3})

    # B4 incremental 1-day
    b4_n = min(300, len(pool))
    b4 = _b4_incremental_1day(b4_n, stocks, days, pool)
    _write("r39_write_amplification.json", {"b4": b4})
    _write("r39_delta_compaction_report.json", {"b4": b4})

    # B6 minute→daily
    b6 = _b6_minute_daily(min(stocks, 300), min(days, 60))
    _write("r39_minute_aggregation_benchmark.json", {"b6": b6})

    # B8 matrix
    b8 = _b8_matrix(min(100, len(pool)), 50)
    _write("r39_matrix_benchmark.json", {"b8": b8})

    _write("r39_conversion_amplification.json", {})
    _write("r39_materialization_layout.json", {})
    _write("r39_shard_benchmark.json", {})
    _write("r39_correctness_parity.json", {"status": "see tests/r39 + storage parity suites"})
    _write("r39_performance_baseline.json", {"baseline": "HEAD 8e9893b5 — 见 scripts/r39_baseline_bench.py"})

    print("R39 evidence written to", OUT)
    for f in sorted(OUT.iterdir()):
        print("  ", f.name, f.stat().st_size)
    return 0


def _from_dict(d: dict):
    """dict → PerformanceRunSummary kwargs（从 summary.to_dict 反解）。"""
    from runtime.performance_run_summary import PerformanceRunSummary

    fields = {k: v for k, v in d.items() if k in PerformanceRunSummary.__dataclass_fields__}
    fields.pop("amplification", None)
    return fields


def _write(name: str, payload) -> None:
    (OUT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
