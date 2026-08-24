# -*- coding: utf-8 -*-
"""R39 §29 baseline harness —— 在 baseline HEAD 8e9893b5 的 worktree 上运行。

只依赖稳定 API（engine.run_many / engine.materialize_many_fast），不 import
任何 R39 新增模块，因此可在旧 commit 上直接跑。输出与
``scripts/r39_benchmark_suite.py`` 对齐的 TTDC 测量，供 before/after 对比。

用法（baseline worktree）：
    cd /tmp/r39_baseline_wt
    PYTHONPATH=factor_engine python factor_engine/scripts/r39_baseline_bench.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from factor_engine.api import rank, ts_mean, ts_std, ts_sum
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

OUT = Path("build/r39_evidence")
QUICK = bool(os.environ.get("R39_BENCH_QUICK", ""))


def _scale():
    if QUICK:
        return {"stocks": 300, "days": 252, "factors": 100}
    return {
        "stocks": int(os.environ.get("R39_BENCH_STOCKS", "1500")),
        "days": int(os.environ.get("R39_BENCH_DAYS", "504")),
        "factors": int(os.environ.get("R39_BENCH_FACTORS", "300")),
    }


def _source(stocks: int, days: int) -> InMemorySeriesSource:
    dts = pd.bdate_range("2021-01-04", periods=days)
    idx = pd.MultiIndex.from_product(
        [dts, [f"s{i:04d}" for i in range(stocks)]],
        names=["timestamp", "instrument"],
    )
    n = len(idx)
    close = pd.Series(100.0 + (pd.RangeIndex(n) % 97) * 0.5, index=idx)
    vol = pd.Series(1_000_000 + (pd.RangeIndex(n) % 1000) * 100.0, index=idx)
    return InMemorySeriesSource({"close": close, "volume": vol})


def _pool():
    return [
        Factor(name=f"c{i}", expr=fn(col("close")))
        for i, fn in enumerate([ts_mean, ts_std, ts_sum, rank] * 400)
    ] + [
        Factor(name=f"v{i}", expr=fn(col("volume")))
        for i, fn in enumerate([ts_mean, ts_std, ts_sum] * 400)
    ]


def _bench(label: str, n_factors: int, stocks: int, days: int, pool) -> dict:
    source = _source(stocks, days)
    engine = FactorEngine(data_source=source, backend=PandasBackend())
    factors = pool[:n_factors]
    run_t0 = time.monotonic()
    out = engine.run_many(factors, enable_cse=True)
    run_ms = (time.monotonic() - run_t0) * 1000.0
    mat_t0 = time.monotonic()
    mat = engine.materialize_many_fast(
        factors,
        materialize_kwargs={
            "lake_root": str(OUT.parent / "bench_lakes" / label),
            "write_target": "local",
            "value_dtype": "float32",
        },
    )
    mat_ms = (time.monotonic() - mat_t0) * 1000.0
    return {
        "label": label,
        "n_factors": n_factors,
        "stocks": stocks,
        "days": days,
        "run_many_ms": round(run_ms, 2),
        "materialize_many_fast_ms": round(mat_ms, 2),
        "total_ttdc_ms": round(run_ms + mat_ms, 2),
        "batch_write_transaction_count": mat.get("batch_write_transaction_count"),
        "scheduler": mat.get("scheduler"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = _scale()
    pool = _pool()
    results = {
        "baseline": "HEAD 8e9893b5 (R39 审计基线)",
        "note": "在 baseline worktree 运行；与 after 树同 scale/同 workload",
        "scale": s,
        "runs": [
            _bench("B1-daily-small", min(100, len(pool)), s["stocks"], s["days"], pool),
            _bench("B2-daily-large", min(s["factors"], len(pool)), s["stocks"], s["days"], pool),
        ],
    }
    (OUT / "r39_performance_baseline.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
