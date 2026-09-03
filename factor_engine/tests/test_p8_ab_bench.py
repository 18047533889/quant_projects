# -*- coding: utf-8 -*-
"""P0#8 (100k GO §11): A/B benchmark — scheduler-forced-thread vs HybridExecutor
classifier-chosen execution policy on a mixed operator workload.

Workload: 8 tasks, mix of
  - native/GIL-releasing (duckdb_sql / polars) -> classifier says thread
  - pandas_numpy GIL-bound -> classifier says process
Run under (a) prefer='thread' (old scheduler force) and (b) prefer=None
(classifier authority). Measure wall-clock.
"""
from __future__ import annotations

import time

import pytest

from factor_engine.runtime.hybrid_executor import HybridExecutor
from factor_engine.runtime.resource_broker import ResourceBroker


def _native_work(n: int) -> int:
    import numpy as np
    x = np.arange(n, dtype=np.float64)
    for _ in range(60):
        x = np.sqrt(x + 1.0)
    return int(x.sum())


def _pandas_work(n: int) -> int:
    import pandas as pd
    s = pd.Series(range(n), dtype="float64")
    acc = 0.0
    for _ in range(80):
        s = s.rolling(5, min_periods=1).mean()
        acc += float(s.iloc[-1])
    return int(acc)


def _run_policy(broker, prefer, n, n_native, n_pandas):
    ex = HybridExecutor(broker=broker)
    try:
        futs = []
        for _ in range(n_native):
            futs.append(ex.submit("duckdb_sql", _native_work, n, prefer=prefer))
        for _ in range(n_pandas):
            futs.append(ex.submit("pandas_numpy", _pandas_work, n, prefer=prefer))
        t0 = time.monotonic()
        for f in futs:
            f.result(timeout=300)
        wall = time.monotonic() - t0
        return wall, ex.summary()
    finally:
        ex.shutdown(wait=True)


def test_ab_bench_classifier_vs_forced_thread():
    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    n = 2_000_000
    n_native, n_pandas = 4, 4
    wall_forced, sum_forced = _run_policy(broker, "thread", n, n_native, n_pandas)
    wall_class, sum_class = _run_policy(broker, None, n, n_native, n_pandas)
    assert sum_class["process_task_count"] >= n_pandas, sum_class
    assert sum_forced["process_task_count"] == 0, sum_forced
    print(f"\nP0#8 A/B: forced-thread wall={wall_forced:.3f}s "
          f"(thread={sum_forced['thread_task_count']},proc={sum_forced['process_task_count']}) | "
          f"classifier wall={wall_class:.3f}s "
          f"(thread={sum_class['thread_task_count']},proc={sum_class['process_task_count']})")
    assert wall_forced > 0 and wall_class > 0
