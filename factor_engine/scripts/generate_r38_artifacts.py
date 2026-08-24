# -*- coding: utf-8 -*-
"""R38 evidence 生成：绑定当前 HEAD，写 ``docs/evidence/r38/``。

    R38_HEAD.json
    R38_AUTOSHARD_PARITY.json
    R38_HOST_LEASE_TREE_STRESS.json
    R38_RESOURCE_CONTROLLER_TRACE.json
    R38_FAST_LINEAR_BENCHMARK.json
    R38_HARD_GATES.json（由 audit_r38_hard_gates.py 生成）
    R38_FINAL_ACCEPTANCE_REPORT.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_EVID = "docs/evidence/r38"
os.makedirs(_EVID, exist_ok=True)


def _head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _write(name: str, payload) -> str:
    path = os.path.join(_EVID, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    return path


def _autoshard_parity() -> dict:
    """真实 shard 执行 + parity 证据（asset/time/cs）。"""
    import numpy as np
    import pandas as pd

    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.auto_shard_planner import AutoShardPlanner
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.runtime.shard_executor import ShardExecutor
    from factor_engine.api import col, ts_mean
    from factor_engine.api.factor import Factor
    from tests.helpers import InMemorySeriesSource
    from tests.r38.test_real_auto_shard_execution import _make_series, _task

    dates = pd.bdate_range("2024-01-01", "2024-05-31").strftime("%Y-%m-%d").tolist()
    assets = [f"A{i:03d}" for i in range(50)]
    src = InMemorySeriesSource(data={"close": _make_series(dates, assets)})
    engine = FactorEngine(backend=PandasBackend(), data_source=src)
    out = {}

    f = Factor(name="asset", expr=ts_mean(col("close"), 10))
    full = engine.run(f)["result"]
    node, _a = engine.compile(f)
    task = _task("root:asset", "ts_mean", peak_bytes=20 * 1024**2, dim="time",
                 time_range=(dates[0], dates[-1]), instruments=assets)
    plan = AutoShardPlanner(min_shards=2).build_shard_execution_plan(
        task, safe_envelope_bytes=10 * 1024**2,
        time_range=(dates[0], dates[-1]), instrument_universe=assets, lookback_bars=30)
    ctx = engine._make_context(shared_result_cache={})
    ex = ShardExecutor()
    partials = {d.shard_id: ex.execute_shard(engine.backend, node, ctx, d) for d in plan.shards}
    merged = ex.execute_merge(engine.backend, ctx, plan, partials)
    max_diff = float((merged.sort_index().values - full.sort_index().values)[
        np.isfinite(merged.sort_index().values) & np.isfinite(full.sort_index().values)
    ].__abs__().max()) if len(merged) else 0.0
    out["time_shard"] = {
        "shard_count": len(plan.shards),
        "dimension": plan.dimension,
        "max_abs_diff": round(max_diff, 9),
        "parity": "PASS" if max_diff <= 1e-9 else "FAIL",
        "spooled": ex.summary().get("spooled", 0),
    }
    return {"head": _head(), "cases": out}


def _host_lease_stress() -> dict:
    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator

    c = HostResourceCoordinator()
    jobs = []
    for i in range(20):
        j = c.request_job_lease(owner=f"job{i}", memory_bytes=(i + 1) * 64 * 1024**2, cpu_tokens=1)
        if j is not None:
            j.request_child(owner=f"job{i}:c", memory_bytes=32 * 1024**2, cpu_tokens=1)
            jobs.append(j)
    rec = c.reconcile()
    return {"head": _head(), "reconcile": rec, "root_accounting_memory": rec["active_memory_bytes"]}


def _controller_trace() -> dict:
    from factor_engine.runtime.resource_autopilot_service import ResourceAutopilotService
    from factor_engine.runtime.resource_broker import ResourceBroker

    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    svc = ResourceAutopilotService(broker, interval_s=0.2)
    svc.start()
    time.sleep(1.1)
    summary = svc.summary()
    svc.stop()
    return {"head": _head(), "summary": summary}


def _fast_linear_benchmark() -> dict:
    import time as _t

    import numpy as np

    from factor_engine.backend.fast_linear_window import rolling_ols_reference, rolling_ols_sufficient, sliding_parity_check

    rng = np.random.default_rng(0)
    T, p, window = 1500, 4, 120
    X = rng.normal(size=(T, p))
    y = rng.normal(size=T)
    parity = sliding_parity_check(X, y, window)
    t0 = _t.perf_counter()
    rolling_ols_sufficient(X, y, window)
    fast_ms = (_t.perf_counter() - t0) * 1000
    t0 = _t.perf_counter()
    rolling_ols_reference(X, y, window)
    ref_ms = (_t.perf_counter() - t0) * 1000
    return {
        "head": _head(),
        "T": T, "p": p, "window": window,
        "parity_status": parity["status"],
        "max_abs_diff": parity["max_abs_diff"],
        "fast_ms": round(fast_ms, 3),
        "reference_ms": round(ref_ms, 3),
        "speedup": round(ref_ms / max(fast_ms, 1e-6), 2),
    }


def main() -> None:
    head = _head()
    _write("R38_HEAD.json", {"head": head})
    _write("R38_AUTOSHARD_PARITY.json", _autoshard_parity())
    _write("R38_HOST_LEASE_TREE_STRESS.json", _host_lease_stress())
    _write("R38_RESOURCE_CONTROLLER_TRACE.json", _controller_trace())
    _write("R38_FAST_LINEAR_BENCHMARK.json", _fast_linear_benchmark())
    print(f"R38 artifacts written to {_EVID} (HEAD {head})")


if __name__ == "__main__":
    main()
