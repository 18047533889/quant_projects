# -*- coding: utf-8 -*-
"""Phase 5 P1-15：统一资源与性能 telemetry。

每次 run / run_many 至少记录：effective CPU slots / memory cap / process budget /
max workers / DuckDB threads+memory / RSS start·peak / 各 cache 层 bytes /
CSE live / result bytes / spill / evictions / throttles / backend route / fallback reason。
写入 ``runtime_stats["resource"]`` 与 lineage audit。
"""
from __future__ import annotations

import os
from typing import Any


def _rss_bytes() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


def snapshot_resource_telemetry(perf: Any | None = None, plan: Any | None = None) -> dict[str, Any]:
    """采集当前资源快照（一次性）。"""
    from runtime.resource_governor import (
        ExecutionResourcePlan,
        effective_cpu_slots,
        effective_memory_limit_bytes,
        spill_disk_available,
        tempfile_dir,
    )

    if plan is None:
        try:
            plan = (
                perf.build_resource_plan()
                if perf is not None
                else ExecutionResourcePlan.auto()
            )
        except Exception:
            plan = ExecutionResourcePlan.auto()
    out: dict[str, Any] = {
        "effective_cpu_slots": effective_cpu_slots(),
        "effective_memory_limit_bytes": effective_memory_limit_bytes(),
        "rss_start_bytes": _rss_bytes(),
    }
    if plan is not None:
        for key, value in plan.to_dict().items():
            out[key] = value
    out["spill_dir"] = tempfile_dir()
    out["spill_disk_bytes"] = spill_disk_available(tempfile_dir())
    return out


def resource_telemetry_summary(perf: Any | None = None) -> dict[str, Any]:
    """快照 + 全局 governor 摘要（供 runtime_stats["resource"]）。"""
    out = snapshot_resource_telemetry(perf=perf)
    try:
        from runtime.resource_governor import global_memory_governor

        gov = global_memory_governor()
        out["governor"] = gov.summary()
    except Exception:
        pass
    return out


def finalize_resource_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    """运行结束时补 RSS 峰值与结束值。"""
    telemetry = dict(telemetry or {})
    telemetry["rss_peak_bytes"] = _rss_bytes()
    try:
        telemetry["rss_end_bytes"] = _rss_bytes()
    except Exception:
        pass
    return telemetry


def record_resource_telemetry(runtime_stats: dict[str, Any] | None, *, finalize: bool = False) -> dict[str, Any]:
    """把资源 telemetry 写入 runtime_stats（幂等，不覆盖已有更完整快照）。"""
    stats = dict(runtime_stats or {})
    current = stats.get("resource")
    if not isinstance(current, dict):
        current = resource_telemetry_summary()
    else:
        current = dict(current)
    if finalize:
        current = finalize_resource_telemetry(current)
    # 环境变量来源（供审计/lineage）
    current["env"] = {
        "QUANT_PRODUCTION_MODE": os.environ.get("QUANT_PRODUCTION_MODE", ""),
        "FACTOR_ENGINE_MAX_WORKERS": os.environ.get("FACTOR_ENGINE_MAX_WORKERS", ""),
        "FACTOR_ENGINE_MAX_MEMORY_BYTES": os.environ.get("FACTOR_ENGINE_MAX_MEMORY_BYTES", ""),
    }
    stats["resource"] = current
    return stats
