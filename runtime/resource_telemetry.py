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
    """当前进程 RSS（R27-035/169）：psutil current → /proc/self/status VmRSS。

    ``ru_maxrss`` 是**历史峰值**，不是当前值；用它在 start/end 采样会把两端都
    变成同一个峰值，不能用于 adaptive scheduling（R27-034/144）。
    """
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _lifetime_peak_rss_bytes() -> int | None:
    """进程历史峰值 RSS（``ru_maxrss``）；与 current RSS 分开（R27-035/144）。"""
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
        "rss_current_start": _rss_bytes(),
        "rss_peak_lifetime": _lifetime_peak_rss_bytes(),
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


def finalize_resource_telemetry(
    telemetry: dict[str, Any],
    *,
    run_peak: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行结束时补 current RSS 结束值 + lifetime peak（R27-144）。

    R36 P0-029（§169）：``rss_peak_run`` 必须来自 **run 专用 sampler** 的窗口
    峰值（``run_peak``），不能混入进程历史 lifetime peak（ru_maxrss）——否则
    无法训练 task memory model。run_peak 未提供时退化为当前已知最大值（诚实
    标注来源）。
    """
    telemetry = dict(telemetry or {})
    current = _rss_bytes()
    lifetime = _lifetime_peak_rss_bytes()
    telemetry["rss_current_end"] = current
    if lifetime is not None:
        telemetry["rss_peak_lifetime"] = lifetime
    if run_peak is not None:
        # §168 true run peak：只统计该 run 时间窗口的 process family PSS。
        peak_pss = int(run_peak.get("peak_family_pss", 0) or 0)
        peak_rss = int(run_peak.get("peak_family_rss", 0) or 0)
        telemetry["rss_peak_run"] = peak_rss
        telemetry["rss_peak_run_pss"] = peak_pss
        telemetry["rss_peak_run_source"] = "run_peak_sampler"
        telemetry["run_peak"] = run_peak
    else:
        known = [v for v in (telemetry.get("rss_current_start"), current, lifetime)
                 if isinstance(v, int) and v > 0]
        telemetry["rss_peak_run"] = max(known) if known else (lifetime or 0)
        telemetry["rss_peak_run_source"] = "current_max_fallback"
    return telemetry


def record_resource_telemetry(
    runtime_stats: dict[str, Any] | None,
    *,
    finalize: bool = False,
    run_peak: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把资源 telemetry 写入 runtime_stats（幂等，不覆盖已有更完整快照）。"""
    stats = dict(runtime_stats or {})
    current = stats.get("resource")
    if not isinstance(current, dict):
        current = resource_telemetry_summary()
    else:
        current = dict(current)
    if finalize:
        current = finalize_resource_telemetry(current, run_peak=run_peak)
    # 环境变量来源（供审计/lineage）
    current["env"] = {
        "QUANT_PRODUCTION_MODE": os.environ.get("QUANT_PRODUCTION_MODE", ""),
        "FACTOR_ENGINE_MAX_WORKERS": os.environ.get("FACTOR_ENGINE_MAX_WORKERS", ""),
        "FACTOR_ENGINE_MAX_MEMORY_BYTES": os.environ.get("FACTOR_ENGINE_MAX_MEMORY_BYTES", ""),
    }
    stats["resource"] = current
    return stats
