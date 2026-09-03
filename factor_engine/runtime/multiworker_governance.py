# -*- coding: utf-8 -*-
"""多 worker 资源治理（100k GO §110 item 9 / P0#9）。

目标：默认**单主进程内部并发**（FE 默认用线程池，不隐式 spawn 子进程）；当多个
FE worker 进程共存时，为每个进程强制 CPU 配额，使 ``N workers × 默认线程 <= 31``
（机器 32 核，OOM 守卫规则：所有子进程合计最多 31 核）。

环境旋钮（与 :mod:`runtime.perf_config` 一致解析，避免口径分歧）：

- ``FACTOR_ENGINE_RESOURCE_PROFILE``：``solo`` / ``shared2`` / ``shared4`` /
  ``balanced`` / ``aggressive`` / ``coexist``。``solo``→31 核，``shared2``→15，
  ``shared4``→7；``balanced``/``coexist`` 默认按 2 进程共存（15 核）保守处理。
- ``FACTOR_ENGINE_COEXIST``：可解析为**正整数**时视为共存进程数（``4``→每进程
  7 核）；否则按布尔（``true``→默认共存，``false``→solo）。
- ``FACTOR_ENGINE_NATIVE_FUSION`` / ``FACTOR_ENGINE_SCHEDULER``：仅参与
  ``PerfConfig`` 缓存签名，本模块不改变其语义。

配额计算：``per_process_cpu_budget = floor(31 / coexist_count)``（至少 1）。
worker 池创建时 ``workers × per_worker_threads <= budget``，超限则 clamp 并
在启动时**只打一次**日志。env 误配（如 ``COEXIST=4`` 但默认 worker=16）→ 告警
+ clamp，不静默放行。
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

from factor_engine.runtime.perf_config import PerfConfig

logger = logging.getLogger("factor_engine.multiworker")

#: 机器 32 核，OOM 守卫：所有子进程合计最多 31 核。
DEFAULT_TOTAL_CORES = 31

#: resource_profile → 共存进程数。默认（balanced/coexist/aggressive）都是**单主进程
#: 内部并发**（GO §110 item 9：默认不隐式 spawn 子进程）；只有显式 ``sharedN``
#: 才按 N 均分。``coexist`` 布尔语义是「给其他程序留内存」，不是 FE worker 进程数。
_PROFILE_COEXIST: dict[str, int] = {
    "solo": 1,
    "shared2": 2,
    "shared4": 4,
    "balanced": 1,  # 默认单主进程内部并发
    "coexist": 1,  # 与外部程序共存 ≠ 多 FE worker 进程
    "aggressive": 1,  # 独占
}

#: 启动时只打一次治理决策日志。
_LOGGED: set[str] = set()
_LOG_LOCK = threading.Lock()


def _coexist_count_from_env(perf: PerfConfig | None = None) -> int:
    """从环境解析共存进程数。

    优先级：``FACTOR_ENGINE_COEXIST`` 正整数 > ``FACTOR_ENGINE_RESOURCE_PROFILE``
    映射 > 默认 1（solo）。``FACTOR_ENGINE_COEXIST`` 为布尔时按 profile 处理。
    """
    raw = os.environ.get("FACTOR_ENGINE_COEXIST", "").strip()
    if raw:
        try:
            n = int(raw)
            if n >= 1:
                return n
        except ValueError:
            pass  # 非整数 → 按布尔/profile 处理
    profile = (perf.resource_profile if perf is not None else None) or os.environ.get(
        "FACTOR_ENGINE_RESOURCE_PROFILE", "balanced"
    ).strip().lower()
    # 显式 ``sharedN`` 才按 N 均分；其余 profile 默认单主进程（1）。
    if profile.startswith("shared"):
        try:
            n = int(profile[len("shared"):])
            if n >= 1:
                return n
        except ValueError:
            pass
    return _PROFILE_COEXIST.get(profile, 1)


def coexistence_process_count(perf: PerfConfig | None = None) -> int:
    """共存 FE worker 进程数（至少 1）。"""
    return max(1, _coexist_count_from_env(perf))


def per_process_cpu_budget(perf: PerfConfig | None = None) -> int:
    """每进程 CPU 配额 = ``floor(31 / coexist_count)``（至少 1）。"""
    return max(1, DEFAULT_TOTAL_CORES // coexistence_process_count(perf))


@dataclass(frozen=True)
class WorkerBudget:
    """一次 worker 池的最终配额决策。"""

    workers: int
    per_worker_threads: int
    coexist_count: int
    cpu_budget: int
    clamped: bool
    warning: str | None = None

    @property
    def total_runnable(self) -> int:
        return self.workers * self.per_worker_threads


def resolve_worker_budget(
    requested_workers: int | None,
    per_worker_threads: int | None = None,
    perf: PerfConfig | None = None,
    *,
    total_cores: int = DEFAULT_TOTAL_CORES,
) -> WorkerBudget:
    """把请求的 worker 数 clamp 到共存配额内。

    ``workers × per_worker_threads <= floor(total_cores / coexist_count)``。
    任一维度超限都 clamp，并返回 ``clamped=True`` + 告警文本（调用方负责打日志）。

    Args:
        requested_workers: 请求的 worker 数；``None`` 表示由调用方默认。
        per_worker_threads: 每 worker 内部线程数；``None`` 表示 1（纯线程池）。
        perf: 已解析的 ``PerfConfig``（含 resource_profile/coexist）；``None`` 时
            从环境现读。
        total_cores: 机器总核数上限（默认 31，OOM 守卫）。
    """
    perf = perf or PerfConfig.from_env()
    coexist = coexistence_process_count(perf)
    budget = max(1, total_cores // coexist)
    threads = max(1, int(per_worker_threads or 1))
    workers = max(1, int(requested_workers or 1))

    clamped = False
    warning: str | None = None
    if workers * threads > budget:
        # 优先降 worker 数（保持每 worker 线程），仍不满足再降线程。
        capped_workers = max(1, budget // threads)
        if capped_workers < workers:
            workers = capped_workers
            clamped = True
        if workers * threads > budget:
            threads = max(1, budget // workers)
            clamped = True
        warning = (
            f"multiworker governance: coexist={coexist} budget={budget} cores; "
            f"requested {requested_workers} workers × {per_worker_threads or 1} "
            f"threads oversubscribes → clamped to {workers} × {threads}"
        )
    return WorkerBudget(
        workers=workers,
        per_worker_threads=threads,
        coexist_count=coexist,
        cpu_budget=budget,
        clamped=clamped,
        warning=warning,
    )


def log_governance_decision(budget: WorkerBudget) -> None:
    """启动时只打一次治理决策日志（进程内幂等）。"""
    key = f"{budget.workers}x{budget.per_worker_threads}@{budget.coexist_count}"
    with _LOG_LOCK:
        if key in _LOGGED:
            return
        _LOGGED.add(key)
    if budget.warning:
        logger.warning(budget.warning)
    else:
        logger.info(
            "multiworker governance: coexist=%d budget=%d cores → %d workers × %d threads",
            budget.coexist_count,
            budget.cpu_budget,
            budget.workers,
            budget.per_worker_threads,
        )


def reset_governance_log() -> None:
    """清空「只打一次」日志集合（测试用）。"""
    with _LOG_LOCK:
        _LOGGED.clear()
