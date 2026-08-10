# -*- coding: utf-8 -*-
"""R27-174: ResourceBroker —— live headroom + CPU/RAM/IO/spill token admission。

目标（R27-022..044）
    - **HardMemoryLimit**（cgroup v2 ``memory.max`` / SLURM / RLIMIT / host RAM /
      用户配置，取最严格）与 **LiveMemoryHeadroom**（cgroup ``memory.current`` /
      host ``MemAvailable`` / 当前进程族 RSS / spill free disk）分开。
    - 进程族内存 = sum(parent + recursive children RSS/PSS)（R27-036/238）。
    - 每 task 申请 memory / CPU / IO / spill token；``sum(running) + candidate
      * uncertainty <= admissible`` 才启动（R27-037..044, R27-119..122）。
    - 压力分档（R27-041）：normal / PRESSURE_1..4 / critical，动作逐档升级。
    - 外部负载感知（R27-130/131）：MemAvailable 下降 / CPU util 升高 → 停止
      admission，而不是等自己 RSS 到 90% 才反应。

本模块只做资源账目与 admission 判定，不直接执行 task（由
:mod:`runtime.adaptive_batch_scheduler` 消费）。
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from runtime.task_resource_contract import DEFAULT_UNCERTAINTY, TaskResourceContract

_logger = logging.getLogger(__name__)

#: 采样节流（R27-033：500ms~2s，默认 1s）。不要每 operator cell 采样。
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
#: 外部 reserve：max(min_host_reserve_gb, RAM * min_host_reserve_fraction)（R27-028）。
DEFAULT_MIN_HOST_RESERVE_GB = 8.0
DEFAULT_MIN_HOST_RESERVE_FRACTION = 0.15
#: 默认 spill 盘保留：max(20GB, 10%)（R27-123）。
DEFAULT_MIN_FREE_GB = 20.0
DEFAULT_MIN_FREE_FRACTION = 0.10


@dataclass(frozen=True)
class ResourceSnapshot:
    """一次资源采样（R27-032）。"""

    timestamp_ms: float
    hard_cpu_slots: int
    system_cpu_util: float
    loadavg: float
    hard_memory_limit: int
    cgroup_memory_current: int | None
    host_mem_available: int
    process_rss: int
    worker_rss: int
    process_family_rss: int
    process_family_pss: int
    spill_free_bytes: int
    disk_busy: float

    @property
    def live_headroom(self) -> int:
        """live headroom = min(cgroup, host, configured)（R27-027）。"""
        candidates: list[int] = []
        if self.cgroup_memory_current is not None and self.hard_memory_limit:
            cgroup_headroom = max(0, self.hard_memory_limit - self.cgroup_memory_current)
            candidates.append(cgroup_headroom)
        # host_headroom = MemAvailable - external_reserve（external reserve 由
        # broker 在 configured_headroom 侧统一扣除；此处只保证非负）。
        candidates.append(max(0, self.host_mem_available))
        # configured_headroom = configured_limit - FE_process_family_RSS。
        if self.hard_memory_limit:
            configured = max(0, self.hard_memory_limit - self.process_family_rss)
            candidates.append(configured)
        return min(candidates) if candidates else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "hard_cpu_slots": self.hard_cpu_slots,
            "system_cpu_util": round(self.system_cpu_util, 3),
            "loadavg": self.loadavg,
            "hard_memory_limit": self.hard_memory_limit,
            "cgroup_memory_current": self.cgroup_memory_current,
            "host_mem_available": self.host_mem_available,
            "process_rss": self.process_rss,
            "worker_rss": self.worker_rss,
            "process_family_rss": self.process_family_rss,
            "process_family_pss": self.process_family_pss,
            "spill_free_bytes": self.spill_free_bytes,
            "disk_busy": round(self.disk_busy, 3),
            "live_headroom": self.live_headroom,
        }


#: 压力阶段（R27-041）
STAGE_NORMAL = "NORMAL"
STAGE_PRESSURE_1 = "PRESSURE_1"   # 停止 speculative prefetch
STAGE_PRESSURE_2 = "PRESSURE_2"   # evict 低价值 cache
STAGE_PRESSURE_3 = "PRESSURE_3"   # 停止新 task admission + spill
STAGE_PRESSURE_4 = "PRESSURE_4"   # 降低并发目标
STAGE_CRITICAL = "CRITICAL"       # 只允许 running task 完成/写出


def _read_int(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read().strip()
        return int(raw)
    except (OSError, ValueError):
        return None


def _cgroup_memory_current() -> int | None:
    return _read_int("/sys/fs/cgroup/memory.current") or _read_int(
        "/sys/fs/cgroup/memory/memory.current"
    )


def _host_mem_available() -> int:
    try:
        import psutil  # type: ignore[import-untyped]

        return max(0, int(psutil.virtual_memory().available))
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return max(0, int(line.split()[1]) * 1024)
    except OSError:
        pass
    return 0


def _cpu_util(interval: float = 0.05) -> float:
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.cpu_percent(interval=interval)) / 100.0
    except Exception:
        return 0.0


def _loadavg() -> float:
    try:
        import os as _os

        return float(_os.getloadavg()[0])
    except (AttributeError, OSError):
        return 0.0


def _disk_busy() -> float:
    return 0.0  # 需要 iostat；第一版 telemetry 占位（R27-032 disk_busy 允许 0）


def _process_family_rss(pss: bool = False) -> int:
    """主进程 + 递归子进程的 RSS/PSS 之和（R27-036）。

    Linux ``/proc/*/smaps_rollup`` 可读时优先 PSS（共享内存更准）；RSS 兜底。
    """
    try:
        import psutil  # type: ignore[import-untyped]

        def _one(proc: Any) -> tuple[int, int]:
            mi = proc.memory_info()
            rss = int(getattr(mi, "rss", 0))
            pss = int(getattr(mi, "pss", 0)) or 0
            try:
                if pss == 0:
                    with open(f"/proc/{proc.pid}/smaps_rollup", encoding="utf-8") as fh:
                        for line in fh:
                            if line.startswith("Pss:"):
                                pss = int(line.split()[1]) * 1024
                                break
            except OSError:
                pass
            return rss, pss

        root = psutil.Process()
        total_rss = 0
        total_pss = 0
        visited: set[int] = set()

        def _walk(proc: Any) -> None:
            nonlocal total_rss, total_pss
            if proc.pid in visited:
                return
            visited.add(proc.pid)
            r, p = _one(proc)
            total_rss += r
            total_pss += p
            try:
                for child in proc.children(recursive=True):
                    _walk(child)
            except Exception:
                pass

        _walk(root)
        return total_pss if pss and total_pss else total_rss
    except Exception:
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            return 0


class _CpuTokenAllocator:
    """CPU token 记账（R27-043/044）。

    DuckDB threads=4 的 task 申请 4 个 token；GIL-bound Python 申请 1 个。
    """

    def __init__(self, hard_slots: int) -> None:
        self.hard_slots = max(1, hard_slots)
        self.soft_budget = self.hard_slots
        self.in_use = 0
        self._lock = threading.RLock()

    def set_soft_budget(self, slots: int) -> None:
        with self._lock:
            self.soft_budget = max(1, min(self.hard_slots, slots))

    def try_acquire(self, tokens: int) -> bool:
        with self._lock:
            if self.in_use + tokens <= self.soft_budget:
                self.in_use += tokens
                return True
            return False

    def release(self, tokens: int) -> None:
        with self._lock:
            self.in_use = max(0, self.in_use - tokens)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hard_slots": self.hard_slots,
                "soft_budget": self.soft_budget,
                "in_use": self.in_use,
            }


class _IoTokenAllocator:
    """IO token（R27-119/120）。本地盘 / 网络盘共享一个记账器（第一版）。"""

    def __init__(self, concurrency: int) -> None:
        self.concurrency = max(1, concurrency)
        self.in_use = 0
        self._lock = threading.RLock()

    def try_acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            if self.in_use + tokens <= self.concurrency:
                self.in_use += tokens
                return True
            return False

    def release(self, tokens: int = 1) -> None:
        with self._lock:
            self.in_use = max(0, self.in_use - tokens)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {"concurrency": self.concurrency, "in_use": self.in_use}


class ResourceBroker:
    """live headroom + token admission 的统一资源代理（R27-174 API）。

    API:
        ``snapshot()`` / ``can_admit(task)`` / ``reserve(task)`` /
        ``release(task)`` / ``pressure_stage()`` / ``recommended_concurrency()``
    """

    def __init__(
        self,
        *,
        hard_memory_limit: int | None = None,
        cpu_slots: int | None = None,
        min_host_reserve_gb: float = DEFAULT_MIN_HOST_RESERVE_GB,
        min_host_reserve_fraction: float = DEFAULT_MIN_HOST_RESERVE_FRACTION,
        spill_min_free_gb: float = DEFAULT_MIN_FREE_GB,
        spill_min_free_fraction: float = DEFAULT_MIN_FREE_FRACTION,
        sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        base_uncertainty: float = DEFAULT_UNCERTAINTY,
    ) -> None:
        from runtime.resource_governor import (
            effective_cpu_slots,
            effective_memory_limit_bytes,
        )

        self.hard_memory_limit = (
            hard_memory_limit or effective_memory_limit_bytes()
        )
        self.hard_cpu_slots = cpu_slots or effective_cpu_slots()
        self.min_host_reserve_gb = float(min_host_reserve_gb)
        self.min_host_reserve_fraction = float(min_host_reserve_fraction)
        self.spill_min_free_gb = float(spill_min_free_gb)
        self.spill_min_free_fraction = float(spill_min_free_fraction)
        self.sample_interval = max(0.05, float(sample_interval_seconds))
        self.base_uncertainty = float(base_uncertainty)
        self._lock = threading.RLock()
        self._cached: ResourceSnapshot | None = None
        self._last_sample_ms = 0.0
        self._cpu = _CpuTokenAllocator(self.hard_cpu_slots)
        self._io = _IoTokenAllocator(max(1, self.hard_cpu_slots))
        self._running: dict[str, TaskResourceContract] = {}
        self._admission_events: list[str] = []
        self._pressure_log: list[str] = []
        # 外部负载平滑（R27-130/131）
        self._external_cpu_ema: float = 0.0
        self._mem_available_ema: int | None = None

    # -- 采样 --

    def _now_ms(self) -> float:
        import time

        return time.monotonic() * 1000.0

    def _refresh(self, *, force: bool = False) -> ResourceSnapshot:
        now = self._now_ms()
        if (
            not force
            and self._cached is not None
            and now - self._last_sample_ms < self.sample_interval * 1000.0
        ):
            return self._cached
        rss = _process_family_rss(pss=False)
        pss = _process_family_rss(pss=True)
        snap = ResourceSnapshot(
            timestamp_ms=now,
            hard_cpu_slots=self.hard_cpu_slots,
            system_cpu_util=_cpu_util(),
            loadavg=_loadavg(),
            hard_memory_limit=self.hard_memory_limit,
            cgroup_memory_current=_cgroup_memory_current(),
            host_mem_available=_host_mem_available(),
            process_rss=rss,
            worker_rss=0,
            process_family_rss=rss,
            process_family_pss=pss,
            spill_free_bytes=self._spill_free(),
            disk_busy=_disk_busy(),
        )
        # 外部负载 EMA：外部 CPU 占用 ≈ max(0, system_cpu_util - our_cpu_share)。
        self._external_cpu_ema = 0.8 * self._external_cpu_ema + 0.2 * snap.system_cpu_util
        self._cached = snap
        self._last_sample_ms = now
        return snap

    def _spill_free(self) -> int:
        try:
            from runtime.resource_governor import spill_disk_available, tempfile_dir

            return spill_disk_available(tempfile_dir()) or 0
        except Exception:
            return 0

    def snapshot(self) -> ResourceSnapshot:
        return self._refresh(force=True)

    # -- 压力阶段（R27-041/131） --

    def pressure_stage(self) -> str:
        snap = self._refresh()
        headroom = snap.live_headroom
        reserve = self._reserve_bytes()
        # R27-131：其他任务内存上涨 → MemAvailable 下降 → 尽早停止 admission。
        if snap.host_mem_available <= reserve * 1.5:
            return STAGE_CRITICAL
        if snap.host_mem_available <= reserve * 2.5:
            return STAGE_PRESSURE_4
        frac = headroom / max(1, snap.hard_memory_limit)
        # live headroom 越接近 reserve 越紧张。
        if headroom <= reserve:
            return STAGE_PRESSURE_3
        if headroom <= reserve * 1.5:
            return STAGE_PRESSURE_2
        if frac < 0.15 or self._external_cpu_ema > 0.6:
            return STAGE_PRESSURE_1
        return STAGE_NORMAL

    def _reserve_bytes(self) -> int:
        ram = self.hard_memory_limit
        raw = max(
            int(self.min_host_reserve_gb * 1024**3),
            int(ram * self.min_host_reserve_fraction),
        )
        # R27-029：默认不要把 MemAvailable 吃到接近 0。小内存机器上
        # ``min_host_reserve_gb=8`` 会把整个预算占满 → reserve 封顶 35% RAM，
        # 大服务器上 8GB min 仍生效（128GB → min(8GB, 44.8GB)=8GB）。
        return min(raw, int(ram * 0.35))

    def _usable_spill(self) -> int:
        snap = self._refresh()
        free = snap.spill_free_bytes
        reserve = max(
            int(self.spill_min_free_gb * 1024**3),
            int(snap.hard_memory_limit * self.spill_min_free_fraction),
        )
        return max(0, free - reserve)

    # -- admission --

    def can_admit(self, task: TaskResourceContract) -> bool:
        snap = self._refresh()
        stage = self.pressure_stage()
        if stage in {STAGE_PRESSURE_3, STAGE_PRESSURE_4, STAGE_CRITICAL}:
            self._admission_events.append(f"blocked:{stage}")
            return False
        peak = task.admissible_peak_bytes
        if peak > 0:
            in_use = sum(
                t.admissible_peak_bytes for t in self._running.values()
            )
            # R27-039: sum(running) + candidate*uncertainty <= admissible_memory
            admissible = max(0, snap.live_headroom - self._reserve_bytes())
            if in_use + peak > admissible:
                self._admission_events.append(f"blocked:memory:{in_use + peak}>{admissible}")
                return False
        # CPU token（R27-043/044）
        if task.cpu_tokens > 0 and not self._cpu.try_acquire(task.cpu_tokens):
            self._admission_events.append(
                f"blocked:cpu:{task.cpu_tokens}@{self._cpu.summary()['soft_budget']}"
            )
            return False
        # IO token（R27-119/120）
        if task.io_tokens > 0 and not self._io.try_acquire(task.io_tokens):
            self._cpu.release(task.cpu_tokens)
            self._admission_events.append(
                f"blocked:io:{task.io_tokens}@{self._io.summary()['concurrency']}"
            )
            return False
        # spill token（R27-121/122）
        if task.spill_bytes > 0 and task.spill_bytes > self._usable_spill():
            self._cpu.release(task.cpu_tokens)
            self._io.release(task.io_tokens)
            self._admission_events.append(
                f"blocked:spill:{task.spill_bytes}>{self._usable_spill()}"
            )
            return False
        return True

    def reserve(self, task: TaskResourceContract, *, task_id: str = "") -> bool:
        """尝试为 task 预留资源；成功才真正登记。"""
        with self._lock:
            if not self.can_admit(task):
                return False
            self._running[task_id or id(task)] = task
            return True

    def release(self, task: TaskResourceContract, *, task_id: str = "") -> None:
        with self._lock:
            self._running.pop(task_id or id(task), None)
            self._cpu.release(task.cpu_tokens)
            self._io.release(task.io_tokens)

    def recommended_concurrency(self) -> int:
        snap = self._refresh()
        stage = self.pressure_stage()
        if stage in {STAGE_CRITICAL, STAGE_PRESSURE_4}:
            return 1
        if stage in {STAGE_PRESSURE_3, STAGE_PRESSURE_2}:
            return max(1, self.hard_cpu_slots // 4)
        headroom = max(0, snap.live_headroom - self._reserve_bytes())
        # 至少 1；最多 CPU slots。内存余量不充足时按 2GB/worker 估算上限。
        by_cpu = self.hard_cpu_slots
        by_mem = max(1, headroom // (2 * 1024**3))
        return max(1, min(by_cpu, by_mem))

    # -- 校准（R27-200..203） --

    def adapt_uncertainty(self, *, underpredict_streak: int, overpredict_streak: int) -> None:
        """连续低估 → 提高；长期过度保守 → 逐渐降低（R27-201..203）。"""
        with self._lock:
            if underpredict_streak >= 3:
                self.base_uncertainty = min(2.5, self.base_uncertainty * 1.15)
                self._pressure_log.append(f"uncertainty_up:{self.base_uncertainty:.2f}")
            elif overpredict_streak >= 3:
                self.base_uncertainty = max(1.05, self.base_uncertainty * 0.93)
                self._pressure_log.append(f"uncertainty_down:{self.base_uncertainty:.2f}")

    # -- 外部负载降并发（R27-130） --

    def lower_soft_cpu_budget(self, factor: float = 0.5) -> None:
        """外部负载升高 → 降低 soft CPU budget（R27-130）。"""
        self._cpu.set_soft_budget(int(self.hard_cpu_slots * max(0.1, factor)))

    def summary(self) -> dict[str, Any]:
        snap = self._refresh()
        return {
            "hard_memory_limit": self.hard_memory_limit,
            "hard_cpu_slots": self.hard_cpu_slots,
            "live_headroom": snap.live_headroom,
            "host_mem_available": snap.host_mem_available,
            "process_family_rss": snap.process_family_rss,
            "process_family_pss": snap.process_family_pss,
            "reserve_bytes": self._reserve_bytes(),
            "usable_spill_bytes": self._usable_spill(),
            "pressure_stage": self.pressure_stage(),
            "recommended_concurrency": self.recommended_concurrency(),
            "cpu": self._cpu.summary(),
            "io": self._io.summary(),
            "running_tasks": len(self._running),
            "running_peak_sum": sum(t.admissible_peak_bytes for t in self._running.values()),
            "admission_events": self._admission_events[-50:],
            "pressure_log": self._pressure_log[-20:],
            "base_uncertainty": round(self.base_uncertainty, 3),
            "external_cpu_ema": round(self._external_cpu_ema, 3),
        }
