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

# R36：PSI / memory slope / swap 信号（§28/29/154）。resource_broker 只采集，
# 判定在 resource_autopilot（惰性 import 避免循环依赖）。
from runtime.resource_monitor import MemorySlopeTracker  # noqa: E402

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
    #: 本进程族（FactorEngine）自身的 CPU 占用分数 [0,1]（R31-P0-010）。
    our_cpu_util: float
    #: 外部负载 ≈ max(0, system_cpu - our_cpu)（R31-P0-010）。
    external_cpu_util: float
    loadavg: float
    hard_memory_limit: int
    cgroup_memory_current: int | None
    host_mem_available: int
    process_rss: int
    worker_rss: int
    process_family_rss: int
    process_family_pss: int
    spill_free_bytes: int
    spill_total_bytes: int
    disk_busy: float
    # R36（§28/29）：PSI / memory slope / swap / memory.events —— 反馈控制输入。
    cpu_psi_some: float = 0.0
    cpu_psi_full: float = 0.0
    memory_psi_some: float = 0.0
    memory_psi_full: float = 0.0
    io_psi_some: float = 0.0
    io_psi_full: float = 0.0
    mem_available_slope: float = 0.0
    memory_events: dict = field(default_factory=dict)
    swap_current: int | None = None
    swap_max: int | None = None
    major_fault_rate: float = 0.0

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
            "our_cpu_util": round(self.our_cpu_util, 3),
            "external_cpu_util": round(self.external_cpu_util, 3),
            "loadavg": self.loadavg,
            "hard_memory_limit": self.hard_memory_limit,
            "cgroup_memory_current": self.cgroup_memory_current,
            "host_mem_available": self.host_mem_available,
            "process_rss": self.process_rss,
            "worker_rss": self.worker_rss,
            "process_family_rss": self.process_family_rss,
            "process_family_pss": self.process_family_pss,
            "spill_free_bytes": self.spill_free_bytes,
            "spill_total_bytes": self.spill_total_bytes,
            "disk_busy": round(self.disk_busy, 3),
            "live_headroom": self.live_headroom,
            "cpu_psi_some": round(self.cpu_psi_some, 4),
            "memory_psi_some": round(self.memory_psi_some, 4),
            "io_psi_some": round(self.io_psi_some, 4),
            "mem_available_slope": round(self.mem_available_slope, 3),
            "swap_current": self.swap_current,
            "swap_max": self.swap_max,
        }


#: 压力阶段（R27-041）
STAGE_NORMAL = "NORMAL"
STAGE_PRESSURE_1 = "PRESSURE_1"   # 停止 speculative prefetch
STAGE_PRESSURE_2 = "PRESSURE_2"   # evict 低价值 cache
STAGE_PRESSURE_3 = "PRESSURE_3"   # 停止新 task admission + spill
STAGE_PRESSURE_4 = "PRESSURE_4"   # 降低并发目标
STAGE_CRITICAL = "CRITICAL"       # 只允许 running task 完成/写出


def peak_hint(task: Any) -> int:
    """task 的 admission 峰值字节（纯 helper，供 admission 事件可读）。"""
    try:
        return int(task.admissible_peak_bytes)
    except Exception:
        return 0


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


def _cpu_util(interval: float = 0.0) -> float:
    try:
        import psutil  # type: ignore[import-untyped]

        # interval=0：非阻塞（返回自上次调用以来的百分比），不 sleep 50ms——
        # scheduler 热循环里每次 admission 都采样，blocking 版本会把小批量拖慢。
        return float(psutil.cpu_percent(interval=interval)) / 100.0
    except Exception:
        return 0.0


def _loadavg() -> float:
    try:
        import os as _os

        return float(_os.getloadavg()[0])
    except (AttributeError, OSError):
        return 0.0


def _psi(namespace: str) -> tuple[float, float]:
    """PSI ``(some_avg10, full_avg10)``；不可读返回 ``(0.0, 0.0)``（R36 §28）。"""
    try:
        from runtime.resource_monitor import psi_cpu, psi_io, psi_memory

        reader = {"cpu": psi_cpu, "memory": psi_memory, "io": psi_io}[namespace]
        out = reader()
        if out is None:
            return 0.0, 0.0
        return float(out[0]), float(out[1])
    except Exception:
        return 0.0, 0.0


def _swap() -> tuple[int | None, int | None]:
    try:
        from runtime.resource_monitor import swap_usage

        return swap_usage()
    except Exception:
        return None, None


def _memory_events() -> dict:
    try:
        from runtime.resource_monitor import read_memory_events

        return read_memory_events()
    except Exception:
        return {}


def _disk_io_counters() -> tuple[int, int] | None:
    """当前磁盘累计读写字节（``read_bytes + write_bytes``）；不可用返回 ``None``。

    R31-P0-012：让 IO pressure 可观测——``disk_busy`` 不再是恒 0 占位。
    """
    try:
        import psutil  # type: ignore[import-untyped]

        counters = psutil.disk_io_counters()
        if counters is None:
            return None
        return int(counters.read_bytes) + int(counters.write_bytes), int(
            getattr(counters, "busy_time", 0)
        )
    except Exception:
        return None


def _process_family_cpu_times() -> float:
    """本进程族累计 CPU 秒（user + system，跨全部进程/核心）。

    R31-P0-010：external CPU 需要「FactorEngine 自己吃了多少 CPU」。
    采样间隔内的增量 / (dt * cores) 即本进程族占用分数。
    """
    try:
        import psutil  # type: ignore[import-untyped]

        root = psutil.Process()
        total = 0.0
        visited: set[int] = set()

        def _walk(proc: Any) -> None:
            nonlocal total
            if proc.pid in visited:
                return
            visited.add(proc.pid)
            try:
                ct = proc.cpu_times()
                total += float(ct.user) + float(ct.system)
            except Exception:
                pass
            try:
                for child in proc.children(recursive=True):
                    _walk(child)
            except Exception:
                pass

        _walk(root)
        return total
    except Exception:
        return 0.0


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

    def can_acquire(self, tokens: int) -> bool:
        """纯判定：无副作用（R31-P0-009）。"""
        with self._lock:
            return self.in_use + tokens <= self.soft_budget

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

    def can_acquire(self, tokens: int = 1) -> bool:
        """纯判定：无副作用（R31-P0-009）。"""
        with self._lock:
            return self.in_use + tokens <= self.concurrency

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


class ReservationLease:
    """R31-005: 幂等 release 的资源预留租约。

    ``broker.try_reserve(task)`` 成功返回 lease；task 终态（SUCCESS / FAILED /
    CANCELLED / BROKEN_WORKER / TIMEOUT）统一 ``lease.release()``。``release``
    幂等：重复调用无害（R31_TASK_FAILURE_RESOURCE_LEAK_ZERO 的基础）。
    """

    def __init__(self, broker: "ResourceBroker", task: TaskResourceContract, task_id: str) -> None:
        self._broker = broker
        self._task = task
        self._task_id = task_id
        self._released = False
        self._lock = threading.RLock()

    @property
    def task(self) -> TaskResourceContract:
        return self._task

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def release(self) -> None:
        """幂等释放：CPU/IO token 归还 + ``_running`` 注销，只做一次。"""
        with self._lock:
            if self._released:
                return
            self._released = True
            self._broker._release_locked(self._task, task_id=self._task_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self._task_id,
            "released": self.released,
            "cpu_tokens": self._task.cpu_tokens,
            "io_tokens": self._task.io_tokens,
            "admissible_peak_bytes": self._task.admissible_peak_bytes,
        }


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
        # P0-016：job/sink 的 live signals（由 scheduler 在 resource_decision() 时
        # 上报；autopilot 后台线程每 tick 聚合，不再硬编码 None/0.0）。
        self._live_signals: dict[str, Any] = {
            "sink_backpressure": 0.0,
            "job_memory_lease_bytes": None,
        }
        self._cpu = _CpuTokenAllocator(self.hard_cpu_slots)
        self._io = _IoTokenAllocator(max(1, self.hard_cpu_slots))
        self._running: dict[str, TaskResourceContract] = {}
        # R31-P0-006: reserve() 兼容路径持有的租约（try_reserve 直接返回租约不登记）。
        self._leases: dict[str, ReservationLease] = {}
        self._admission_events: list[str] = []
        self._pressure_log: list[str] = []
        # 外部负载平滑（R27-130/131）
        self._external_cpu_ema: float = 0.0
        self._mem_available_ema: int | None = None
        # R31-P0-010: 本进程族 CPU 时间增量（采样间隔内算自身占用分数）。
        self._last_family_cpu_times: float | None = None
        # R31-P0-012: 磁盘 IO 计数增量（busy 观测）。
        self._last_io_bytes: int | None = None
        self._last_io_sample_ms: float | None = None
        # R36（§69）：MemAvailable 时间斜率（预测性压力）。
        self._slope_tracker = MemorySlopeTracker()
        # R36：ResourceController（惰性构造，避免循环 import）。
        self._controller: Any = None

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
        system_cpu = _cpu_util()
        # R31-P0-010：外部 CPU = max(0, system - 本进程族 share)。
        family_times = _process_family_cpu_times()
        our_cpu = 0.0
        if self._last_family_cpu_times is not None:
            dt = max(0.0001, (now - self._last_sample_ms) / 1000.0)
            cores = max(1, self.hard_cpu_slots)
            our_cpu = max(
                0.0,
                min(1.0, (family_times - self._last_family_cpu_times) / (dt * cores)),
            )
        self._last_family_cpu_times = family_times
        external_cpu = max(0.0, system_cpu - our_cpu)
        self._external_cpu_ema = 0.8 * self._external_cpu_ema + 0.2 * external_cpu
        # R31-P0-012：磁盘 busy ≈ 读写吞吐 / 参考带宽（饱和 → 1.0）。
        disk_busy = self._disk_io_busy(now)
        mem_available = _host_mem_available()
        # R36（§69）：MemAvailable 斜率——外部任务快速吃内存时提前让路。
        slope = self._slope_tracker.update(now / 1000.0, mem_available)
        # R36（§28/29）：PSI + swap + memory.events。
        cpu_psi = _psi("cpu")
        mem_psi = _psi("memory")
        io_psi = _psi("io")
        sw_cur, sw_max = _swap()
        snap = ResourceSnapshot(
            timestamp_ms=now,
            hard_cpu_slots=self.hard_cpu_slots,
            system_cpu_util=system_cpu,
            our_cpu_util=our_cpu,
            external_cpu_util=external_cpu,
            loadavg=_loadavg(),
            hard_memory_limit=self.hard_memory_limit,
            cgroup_memory_current=_cgroup_memory_current(),
            host_mem_available=mem_available,
            process_rss=rss,
            worker_rss=0,
            process_family_rss=rss,
            process_family_pss=pss,
            spill_free_bytes=self._spill_free(),
            spill_total_bytes=self._spill_total(),
            disk_busy=disk_busy,
            cpu_psi_some=cpu_psi[0],
            cpu_psi_full=cpu_psi[1],
            memory_psi_some=mem_psi[0],
            memory_psi_full=mem_psi[1],
            io_psi_some=io_psi[0],
            io_psi_full=io_psi[1],
            mem_available_slope=slope,
            memory_events=_memory_events(),
            swap_current=sw_cur,
            swap_max=sw_max,
        )
        self._cached = snap
        self._last_sample_ms = now
        return snap

    def _spill_free(self) -> int:
        try:
            from runtime.resource_governor import spill_disk_available, tempfile_dir

            return spill_disk_available(tempfile_dir()) or 0
        except Exception:
            return 0

    def _spill_total(self) -> int:
        try:
            from runtime.resource_governor import spill_disk_total, tempfile_dir

            return spill_disk_total(tempfile_dir()) or 0
        except Exception:
            return 0

    def _disk_io_busy(self, now: float) -> float:
        """磁盘 busy 观测（R31-P0-012）：读写吞吐占参考带宽的比例。

        参考带宽（500 MB/s）只是归一化常量；``_last_io_bytes is None``（首次采样）
        时返回 0.0。探测失败同样返回 0.0（不误伤 admission）。
        """
        counters = _disk_io_counters()
        if counters is None:
            return 0.0
        delta_bytes, _busy_time = counters
        if self._last_io_bytes is None:
            self._last_io_bytes = delta_bytes
            self._last_io_sample_ms = now
            return 0.0
        dt = max(0.0001, (now - (self._last_io_sample_ms or now)) / 1000.0)
        bps = max(0, delta_bytes - self._last_io_bytes) / dt
        self._last_io_bytes = delta_bytes
        self._last_io_sample_ms = now
        # 归一化：500 MB/s 视为满速（本地 SSD / NVMe 参考）。
        return max(0.0, min(1.0, bps / (500 * 1024**2)))

    def snapshot(self) -> ResourceSnapshot:
        return self._refresh(force=True)

    def cpu_budget(self) -> int:
        """R33-P0-038：当前 soft CPU budget（调度器显式并发上限依据）。

        ``lower_soft_cpu_budget`` 动态降预算时，scheduler 的
        ``running_futures < dynamic_concurrency_limit`` 用本值限制新 admission。
        """
        return max(1, int(self._cpu.soft_budget))

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
        """可用 spill = free - reserve；reserve 针对 **spill 盘容量** 计算（R31-P0-011）。

        ``spill_min_free_fraction`` 的基准从 RAM 硬上限改成 spill 文件系统总容量——
        spill 是独立容量维度，不能用内存比例预留。
        """
        snap = self._refresh()
        free = snap.spill_free_bytes
        total = snap.spill_total_bytes or snap.hard_memory_limit
        reserve = max(
            int(self.spill_min_free_gb * 1024**3),
            int(total * self.spill_min_free_fraction),
        )
        return max(0, free - reserve)

    # -- admission --

    def can_admit(self, task: TaskResourceContract) -> bool:
        """R31-P0-009：**纯函数**——只判定，不获取任何 token（无副作用）。

        旧实现 ``can_admit`` 内部调用 ``try_acquire``，调用方只询问不随后
        reserve/release 时 CPU/IO token 泄漏。现在判定与获取分离：
        ``can_admit`` 纯判定，``try_reserve`` 原子 check+acquire 并返回 lease。
        """
        snap = self._refresh()
        stage = self.pressure_stage()
        if stage in {STAGE_PRESSURE_3, STAGE_PRESSURE_4, STAGE_CRITICAL}:
            return False
        peak = task.admissible_peak_bytes
        if peak > 0:
            in_use = sum(
                t.admissible_peak_bytes for t in self._running.values()
            )
            # R27-039: sum(running) + candidate*uncertainty <= admissible_memory
            admissible = max(0, snap.live_headroom - self._reserve_bytes())
            if in_use + peak > admissible:
                return False
        with self._lock:
            # CPU token（R27-043/044）—— 纯判定
            if task.cpu_tokens > 0 and not self._cpu.can_acquire(task.cpu_tokens):
                return False
            # IO token（R27-119/120）—— 纯判定
            if task.io_tokens > 0 and not self._io.can_acquire(task.io_tokens):
                return False
        # spill token（R27-121/122）
        if task.spill_bytes > 0 and task.spill_bytes > self._usable_spill():
            return False
        return True

    def try_reserve(
        self, task: TaskResourceContract, *, task_id: str = ""
    ) -> ReservationLease | None:
        """原子 check + acquire（R31-P0-009/006）：成功返回 lease，失败返回 ``None``。

        在同一把锁内判定并获取：``can_admit`` True 后 ``try_acquire`` 必然成功，
        不存在「check 通过但 token 被抢走」的窗口。
        """
        tid = str(task_id or id(task))
        with self._lock:
            if not self.can_admit(task):
                self._admission_events.append(
                    f"rejected:{task.backend}:cpu={task.cpu_tokens}:mem={peak_hint(task)}"
                )
                return None
            if task.cpu_tokens > 0:
                self._cpu.try_acquire(task.cpu_tokens)
            if task.io_tokens > 0:
                self._io.try_acquire(task.io_tokens)
            self._running[tid] = task
        self._admission_events.append(
            f"admitted:{task.backend}:cpu={task.cpu_tokens}:mem={peak_hint(task)}"
        )
        return ReservationLease(self, task, tid)

    def _release_locked(self, task: TaskResourceContract, *, task_id: str) -> None:
        """租约/旧 API 共用的释放原语（只在 ``ReservationLease.release`` 内幂等化）。"""
        self._running.pop(task_id, None)
        self._cpu.release(task.cpu_tokens)
        self._io.release(task.io_tokens)

    def reserve(self, task: TaskResourceContract, *, task_id: str = "") -> bool:
        """向后兼容：返回 bool。内部经 ``try_reserve`` 拿到 lease 并持有，
        ``release()`` 释放。旧调用方 ``reserve→release`` 语义不变，且不再有
        ``can_admit`` 单独调用泄漏 token 的问题。
        """
        tid = str(task_id or id(task))
        with self._lock:
            if self._running.get(tid) is not None:
                return True
        lease = self.try_reserve(task, task_id=tid)
        if lease is None:
            return False
        with self._lock:
            self._leases[tid] = lease
        return True

    def release(self, task: TaskResourceContract, *, task_id: str = "") -> None:
        """向后兼容：按 task_id 释放租约（幂等）。"""
        tid = str(task_id or id(task))
        with self._lock:
            lease = self._leases.pop(tid, None)
        if lease is not None:
            lease.release()
            return
        # 无租约时直接归还（覆盖直接 try_reserve 后手工 release 的调用方）。
        self._release_locked(task, task_id=tid)

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

    # -- R36 双向控制器（P0-003：压力恢复后自动升速） --

    def raise_soft_cpu_budget(self, amount: int = 1) -> None:
        """压力解除后**升** soft CPU budget（每次 +1，slow additive increase）。

        R36 P0-003：旧实现只有 ``lower_soft_cpu_budget``（只降不升），下午另一个
        算法结束后 FE 会永远停在低预算。本方法由 ResourceController 在稳定 N 样本
        后调用（§66 fast down / slow up）。
        """
        self._cpu.set_soft_budget(min(self.hard_cpu_slots, self._cpu.soft_budget + int(amount)))

    def restore_cpu_budget(self) -> None:
        """一次性恢复 full budget（异常回退 conservative 时用）。"""
        self._cpu.set_soft_budget(self.hard_cpu_slots)

    # -- R36 ResourceDecision（P0-002：broker 算出建议值，scheduler 必须消费） --

    def _resource_controller(self) -> Any:
        if self._controller is None:
            from runtime.resource_autopilot import ResourceController

            self._controller = ResourceController(self)
        return self._controller

    def _signals_from_snapshot(self, snap: ResourceSnapshot) -> Any:
        """从单个采样构造 ResourceSignals（避免一个 tick 内多次 force 采样）。"""
        from runtime.resource_monitor import ResourceSignals

        return ResourceSignals(
            host_mem_available=snap.host_mem_available,
            cgroup_mem_current=snap.cgroup_memory_current,
            family_rss=snap.process_family_rss,
            family_pss=snap.process_family_pss,
            memory_psi_some=snap.memory_psi_some,
            memory_psi_full=snap.memory_psi_full,
            cpu_util=snap.system_cpu_util,
            cpu_psi_some=snap.cpu_psi_some,
            io_psi_some=snap.io_psi_some,
            disk_latency_ms=0.0,
            writer_backpressure=0.0,
            scan_backpressure=0.0,
            mem_available_slope=snap.mem_available_slope,
            memory_events=dict(snap.memory_events),
            swap_current=snap.swap_current,
            swap_max=snap.swap_max,
        )

    def _envelope_from_snapshot(self, snap: ResourceSnapshot) -> Any:
        """从单个采样构造 Safe Envelope（§303/17）。"""
        from runtime.resource_monitor import compute_safe_envelope

        env = compute_safe_envelope(
            hard_memory_bytes=self.hard_memory_limit,
            live_headroom_bytes=snap.live_headroom,
            emergency_reserve_bytes=self._reserve_bytes(),
            family_pss=snap.process_family_pss,
        )
        from dataclasses import replace

        return replace(
            env,
            hard_cpu_tokens=self.hard_cpu_slots,
            target_cpu_tokens=max(1, self._cpu.soft_budget),
            io_capacity_score=1.0,
            remote_capacity_score=1.0,
            spill_free_bytes=snap.spill_free_bytes,
        )

    def resource_signals(self) -> Any:
        """当前 ResourceSignals（PSI + slope + swap + family memory）。"""
        return self._signals_from_snapshot(self._refresh(force=True))

    def resource_envelope(self) -> Any:
        """当前 Safe Envelope（§303/17）。"""
        return self._envelope_from_snapshot(self._refresh(force=True))

    def resource_decision(
        self,
        *,
        job_memory_lease_bytes: int | None = None,
        sink_backpressure: float = 0.0,
    ) -> Any:
        """R36 P0-002 + R38 P0-011/013：ResourceDecision 消费。

        R38：ResourceAutopilotService 激活时（进程级 fixed-cadence 控制循环），
        **只读**最新 snapshot —— scheduler 绝不自己 tick controller（否则 stable/
        cooldown 由 loop 次数决定，多个 scheduler 高频 tick 会失控）。snapshot 太旧
        → conservative fallback（§P0-012）。服务未激活（研究/standalone）才做
        one-off tick。

        P0-016：调用方提供的 ``sink_backpressure`` / ``job_memory_lease_bytes``
        先写入 live signals——autopilot 后台线程每 tick 聚合（不再被硬编码
        None/0.0 覆盖）。snapshot 过期时**直接** conservative fallback，不再把
        同一个 stale last decision 当 fresh 返回。
        """
        self._record_live_signals(
            sink_backpressure=sink_backpressure,
            job_memory_lease_bytes=job_memory_lease_bytes,
        )
        try:
            from runtime.resource_autopilot_service import get_resource_autopilot

            # 优先 broker 自己绑定的 autopilot service（R39：测试/部署常用本地
            # service 而非全局单例；若只查全局单例，未注册时每次调用都 fall
            # through 到 controller.tick —— scheduler 热循环会放大控制循环）。
            autopilot = getattr(self, "_autopilot_service", None)
            if autopilot is None or not autopilot.started:
                autopilot = get_resource_autopilot()
            if autopilot is not None and autopilot.started and autopilot.last_decision() is not None:
                snap = autopilot.last_decision()
                if not snap.is_stale:
                    return snap.decision
                # 太旧 → **直接** conservative（旧 last_decision 同样过期，不能当 fresh）。
                return self._conservative_decision()
        except Exception:
            pass
        snap = self._refresh(force=True)
        return self._resource_controller().tick(
            self._signals_from_snapshot(snap),
            self._envelope_from_snapshot(snap),
            job_memory_lease_bytes=job_memory_lease_bytes,
            sink_backpressure=sink_backpressure,
        )

    def _record_live_signals(
        self,
        *,
        sink_backpressure: float = 0.0,
        job_memory_lease_bytes: int | None = None,
    ) -> None:
        """记录调用方上报的 live signals（P0-016，供 autopilot 后台聚合）。"""
        with self._lock:
            self._live_signals["sink_backpressure"] = float(sink_backpressure or 0.0)
            self._live_signals["job_memory_lease_bytes"] = job_memory_lease_bytes

    def live_signals(self) -> dict[str, Any]:
        """当前最新 live signals（autopilot ``_tick`` 消费，P0-016）。"""
        with self._lock:
            return dict(self._live_signals)

    def _conservative_decision(self) -> Any:
        """decision 过期时的保守回退（§P0-012：不自行创建第二套 decision）。"""
        from runtime.resource_autopilot import ResourceDecision

        env = self.resource_envelope()
        return ResourceDecision(
            target_concurrency=1,
            target_cpu_tokens=max(1, self.hard_cpu_slots // 4),
            read_wave_bytes=256 * 1024**2,
            factor_block_bytes=64 * 1024**2,
            result_queue_bytes=128 * 1024**2,
            io_concurrency=1,
            remote_concurrency=1,
            cache_budget_bytes=128 * 1024**2,
            spill_budget_bytes=0,
            pressure_state="NORMAL",
            memory_constrained=True,
            reasons=("stale_decision_conservative_fallback",),
        )

    def resource_controller_summary(self) -> dict[str, Any]:
        try:
            return self._resource_controller().to_dict()
        except Exception:
            return {}

    def summary(self) -> dict[str, Any]:
        snap = self._refresh()
        return {
            "hard_memory_limit": self.hard_memory_limit,
            "hard_cpu_slots": self.hard_cpu_slots,
            "live_headroom": snap.live_headroom,
            "host_mem_available": snap.host_mem_available,
            "process_family_rss": snap.process_family_rss,
            "process_family_pss": snap.process_family_pss,
            "spill_total_bytes": snap.spill_total_bytes,
            "spill_free_bytes": snap.spill_free_bytes,
            "disk_busy": round(snap.disk_busy, 3),
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
