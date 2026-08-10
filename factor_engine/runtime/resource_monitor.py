# -*- coding: utf-8 -*-
"""R36 Resource Monitor：PSI + memory slope + Safe Envelope 信号采集。

R36 附录 D 的核心升级是把资源治理从「配置系统」升级为「实时反馈控制系统」。
本模块提供反馈控制所需的**输入信号**（§28/29/304/303）：
    - Linux PSI（``/proc/pressure/{cpu,memory,io}`` + cgroup ``*.pressure``）
    - cgroup ``memory.events``（low/high/oom/oom_kill）
    - swap current/max
    - host MemAvailable 随时间变化的**斜率**（§69：外部任务快速吃内存时提前让路）
    - Safe Envelope（§16/17/18/19）：Hard / Soft / Emergency 三层 + DuckDB
      outside-buffer reserve + writer burst 预留

不在这里做任何 admission 判定——判定在 :mod:`runtime.resource_autopilot`。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# PSI 读取（§28）
# ---------------------------------------------------------------------------


def _parse_psi(path: str) -> tuple[float, float] | None:
    """读取 ``/proc/pressure/X``：返回 ``(some_avg10, full_avg10)``；不可读 ``None``。"""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("some "):
                    some = _avg10_from_line(line)
                elif line.startswith("full "):
                    full = _avg10_from_line(line)
                else:
                    continue
            return (some, full)
    except (OSError, ValueError):
        return None


def _avg10_from_line(line: str) -> float:
    for tok in line.split():
        if tok.startswith("avg10="):
            return float(tok.split("=")[1])
    return 0.0


def psi_cpu() -> tuple[float, float] | None:
    return _parse_psi("/proc/pressure/cpu")


def psi_memory() -> tuple[float, float] | None:
    return _parse_psi("/proc/pressure/memory")


def psi_io() -> tuple[float, float] | None:
    return _parse_psi("/proc/pressure/io")


def cgroup_pressure(namespace: str) -> tuple[float, float] | None:
    """cgroup v2 ``cpu.pressure`` / ``memory.pressure`` / ``io.pressure``。"""
    return _parse_psi(f"/sys/fs/cgroup/{namespace}.pressure")


def read_memory_events() -> dict[str, int]:
    """cgroup ``memory.events``（low/high/max/oom/oom_kill）；不可读返回空 dict。"""
    for path in ("/sys/fs/cgroup/memory.events", "/sys/fs/cgroup/memory/memory.events"):
        try:
            with open(path, encoding="utf-8") as fh:
                return {k: int(v) for k, v in (line.split() for line in fh if line.strip())}
        except (OSError, ValueError):
            continue
    return {}


def swap_usage() -> tuple[int | None, int | None]:
    """当前 swap 使用 / swap 上限（字节）；无 swap 返回 ``(None, None)``。"""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            total = free = None
            for line in fh:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1]) * 1024
            if total in (None, 0):
                return None, None
            return max(0, total - (free or 0)), total
    except (OSError, ValueError):
        return None, None


def major_page_fault_rate(interval_s: float = 0.0) -> float:
    """当前 major page fault 速率（faults/sec，增量采样）。"""
    try:
        from resource import getrusage, RUSAGE_SELF

        now = time.monotonic()
        cur = getrusage(RUSAGE_SELF).ru_majflt
        last = getattr(major_page_fault_rate, "_last", None)
        last_t = getattr(major_page_fault_rate, "_last_t", None)
        major_page_fault_rate._last = cur  # type: ignore[attr-defined]
        major_page_fault_rate._last_t = now  # type: ignore[attr-defined]
        if last is None or last_t is None:
            return 0.0
        dt = max(0.001, now - last_t)
        return max(0.0, (cur - last) / dt)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 信号结构（§304）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSignals:
    """一次控制 tick 的全部资源输入信号（§304）。"""

    host_mem_available: int = 0
    cgroup_mem_current: int | None = None
    family_rss: int = 0
    family_pss: int = 0
    memory_psi_some: float = 0.0
    memory_psi_full: float = 0.0
    cpu_util: float = 0.0
    cpu_psi_some: float = 0.0
    io_psi_some: float = 0.0
    disk_latency_ms: float = 0.0
    writer_backpressure: float = 0.0
    scan_backpressure: float = 0.0
    mem_available_slope: float = 0.0  # bytes/sec（负 = 快速被吃）
    memory_events: dict[str, int] = field(default_factory=dict)
    swap_current: int | None = None
    swap_max: int | None = None
    major_fault_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out = {
            "host_mem_available": self.host_mem_available,
            "cgroup_mem_current": self.cgroup_mem_current,
            "family_rss": self.family_rss,
            "family_pss": self.family_pss,
            "memory_psi_some": round(self.memory_psi_some, 4),
            "memory_psi_full": round(self.memory_psi_full, 4),
            "cpu_util": round(self.cpu_util, 4),
            "cpu_psi_some": round(self.cpu_psi_some, 4),
            "io_psi_some": round(self.io_psi_some, 4),
            "disk_latency_ms": round(self.disk_latency_ms, 3),
            "writer_backpressure": round(self.writer_backpressure, 4),
            "scan_backpressure": round(self.scan_backpressure, 4),
            "mem_available_slope": round(self.mem_available_slope, 3),
            "memory_events": dict(self.memory_events),
            "swap_current": self.swap_current,
            "swap_max": self.swap_max,
            "major_fault_rate": round(self.major_fault_rate, 3),
        }
        return out


# ---------------------------------------------------------------------------
# Safe Envelope（§16/17/18/19）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostResourceEnvelope:
    """三层内存 Envelope + CPU/IO 预算（§303）。"""

    hard_memory_bytes: int
    safe_memory_bytes: int
    emergency_reserve_bytes: int
    hard_cpu_tokens: int
    target_cpu_tokens: int
    io_capacity_score: float
    remote_capacity_score: float
    spill_free_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_memory_bytes": self.hard_memory_bytes,
            "safe_memory_bytes": self.safe_memory_bytes,
            "emergency_reserve_bytes": self.emergency_reserve_bytes,
            "hard_cpu_tokens": self.hard_cpu_tokens,
            "target_cpu_tokens": self.target_cpu_tokens,
            "io_capacity_score": round(self.io_capacity_score, 3),
            "remote_capacity_score": round(self.remote_capacity_score, 3),
            "spill_free_bytes": self.spill_free_bytes,
        }


#: DuckDB 官方：memory_limit 只约束 buffer manager；vectors/query results/
#: aggregate states 在其外分配（§19）。预留为 process budget 的 10%（R36-P0-*）。
DUCKDB_OUTSIDE_BUFFER_RESERVE_FRACTION = 0.10
#: writer burst 预留：Parquet writer / CH 写突发（§17）。
WRITER_BURST_FRACTION = 0.03
#: native/untracked 分配（Arrow buffers / 共享库 / 内核临时）预留。
UNTRACKED_NATIVE_FRACTION = 0.05


def compute_safe_envelope(
    *,
    hard_memory_bytes: int,
    live_headroom_bytes: int,
    emergency_reserve_bytes: int | None = None,
    family_pss: int = 0,
) -> HostResourceEnvelope:
    """Safe Envelope = live headroom - 各层 reserve（§17）。

    EmergencyReserve 不能只用固定百分比（§18）：提供显式值时优先，否则按
    max(absolute_floor, fraction_of_hard) 计算。
    """
    hard = max(1, int(hard_memory_bytes))
    if emergency_reserve_bytes is None or emergency_reserve_bytes <= 0:
        floor = max(512 * 1024**2, int(hard * 0.05))
        emergency = max(floor, int(hard * 0.15))
    else:
        emergency = max(0, int(emergency_reserve_bytes))
    untracked = int(hard * UNTRACKED_NATIVE_FRACTION)
    writer_burst = int(hard * WRITER_BURST_FRACTION)
    safe = max(0, int(live_headroom_bytes) - emergency - untracked - writer_burst)
    return HostResourceEnvelope(
        hard_memory_bytes=hard,
        safe_memory_bytes=safe,
        emergency_reserve_bytes=emergency,
        hard_cpu_tokens=1,
        target_cpu_tokens=1,
        io_capacity_score=0.0,
        remote_capacity_score=0.0,
        spill_free_bytes=0,
    )


# ---------------------------------------------------------------------------
# Memory 斜率（§69）
# ---------------------------------------------------------------------------


class MemorySlopeTracker:
    """MemAvailable 时间斜率（bytes/sec），EWMA 平滑。

    若过去数秒 MemAvailable 快速下降（外部任务正在吃内存），即使当前值还高，
    也要**提前**降 admission（§69/70 predictive pressure）。
    """

    def __init__(self, *, window: int = 5, ema_alpha: float = 0.3) -> None:
        self._window = max(2, int(window))
        self._alpha = float(ema_alpha)
        self._points: list[tuple[float, float]] = []  # (t_s, avail)
        self._ema: float = 0.0
        self._seen = False

    def update(self, t_s: float, mem_available: int) -> float:
        """记录一个采样点，返回当前斜率（bytes/sec，负 = 正在被吃）。

        同时间戳的重复点直接忽略（一个 control tick 内多次 force 采样不稀释斜率）。
        """
        if self._points and t_s == self._points[-1][0]:
            return self._ema
        self._points.append((t_s, float(mem_available)))
        if len(self._points) > self._window:
            self._points.pop(0)
        if len(self._points) < 2:
            return 0.0
        t0, a0 = self._points[0]
        t1, a1 = self._points[-1]
        dt = max(0.001, t1 - t0)
        slope = (a1 - a0) / dt
        if not self._seen:
            self._ema = slope
            self._seen = True
        else:
            self._ema = self._alpha * slope + (1 - self._alpha) * self._ema
        return self._ema

    @property
    def slope(self) -> float:
        return self._ema
