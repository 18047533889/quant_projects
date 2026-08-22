"""DuckDB 连接级 PRAGMA 配置（从 engine 抽离，便于测试与 isolated connection 复用）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

from data_access.core.exceptions import ValidationError


def _sql_string(value: str) -> str:
    """DuckDB PRAGMA 字符串字面量转义。"""
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB 进程/连接级性能与 spill 配置。"""

    threads: int
    memory_limit: str | None = None
    enable_object_cache: bool = True
    preserve_insertion_order: bool = False
    temp_directory: str | None = None
    max_temp_directory_size: str | None = None


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValidationError(f"{name} 必须是数字") from exc
    if value <= 0:
        raise ValidationError(f"{name} 必须大于 0")
    return value


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValidationError(f"{name} 必须是整数") from exc
    if value <= 0:
        raise ValidationError(f"{name} 必须大于 0")
    return value


def _cgroup_memory_limit_bytes() -> int | None:
    """#P1-26 cgroup 内存上限：v2 ``memory.max`` / v1 ``memory.limit_in_bytes``。

    Docker/K8s/Slurm 里 host RAM 可能 512GB 而容器只有 32GB——按 host 估会
    OOM。effective = min(host, cgroup, RLIMIT_AS)。
    """
    candidates = []
    try:
        p = Path("/sys/fs/cgroup/memory.max")
        if p.exists():
            raw = p.read_text(encoding="ascii").strip()
            if raw.isdigit():
                candidates.append(int(raw))
    except (OSError, ValueError):
        pass
    try:
        p = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if p.exists():
            raw = p.read_text(encoding="ascii").strip()
            if raw.isdigit():
                candidates.append(int(raw))
    except (OSError, ValueError):
        pass
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY and hard > 0:
            candidates.append(hard)
    except (ValueError, OSError, ImportError):
        pass
    finite = [c for c in candidates if c > 0 and c != (1 << 63) - 1]
    return min(finite) if finite else None


def _cgroup_cpu_slots() -> int | None:
    """#P1-26 cgroup/cpuset 有效 CPU 数：``cpu.max`` quota 与
    ``cpuset.cpus.effective`` 取交集下限。
    """
    slots: list[int] = []
    try:
        p = Path("/sys/fs/cgroup/cpu.max")
        if p.exists():
            raw = p.read_text(encoding="ascii").split()
            if len(raw) >= 2 and raw[0].isdigit() and raw[1].isdigit():
                quota, period = int(raw[0]), int(raw[1])
                if period > 0 and quota > 0:
                    slots.append(max(1, quota // period))
    except (OSError, ValueError):
        pass
    try:
        p = Path("/sys/fs/cgroup/cpuset.cpus.effective")
        if p.exists():
            raw = p.read_text(encoding="ascii").strip()
            count = 0
            for part in raw.split(","):
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    if lo.isdigit() and hi.isdigit():
                        count += int(hi) - int(lo) + 1
                elif part.isdigit():
                    count += 1
            if count > 0:
                slots.append(count)
    except (OSError, ValueError):
        pass
    if not slots:
        return None
    effective = min(slots)
    host = os.cpu_count() or 1
    return max(1, min(effective, host))


def _available_memory_bytes() -> int | None:
    """Return host memory available to this process when it can be determined.

    #P1-26 effective = min(host MemAvailable, cgroup limit, RLIMIT_AS)——容器/
    Slurm 下按 host 估会 OOM。
    """
    host: int | None = None
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    value = int(line.split()[1])
                    host = value * 1024
    except (OSError, ValueError, IndexError):
        pass
    if host is None:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                host = pages * page_size
        except (OSError, ValueError, AttributeError):
            pass
    cgroup = _cgroup_memory_limit_bytes()
    if host is None:
        return cgroup
    if cgroup is None:
        return host
    return min(host, cgroup)


def _auto_memory_limit() -> str | None:
    available = _available_memory_bytes()
    if available is None:
        return None
    fraction = _parse_float_env("DUCKDB_MEMORY_FRACTION", 0.60)
    minimum = _parse_int_env("DUCKDB_MEMORY_MIN_MB", 512) * 1024 * 1024
    maximum = _parse_int_env("DUCKDB_MEMORY_MAX_MB", 65536) * 1024 * 1024
    if minimum > maximum:
        raise ValidationError("DUCKDB_MEMORY_MIN_MB 不能大于 DUCKDB_MEMORY_MAX_MB")
    limit = min(max(int(available * fraction), minimum), maximum)
    return f"{max(1, limit // (1024 * 1024))}MB"

def resolve_duckdb_config(
    *,
    threads: int | None = None,
    memory_limit: str | None = None,
    enable_object_cache: bool = True,
) -> DuckDBConfig:
    """从显式参数 + 环境变量解析 DuckDBConfig。"""
    env_threads = os.environ.get("DUCKDB_THREADS")
    if threads is not None:
        effective_threads = threads
    elif env_threads:
        try:
            effective_threads = int(env_threads)
        except ValueError as exc:
            raise ValidationError("DUCKDB_THREADS 必须是整数") from exc
    else:
        # #P1-26 cpuset/cpu.max 感知：容器 quota 限制下不能按 host CPU 数开线程。
        effective_threads = min(
            _cgroup_cpu_slots() or os.cpu_count() or 4,
            _parse_int_env("DUCKDB_MAX_THREADS", 8),
        )
    if effective_threads <= 0:
        raise ValidationError("DuckDB threads 必须大于 0")
    effective_mem = memory_limit or os.environ.get("DUCKDB_MEMORY_LIMIT")
    if effective_mem is None:
        effective_mem = _auto_memory_limit()
    temp_dir = os.environ.get("DUCKDB_TEMP_DIRECTORY") or None
    max_temp = os.environ.get("DUCKDB_MAX_TEMP_DIRECTORY_SIZE") or None
    preserve = os.environ.get("DUCKDB_PRESERVE_INSERTION_ORDER", "").lower() in {
        "1",
        "true",
        "yes",
    }
    return DuckDBConfig(
        threads=effective_threads,
        memory_limit=effective_mem,
        enable_object_cache=enable_object_cache,
        preserve_insertion_order=preserve,
        temp_directory=temp_dir,
        max_temp_directory_size=max_temp,
    )


def apply_pragmas(conn: duckdb.DuckDBPyConnection, cfg: DuckDBConfig) -> None:
    """对连接应用 PRAGMA；调用方负责在需要时加 catalog 写锁。

    依赖版本能力层：``enable_object_cache`` 在当前 DuckDB 已 deprecated 时跳过
    并告警一次（不能假设一个 PRAGMA 从 0.10 到未来版本语义永远一样）。
    """
    from .duckdb_capabilities import (
        get_duckdb_capabilities,
        warn_object_cache_once,
    )

    caps = get_duckdb_capabilities()
    conn.execute(f"PRAGMA threads={cfg.threads}")
    if cfg.memory_limit:
        conn.execute(f"PRAGMA memory_limit={_sql_string(cfg.memory_limit)}")
    if cfg.enable_object_cache:
        if caps.object_cache_deprecated:
            warn_object_cache_once()
        else:
            conn.execute("PRAGMA enable_object_cache=true")
    conn.execute(
        "PRAGMA preserve_insertion_order="
        f"{'true' if cfg.preserve_insertion_order else 'false'}"
    )
    if cfg.temp_directory:
        conn.execute(f"PRAGMA temp_directory={_sql_string(cfg.temp_directory)}")
    if cfg.max_temp_directory_size:
        conn.execute(
            "PRAGMA max_temp_directory_size="
            f"{_sql_string(cfg.max_temp_directory_size)}"
        )
