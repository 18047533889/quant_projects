# -*- coding: utf-8 -*-
"""全局资源治理：真实资源发现 + ExecutionResourcePlan + 运行时 RSS Governor。

Phase 5 R1-R3 / P1-3 / P1-13 / P1-15。目标：让 FactorEngine 在不同配置服务器
（8GB~128GB、Docker/K8s/SLURM、本地盘 / COS）上不 OOM、不 oversubscribe。

资源上限按**最严格来源**取最小值（用户显式配置 > cgroup v2 > cgroup v1 >
SLURM > RLIMIT > host RAM），绝不能只信 ``psutil.virtual_memory().total``——
比如物理内存 256GB、容器只给 16GB 时拿 256GB×0.7 必炸。

设计
    - :func:`effective_cpu_slots` / :func:`effective_memory_limit_bytes`：进程真实上限
    - :func:`build_execution_resource_plan`：单次执行的完整预算（含各 cache 层/结果/spill）
    - :class:`MemoryGovernor`：统一 byte 预算 + 运行时 RSS 分档治理，可挂到各 cache 层
    - :class:`ExecutionResourceScope`：context manager，设置资源 → 执行 → finally 恢复
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from runtime.resource_errors import ResourceBudgetExceeded, ResourceContractApplyError

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 真实资源发现
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _read_cgroup_v2_max() -> int | None:
    """cgroup v2 ``memory.max``（字节）；``max`` 表示无限。"""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.max"):
        try:
            raw = Path(p).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw == "max":
            return None
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _read_cgroup_v1_max() -> int | None:
    """cgroup v1 ``memory.limit_in_bytes``。"""
    for p in ("/sys/fs/cgroup/memory/memory.limit_in_bytes",):
        try:
            raw = Path(p).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        try:
            value = int(raw)
            # 极大值表示无限制（如 2^63-1 或 2^63）
            if value <= 0 or value >= (1 << 62):
                return None
            return value
        except ValueError:
            continue
    return None


def _read_cgroup_cpu_quota() -> tuple[int, int] | None:
    """cgroup v2 ``cpu.max``（quota, period）；无限返回 ``None``。"""
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        quota, period = (int(x) for x in raw.split())
    except ValueError:
        return None
    if quota < 0 or period <= 0:
        return None
    return quota, period


def _read_rlimit_mem() -> int | None:
    """RLIMIT_AS / RLIMIT_DATA 软上限（字节）；两者均无限返回 ``None``。

    审计 #338：对 RLIMIT_AS 与 RLIMIT_DATA 都取 finite 的 soft 值，返回其中
    **最小值**（不是第一个找到的）；两者都是 ``RLIM_INFINITY`` → ``None``。
    """
    try:
        import resource

        finite: list[int] = []
        for which in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            soft, _hard = resource.getrlimit(which)
            if soft != resource.RLIM_INFINITY:
                finite.append(int(soft))
        if not finite:
            return None
        return min(finite)
    except Exception:
        return None


def _host_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _host_mem_available_bytes() -> int:
    """当前可用内存（MemAvailable）——live headroom 的主机侧输入（R27-023/025）。"""
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


def _cgroup_memory_current_bytes() -> int | None:
    """cgroup v2 ``memory.current``（R27-024）；不可读返回 ``None``。"""
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.current"):
        try:
            raw = Path(p).read_text(encoding="utf-8").strip()
            return int(raw)
        except (OSError, ValueError):
            continue
    return None


def process_family_rss_bytes(*, prefer_pss: bool = False) -> int:
    """主进程 + 递归子进程的 RSS/PSS 之和（R27-036/238）。

    Linux ``/proc/*/smaps_rollup`` 可读时优先 PSS（共享内存更准）；否则 RSS。
    """
    try:
        import psutil  # type: ignore[import-untyped]

        def _one(proc: Any) -> tuple[int, int]:
            mi = proc.memory_info()
            rss = int(getattr(mi, "rss", 0))
            pss = int(getattr(mi, "pss", 0)) or 0
            if pss == 0:
                try:
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
        return total_pss if prefer_pss and total_pss else total_rss
    except Exception:
        return 0


def live_memory_headroom_bytes(
    *,
    hard_limit: int | None = None,
    min_host_reserve_gb: float = 8.0,
    min_host_reserve_fraction: float = 0.15,
) -> int:
    """live headroom = min(cgroup_headroom, host_headroom, configured_headroom)（R27-024..027）。

    - ``cgroup_headroom = memory.max - memory.current``（无限则略过）
    - ``host_headroom = MemAvailable - external_reserve``
    - ``configured_headroom = hard_limit - FE_process_family_RSS``
    """
    limit = hard_limit or effective_memory_limit_bytes()
    candidates: list[int] = []
    cur = _cgroup_memory_current_bytes()
    if cur is not None:
        cgroup_headroom = max(0, limit - cur)
        candidates.append(cgroup_headroom)
    reserve = max(int(min_host_reserve_gb * 1024**3),
                  int(limit * min_host_reserve_fraction))
    host_headroom = max(0, _host_mem_available_bytes() - reserve)
    candidates.append(host_headroom)
    configured = max(0, limit - process_family_rss_bytes())
    candidates.append(configured)
    return min(candidates) if candidates else 0


def _default_per_worker_peak_bytes() -> int:
    """每 worker 峰值内存（字节）：env 覆盖，默认 3GiB（R27-021/167）。"""
    env = os.environ.get("FACTOR_ENGINE_PER_WORKER_PEAK_BYTES", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    legacy = os.environ.get("FACTOR_ENGINE_PER_WORKER_PEAK_MB", "").strip()
    if legacy.isdigit() and float(legacy) > 0:
        return int(float(legacy) * 1024 * 1024)
    return 3 * 1024**3


def effective_memory_limit_bytes(
    *,
    explicit_bytes: int | None = None,
    explicit_env: str = "FACTOR_ENGINE_MAX_MEMORY_BYTES",
) -> int:
    """进程真实内存上限（最严格来源取最小值）。

    审计 #336：显式配置 / 环境变量都不能超过探测到的 **hard limit**。
    流程：
        1. 探测 hard 候选（cgroup v2 > cgroup v1 > SLURM > RLIMIT > host RAM），
           取 min 作为 ``hard_limit``；
        2. ``requested = explicit_bytes(>0) or env or hard_limit``；
        3. 返回 ``min(requested, hard_limit)``。

    即显式配置 64GB + 容器 8GB → 8GB。完全无法探测时 hard_limit 回退 8GiB。
    """
    candidates: list[int] = []
    cg2 = _read_cgroup_v2_max()
    if cg2 is not None:
        candidates.append(cg2)
    cg1 = _read_cgroup_v1_max()
    if cg1 is not None:
        candidates.append(cg1)
    slurm = _env_int("SLURM_MEM_PER_NODE", None)
    if slurm is not None:
        candidates.append(slurm * 1024 * 1024)  # MB → bytes
    rlimit = _read_rlimit_mem()
    if rlimit is not None:
        candidates.append(rlimit)
    host = _host_memory_bytes()
    if host is not None:
        candidates.append(host)
    hard_limit = min(candidates) if candidates else 8 * 1024**3

    requested = hard_limit
    if explicit_bytes is not None and explicit_bytes > 0:
        requested = explicit_bytes
    else:
        env = _env_int(explicit_env, None)
        if env is not None:
            requested = env
    return min(requested, hard_limit)


def _probe_hard_cpu_limit() -> int:
    """探测进程真实 CPU 硬上限（取最严格约束的最小值）。

    与内存的 ``hard_limit`` 对齐：cgroup v2 ``cpu.max`` quota 与 sched affinity
    都是硬约束，取 **min**（容器 quota 2 + affinity 4 → 2；quota 8 + affinity 4
    → 4）。没有 cgroup 时用 ``sched_getaffinity``；再不行回退 ``cpu_count``。
    """
    candidates: list[int] = []
    qp = _read_cgroup_cpu_quota()
    if qp is not None:
        quota, period = qp
        # 审计 #337：小数 quota 用 ceil（0.5 quota → 1 slot）。
        candidates.append(max(1, math.ceil(quota / period)))
    try:
        import os as _os

        candidates.append(max(1, len(_os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        try:
            import multiprocessing

            candidates.append(max(1, multiprocessing.cpu_count()))
        except Exception:  # pragma: no cover
            pass
    if not candidates:
        return 4
    return min(candidates)


def effective_cpu_slots(
    *,
    explicit: int | None = None,
    env: str = "FACTOR_ENGINE_CPU_BUDGET",
    hard_limit: int | None = None,
) -> int:
    """进程真实 CPU slot 数（cgroup quota 感知），替代 naive ``physical_cores``。

    R20-138：与 :func:`effective_memory_limit_bytes` 同构——先探测 **hard limit**
    （显式 ``hard_limit`` 参数 > cgroup v2 ``cpu.max`` quota > sched affinity >
    host CPU），再取 ``min(requested, hard_limit)``。显式配置 / 环境变量都**不能**
    超过 hard limit：容器 2 CPU + explicit 64 → 最终 2（不再 explicit 直接 return）。

    - 显式参数 / ``FACTOR_ENGINE_CPU_BUDGET`` 优先作为 requested
    - ``hard_limit`` 显式参数供测试注入模拟容器配额
    """
    hard = hard_limit if hard_limit is not None and hard_limit > 0 else _probe_hard_cpu_limit()
    requested = hard
    if explicit is not None and explicit > 0:
        requested = explicit
    else:
        env_v = _env_int(env, None)
        if env_v is not None:
            requested = env_v
    return max(1, min(requested, hard))


def spill_disk_available(path: str | os.PathLike | None = None) -> int | None:
    """目标目录剩余可写字节数；探测失败返回 ``None``。"""
    target = Path(path or tempfile_dir())
    try:
        st = os.statvfs(target)
        return int(st.f_bavail * st.f_frsize)
    except OSError:
        return None


def tempfile_dir() -> str:
    """spill 临时目录（``FACTOR_ENGINE_SPILL_DIR`` 优先，否则系统 temp）。"""
    raw = os.environ.get("FACTOR_ENGINE_SPILL_DIR", "").strip()
    if raw:
        return raw
    import tempfile

    return tempfile.gettempdir()


def spill_disk_speed_class() -> str:
    """spill 盘速度等级（nvme / ssd / network / unknown）。"""
    target = Path(tempfile_dir())
    try:
        import psutil  # type: ignore[import-untyped]

        for part in psutil.disk_partitions(all=True):
            try:
                if target.relative_to(part.mountpoint):
                    fstype = (part.fstype or "").lower()
                    if any(k in fstype for k in ("nfs", "cifs", "smb", "fuse", "sshfs")):
                        return "network"
                    if "nvme" in (part.device or "").lower():
                        return "nvme"
                    return "ssd"
            except ValueError:
                continue
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# ExecutionResourcePlan
# ---------------------------------------------------------------------------

_DEFAULT_FRACTIONS = {
    "process": 0.75,
    "duckdb": 0.45,  # 相对 process budget
    "factor_cache": 0.30,
    "result": 0.10,
    "reserve": 0.15,  # 相对 process budget（进程外开销）
}


@dataclass(frozen=True)
class ExecutionResourcePlan:
    """一次执行的完整资源预算（R1）。"""

    effective_memory_limit_bytes: int
    process_budget_bytes: int
    reserve_bytes: int
    duckdb_budget_bytes: int
    data_cache_budget_bytes: int
    cse_budget_bytes: int
    panel_budget_bytes: int
    result_budget_bytes: int
    spill_budget_bytes: int
    max_workers: int
    duckdb_threads: int
    polars_threads: int
    io_concurrency: int
    spill_dir: str
    spill_disk_bytes: int | None
    # R27-021/167：每 worker 峰值内存（字节）——RAM 硬约束 worker 数的真实依据。
    per_worker_peak_bytes: int = 3 * 1024**3

    @classmethod
    def _memory_bounded_workers(
        cls,
        workers: int,
        process_budget_bytes: int,
        per_worker_peak_bytes: int | None,
    ) -> int:
        """``n_jobs <= floor(process_budget / per_worker_peak)``（R27-021/167）。

        让每 worker 峰值内存**真正**限制并发，而不是只有注释声称。32 核 16GB
        机器、每 worker 3GB → 最多 5 个 worker。
        """
        if per_worker_peak_bytes is None or per_worker_peak_bytes <= 0:
            return workers
        by_memory = max(1, process_budget_bytes // per_worker_peak_bytes)
        return max(1, min(workers, by_memory))

    @classmethod
    def auto(cls, *, max_workers: int | None = None) -> "ExecutionResourcePlan":
        """按真实资源自动分配（``resources.mode=auto``）。"""
        mem = effective_memory_limit_bytes()
        process = int(mem * _DEFAULT_FRACTIONS["process"])
        duckdb = int(process * _DEFAULT_FRACTIONS["duckdb"])
        factor_cache = int(process * _DEFAULT_FRACTIONS["factor_cache"])
        result = int(process * _DEFAULT_FRACTIONS["result"])
        reserve = int(mem * 0.20)  # Python interpreter / allocator / OS page cache
        spill = int(mem * 0.15)
        cpu = effective_cpu_slots()
        workers = max_workers if max_workers and max_workers > 0 else cpu
        workers = max(1, min(workers, cpu))
        # R27-021/167：每 worker 峰值内存真正约束 worker 数。
        per_worker_peak = _default_per_worker_peak_bytes()
        workers = cls._memory_bounded_workers(workers, process, per_worker_peak)
        # DuckDB 官方：memory_limit 只覆盖部分分配，建议 50-60% 系统内存 + 降 threads。
        # 避免 oversubscription：n_jobs × duckdb_threads <= cpu。
        duckdb_threads = max(1, min(cpu, cpu // workers if workers else cpu))
        io = max(1, min(cpu, workers * 2))
        spill_dir = tempfile_dir()
        return cls(
            effective_memory_limit_bytes=mem,
            process_budget_bytes=process,
            reserve_bytes=reserve,
            duckdb_budget_bytes=duckdb,
            data_cache_budget_bytes=int(factor_cache * 0.35),
            cse_budget_bytes=int(factor_cache * 0.40),
            panel_budget_bytes=int(factor_cache * 0.25),
            result_budget_bytes=result,
            spill_budget_bytes=spill,
            max_workers=workers,
            duckdb_threads=duckdb_threads,
            polars_threads=duckdb_threads,
            io_concurrency=io,
            spill_dir=spill_dir,
            spill_disk_bytes=spill_disk_available(spill_dir),
            per_worker_peak_bytes=per_worker_peak,
        )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "ExecutionResourcePlan | None":
        """从 ``resources`` 配置 dict 构造；非法配置在 production 下必须显式失败。"""
        if not cfg:
            return cls.auto()
        mode = str(cfg.get("mode") or "auto").lower()
        if mode == "auto" and not any(
            k in cfg for k in ("memory", "concurrency", "cache", "spill", "results")
        ):
            return cls.auto()
        base = cls.auto()
        mem_cfg = cfg.get("memory") if isinstance(cfg.get("memory"), dict) else {}
        limit_bytes = _cfg_bytes(mem_cfg.get("limit"), base.effective_memory_limit_bytes)
        process_frac = float(mem_cfg.get("process_fraction", _DEFAULT_FRACTIONS["process"]))
        reserve_gb = float(mem_cfg.get("reserve_gb", 0.0))
        if not (0.05 <= process_frac <= 0.95):
            raise ValueError(f"resources.memory.process_fraction 必须在 [0.05, 0.95]，得到 {process_frac}")
        process = int(limit_bytes * process_frac)
        reserve = int(reserve_gb * 1024**3) if reserve_gb > 0 else int(limit_bytes * 0.20)
        concurrency = cfg.get("concurrency") if isinstance(cfg.get("concurrency"), dict) else {}
        cache_cfg = cfg.get("cache") if isinstance(cfg.get("cache"), dict) else {}
        spill_cfg = cfg.get("spill") if isinstance(cfg.get("spill"), dict) else {}
        results_cfg = cfg.get("results") if isinstance(cfg.get("results"), dict) else {}

        total_cache = int(process * float(cache_cfg.get("total_fraction", 0.20)))
        cse_frac = float(cache_cfg.get("cse_fraction", 0.40))
        panel_frac = float(cache_cfg.get("panel_fraction", 0.25))
        column_frac = float(cache_cfg.get("column_fraction", 0.35))
        duckdb_frac = float(cache_cfg.get("duckdb_fraction", 0.45))
        if not (0.05 <= duckdb_frac <= 0.95):
            raise ValueError(
                f"resources.cache.duckdb_fraction 必须在 [0.05, 0.95]，得到 {duckdb_frac}"
            )

        workers_cfg = int(concurrency.get("max_workers") or 0) or base.max_workers
        workers = max(1, min(workers_cfg, effective_cpu_slots()))
        # R27-021/167：每 worker 峰值内存真正约束 worker 数（RAM 硬约束）。
        per_worker_peak = int(
            mem_cfg.get("per_worker_peak_bytes")
            or _default_per_worker_peak_bytes()
        )
        workers = cls._memory_bounded_workers(workers, process, per_worker_peak)
        spill_dir = str(spill_cfg.get("directory") or tempfile_dir())
        spill_enabled = bool(spill_cfg.get("enabled", True))
        spill_budget = _cfg_bytes(spill_cfg.get("max_size"), int(limit_bytes * 0.15)) if spill_enabled else 0
        result_frac = float(results_cfg.get("max_in_memory_fraction", 0.10))
        duckdb_threads = max(1, int(concurrency.get("duckdb_threads") or 0) or base.duckdb_threads)

        return ExecutionResourcePlan(
            effective_memory_limit_bytes=limit_bytes,
            process_budget_bytes=process,
            reserve_bytes=reserve,
            duckdb_budget_bytes=int(process * duckdb_frac),
            data_cache_budget_bytes=int(total_cache * column_frac),
            cse_budget_bytes=int(total_cache * cse_frac),
            panel_budget_bytes=int(total_cache * panel_frac),
            result_budget_bytes=int(process * result_frac),
            spill_budget_bytes=spill_budget,
            max_workers=workers,
            duckdb_threads=duckdb_threads,
            polars_threads=max(1, int(concurrency.get("polars_threads") or 0) or duckdb_threads),
            io_concurrency=max(1, int(concurrency.get("io_concurrency") or 0) or base.io_concurrency),
            spill_dir=spill_dir,
            spill_disk_bytes=spill_disk_available(spill_dir),
            per_worker_peak_bytes=per_worker_peak,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_memory_limit_bytes": self.effective_memory_limit_bytes,
            "process_budget_bytes": self.process_budget_bytes,
            "reserve_bytes": self.reserve_bytes,
            "duckdb_budget_bytes": self.duckdb_budget_bytes,
            "data_cache_budget_bytes": self.data_cache_budget_bytes,
            "cse_budget_bytes": self.cse_budget_bytes,
            "panel_budget_bytes": self.panel_budget_bytes,
            "result_budget_bytes": self.result_budget_bytes,
            "spill_budget_bytes": self.spill_budget_bytes,
            "max_workers": self.max_workers,
            "duckdb_threads": self.duckdb_threads,
            "polars_threads": self.polars_threads,
            "io_concurrency": self.io_concurrency,
            "spill_dir": self.spill_dir,
            "spill_disk_bytes": self.spill_disk_bytes,
            "per_worker_peak_bytes": self.per_worker_peak_bytes,
        }


def _cfg_bytes(value: Any, default: int) -> int:
    """解析配置字节值：``"8GB"`` / ``"512MB"`` / 整数（字节） / ``"auto"``。"""
    if value is None or str(value).lower() in {"auto", ""}:
        return default
    if isinstance(value, int):
        return value
    s = str(value).strip().upper()
    mult = 1
    for suffix, m in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if s.endswith(suffix):
            mult = m
            s = s[: -len(suffix)]
            break
    try:
        return int(float(s) * mult)
    except ValueError as exc:
        raise ValueError(f"非法内存配置值: {value!r}") from exc


# ---------------------------------------------------------------------------
# 对象字节估算（R7）
# ---------------------------------------------------------------------------


def estimate_object_bytes(value: Any) -> int:
    """按类型估算对象字节占用（比 ``nbytes`` 准）。

    - pd.Series：``memory_usage(index=True, deep=True)``
    - pd.DataFrame：``memory_usage(index=True, deep=True).sum()``
    - np.ndarray / pyarrow.Table：``nbytes``
    - polars DataFrame：``estimated_size()``
    - dict/list：浅层累加（避免递归爆炸）
    """
    if value is None:
        return 0
    try:
        import pandas as pd

        if isinstance(value, pd.Series):
            return int(value.memory_usage(index=True, deep=True))
        if isinstance(value, pd.DataFrame):
            return int(value.memory_usage(index=True, deep=True).sum())
    except ImportError:  # pragma: no cover
        pass
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and nbytes > 0:
        return nbytes
    est = getattr(value, "estimated_size", None)
    if callable(est):
        try:
            return int(est())
        except Exception:
            pass
    if isinstance(value, dict):
        return sum(estimate_object_bytes(v) for v in value.values()) + 256
    if isinstance(value, (list, tuple)):
        return sum(estimate_object_bytes(v) for v in value[:512]) + 128
    return 0


# ---------------------------------------------------------------------------
# MemoryGovernor（R2 / R6）
# ---------------------------------------------------------------------------

#: 运行时 RSS 分档（相对 process budget）——stage 边界（越往上压力越大）：
#: normal < stop_warmup < evict_lru < spill < throttle < critical
_STAGE_STOP_WARMUP = 0.70
_STAGE_EVICT_LRU = 0.80
_STAGE_SPILL = 0.85
_STAGE_THROTTLE = 0.90
_STAGE_CRITICAL = 0.95


@dataclass
class MemoryGovernor:
    """统一 byte 预算 + 运行时 RSS 分档治理。

    可被多个 cache 层共享：每层 ``reserve/release`` 记账，超 budget 时触发
    注册的 evict 回调（LRU）；RSS 超过阈值时分档 throttle。**不**用
    ``gc.collect()`` 当治理手段。

    R20-119..124：``_usage`` / ``_evict_hooks`` / ``throttles`` / ``evictions``
    是共享 mutable state，threading 下由 ``_lock``（``threading.RLock``）保护。
    所有 mutate 方法（reserve/release/reserve_accounting/release_accounting/
    _evict_for/release_all/register_layer/unregister_layer/throttle）都在锁内执行，
    ``total_usage`` 读也加锁。RLock 可重入：evict hook 内部再调
    ``release_accounting`` 不会死锁。
    """

    process_budget_bytes: int
    duckdb_budget_bytes: int
    rss_probe: Callable[[], int] | None = None
    _usage: dict[str, int] = field(default_factory=dict)
    _evict_hooks: dict[str, Callable[[int], int]] = field(default_factory=dict)
    throttles: list[str] = field(default_factory=list)
    evictions: list[str] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @property
    def lock(self) -> threading.RLock:
        """Governor 内部锁（供 cache 层以相同顺序加锁，避免锁顺序反转死锁）。"""
        return self._lock

    @property
    def total_usage(self) -> int:
        """全部登记层的字节占用之和。"""
        with self._lock:
            return sum(self._usage.values())

    def register_layer(self, name: str, evict: Callable[[int], int]) -> None:
        """注册缓存层：``evict(target_bytes) -> freed_bytes``。"""
        with self._lock:
            self._evict_hooks[name] = evict

    def unregister_layer(self, name: str) -> None:
        """从 evict hooks 与记账中移除某层（审计 #333/#334）。"""
        with self._lock:
            self._evict_hooks.pop(name, None)
            self._usage.pop(name, None)

    def reserve_accounting(self, name: str, bytes_: int) -> None:
        """**仅记账**（不触发 evict hook）：cache 内部 ``set`` 时调用（审计 #333）。

        避免递归：cache 已有自己的 LRU 预算逐出，governor 不应在每次 set 时
        再触发一轮 evict hook。
        """
        bytes_ = max(0, int(bytes_))
        with self._lock:
            self._usage[name] = self._usage.get(name, 0) + bytes_

    def release_accounting(self, name: str, bytes_: int) -> None:
        """**仅记账**（不触发 evict hook）：cache 内部 evict/release 时调用（审计 #333）。"""
        bytes_ = max(0, int(bytes_))
        with self._lock:
            self._usage[name] = max(0, self._usage.get(name, 0) - bytes_)

    def reserve(self, name: str, bytes_: int) -> bool:
        """某层尝试占用 ``bytes_`` 字节；超出全局 budget 时触发 evict。

        返回是否成功；失败时调用方应放弃缓存该对象（宁可重算，不要 OOM）。
        """
        bytes_ = max(0, int(bytes_))
        with self._lock:
            if self.total_usage + bytes_ > self.process_budget_bytes:
                freed = self._evict_for(bytes_)
                if self.total_usage + bytes_ > self.process_budget_bytes:
                    return False
            self.reserve_accounting(name, bytes_)
            return True

    def _evict_for(self, needed: int) -> int:
        """逐出 LRU 缓存直到腾出 ``needed`` 字节。

        审计 #333：逐出记账统一走 ``release_accounting``（hook 本身只逐出并返回
        释放字节数，不直接改 ``_usage``，避免与这里重复扣减）。

        在 ``_lock`` 内调用 evict hook（hook 通过 ``release_accounting`` 重入
        RLock 安全）；lock 顺序恒为 governor → cache（调用方 ``set`` 也先取
        ``gov.lock``），不会锁反转。
        """
        freed = 0
        for name, hook in self._evict_hooks.items():
            if self.total_usage + needed <= self.process_budget_bytes:
                break
            target = max(0, self.total_usage + needed - self.process_budget_bytes)
            n = hook(target)
            if n > 0:
                self.release_accounting(name, n)
                freed += n
                self.evictions.append(f"{name}:{n}")
        return freed

    def release(self, name: str, bytes_: int) -> None:
        """释放某层占用的字节。"""
        self.release_accounting(name, bytes_)

    def release_all(self, name: str) -> None:
        """清空某层全部记账。"""
        with self._lock:
            self._usage.pop(name, None)

    # -- 运行时 RSS 分档 --

    def _current_rss(self) -> int:
        if self.rss_probe is not None:
            try:
                rss = self.rss_probe()
                if rss is not None and rss > 0:
                    return int(rss)
            except Exception:
                pass
        # 审计 #341：优先用真实当前 RSS；不可用时读 /proc/self/status VmRSS；
        # 最后才 fallback ru_maxrss。
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
        try:
            import resource

            # 注意：ru_maxrss 是历史峰值，不是当前值；仅在以上真实 RSS 探测
            # 均不可用时才作兜底。
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            return 0

    def pressure_stage(self) -> str:
        """返回当前压力档位：normal / stop_warmup / evict_lru / spill / throttle / critical。"""
        budget = self.process_budget_bytes or 1
        frac = self._current_rss() / budget
        if frac >= _STAGE_CRITICAL:
            return "critical"
        if frac >= _STAGE_THROTTLE:
            return "throttle"
        if frac >= _STAGE_SPILL:
            return "spill"
        if frac >= _STAGE_EVICT_LRU:
            return "evict_lru"
        if frac >= _STAGE_STOP_WARMUP:
            return "stop_warmup"
        return "normal"

    def throttle(self, stage: str) -> None:
        """记录一次 throttle 事件（调用方根据 stage 执行对应动作）。"""
        with self._lock:
            self.throttles.append(stage)

    def check_pre_warmup(self) -> bool:
        """是否允许继续预热新 cache。

        R20-128..131：``stop_warmup`` 及以上**不允许**继续预热 —— 文档语义是
        stop_warmup 代表「内存吃紧，别再往 cache 里加东西」。只有 ``normal``
        档才返回 True。
        """
        return self.pressure_stage() == "normal"

    def check_admit(self, size_bytes: int, *, factor: float = 1.0) -> bool:
        """新对象是否可 admit；超 process budget 时触发 evict 后仍不够则拒绝。

        R20-125..127：旧实现超预算时 ``reserve("__probe__", size)`` 会留下
        ``__probe__`` ghost accounting（只探测、不真的存对象却记账）。现改为
        无副作用的 :meth:`can_admit`：必要时先 ``_evict_for``，再判断
        ``total_usage + size <= budget``；成功返回 True 且不记账，失败返回 False
        也不记账。
        """
        return self.can_admit(size_bytes, factor=factor)

    def can_admit(self, size_bytes: int, *, factor: float = 1.0) -> bool:
        """无副作用 admission probe（R20-125..127）。

        只在必要时触发 evict（真实逐出缓存），**从不**为探测本身记账。返回
        True/False 且 ``_usage`` 不含 ``__probe__``。
        """
        size = max(0, int(size_bytes * factor))
        with self._lock:
            if self.total_usage + size > self.process_budget_bytes:
                self._evict_for(size)
            return self.total_usage + size <= self.process_budget_bytes

    # -- R27-031/168/236：live headroom 动态 admission（不只 this-process RSS vs
    #    static budget，还要看 MemAvailable / cgroup current / 进程族 RSS） --

    def live_headroom_bytes(
        self,
        *,
        min_host_reserve_gb: float = 8.0,
        min_host_reserve_fraction: float = 0.15,
    ) -> int:
        """当前 live headroom（R27-024..027），作为动态 admission 依据。"""
        return live_memory_headroom_bytes(
            hard_limit=self.process_budget_bytes,
            min_host_reserve_gb=min_host_reserve_gb,
            min_host_reserve_fraction=min_host_reserve_fraction,
        )

    def can_admit_live(
        self,
        size_bytes: int,
        *,
        factor: float = 1.0,
        min_host_reserve_gb: float = 8.0,
        min_host_reserve_fraction: float = 0.15,
    ) -> bool:
        """live-headroom 感知 admission（R27-131：外部任务内存上涨 → MemAvailable
        下降 → 尽早拒绝，而不是等自己 RSS 到 90% 才反应）。

        同时满足两个条件才 admit：
            1. 内部记账 ``total_usage + size <= process_budget``（旧约束）；
            2. ``size * factor <= live_headroom``（外部共存约束）。
        """
        size = max(0, int(size_bytes * factor))
        with self._lock:
            if self.total_usage + size > self.process_budget_bytes:
                self._evict_for(size)
            if self.total_usage + size > self.process_budget_bytes:
                return False
        headroom = self.live_headroom_bytes(
            min_host_reserve_gb=min_host_reserve_gb,
            min_host_reserve_fraction=min_host_reserve_fraction,
        )
        if size > headroom:
            self.throttle("live_headroom_blocked")
            return False
        return True

    def external_pressure_stage(self) -> str:
        """R27-131：外部负载导致的内存压力档（normal / blocked / critical）。

        仅看 host MemAvailable 相对外部 reserve 的比例——外部任务突然吃内存时，
        即使本进程 RSS 不高也会进入更高档，触发停止 admission。
        """
        headroom = self.live_headroom_bytes()
        budget = max(1, self.process_budget_bytes)
        reserve = int(budget * 0.15)
        if headroom <= 0:
            return "critical"
        if headroom <= reserve * 1.5:
            return "throttle"
        if headroom <= reserve * 3.0:
            return "stop_warmup"
        return "normal"

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "process_budget_bytes": self.process_budget_bytes,
                "usage_bytes": self.total_usage,
                "usage_by_layer": dict(self._usage),
                "pressure_stage": self.pressure_stage(),
                "throttles": list(self.throttles),
                "evictions": list(self.evictions),
            }


# ---------------------------------------------------------------------------
# ExecutionResourceScope（R15）
# ---------------------------------------------------------------------------


#: 进程级 RLock：``ExecutionResourceScope`` 修改 DUCKDB/POLARS 等 process-global
#: env 时加锁，避免两个并发 engine request 的 enter/exit 交错恢复 previous
#: （R20-138..145）。
_ENV_LOCK = threading.RLock()


class ExecutionResourceScope:
    """context manager：进入时应用 ExecutionResourcePlan，退出时恢复环境。

    负责把 DuckDB threads / Polars threads 等动态设置写入环境并在 finally 恢复，
    避免并行 worker 共享 mutable 状态、改完不还原。

    R20-132..137：``strict`` 显式传入真实 run_mode 语义（production → fail-closed），
    不再从 ``FACTOR_ENGINE_RUN_MODE`` 环境变量猜测（env 只在未显式给出时兜底）。
    R20-138..145：enter/exit 对 process-global env 的修改由 ``_ENV_LOCK`` 串行化。
    """

    def __init__(
        self,
        plan: ExecutionResourcePlan | None = None,
        *,
        duckdb_threads: int | None = None,
        strict: bool | None = None,
    ) -> None:
        self.plan = plan or ExecutionResourcePlan.auto()
        self._duckdb_threads = duckdb_threads
        # R20-132..137：显式 strict 优先；未给出时从 env 兜底。
        if strict is not None:
            self.strict = bool(strict)
        else:
            self.strict = os.environ.get("FACTOR_ENGINE_RUN_MODE") == "production"
        self._prev: dict[str, str | None] = {}
        self._prev_pragma: int | None = None
        #: telemetry：requested threads 与 effective threads 分开，不虚假标已生效。
        self.requested_threads: int | None = None
        self.effective_threads: int | None = None
        self.polars_live_effective: bool = False

    def __enter__(self) -> "ExecutionResourceScope":
        with _ENV_LOCK:
            self._prev = {
                "DUCKDB_MAX_THREADS": os.environ.get("DUCKDB_MAX_THREADS"),
                "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS"),
                "DUCKDB_MEMORY_LIMIT": os.environ.get("DUCKDB_MEMORY_LIMIT"),
                "DUCKDB_TEMP_DIRECTORY": os.environ.get("DUCKDB_TEMP_DIRECTORY"),
                "DUCKDB_MAX_TEMP_DIRECTORY_SIZE": os.environ.get("DUCKDB_MAX_TEMP_DIRECTORY_SIZE"),
            }
            threads = self._duckdb_threads or self.plan.duckdb_threads
            self.requested_threads = int(threads)
            os.environ["DUCKDB_MAX_THREADS"] = str(threads)
            # 审计 #348：polars 线程数是进程级启动期设置，若 polars 已被 import，
            # 事后改 POLARS_MAX_THREADS 不保证生效。best-effort 仍设置 env，
            # 但明确告警需要不同 CPU 配额时应走 worker 进程隔离。
            self.polars_live_effective = "polars" not in sys.modules
            if "polars" in sys.modules:
                _logger.warning(
                    "POLARS_MAX_THREADS 在 polars 已导入后不保证生效；需要不同 "
                    "CPU 配额请用 worker 进程隔离"
                )
            os.environ["POLARS_MAX_THREADS"] = str(max(1, self.plan.polars_threads))
            os.environ["DUCKDB_MEMORY_LIMIT"] = str(self.plan.duckdb_budget_bytes)
            os.environ["DUCKDB_TEMP_DIRECTORY"] = self.plan.spill_dir
            if self.plan.spill_budget_bytes > 0:
                os.environ["DUCKDB_MAX_TEMP_DIRECTORY_SIZE"] = str(self.plan.spill_budget_bytes)
            self.effective_threads = int(threads)
            self._apply_live_pragma(threads)
        return self

    def _apply_live_pragma(self, threads: int) -> None:
        """对当前 DuckDB 连接应用 live ``PRAGMA threads``。

        审计 #348/#349：成功则记录 ``_prev_pragma``；失败不再静默——production
        下抛 ``ResourceContractApplyError``（fail-closed），非 production 记 warning。
        """
        try:
            from data_access import get_store

            engine = get_store()._engine
            if hasattr(engine, "_write_lock"):
                with engine._write_lock:
                    self._prev_pragma = int(
                        engine._conn.execute("SELECT current_setting('threads')").fetchone()[0]
                    )
                    engine._conn.execute(f"PRAGMA threads={int(threads)}")
        except Exception as exc:  # noqa: BLE001
            self._prev_pragma = None
            if self.strict:
                raise ResourceContractApplyError(
                    f"无法应用 DuckDB live PRAGMA threads={int(threads)}: {exc}"
                ) from exc
            _logger.warning(
                "无法应用 DuckDB live PRAGMA threads=%d: %s（非 production，继续）",
                int(threads),
                exc,
            )

    def __exit__(self, exc_type, exc, tb) -> None:
        with _ENV_LOCK:
            for key, value in self._prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if self._prev_pragma is not None:
                try:
                    from data_access import get_store

                    engine = get_store()._engine
                    if hasattr(engine, "_write_lock"):
                        with engine._write_lock:
                            engine._conn.execute(f"PRAGMA threads={int(self._prev_pragma)}")
                except Exception as exc2:  # noqa: BLE001
                    # 审计 #349：恢复失败只在非 production 下吞掉；production 不吞。
                    if self.strict:
                        raise ResourceContractApplyError(
                            f"无法恢复 DuckDB PRAGMA threads={int(self._prev_pragma)}: {exc2}"
                        ) from exc2
                    _logger.warning(
                        "无法恢复 DuckDB PRAGMA threads=%d: %s",
                        int(self._prev_pragma),
                        exc2,
                    )


# ---------------------------------------------------------------------------
# 结果字节预算（R18）
# ---------------------------------------------------------------------------


def assert_result_within_budget(value: Any, plan: ExecutionResourcePlan, *, factor_name: str = "") -> None:
    """最终结果字节预算检查：超限抛 ``ResourceBudgetExceeded``（production fail-closed）。"""
    size = estimate_object_bytes(value)
    if size <= plan.result_budget_bytes:
        return
    raise ResourceBudgetExceeded(
        f"final result {size / 1024**2:.1f} MiB exceeds result_budget "
        f"{plan.result_budget_bytes / 1024**2:.1f} MiB"
        + (f" (factor={factor_name})" if factor_name else "")
        + "; use result_policy=sink/stream/materialize instead of returning in-memory"
    )


# 全局单例（进程内各 engine 默认共享，逐层 reserve/release 记账）
_GLOBAL_GOVERNOR: MemoryGovernor | None = None


def global_memory_governor() -> MemoryGovernor:
    """进程级全局 governor（懒构造，跟随真实资源上限）。"""
    global _GLOBAL_GOVERNOR
    if _GLOBAL_GOVERNOR is None:
        plan = ExecutionResourcePlan.auto()
        _GLOBAL_GOVERNOR = MemoryGovernor(
            process_budget_bytes=plan.process_budget_bytes,
            duckdb_budget_bytes=plan.duckdb_budget_bytes,
        )
    return _GLOBAL_GOVERNOR


def reset_global_governor() -> None:
    """清空全局 governor（测试用）。"""
    global _GLOBAL_GOVERNOR
    _GLOBAL_GOVERNOR = None
