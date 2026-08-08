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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from runtime.resource_errors import ResourceBudgetExceeded

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
    """RLIMIT_AS / RLIMIT_DATA 软上限（字节）；无限返回 ``None``。"""
    try:
        import resource

        for which in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            soft, hard = resource.getrlimit(which)
            if soft != resource.RLIM_INFINITY:
                return int(soft)
        return None
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


def effective_memory_limit_bytes(
    *,
    explicit_bytes: int | None = None,
    explicit_env: str = "FACTOR_ENGINE_MAX_MEMORY_BYTES",
) -> int:
    """进程真实内存上限（最严格来源取最小值）。

    优先级：显式参数/环境变量 > cgroup v2 > cgroup v1 > SLURM > RLIMIT > host RAM。
    完全无法探测时回退 8GiB。
    """
    if explicit_bytes is not None and explicit_bytes > 0:
        return explicit_bytes
    env = _env_int(explicit_env, None)
    if env is not None:
        return env
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
    if not candidates:
        return 8 * 1024**3
    return min(candidates)


def effective_cpu_slots(
    *,
    explicit: int | None = None,
    env: str = "FACTOR_ENGINE_CPU_BUDGET",
) -> int:
    """进程真实 CPU slot 数（cgroup quota 感知），替代 naive ``physical_cores``。

    - 显式参数 / ``FACTOR_ENGINE_CPU_BUDGET`` 优先
    - cgroup v2 ``cpu.max`` quota/period 折算（period 内只能跑 quota 个微秒）
    - sched affinity（容器亲和性）
    - host CPU 兜底
    """
    if explicit is not None and explicit > 0:
        return explicit
    env_v = _env_int(env, None)
    if env_v is not None:
        return env_v
    qp = _read_cgroup_cpu_quota()
    if qp is not None:
        quota, period = qp
        slots = quota // period
        if slots > 0:
            return max(1, slots)
    try:
        import os as _os

        return max(1, len(_os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        try:
            import multiprocessing

            return max(1, multiprocessing.cpu_count())
        except Exception:  # pragma: no cover
            return 4


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
        duckdb_frac = float(cache_cfg.get("duckdb_fraction", 0.45)) if False else 0.45

        workers_cfg = int(concurrency.get("max_workers") or 0) or base.max_workers
        workers = max(1, min(workers_cfg, effective_cpu_slots()))
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

#: 运行时 RSS 分档（相对 process budget）
_STAGE_NORMAL = 0.70
_STAGE_STOP_WARMUP = 0.80
_STAGE_EVICT_LRU = 0.85
_STAGE_SPILL = 0.90
_STAGE_THROTTLE = 0.95


@dataclass
class MemoryGovernor:
    """统一 byte 预算 + 运行时 RSS 分档治理。

    可被多个 cache 层共享：每层 ``reserve/release`` 记账，超 budget 时触发
    注册的 evict 回调（LRU）；RSS 超过阈值时分档 throttle。**不**用
    ``gc.collect()`` 当治理手段。
    """

    process_budget_bytes: int
    duckdb_budget_bytes: int
    rss_probe: Callable[[], int] | None = None
    _usage: dict[str, int] = field(default_factory=dict)
    _evict_hooks: dict[str, Callable[[int], int]] = field(default_factory=dict)
    throttles: list[str] = field(default_factory=list)
    evictions: list[str] = field(default_factory=list)

    @property
    def total_usage(self) -> int:
        """全部登记层的字节占用之和。"""
        return sum(self._usage.values())

    def register_layer(self, name: str, evict: Callable[[int], int]) -> None:
        """注册缓存层：``evict(target_bytes) -> freed_bytes``。"""
        self._evict_hooks[name] = evict

    def reserve(self, name: str, bytes_: int) -> bool:
        """某层尝试占用 ``bytes_`` 字节；超出全局 budget 时触发 evict。

        返回是否成功；失败时调用方应放弃缓存该对象（宁可重算，不要 OOM）。
        """
        bytes_ = max(0, int(bytes_))
        if self.total_usage + bytes_ > self.process_budget_bytes:
            freed = self._evict_for(bytes_)
            if self.total_usage + bytes_ > self.process_budget_bytes:
                return False
        self._usage[name] = self._usage.get(name, 0) + bytes_
        return True

    def _evict_for(self, needed: int) -> int:
        """逐出 LRU 缓存直到腾出 ``needed`` 字节。"""
        freed = 0
        for name, hook in self._evict_hooks.items():
            if self.total_usage + needed <= self.process_budget_bytes:
                break
            target = max(0, self.total_usage + needed - self.process_budget_bytes)
            n = hook(target)
            if n > 0:
                self._usage[name] = max(0, self._usage.get(name, 0) - n)
                freed += n
                self.evictions.append(f"{name}:{n}")
        return freed

    def release(self, name: str, bytes_: int) -> None:
        """释放某层占用的字节。"""
        bytes_ = max(0, int(bytes_))
        self._usage[name] = max(0, self._usage.get(name, 0) - bytes_)

    def release_all(self, name: str) -> None:
        """清空某层全部记账。"""
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
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            return 0

    def pressure_stage(self) -> str:
        """返回当前压力档位：normal / stop_warmup / evict_lru / spill / throttle / critical。"""
        budget = self.process_budget_bytes or 1
        frac = self._current_rss() / budget
        if frac >= _STAGE_THROTTLE:
            return "critical"
        if frac >= _STAGE_SPILL:
            return "throttle"
        if frac >= _STAGE_EVICT_LRU:
            return "spill"
        if frac >= _STAGE_STOP_WARMUP:
            return "evict_lru"
        if frac >= _STAGE_NORMAL:
            return "stop_warmup"
        return "normal"

    def throttle(self, stage: str) -> None:
        """记录一次 throttle 事件（调用方根据 stage 执行对应动作）。"""
        self.throttles.append(stage)

    def check_pre_warmup(self) -> bool:
        """是否允许继续预热新 cache（stop_warmup 及以上不允许）。"""
        return self.pressure_stage() in ("normal", "stop_warmup")

    def check_admit(self, size_bytes: int, *, factor: float = 1.0) -> bool:
        """新对象是否可 admit；超 process budget 时触发 evict 后仍不够则拒绝。"""
        if self.total_usage + size_bytes * factor <= self.process_budget_bytes:
            return True
        return self.reserve("__probe__", int(size_bytes * factor))

    def summary(self) -> dict[str, Any]:
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


class ExecutionResourceScope:
    """context manager：进入时应用 ExecutionResourcePlan，退出时恢复环境。

    负责把 DuckDB threads / Polars threads 等动态设置写入环境并在 finally 恢复，
    避免并行 worker 共享 mutable 状态、改完不还原。
    """

    def __init__(
        self,
        plan: ExecutionResourcePlan | None = None,
        *,
        duckdb_threads: int | None = None,
    ) -> None:
        self.plan = plan or ExecutionResourcePlan.auto()
        self._duckdb_threads = duckdb_threads
        self._prev: dict[str, str | None] = {}
        self._prev_pragma: int | None = None

    def __enter__(self) -> "ExecutionResourceScope":
        self._prev = {
            "DUCKDB_MAX_THREADS": os.environ.get("DUCKDB_MAX_THREADS"),
            "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS"),
            "DUCKDB_MEMORY_LIMIT": os.environ.get("DUCKDB_MEMORY_LIMIT"),
            "DUCKDB_TEMP_DIRECTORY": os.environ.get("DUCKDB_TEMP_DIRECTORY"),
            "DUCKDB_MAX_TEMP_DIRECTORY_SIZE": os.environ.get("DUCKDB_MAX_TEMP_DIRECTORY_SIZE"),
        }
        threads = self._duckdb_threads or self.plan.duckdb_threads
        os.environ["DUCKDB_MAX_THREADS"] = str(threads)
        os.environ["POLARS_MAX_THREADS"] = str(max(1, self.plan.polars_threads))
        os.environ["DUCKDB_MEMORY_LIMIT"] = str(self.plan.duckdb_budget_bytes)
        os.environ["DUCKDB_TEMP_DIRECTORY"] = self.plan.spill_dir
        if self.plan.spill_budget_bytes > 0:
            os.environ["DUCKDB_MAX_TEMP_DIRECTORY_SIZE"] = str(self.plan.spill_budget_bytes)
        self._apply_live_pragma(threads)
        return self

    def _apply_live_pragma(self, threads: int) -> None:
        try:
            from data_access import get_store

            engine = get_store()._engine
            if hasattr(engine, "_write_lock"):
                with engine._write_lock:
                    self._prev_pragma = int(
                        engine._conn.execute("SELECT current_setting('threads')").fetchone()[0]
                    )
                    engine._conn.execute(f"PRAGMA threads={int(threads)}")
        except Exception:
            self._prev_pragma = None

    def __exit__(self, exc_type, exc, tb) -> None:
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
            except Exception:
                pass


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
