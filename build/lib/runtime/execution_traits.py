# -*- coding: utf-8 -*-
"""R35 §110 / §119: ExecutionTraits — real GIL / thread / parallelism contract.

The scheduler must know, per backend kind, whether a task releases the GIL,
how many internal threads it may spawn, on which dimension it can be
parallelised, whether its payload is picklable (for a process lane), and where
its memory lives.  This replaces guessing from ``backend == pandas_numpy``.

Also provides :func:`thread_budget` — a context manager wrapping
``threadpoolctl.threadpool_limits`` so a task's inner BLAS/OpenMP threads are
pinned to its broker-granted CPU tokens (§23 / §24).  When threadpoolctl is
absent the limits are a no-op (honest: it is an optional performance dep).
"""
from __future__ import annotations

import dataclasses
import enum
import os
from dataclasses import dataclass
from typing import Any, Iterator


class NumericDeterminismLevel(str, enum.Enum):
    """R40 #256：production 数值确定性的三级分类。

    * ``BITWISE_DETERMINISTIC`` — 相同输入 + 相同环境位级复现（纯 numpy /
      numba 单线程内核）；
    * ``DETERMINISTIC_WITHIN_TOLERANCE`` — 结果在容差内复现（BLAS 线程数 /
      vendor 变化会带来浮点舍入差异，但 allclose 通过）；
    * ``NONDETERMINISTIC_RESEARCH_ONLY`` — 不可确定性（某些 research 内核），
      production 禁止。

    production evidence 必须声明每个算子的确定级别；同时记录 BLAS
    vendor/version/thread config（线程数变化会让位级复现失效）。
    """

    BITWISE_DETERMINISTIC = "bitwise_deterministic"
    DETERMINISTIC_WITHIN_TOLERANCE = "deterministic_within_tolerance"
    NONDETERMINISTIC_RESEARCH_ONLY = "nondeterministic_research_only"


def determinism_level_for(backend: str) -> NumericDeterminismLevel:
    """按 backend 分类确定级别（#256）。

    * duckdb/polars native 与 numpy vectorized 在固定线程数下是
      ``DETERMINISTIC_WITHIN_TOLERANCE``（内部 BLAS/向量化有舍入）；
    * numba nogil 单线程内核 = ``BITWISE_DETERMINISTIC``；
    * research_python（纯 Python 任意迭代）保守归
      ``NONDETERMINISTIC_RESEARCH_ONLY``。
    """
    b = (backend or "").lower()
    if "research" in b:
        return NumericDeterminismLevel.NONDETERMINISTIC_RESEARCH_ONLY
    if "numba" in b:
        return NumericDeterminismLevel.BITWISE_DETERMINISTIC
    if "duckdb" in b or "polars" in b or "numpy" in b or b == "sql":
        return NumericDeterminismLevel.DETERMINISTIC_WITHIN_TOLERANCE
    return NumericDeterminismLevel.DETERMINISTIC_WITHIN_TOLERANCE


def record_blas_config() -> dict[str, str]:
    """记录 BLAS vendor/version/thread config（#256 evidence）。

    线程环境变量 / threadpoolctl 可用信息都收进一个 dict，供 production
    evidence 与 cross-process determinism 对照。
    """
    config: dict[str, str] = {}
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "DUCKDB_MAX_THREADS",
    ):
        config[var] = os.environ.get(var, "")
    config["threadpoolctl_available"] = str(THREADPOOLCTL_AVAILABLE)
    if THREADPOOLCTL_AVAILABLE:
        try:
            import threadpoolctl

            libs = threadpoolctl.threadpool_info()
            config["blas_libraries"] = repr(
                [
                    {
                        "name": getattr(l, "name", ""),
                        "version": getattr(l, "version", ""),
                        "threads": getattr(l, "num_threads", ""),
                    }
                    for l in libs
                ]
            )
        except Exception:  # pragma: no cover - defensive
            config["blas_libraries"] = "unavailable"
    return config

try:
    from threadpoolctl import threadpool_limits

    THREADPOOLCTL_AVAILABLE = True
except Exception:  # pragma: no cover - performance extra not installed
    threadpool_limits = None  # type: ignore
    THREADPOOLCTL_AVAILABLE = False

from contextlib import contextmanager


@dataclass(frozen=True)
class ExecutionTraits:
    """Real execution properties of a backend kind (taskbook §110)."""

    backend_kind: str                # sql_native / polars_native / numpy_blas / numba_kernel / python_specialized
    releases_gil: bool
    internal_threads: int            # 1 = single-threaded inner compute
    parallelizable_dimension: str    # factor / block / column / none
    picklable_payload: bool          # true only for a safe ExecutionCapsule
    memory_locality: str             # buffer / mmap / relation / contiguous / object

    @property
    def thread_safe(self) -> bool:
        return self.releases_gil or self.internal_threads == 1


#: Static traits per backend kind (§109 backend taxonomy).
EXECUTION_TRAITS: dict[str, ExecutionTraits] = {
    "sql_native": ExecutionTraits(
        "sql_native", releases_gil=True, internal_threads=1,
        parallelizable_dimension="factor", picklable_payload=False, memory_locality="relation",
    ),
    "polars_native": ExecutionTraits(
        "polars_native", releases_gil=True, internal_threads=1,
        parallelizable_dimension="factor", picklable_payload=False, memory_locality="relation",
    ),
    "numpy_blas": ExecutionTraits(
        "numpy_blas", releases_gil=True, internal_threads=1,
        parallelizable_dimension="factor", picklable_payload=True, memory_locality="contiguous",
    ),
    "numba_kernel": ExecutionTraits(
        "numba_kernel", releases_gil=True, internal_threads=1,
        parallelizable_dimension="factor", picklable_payload=True, memory_locality="contiguous",
    ),
    "python_specialized": ExecutionTraits(
        "python_specialized", releases_gil=False, internal_threads=1,
        parallelizable_dimension="factor", picklable_payload=False, memory_locality="object",
    ),
}

#: Backend-kind labels for canonical surfaces (SQL- and Polars-native operators).
SQL_NATIVE_KIND = "sql_native"
POLARS_NATIVE_KIND = "polars_native"
NUMPY_BLAS_KIND = "numpy_blas"
NUMBA_KERNEL_KIND = "numba_kernel"
PYTHON_SPECIALIZED_KIND = "python_specialized"


def traits_for_backend(backend: str) -> ExecutionTraits:
    """Map an engine backend name to its execution traits (best-effort)."""
    b = (backend or "").lower()
    if "duckdb" in b or b == "sql":
        return EXECUTION_TRAITS[SQL_NATIVE_KIND]
    if "polars" in b:
        return EXECUTION_TRAITS[POLARS_NATIVE_KIND]
    if b in ("numpy", "numpy_numba", "numba"):
        return EXECUTION_TRAITS[NUMBA_KERNEL_KIND]
    if "numpy" in b or "blas" in b or b == "pandas_numpy":
        return EXECUTION_TRAITS[NUMPY_BLAS_KIND]
    return EXECUTION_TRAITS[PYTHON_SPECIALIZED_KIND]


@contextmanager
def thread_budget(backend_threads: int = 1) -> Iterator[None]:
    """Pin the inner BLAS/OpenMP library threads to the task's broker-granted
    CPU budget (§23 / §24).  No-op when threadpoolctl is unavailable."""
    if not THREADPOOLCTL_AVAILABLE or backend_threads is None or backend_threads < 1:
        yield
        return
    with threadpool_limits(limits=backend_threads):
        yield


def nested_thread_oversubscription_error(
    fe_workers: int, blas_threads: int, available_cpus: int
) -> list[str]:
    """§23: detect FE_workers × inner_threads oversubscription and report it."""
    errs: list[str] = []
    if fe_workers > 1 and blas_threads > 1:
        total = fe_workers * blas_threads
        if total > max(available_cpus, 1):
            errs.append(
                f"oversubscription: FE={fe_workers} × BLAS={blas_threads} = {total} "
                f"> available_cpus={available_cpus}"
            )
    return errs
