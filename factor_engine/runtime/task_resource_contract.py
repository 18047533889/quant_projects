# -*- coding: utf-8 -*-
"""R27-009: TaskResourceContract —— 每个 DAG task 执行前的统一资源契约。

调度器 admission 的唯一依据是「task 声明要多少 CPU / IO / 内存 / 输出字节」，
而不是 worker 数量。本模块定义该契约与字节估算 helper，供
:mod:`runtime.resource_broker` 的 memory/CPU/IO/spill token admission 消费。

设计要点（R27-009/014/015）
    - ``predicted_peak_memory_bytes`` 是**数值**（R27-014：不接受静态
      low/medium/high 档作为最终 admission）。
    - ``shardable`` / ``shard_dimension`` 是语义属性（R27-084），不是性能属性。
    - ``uncertainty`` 随 calibration 收敛（初始 1.25~1.50，收敛后 1.10~1.20，
      R27-040）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_UNCERTAINTY = 1.30
UNCERTAINTY_WARM = 1.50
UNCERTAINTY_COLD = 1.25
UNCERTAINTY_CALIBRATED = 1.15


@dataclass(frozen=True)
class TaskResourceContract:
    """每个 DAG task 执行前的资源契约（R27-009）。"""

    # 计算资源
    predicted_elapsed_ms: float = 0.0
    cpu_tokens: int = 1
    io_tokens: int = 1
    # 字节
    input_bytes: int = 0
    peak_memory_bytes: int = 0
    output_bytes: int = 0
    spill_bytes: int = 0
    # 执行性质
    gil_bound: bool = False
    releases_gil: bool = False
    # backend 线程
    backend: str = "pandas_numpy"
    backend_threads: int = 1
    # shard 语义（R27-084：shardability 是语义属性）
    shardable: bool = False
    shard_dimension: str | None = None
    # 校准
    uncertainty: float = DEFAULT_UNCERTAINTY
    # 可解释性（R27-143：调度器必须可解释）
    estimate_basis: str = "static"

    @property
    def admissible_peak_bytes(self) -> int:
        """admission 用峰值：估算 × uncertainty（R27-019/039）。"""
        return max(0, int(self.peak_memory_bytes * self.uncertainty))

    def with_uncertainty(self, uncertainty: float) -> "TaskResourceContract":
        return TaskResourceContract(
            predicted_elapsed_ms=self.predicted_elapsed_ms,
            cpu_tokens=self.cpu_tokens,
            io_tokens=self.io_tokens,
            input_bytes=self.input_bytes,
            peak_memory_bytes=self.peak_memory_bytes,
            output_bytes=self.output_bytes,
            spill_bytes=self.spill_bytes,
            gil_bound=self.gil_bound,
            releases_gil=self.releases_gil,
            backend=self.backend,
            backend_threads=self.backend_threads,
            shardable=self.shardable,
            shard_dimension=self.shard_dimension,
            uncertainty=uncertainty,
            estimate_basis=self.estimate_basis,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_elapsed_ms": self.predicted_elapsed_ms,
            "cpu_tokens": self.cpu_tokens,
            "io_tokens": self.io_tokens,
            "input_bytes": self.input_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "admissible_peak_bytes": self.admissible_peak_bytes,
            "output_bytes": self.output_bytes,
            "spill_bytes": self.spill_bytes,
            "gil_bound": self.gil_bound,
            "releases_gil": self.releases_gil,
            "backend": self.backend,
            "backend_threads": self.backend_threads,
            "shardable": self.shardable,
            "shard_dimension": self.shard_dimension,
            "uncertainty": round(self.uncertainty, 3),
            "estimate_basis": self.estimate_basis,
        }


# ---------------------------------------------------------------------------
# 面板字节估算（R27-014：panel_bytes = rows * instruments * dtype_size）
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {"float64": 8, "float32": 4, "int64": 8, "int32": 4, "bool": 1}


def panel_bytes(rows: int, instruments: int, *, dtype_size: float = 8.0) -> int:
    """面板原始字节：``rows * instruments * dtype_size``（R27-014 初始估计）。"""
    return max(0, int(rows * max(0, instruments) * dtype_size))


def estimate_operator_peak_bytes(
    *,
    input_bytes: int = 0,
    input_count: int = 1,
    panel_bytes_: int = 0,
    output_bytes: int = 0,
    temporary_multiplier: float = 1.0,
    backend_multiplier: float = 1.0,
    window_workspace_multiplier: float = 1.0,
) -> int:
    """按 R27-014/015 的乘数模型估算算子峰值字节。

    ``peak ≈ (inputA + inputB + output + rolling workspace) * backend conversion``
    的机械近似：输入乘 input multiplicity，输出加临时量（temporary ×
    backend × window workspace）。
    """
    base = (
        max(0, input_bytes) * max(1, input_count)
        + max(0, panel_bytes_)
        + max(0, output_bytes) * temporary_multiplier
    )
    return max(0, int(base * backend_multiplier * window_workspace_multiplier))


def cheap_operator_contract(*, output_bytes: int = 0, input_bytes: int = 0) -> TaskResourceContract:
    """cheap elementwise / literal 算子契约：峰值≈输入+输出，不 spill（R27-075）。"""
    peak = max(0, input_bytes) + max(0, output_bytes)
    return TaskResourceContract(
        predicted_elapsed_ms=1.0,
        cpu_tokens=1,
        io_tokens=0,
        input_bytes=input_bytes,
        peak_memory_bytes=peak,
        output_bytes=output_bytes,
        spill_bytes=0,
        gil_bound=False,
        releases_gil=True,
        shardable=True,
        shard_dimension="time",
        uncertainty=UNCERTAINTY_COLD,
        estimate_basis="cheap-elementwise",
    )


#: 模块级默认契约为不可 shard（语义未知时禁止任意切分，R27-084/090）。
def unsafe_contract(reason: str = "semantic shardability unknown") -> TaskResourceContract:
    return TaskResourceContract(
        shardable=False,
        shard_dimension=None,
        estimate_basis=f"unsafe:{reason}",
    )
