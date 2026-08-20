# -*- coding: utf-8 -*-
"""R38 P0-007/008/009（§6）：TaskRunObservation —— calibration 只接收真实观测。

R36 的 ``_record_task_calibration`` 把：
    - ``elapsed_ms`` = 绝对 monotonic timestamp（不是 duration）；
    - ``peak_mem`` = contract 预测值（拿预测当真实值，再学自己的预测）；
    - ``output_bytes`` = contract 预测值。

本模块定义真实观测结构（§P0-007 必须字段），并让
:func:`ResourceCalibrationStore.record` 只把**可归因**的峰值内存喂进 P99 模型
（§P0-008：``attribution_quality`` 区分 isolated / low-concurrency /
concurrent-marginal / run-level-only / unattributed）。

- ``elapsed_ms``：真实 duration（``finished_at_monotonic - started_at_monotonic``）。
- ``peak_family_pss``：真实观测峰值（可归因时才有；否则 ``None``）。
- ``predicted_peak_mem``：预测值（仅供对比 / correction 诊断，不进 P99 主模型）。
- ``output_bytes_actual``：``estimate_object_bytes(result)``（真实输出字节）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: attribution quality 分级（§P0-008 可选值）。
ATTRIBUTION_ISOLATED = "isolated"
ATTRIBUTION_LOW_CONCURRENCY = "low-concurrency"
ATTRIBUTION_CONCURRENT_MARGINAL = "concurrent-marginal"
ATTRIBUTION_RUN_LEVEL = "run-level-only"
ATTRIBUTION_UNATTRIBUTED = "unattributed"

#: 只有这些质量级别的内存观测可以进入 shape P99 主模型（§P0-008）。
P99_TRUSTED_ATTRIBUTION: frozenset[str] = frozenset({
    ATTRIBUTION_ISOLATED,
    ATTRIBUTION_LOW_CONCURRENCY,
    ATTRIBUTION_CONCURRENT_MARGINAL,
})


@dataclass(frozen=True)
class TaskRunObservation:
    """一次 task 运行的真实观测（§P0-007 必须字段全给出）。"""

    task_id: str
    started_at_monotonic: float
    finished_at_monotonic: float
    elapsed_ms: float
    baseline_family_pss: int
    peak_family_pss: int | None
    output_bytes_actual: int
    spill_bytes_actual: int
    read_bytes_actual: int
    write_bytes_actual: int
    backend_threads_actual: int
    #: 预测值（用于 correction 诊断，不作为真实值）。
    predicted_peak_mem: int = 0
    attribution_quality: str = ATTRIBUTION_UNATTRIBUTED

    @property
    def peak_is_trusted(self) -> bool:
        """该观测的峰值内存是否可信（可喂进 P99 主模型）。"""
        return (
            self.peak_family_pss is not None
            and self.attribution_quality in P99_TRUSTED_ATTRIBUTION
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "started_at_monotonic": round(self.started_at_monotonic, 3),
            "finished_at_monotonic": round(self.finished_at_monotonic, 3),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "baseline_family_pss": self.baseline_family_pss,
            "peak_family_pss": self.peak_family_pss,
            "output_bytes_actual": self.output_bytes_actual,
            "spill_bytes_actual": self.spill_bytes_actual,
            "read_bytes_actual": self.read_bytes_actual,
            "write_bytes_actual": self.write_bytes_actual,
            "backend_threads_actual": self.backend_threads_actual,
            "predicted_peak_mem": self.predicted_peak_mem,
            "attribution_quality": self.attribution_quality,
            "peak_is_trusted": self.peak_is_trusted,
        }


def estimate_output_bytes(result: Any) -> int:
    """真实输出字节（R38-P0-009：不用 contract.output_bytes 预测值）。"""
    try:
        from runtime.resource_governor import estimate_object_bytes

        return max(0, int(estimate_object_bytes(result)))
    except Exception:
        return 0
