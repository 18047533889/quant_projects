# -*- coding: utf-8 -*-
"""R39-P0-PERF-012：read-wave 失败 typed recovery —— 禁止退化成 N 个 root 重复读。

coalesced wave 一旦失败，如果后续 root 回到 on-demand read 会发生：
    1 次失败的大 scan + N 次 root 重复 scan（FAILED_WAVE_TO_N_ROOT_SCANS > 0）。

typed recovery 决策表（R39 §4.12）：
    OVERSIZED                    → split wave 2-way
    REPRESENTATION_UNSUPPORTED   → downgrade representation only
    TRANSIENT_IO                 → bounded retry
    MEMORY_PRESSURE              → shrink wave
    PERMANENT_SOURCE_ERROR       → abort（唯一允许 fallback 的类别）

``would_fallback_to_n_root_scans``：只要存在 typed recovery plan（非 abort）就返回
False —— 这是 hard gate ``FAILED_WAVE_TO_N_ROOT_SCANS == 0`` 的判定函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class WaveRecoveryCategory(Enum):
    """wave 失败的分类（R39-P0-PERF-012）。"""

    OVERSIZED = "OVERSIZED"
    REPRESENTATION_UNSUPPORTED = "REPRESENTATION_UNSUPPORTED"
    TRANSIENT_IO = "TRANSIENT_IO"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    PERMANENT_SOURCE_ERROR = "PERMANENT_SOURCE_ERROR"


#: 类别 → 恢复动作（唯一权威映射）。
_RECOVERY_ACTIONS: dict[WaveRecoveryCategory, str] = {
    WaveRecoveryCategory.OVERSIZED: "split_2_way",
    WaveRecoveryCategory.REPRESENTATION_UNSUPPORTED: "downgrade_representation",
    WaveRecoveryCategory.TRANSIENT_IO: "bounded_retry",
    WaveRecoveryCategory.MEMORY_PRESSURE: "shrink",
    WaveRecoveryCategory.PERMANENT_SOURCE_ERROR: "abort",
}

_RETRY_LIMITS: dict[WaveRecoveryCategory, int] = {
    WaveRecoveryCategory.OVERSIZED: 0,
    WaveRecoveryCategory.REPRESENTATION_UNSUPPORTED: 0,
    WaveRecoveryCategory.TRANSIENT_IO: 3,
    WaveRecoveryCategory.MEMORY_PRESSURE: 0,
    WaveRecoveryCategory.PERMANENT_SOURCE_ERROR: 0,
}

_SPLIT_FACTORS: dict[WaveRecoveryCategory, int] = {
    WaveRecoveryCategory.OVERSIZED: 2,
    WaveRecoveryCategory.REPRESENTATION_UNSUPPORTED: 1,
    WaveRecoveryCategory.TRANSIENT_IO: 1,
    WaveRecoveryCategory.MEMORY_PRESSURE: 2,
    WaveRecoveryCategory.PERMANENT_SOURCE_ERROR: 1,
}


@dataclass(frozen=True)
class WaveRecoveryPlan:
    """一次 typed wave 恢复决策（R39-P0-PERF-012）。"""

    category: WaveRecoveryCategory
    action: str
    detail: str = ""
    retry_limit: int = 0
    split_factor: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "action": self.action,
            "detail": self.detail,
            "retry_limit": self.retry_limit,
            "split_factor": self.split_factor,
        }


#: 被认作 transient IO 的异常类型名（避免强依赖运行时异常模块）。
_TRANSIENT_IO_TYPES = frozenset({
    "OSError",
    "IOError",
    "ConnectionError",
    "ConnectionResetError",
    "TimeoutError",
    "TransientError",
    "RetryableError",
    "DataAccessRetryableError",
})


def classify_wave_failure(exc: BaseException) -> WaveRecoveryCategory:
    """把 wave 执行异常分类到 typed recovery 类别（R39-P0-PERF-012）。"""
    name = type(exc).__name__
    msg = str(exc).lower()

    if isinstance(exc, MemoryError) or (
        "memory" in msg
        and any(k in msg for k in ("pressure", "limit", "exceeded", "budget", "oom"))
    ):
        return WaveRecoveryCategory.MEMORY_PRESSURE

    if "oversiz" in msg or "too large" in msg or (
        "exceeds" in msg and "memory" in msg
    ):
        return WaveRecoveryCategory.OVERSIZED

    if name in _TRANSIENT_IO_TYPES or any(
        k in msg for k in ("transient", "timeout", "retry", "connection", "temporar")
    ):
        return WaveRecoveryCategory.TRANSIENT_IO

    if (
        "representation" in msg
        or "unsupported" in msg
        or ("backend" in msg and "support" in msg)
    ):
        return WaveRecoveryCategory.REPRESENTATION_UNSUPPORTED

    return WaveRecoveryCategory.PERMANENT_SOURCE_ERROR


def recover_wave_failure(
    failure: BaseException,
    wave: Any,
) -> WaveRecoveryPlan:
    """typed 决策表：failure + wave → WaveRecoveryPlan（R39-P0-PERF-012）。"""
    cat = classify_wave_failure(failure)
    return WaveRecoveryPlan(
        category=cat,
        action=_RECOVERY_ACTIONS[cat],
        detail=f"{type(failure).__name__}: {failure}",
        retry_limit=_RETRY_LIMITS[cat],
        split_factor=_SPLIT_FACTORS[cat],
    )


def would_fallback_to_n_root_scans(failure: BaseException, wave: Any) -> bool:
    """hard gate 判定：只要存在 typed recovery plan（非 abort）就不 fallback。

    R39-P0-PERF-012：禁止 uncontrolled per-root fallback。只有
    PERMANENT_SOURCE_ERROR（abort）会退化成 N 个 root 重复读。
    """
    plan = recover_wave_failure(failure, wave)
    return plan.category is WaveRecoveryCategory.PERMANENT_SOURCE_ERROR


def hard_gate_failed_wave_to_n_root_scans(failure: BaseException, wave: Any) -> int:
    """hard gate counter：0 = 有 typed recovery，1 = 退化成 N root scans。"""
    return 1 if would_fallback_to_n_root_scans(failure, wave) else 0
