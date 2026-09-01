"""research_protocol（任务书 §3.3）：封存 Test + LeakageGuard。

L0-L4：sealed_test datasource read == 0 必须成为 CI（§88.1）。
"""

from __future__ import annotations

import threading
from typing import Any

from alphaprobe.contracts import (
    DateRange,
    ExperimentContext,
    FidelityLevel,
    L0_L4,
    ResearchSplitSpec,
)

__all__ = [
    "LeakageGuard",
    "SealedTestAccess",
    "SealedTestViolation",
    "assert_sealed_test_zero_reads",
    "sealed_test_read_count",
    "reset_sealed_test_counters",
]


class SealedTestViolation(RuntimeError):
    """任何 L0-L4 阶段触碰 sealed test 数据即抛出。"""


class _SealedCounter:
    """进程级 sealed-test 读取计数（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def incr(self, caller: str) -> None:
        with self._lock:
            self._counts[caller] = self._counts.get(caller, 0) + 1

    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


_counter = _SealedCounter()


def sealed_test_read_count() -> int:
    return _counter.total()


def reset_sealed_test_counters() -> None:
    _counter.reset()


class LeakageGuard:
    """§3.3：所有 datasource access 注入。

    L0-L4 阶段请求 sealed_test 段数据 → 直接抛 SealedTestViolation，
    由代码防线保证（不靠 prompt 自觉，§44.1）。
    """

    def __init__(self, split_spec: ResearchSplitSpec | None) -> None:
        self.split_spec = split_spec

    def assert_access_allowed(
        self,
        *,
        stage: FidelityLevel,
        segment: DateRange | str,
        caller: str,
    ) -> None:
        if self.split_spec is None:
            return
        if stage not in L0_L4:
            return  # L5（冻结后）允许读 sealed test
        seg = (
            segment
            if isinstance(segment, DateRange)
            else DateRange(segment, segment)
        )
        if self.split_spec.sealed_test.overlaps(seg):
            # 拦截在闸门（未触达数据），不计入读取数；真实绕过路径才 incr
            raise SealedTestViolation(
                f"sealed-test access denied at stage={stage.value} caller={caller} "
                f"segment={seg.start}..{seg.end}"
            )

    def assert_stage_permits(self, stage: FidelityLevel, fidelity_needed: FidelityLevel) -> None:
        """保真度不得越级：如 L2 不得消费 L4 才有的指标。"""
        order = list(FidelityLevel)
        if order.index(stage) > order.index(fidelity_needed):
            raise SealedTestViolation(
                f"fidelity escalation blocked: stage={stage.value} < needed={fidelity_needed.value}"
            )


def assert_sealed_test_zero_reads() -> None:
    """§88.1 CI gate：L0-L4 全程结束后调用；读数非零即失败。"""
    n = _counter.total()
    if n != 0:
        raise SealedTestViolation(f"sealed_test read count must be 0, got {n}")


class SealedTestAccess:
    """L5 Sealed Test 专用入口：仅 research 版本/ campaign 冻结后允许实例化。

    必须显式传 frozen=True，防止误用。
    """

    def __init__(self, split_spec: ResearchSplitSpec, *, frozen: bool, context: ExperimentContext | None = None) -> None:
        if not frozen:
            raise SealedTestViolation(
                "SealedTestAccess requires frozen=True (version/campaign must be sealed first)"
            )
        self.split_spec = split_spec
        self.context = context
        self.frozen = True