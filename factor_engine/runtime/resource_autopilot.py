# -*- coding: utf-8 -*-
"""R36 Resource Autopilot：ResourceDecision + AIMD 双向控制器（§5/6/65..70）。

把 ResourceBroker 从「threshold-based passive throttle」升级为
「closed-loop adaptive resource controller」（§3）：
    - :class:`ResourceDecision`（§305）：每个 control tick 返回的全部控制目标。
    - :class:`ResourceController`（§65/66/67/68）：AIMD + hysteresis + cooldown。
        - 压力升高 → 立即 multiplicative decrease（target ×0.5）
        - 压力解除且稳定 N 样本 → additive increase（+1）
        - 大幅缩容后 cooldown 至少一个 control window 再升（§68）
    - 动态 budgets（§35..39/49）：read wave / factor block / sink queue / spill /
      cache 全部从 Safe Envelope 派生，不再固定 4GB。
    - 决策可解释（§166/167）：每次变动记录 why/before/after/signal。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from runtime.resource_broker import (
    STAGE_CRITICAL,
    STAGE_NORMAL,
    STAGE_PRESSURE_1,
    STAGE_PRESSURE_2,
    STAGE_PRESSURE_3,
    STAGE_PRESSURE_4,
    ResourceBroker,
)
from runtime.resource_monitor import (
    HostResourceEnvelope,
    MemorySlopeTracker,
    ResourceSignals,
)

#: 双向控制器默认参数（§66/67/68）。
_DOWN_FACTOR = 0.5            # 压力升高 → target ×0.5（fast down）
_UP_STEP = 1                  # 稳定 → +1（slow up）
_STABLE_SAMPLES_REQUIRED = 3  # 稳定 N 样本才升一次（§66：2~5）
_COOLDOWN_TICKS = 2           # 大幅缩容后至少等一个 control window（§68）

#: 动态 budgets 的 Safe Envelope 分数（§35/38/39/49）。
WAVE_FRACTION = 0.10
BLOCK_FRACTION = 0.05
SINK_FRACTION = 0.05
SINK_JOB_LEASE_FRACTION = 0.10
CACHE_FRACTION = 0.15
SPILL_FRACTION = 0.15

#: 绝对 bounds。
WAVE_ABS_MIN = 256 * 1024**2       # 256MB
WAVE_ABS_MAX = 16 * 1024**3        # 16GB
BLOCK_ABS_MIN = 64 * 1024**2       # 64MB
BLOCK_ABS_MAX = 2 * 1024**3        # 2GB
SINK_ABS_MIN = 128 * 1024**2       # 128MB
SINK_ABS_MAX = 8 * 1024**3         # 8GB

#: 预测性压力阈值（§69/70）。
_MEM_SLOPE_THROTTLE_BPS = -1 * 1024**3      # -1GB/s → 提前 throttle
_MEM_SLOPE_STOP_BPS = -3 * 1024**3          # -3GB/s → 停止新大任务
_CPU_PSI_ESCALATE = 0.6                     # cpu psi some avg10 > 0.6 → 让路
_IO_PSI_ESCALATE = 0.5                      # io psi some avg10 > 0.5 → 让路
_MEM_PSI_ESCALATE = 0.3                     # memory psi some avg10 > 0.3 → 让路


@dataclass(frozen=True)
class ResourceDecision:
    """§305：一个 control tick 的全部控制目标（scheduler 必须消费）。"""

    target_concurrency: int
    target_cpu_tokens: int
    read_wave_bytes: int
    factor_block_bytes: int
    result_queue_bytes: int
    io_concurrency: int
    remote_concurrency: int
    cache_budget_bytes: int
    spill_budget_bytes: int
    pressure_state: str
    #: R36 §244/245：低内存不是单独手工配置——controller 自动进入 memory-
    #: constrained mode（内部仍连续控制；本字段是 explain 标签）。
    memory_constrained: bool = False
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_concurrency": self.target_concurrency,
            "target_cpu_tokens": self.target_cpu_tokens,
            "read_wave_bytes": self.read_wave_bytes,
            "factor_block_bytes": self.factor_block_bytes,
            "result_queue_bytes": self.result_queue_bytes,
            "io_concurrency": self.io_concurrency,
            "remote_concurrency": self.remote_concurrency,
            "cache_budget_bytes": self.cache_budget_bytes,
            "spill_budget_bytes": self.spill_budget_bytes,
            "pressure_state": self.pressure_state,
            "memory_constrained": self.memory_constrained,
            "reasons": list(self.reasons),
        }


@dataclass
class _DecisionLogEntry:
    timestamp_ms: float
    field: str
    before: Any
    after: Any
    signal: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": round(self.timestamp_ms, 3),
            "field": self.field,
            "before": self.before,
            "after": self.after,
            "signal": self.signal,
        }


class ResourceController:
    """AIMD + hysteresis + cooldown 的实时资源控制器（§65..70）。

    持有：
        - ``_target_concurrency`` / ``_target_cpu_tokens``：AIMD 控制量
        - ``_stable_count`` / ``_cooldown``：稳定样本计数 / 缩容冷却
        - 决策可解释日志（§166）
    """

    def __init__(
        self,
        broker: ResourceBroker,
        *,
        stable_samples_required: int = _STABLE_SAMPLES_REQUIRED,
        cooldown_ticks: int = _COOLDOWN_TICKS,
        down_factor: float = _DOWN_FACTOR,
        up_step: int = _UP_STEP,
    ) -> None:
        self._broker = broker
        self._stable_required = max(1, int(stable_samples_required))
        self._cooldown_ticks = max(0, int(cooldown_ticks))
        self._down_factor = float(down_factor)
        self._up_step = max(1, int(up_step))
        self._target_cpu_tokens = max(1, broker.cpu_budget())
        self._target_concurrency = max(1, broker.hard_cpu_slots)
        self._stable_count = 0
        self._cooldown = 0
        self._last_stage = STAGE_NORMAL
        self._slope = MemorySlopeTracker()
        self._decision_log: list[_DecisionLogEntry] = []
        self._decision: ResourceDecision | None = None
        self._tick_count = 0

    # -- 控制循环 --

    def tick(
        self,
        signals: ResourceSignals,
        envelope: HostResourceEnvelope,
        *,
        job_memory_lease_bytes: int | None = None,
        sink_backpressure: float = 0.0,
    ) -> ResourceDecision:
        """一个 control tick：读信号 → 分档 → AIMD → 派生 budgets → 返回决策。"""
        self._tick_count += 1
        reasons: list[str] = []
        now = time.monotonic()

        # 1) 基础压力分档（来自 broker 的 live headroom / 外部 CPU）。
        stage = self._broker.pressure_stage()

        # 2) PSI 预测性升级（§30/73/75：外部负载吃 RAM/NVMe 但 FE 自身 RSS 不高）。
        if signals.memory_psi_some > _MEM_PSI_ESCALATE:
            stage = _escalate(stage)
            reasons.append(f"memory_psi_some={signals.memory_psi_some:.3f}>{_MEM_PSI_ESCALATE}")
        if signals.cpu_psi_some > _CPU_PSI_ESCALATE:
            stage = _escalate(stage)
            reasons.append(f"cpu_psi_some={signals.cpu_psi_some:.3f}>{_CPU_PSI_ESCALATE}")
        if signals.io_psi_some > _IO_PSI_ESCALATE:
            stage = _escalate(stage)
            reasons.append(f"io_psi_some={signals.io_psi_some:.3f}>{_IO_PSI_ESCALATE}")

        # 3) Memory 斜率预测性压力（§69/70：现在还有余量 ≠ 未来还有余量）。
        slope = signals.mem_available_slope
        if slope <= _MEM_SLOPE_STOP_BPS:
            stage = _escalate(stage, 2)
            reasons.append(f"mem_slope={slope / 1024**2:.0f}MB/s<={_MEM_SLOPE_STOP_BPS / 1024**2:.0f}MB/s")
        elif slope <= _MEM_SLOPE_THROTTLE_BPS:
            stage = _escalate(stage)
            reasons.append(f"mem_slope={slope / 1024**2:.0f}MB/s<={_MEM_SLOPE_THROTTLE_BPS / 1024**2:.0f}MB/s")

        # 4) 高 swap 活动（§154）：大量 swap-in/out 通常意味性能已恶化。
        if signals.swap_max and signals.swap_current and signals.swap_current > signals.swap_max * 0.5:
            stage = _escalate(stage)
            reasons.append("high_swap_activity")

        # 5) writer backpressure → 减少 compute（§40：>0.70 reduce，>0.90 stop）。
        if sink_backpressure > 0.90:
            stage = _escalate(stage, 2)
            reasons.append(f"sink_backpressure={sink_backpressure:.2f}>0.90")
        elif sink_backpressure > 0.70:
            stage = _escalate(stage)
            reasons.append(f"sink_backpressure={sink_backpressure:.2f}>0.70")

        # 6) AIMD：压力升高 → fast multiplicative decrease；稳定 → slow add up。
        if stage in {STAGE_PRESSURE_3, STAGE_PRESSURE_4, STAGE_CRITICAL}:
            self._fast_down(strong=True, stage=stage, now=now)
            self._cooldown = self._cooldown_ticks
            self._stable_count = 0
        elif stage in {STAGE_PRESSURE_1, STAGE_PRESSURE_2}:
            self._fast_down(strong=False, stage=stage, now=now)
            self._cooldown = max(self._cooldown, 1)
            self._stable_count = 0
        else:
            if self._cooldown > 0:
                self._cooldown -= 1
                reasons.append(f"cooldown={self._cooldown}")
            else:
                self._stable_count += 1
                if self._stable_count >= self._stable_required:
                    self._slow_up(now=now)
                    self._stable_count = 0
                    reasons.append("stable_recovery_add_1")
        self._last_stage = stage

        # 7) 派生动态 budgets（§35..39/49）——全部从 Safe Envelope 来。
        safe = max(0, envelope.safe_memory_bytes)
        hard = max(1, envelope.hard_memory_bytes)
        wave = _clamp(int(safe * WAVE_FRACTION), WAVE_ABS_MIN, WAVE_ABS_MAX)
        block = _clamp(int(safe * BLOCK_FRACTION), BLOCK_ABS_MIN, BLOCK_ABS_MAX)
        sink_q = _clamp(int(safe * SINK_FRACTION), SINK_ABS_MIN, SINK_ABS_MAX)
        if job_memory_lease_bytes:
            sink_q = min(sink_q, max(SINK_ABS_MIN, int(job_memory_lease_bytes * SINK_JOB_LEASE_FRACTION)))
        cache_budget = int(hard * CACHE_FRACTION)
        spill_budget = int(hard * SPILL_FRACTION)

        # 压力档位下调 budgets（§66：reduce wave / reduce block）。
        if stage in {STAGE_PRESSURE_2, STAGE_PRESSURE_3}:
            wave = _clamp(int(wave * 0.5), WAVE_ABS_MIN, WAVE_ABS_MAX)
            block = _clamp(int(block * 0.5), BLOCK_ABS_MIN, BLOCK_ABS_MAX)
        if stage in {STAGE_PRESSURE_3, STAGE_PRESSURE_4, STAGE_CRITICAL}:
            cache_budget = int(cache_budget * 0.5)
            spill_budget = int(spill_budget * 0.5)

        # R36 §244/245：低内存自动进入 memory-constrained mode（explain 标签）。
        memory_constrained = (
            stage in {STAGE_PRESSURE_2, STAGE_PRESSURE_3, STAGE_PRESSURE_4, STAGE_CRITICAL}
            or safe < hard * 0.25
        )
        if memory_constrained:
            reasons.append("memory_constrained")

        decision = ResourceDecision(
            target_concurrency=max(1, self._target_concurrency),
            target_cpu_tokens=max(1, self._target_cpu_tokens),
            read_wave_bytes=wave,
            factor_block_bytes=block,
            result_queue_bytes=sink_q,
            io_concurrency=max(1, min(self._broker.hard_cpu_slots, int(self._target_cpu_tokens))),
            remote_concurrency=max(1, self._broker.hard_cpu_slots // 2),
            cache_budget_bytes=cache_budget,
            spill_budget_bytes=spill_budget,
            pressure_state=stage,
            memory_constrained=memory_constrained,
            reasons=tuple(reasons),
        )
        self._decision = decision
        return decision

    # -- AIMD 原语 --

    def _fast_down(self, *, strong: bool, stage: str, now: float) -> None:
        before = self._target_cpu_tokens
        factor = 0.5 if strong else 0.6
        new_cpu = max(1, int(self._target_cpu_tokens * factor))
        self._apply_cpu_budget(new_cpu, stage, now)
        before_cc = self._target_concurrency
        new_cc = max(1, int(self._target_concurrency * factor))
        if new_cc != before_cc:
            self._log("target_concurrency", before_cc, new_cc, f"pressure={stage} fast_down x{factor}")
            self._target_concurrency = new_cc

    def _slow_up(self, *, now: float) -> None:
        before = self._target_cpu_tokens
        new_cpu = min(self._broker.hard_cpu_slots, before + self._up_step)
        self._apply_cpu_budget(new_cpu, "stable_recovery", now)
        before_cc = self._target_concurrency
        new_cc = min(self._broker.hard_cpu_slots, before_cc + self._up_step)
        if new_cc != before_cc:
            self._log("target_concurrency", before_cc, new_cc, "stable_recovery slow_up +1")
            self._target_concurrency = new_cc

    def _apply_cpu_budget(self, new_cpu: int, stage: str, now: float) -> None:
        before = self._target_cpu_tokens
        if new_cpu == before:
            return
        self._target_cpu_tokens = new_cpu
        self._broker._cpu.set_soft_budget(new_cpu)
        self._log("target_cpu_tokens", before, new_cpu, f"stage={stage}")

    # -- 可解释性 --

    def _log(self, field: str, before: Any, after: Any, signal: str) -> None:
        self._decision_log.append(
            _DecisionLogEntry(
                timestamp_ms=time.monotonic() * 1000.0,
                field=field,
                before=before,
                after=after,
                signal=signal,
            )
        )
        if len(self._decision_log) > 200:
            self._decision_log.pop(0)

    def decision_log(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._decision_log]

    def last_decision(self) -> ResourceDecision | None:
        return self._decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_cpu_tokens": self._target_cpu_tokens,
            "target_concurrency": self._target_concurrency,
            "stable_count": self._stable_count,
            "cooldown": self._cooldown,
            "last_stage": self._last_stage,
            "tick_count": self._tick_count,
            "last_decision": self._decision.to_dict() if self._decision else None,
            "decision_log": self.decision_log()[-20:],
        }


def _escalate(stage: str, steps: int = 1) -> str:
    order = [STAGE_NORMAL, STAGE_PRESSURE_1, STAGE_PRESSURE_2, STAGE_PRESSURE_3, STAGE_PRESSURE_4, STAGE_CRITICAL]
    try:
        idx = order.index(stage)
    except ValueError:
        return stage
    return order[min(len(order) - 1, idx + steps)]


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))
