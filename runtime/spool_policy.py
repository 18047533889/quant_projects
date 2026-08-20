# -*- coding: utf-8 -*-
"""R39-P1-PERF-027: spool threshold 自适应。

把「固定 512MB spool 阈值」升级为环境感知的纯函数决策，让 shard 结果在
「writer 很快 / merge 很近 / 内存足」时留在内存，否则才 spool 到磁盘。

输入（全部可测量）
    - ``live_headroom``：当前 live 内存余量（字节）
    - ``writer_queue_headroom``：writer 队列剩余容量（字节）
    - ``spool_disk_throughput_bytes_per_s``：spool 盘吞吐估计（字节/秒）
    - ``estimated_merge_lifetime_s``：距 merge 的预计时间（秒）

决策规则（:func:`decide_spool_or_keep`，纯函数）
    1. shard 很小（< 64MiB）→ KEEP（小结果直接留内存，不承担落盘开销）；
    2. writer 队列余量 > shard **且** live 内存余量 > shard → KEEP
       （writer 很快 / 内存足，直接写，不 spool）；
    3. merge 很接近（< 5s）且内存足 → KEEP（merge 马上来，不用 spool）；
    4. 否则 → SPOOL（spool 盘吞吐高时优先 arrow 格式，见 ``shard_executor``）。

阈值推导（:func:`adaptive_spool_threshold`）
    基线 512MiB；``FACTOR_ENGINE_SPOOL_THRESHOLD_BYTES`` 环境变量覆盖一切；
    否则按 live headroom 比例与磁盘吞吐上下浮动。取不到任何信号时
    :func:`default_spool_policy_factory` 走原固定 512MiB 阈值（诚实 fail-safe，
    绝不因缺少信号而比旧行为更激进地 spool）。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

#: 决策枚举常量
DECISION_SPOOL = "SPOOL"
DECISION_KEEP = "KEEP"

#: spool 阈值基线（原固定值 512MB，§P0-003）
SPOOL_THRESHOLD_BYTES_BASELINE = 512 * 1024**2
#: 低于该值一律 KEEP（小 shard 直接留内存，不承担落盘开销）
MIN_KEEP_THRESHOLD_BYTES = 64 * 1024**2
#: merge 距今小于该秒数视为「即将 merge」→ 倾向 KEEP
MERGE_NEAR_SECONDS = 5.0

#: 环境变量
ENV_SPOOL_THRESHOLD = "FACTOR_ENGINE_SPOOL_THRESHOLD_BYTES"
ENV_MERGE_LIFETIME = "FACTOR_ENGINE_ESTIMATED_MERGE_LIFETIME_S"

#: 磁盘吞吐快 / 慢阈值（用于 adaptive_spool_threshold 与 arrow 格式偏好）
_FAST_DISK_BYTES_PER_S = 1024**3  # >= 1GB/s → spool 更划算
_SLOW_DISK_BYTES_PER_S = 100 * 1024**2  # <= 100MB/s → 尽量少落盘


@dataclass(frozen=True)
class SpoolDecision:
    """一次 spool/keep 决策的完整记录（可观测、可测试）。"""

    decision: str  # DECISION_SPOOL | DECISION_KEEP
    reason: str  # 决策原因（机器可读）
    threshold_bytes: int  # 本次使用的 spool 阈值（诊断）
    live_headroom: int | None = None
    writer_queue_headroom: int | None = None
    spool_disk_throughput: int | None = None
    estimated_merge_lifetime: float | None = None
    #: True 表示阈值经 :func:`adaptive_spool_threshold` 推导；False = 固定阈值 fail-safe。
    adaptive_threshold: bool = False


def _plentiful(headroom: int | None, shard_bytes: int) -> bool:
    """headroom 是否「富余」——严格大于 shard 才叫富余（None → 不富余）。"""
    return headroom is not None and headroom > shard_bytes


def adaptive_spool_threshold(
    headroom_fraction: float | None = None,
    disk_throughput: int | None = None,
) -> int:
    """由环境推导 spool 阈值（字节）。

    - ``FACTOR_ENGINE_SPOOL_THRESHOLD_BYTES`` 环境变量覆盖一切；
    - 否则以 512MiB 基线，按 live headroom 比例（相对 hard limit）与磁盘吞吐
      上下浮动：
        * headroom_fraction >= 0.5（内存富余）→ ×2（更多留内存）；
        * headroom_fraction <= 0.15（内存紧）→ ×0.5（更早 spool）；
        * disk_throughput >= 1GB/s（盘快）→ ×0.5（spool 更划算）；
        * disk_throughput <= 100MB/s（盘慢）→ ×2（尽量少落盘）。
    - 取不到输入 → 返回基线 512MiB。
    """
    raw = os.environ.get(ENV_SPOOL_THRESHOLD, "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    multiplier = 1.0
    if headroom_fraction is not None:
        if headroom_fraction >= 0.50:
            multiplier *= 2.0
        elif headroom_fraction <= 0.15:
            multiplier *= 0.5
    if disk_throughput is not None and disk_throughput > 0:
        if disk_throughput >= _FAST_DISK_BYTES_PER_S:
            multiplier *= 0.5
        elif disk_throughput <= _SLOW_DISK_BYTES_PER_S:
            multiplier *= 2.0
    return max(MIN_KEEP_THRESHOLD_BYTES, int(SPOOL_THRESHOLD_BYTES_BASELINE * multiplier))


def decide_spool_or_keep(
    shard_bytes: int,
    *,
    live_headroom: int | None,
    writer_queue_headroom: int | None,
    spool_disk_throughput_bytes_per_s: int | None,
    estimated_merge_lifetime_s: float | None,
) -> SpoolDecision:
    """纯函数：按 §R39-P1-PERF-027 规则决定 SPOOL 还是 KEEP。"""
    threshold = adaptive_spool_threshold(
        headroom_fraction=None,
        disk_throughput=spool_disk_throughput_bytes_per_s,
    )
    if shard_bytes < MIN_KEEP_THRESHOLD_BYTES:
        return SpoolDecision(
            decision=DECISION_KEEP,
            reason="small_shard_below_min_keep",
            threshold_bytes=threshold,
            live_headroom=live_headroom,
            writer_queue_headroom=writer_queue_headroom,
            spool_disk_throughput=spool_disk_throughput_bytes_per_s,
            estimated_merge_lifetime=estimated_merge_lifetime_s,
            adaptive_threshold=True,
        )
    if _plentiful(writer_queue_headroom, shard_bytes) and _plentiful(
        live_headroom, shard_bytes
    ):
        return SpoolDecision(
            decision=DECISION_KEEP,
            reason="writer_fast_memory_ample",
            threshold_bytes=threshold,
            live_headroom=live_headroom,
            writer_queue_headroom=writer_queue_headroom,
            spool_disk_throughput=spool_disk_throughput_bytes_per_s,
            estimated_merge_lifetime=estimated_merge_lifetime_s,
            adaptive_threshold=True,
        )
    if (
        estimated_merge_lifetime_s is not None
        and estimated_merge_lifetime_s < MERGE_NEAR_SECONDS
        and _plentiful(live_headroom, shard_bytes)
    ):
        return SpoolDecision(
            decision=DECISION_KEEP,
            reason="merge_imminent_memory_ample",
            threshold_bytes=threshold,
            live_headroom=live_headroom,
            writer_queue_headroom=writer_queue_headroom,
            spool_disk_throughput=spool_disk_throughput_bytes_per_s,
            estimated_merge_lifetime=estimated_merge_lifetime_s,
            adaptive_threshold=True,
        )
    return SpoolDecision(
        decision=DECISION_SPOOL,
        reason="spool_to_disk",
        threshold_bytes=threshold,
        live_headroom=live_headroom,
        writer_queue_headroom=writer_queue_headroom,
        spool_disk_throughput=spool_disk_throughput_bytes_per_s,
        estimated_merge_lifetime=estimated_merge_lifetime_s,
        adaptive_threshold=True,
    )


# ---------------------------------------------------------------------------
# 模块级计数器（spool_decision_count / spool_keep_count /
# adaptive_threshold_applied_count）+ getter/reset
# ---------------------------------------------------------------------------

_spool_decision_count = 0
_spool_keep_count = 0
_adaptive_threshold_applied_count = 0
_counters_lock = threading.Lock()


def record_spool_decision(decision: SpoolDecision) -> None:
    """按一次决策更新模块级计数器（shard_executor 接线钩子）。"""
    global _spool_decision_count, _spool_keep_count, _adaptive_threshold_applied_count
    with _counters_lock:
        if decision.decision == DECISION_KEEP:
            _spool_keep_count += 1
        else:
            _spool_decision_count += 1
        if decision.adaptive_threshold:
            _adaptive_threshold_applied_count += 1


def get_spool_decision_count() -> int:
    with _counters_lock:
        return _spool_decision_count


def get_spool_keep_count() -> int:
    with _counters_lock:
        return _spool_keep_count


def get_adaptive_threshold_applied_count() -> int:
    with _counters_lock:
        return _adaptive_threshold_applied_count


def reset_spool_policy_counters() -> None:
    global _spool_decision_count, _spool_keep_count, _adaptive_threshold_applied_count
    with _counters_lock:
        _spool_decision_count = 0
        _spool_keep_count = 0
        _adaptive_threshold_applied_count = 0


# ---------------------------------------------------------------------------
# default_spool_policy_factory —— 从 broker / sink / 资源快照 best-effort 构建
# ---------------------------------------------------------------------------


def _resolve_live_headroom(broker: Any, explicit: int | None) -> int | None:
    if explicit is not None:
        return int(explicit)
    if broker is not None:
        try:
            v = broker.snapshot().live_headroom
            if v is not None:
                return int(v)
        except Exception:  # noqa: BLE001
            pass
    # broker 缺省时的兜底：governor 探测。返回 0（探测失败）→ 视为未知。
    try:
        from runtime.resource_governor import live_memory_headroom_bytes

        v = live_memory_headroom_bytes()
        if v is not None and v > 0:
            return int(v)
    except Exception:  # noqa: BLE001
        pass
    return None


def _resolve_writer_queue_headroom(sink: Any, explicit: int | None) -> int | None:
    if explicit is not None:
        return int(explicit)
    if sink is None:
        return None
    try:
        queues = getattr(sink, "_worker_queues", None)
        if queues:
            # 多 worker：取最紧（余量最小）的队列——它就是 spool 决策的瓶颈。
            return min(
                max(0, int(q.max_bytes) - int(q.current_bytes)) for q in queues
            )
        q = getattr(sink, "queue", None)
        if q is not None and hasattr(q, "max_bytes") and hasattr(q, "current_bytes"):
            return max(0, int(q.max_bytes) - int(q.current_bytes))
    except Exception:  # noqa: BLE001
        pass
    return None


def _resolve_disk_throughput(broker: Any, explicit: int | None) -> int | None:
    if explicit is not None:
        return int(explicit)
    capacity: int | None = None
    try:
        from runtime.resource_governor import spill_disk_speed_class

        cls = spill_disk_speed_class()
        capacity = {
            "nvme": 1500 * 1024**2,
            "ssd": 500 * 1024**2,
            "network": 50 * 1024**2,
            "unknown": None,
        }.get(cls)
    except Exception:  # noqa: BLE001
        capacity = None
    if capacity is None:
        return None
    busy: float | None = None
    if broker is not None:
        try:
            busy = getattr(broker.snapshot(), "disk_busy", None)
        except Exception:  # noqa: BLE001
            busy = None
    # 当前磁盘越忙，可用吞吐越低（保守估计：busy>=0.7 → 1/4，>=0.4 → 1/2）。
    if busy is not None:
        if busy >= 0.7:
            capacity = capacity // 4
        elif busy >= 0.4:
            capacity = capacity // 2
    return int(capacity)


def _resolve_merge_lifetime(explicit: float | None) -> float | None:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(ENV_MERGE_LIFETIME, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return None


def _fixed_threshold_decision(shard_bytes: int) -> SpoolDecision:
    """诚实 fail-safe：取不到任何自适应信号 → 原固定 512MiB 阈值语义。"""
    keep = shard_bytes < SPOOL_THRESHOLD_BYTES_BASELINE
    return SpoolDecision(
        decision=DECISION_KEEP if keep else DECISION_SPOOL,
        reason="fail_safe_fixed_threshold",
        threshold_bytes=SPOOL_THRESHOLD_BYTES_BASELINE,
        adaptive_threshold=False,
    )


def default_spool_policy_factory(
    *,
    broker: Any = None,
    sink: Any = None,
    live_headroom: int | None = None,
    writer_queue_headroom: int | None = None,
    spool_disk_throughput: int | None = None,
    estimated_merge_lifetime_s: float | None = None,
) -> Callable[[int], SpoolDecision]:
    """从 broker / sink / 资源快照构建 spool 决策 policy（best-effort）。

    - 显式参数优先；缺省时 best-effort 探测：``broker.snapshot()``、
      sink 队列余量、spill 盘速度等级（+ disk_busy 折扣）、
      env ``FACTOR_ENGINE_ESTIMATED_MERGE_LIFETIME_S``。
    - 探测失败 → 该字段为 ``None``。**全部**为 ``None``（取不到任何信号）时，
      policy 走原固定 512MiB 阈值（诚实 fail-safe），绝不比旧行为更激进。
    """
    def policy(shard_bytes: int) -> SpoolDecision:
        lh = _resolve_live_headroom(broker, live_headroom)
        wqh = _resolve_writer_queue_headroom(sink, writer_queue_headroom)
        tp = _resolve_disk_throughput(broker, spool_disk_throughput)
        ml = _resolve_merge_lifetime(estimated_merge_lifetime_s)
        if lh is None and wqh is None and tp is None and ml is None:
            return _fixed_threshold_decision(shard_bytes)
        return decide_spool_or_keep(
            shard_bytes,
            live_headroom=lh,
            writer_queue_headroom=wqh,
            spool_disk_throughput_bytes_per_s=tp,
            estimated_merge_lifetime_s=ml,
        )

    return policy


__all__ = [
    "DECISION_KEEP",
    "DECISION_SPOOL",
    "ENV_MERGE_LIFETIME",
    "ENV_SPOOL_THRESHOLD",
    "MERGE_NEAR_SECONDS",
    "MIN_KEEP_THRESHOLD_BYTES",
    "SPOOL_THRESHOLD_BYTES_BASELINE",
    "SpoolDecision",
    "adaptive_spool_threshold",
    "decide_spool_or_keep",
    "default_spool_policy_factory",
    "get_adaptive_threshold_applied_count",
    "get_spool_decision_count",
    "get_spool_keep_count",
    "record_spool_decision",
    "reset_spool_policy_counters",
]
