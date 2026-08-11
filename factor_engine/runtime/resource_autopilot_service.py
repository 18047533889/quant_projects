# -*- coding: utf-8 -*-
"""R38 P0-011/012/013（§7）：ResourceAutopilotService —— 固定 cadence 的单一控制循环。

R36 的 controller 被多个 scheduler 热循环直接 tick：
    - ``_stable_count`` 增长速度由 job 数决定；
    - ``cooldown`` 由 loop 次数决定；
    - recovery 可能在几毫秒内完成；pressure down 也可能被重复乘很多次；
    - controller mutable state 无统一锁；PSS/PSI 采样被频繁执行。

R38 正确架构（§42）::

    ResourceMonitor
        ↓ fixed cadence (500ms~1s)
    ResourceController.tick()
        ↓
    atomic publish ResourceDecisionSnapshot
        ↓
    Schedulers / DA / Writer 只读 last_decision()

本模块实现：
    - :class:`ResourceDecisionSnapshot`（decision_id / generated_at / valid_until /
      signals_version，§P0-012）——scheduler 若 decision 太旧 → conservative fallback。
    - :class:`ResourceAutopilotService` —— 进程级后台线程，每 ``interval_s`` tick
      一次 controller 并原子发布 snapshot。
    - :func:`get_resource_autopilot` / :func:`start_resource_autopilot` —— 进程级
      单例；``ResourceBroker.resource_decision()`` 在服务激活时**只读** snapshot，
      不再自行 tick（§P0-013：不要在 scheduler 热循环 force 采样）。
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

#: 默认固定 cadence（§7：500ms~1s）。
DEFAULT_INTERVAL_S = 0.5
#: decision 过期阈值：超过该时长未刷新 → conservative fallback（§P0-012）。
DECISION_STALE_S = 2.0


@dataclass(frozen=True)
class ResourceDecisionSnapshot:
    """一次 atomic publish 的 control snapshot（§P0-012）。"""

    decision_id: str
    generated_at_monotonic: float
    valid_until: float
    signals_version: str
    decision: Any

    @property
    def is_stale(self) -> bool:
        return time.monotonic() > self.valid_until

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "generated_at_monotonic": round(self.generated_at_monotonic, 3),
            "valid_until": round(self.valid_until, 3),
            "signals_version": self.signals_version,
            "decision": self.decision.to_dict(),
        }


class ResourceAutopilotService:
    """进程级固定 cadence 控制循环（单一 tick source，§7）。"""

    def __init__(
        self,
        broker: Any,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        self._broker = broker
        self._interval = max(0.1, float(interval_s))
        self._controller = broker._resource_controller()
        # R39：把 service 挂到 broker 上——scheduler 调 ``resource_decision()``
        # 时优先用它做**只读** snapshot（即使本 service 不是全局单例），绝不
        # fall through 到每调用一次 controller.tick（多 scheduler 热循环放大）。
        try:
            broker._autopilot_service = self
        except Exception:
            pass
        self._lock = threading.RLock()
        self._snapshot: ResourceDecisionSnapshot | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False
        self._tick_count = 0
        self._ticks: list[float] = []
        self._last_decision_sec = 0.0

    # -- lifecycle --

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="fe-resource-autopilot", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._started = False
            self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=min(2.0, self._interval * 2 + 0.2))

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    # -- control loop --

    def _run(self) -> None:
        # 先立即 tick 一次，让第一个 snapshot 立即可用。
        try:
            self._tick()
        except Exception:
            pass
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if self._stop.is_set():
                break
            try:
                self._tick()
            except Exception:
                # 控制循环失败不能 kill 线程（兜底：保留上一个 snapshot）。
                pass

    def _tick(self) -> None:
        """一个固定 cadence 的 control tick：采样一次 → controller.tick() → 发布。

        P0-016：消费 scheduler 上报的 live signals（sink backpressure / job lease
        字节）——不再硬编码 ``None/0.0`` 让 controller 永远看不到真实 backpressure。
        """
        now = time.monotonic()
        snap = self._broker._refresh(force=True)
        live = self._broker.live_signals()
        decision = self._controller.tick(
            self._broker._signals_from_snapshot(snap),
            self._broker._envelope_from_snapshot(snap),
            job_memory_lease_bytes=live.get("job_memory_lease_bytes"),
            sink_backpressure=float(live.get("sink_backpressure", 0.0) or 0.0),
        )
        snapshot = ResourceDecisionSnapshot(
            decision_id=uuid.uuid4().hex[:12],
            generated_at_monotonic=now,
            valid_until=now + DECISION_STALE_S,
            signals_version=f"sig:{snap.timestamp_ms:.0f}",
            decision=decision,
        )
        with self._lock:
            self._snapshot = snapshot
            self._tick_count += 1
            self._ticks.append(now)
            if len(self._ticks) > 1000:
                self._ticks = self._ticks[-500:]
            self._last_decision_sec = now
        # R39 #72：把 cache_budget_bytes 应用到已注册的 cache 消费者（DA
        # QueryResultCache）。失败只记不炸——控制循环必须存活。
        self._apply_cache_consumers(decision)

    def _apply_cache_consumers(self, decision: Any) -> None:
        """把 decision.cache_budget_bytes 推给已注册的 cache 消费者。

        DA QueryResultCache 通过 ``apply_resource_cache_budget`` 注册（懒加载）。
        消费者异常 → 只记日志，不影响 control tick。
        """
        budget = getattr(decision, "cache_budget_bytes", None)
        if not budget:
            return
        try:
            from data_access.read.query_cache import apply_resource_cache_budget

            apply_resource_cache_budget(int(budget))
        except Exception:
            # DA 不可用（standalone FE）或 DA 缓存模块缺失 → 跳过，非致命。
            pass

    # -- read-only consumer API --

    def last_decision(self) -> ResourceDecisionSnapshot | None:
        with self._lock:
            return self._snapshot

    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    def summary(self) -> dict[str, Any]:
        with self._lock:
            out = {
                "started": self._started,
                "interval_s": self._interval,
                "tick_count": self._tick_count,
                "last_decision_sec": round(self._last_decision_sec, 3),
                "last_snapshot": self._snapshot.to_dict() if self._snapshot else None,
                "controller": self._controller.to_dict(),
            }
        # R39 #74：统一 cache inventory 总可回收字节（DA QueryResultCache 等）。
        try:
            from data_access.runtime.cache_inventory import inventory_summary

            out["cache_inventory"] = inventory_summary()
        except Exception:
            pass
        return out


#: 进程级单例（一个进程一个 autopilot；绑定 host coordinator 的 broker）。
_AUTOPILOT: ResourceAutopilotService | None = None
_AUTOPILOT_LOCK = threading.Lock()


def get_resource_autopilot() -> ResourceAutopilotService | None:
    return _AUTOPILOT


def start_resource_autopilot(
    broker: Any | None = None, *, interval_s: float = DEFAULT_INTERVAL_S
) -> ResourceAutopilotService:
    """启动进程级 autopilot（幂等）。绑定 host coordinator 的 broker。"""
    global _AUTOPILOT
    with _AUTOPILOT_LOCK:
        if _AUTOPILOT is not None and _AUTOPILOT.started:
            return _AUTOPILOT
        if broker is None:
            try:
                from runtime.host_resource_coordinator import get_host_coordinator

                broker = get_host_coordinator().broker
            except Exception:
                from runtime.resource_broker import ResourceBroker

                broker = ResourceBroker()
        _AUTOPILOT = ResourceAutopilotService(broker, interval_s=interval_s)
        _AUTOPILOT.start()
        return _AUTOPILOT


def stop_resource_autopilot() -> None:
    global _AUTOPILOT
    with _AUTOPILOT_LOCK:
        if _AUTOPILOT is not None:
            _AUTOPILOT.stop()
            _AUTOPILOT = None


def reset_resource_autopilot() -> None:
    stop_resource_autopilot()
