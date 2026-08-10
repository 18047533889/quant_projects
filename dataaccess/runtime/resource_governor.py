"""R25 §26/27 —— GlobalResourceGovernor 与 ResourceReservation。

QueryBudget 只限制单次查询。本模块实现**进程级最小全局 governor**：
    - max_active_queries
    - max_total_reserved_memory
    - max_total_scan_bytes_inflight
    - max_remote_concurrency
    - per_principal_active

语义（§27）：
    - 无法证明同一限制跨多 worker → **必须**明确 single-worker contract 或共享
      limiter。这里实现进程级 governor + ``single_worker_contract()`` 部署检查
      （§63：N workers × max_concurrency 会被放大）。
    - 入场顺序（§26）：resolve exact objects → 计算 object count/bytes → admission
      → execute。**不要**执行完才知道扫了多少。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

from data_access.core.exceptions import ResourceAdmissionError

logger = logging.getLogger("data_access.resource_governor")


@dataclass
class ResourceReservation:
    """一个查询/操作持有的一段全局资源。"""

    query_id: str
    principal_id: str
    estimated_scan_bytes: int = 0
    estimated_memory: int = 0
    remote_requests: int = 0

    released: bool = field(default=False)


class GlobalResourceGovernor:
    """进程级全局资源治理（R25 §27）。

    多 worker 部署下这是 **single-worker contract** 的一部分——每进程一个
    governor，跨进程限制必须用共享 Redis/Postgres limiter 或强制
    ``DATA_ACCESS_SINGLE_WORKER=1``。
    """

    def __init__(
        self,
        *,
        max_active_queries: int = 64,
        max_total_reserved_memory: int | None = None,   # bytes
        max_total_scan_bytes_inflight: int | None = None,
        max_remote_concurrency: int = 16,
        max_duckdb_concurrency: int = 8,
        per_principal_active: int | None = None,
    ) -> None:
        self._max_active_queries = max_active_queries
        self._max_memory = max_total_reserved_memory
        self._max_scan = max_total_scan_bytes_inflight
        self._max_remote = max_remote_concurrency
        self._max_duckdb = max_duckdb_concurrency
        self._per_principal_active = per_principal_active

        self._lock = threading.Lock()
        self._active: dict[str, ResourceReservation] = {}
        self._remote_inflight = 0

    # ---- 准入 ----

    def admit(self, reservation: ResourceReservation) -> ResourceReservation:
        """入场前准入（admission）。超限抛 ``ResourceAdmissionError``。

        入场顺序（§26）：``resolve exact objects → 计算 object count/bytes →
        admission → execute``。这里在 execute 前拦截明显超限。
        """
        with self._lock:
            if len(self._active) >= self._max_active_queries:
                raise ResourceAdmissionError(
                    f"resource admission: active queries {len(self._active)} 达到上限 "
                    f"{self._max_active_queries}（T-RES-001，global governor）。"
                )
            if self._per_principal_active is not None:
                per = sum(
                    1
                    for r in self._active.values()
                    if r.principal_id == reservation.principal_id
                )
                if per >= self._per_principal_active:
                    raise ResourceAdmissionError(
                        f"resource admission: principal {reservation.principal_id!r} "
                        f"active queries {per} 达到 per-principal 上限 "
                        f"{self._per_principal_active}。"
                    )
            if self._max_memory is not None:
                used = sum(r.estimated_memory for r in self._active.values())
                if used + reservation.estimated_memory > self._max_memory:
                    raise ResourceAdmissionError(
                        f"resource admission: 全局保留内存 {used + reservation.estimated_memory}"
                        f" > 上限 {self._max_memory}（T-RES-001）。"
                    )
            if self._max_scan is not None:
                inflight = sum(r.estimated_scan_bytes for r in self._active.values())
                if inflight + reservation.estimated_scan_bytes > self._max_scan:
                    raise ResourceAdmissionError(
                        f"resource admission: 全局 inflight scan bytes "
                        f"{inflight + reservation.estimated_scan_bytes} > 上限 "
                        f"{self._max_scan}（R25 §27）。"
                    )
            self._active[reservation.query_id] = reservation
            return reservation

    # ---- 远程并发 ----

    def acquire_remote_slot(self) -> bool:
        """限制远程请求并发（COS LIST/HEAD/httpfs）。"""
        with self._lock:
            if self._remote_inflight >= self._max_remote:
                return False
            self._remote_inflight += 1
            return True

    def release_remote_slot(self) -> None:
        with self._lock:
            if self._remote_inflight > 0:
                self._remote_inflight -= 1

    # ---- 释放 ----

    def release(self, query_id: str) -> None:
        with self._lock:
            self._active.pop(query_id, None)

    # ---- 状态 ----

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def inflight_scan_bytes(self) -> int:
        with self._lock:
            return sum(r.estimated_scan_bytes for r in self._active.values())

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_active_queries": self._max_active_queries,
                "max_total_reserved_memory": self._max_memory,
                "max_total_scan_bytes_inflight": self._max_scan,
                "max_remote_concurrency": self._max_remote,
                "per_principal_active": self._per_principal_active,
                "active": len(self._active),
                "remote_inflight": self._remote_inflight,
            }


_governor: GlobalResourceGovernor | None = None
_governor_lock = threading.Lock()


def get_global_governor() -> GlobalResourceGovernor:
    """进程级 governor 单例。"""
    global _governor
    if _governor is None:
        with _governor_lock:
            if _governor is None:
                _governor = GlobalResourceGovernor()
    return _governor


def reset_global_governor() -> None:
    """测试用：重置 governor 单例。"""
    global _governor
    with _governor_lock:
        _governor = None


def enforce_single_worker_contract() -> bool:
    """R25 §63：multi-worker 部署下进程级 semaphore 会被放大。

    生产部署必须 ``DATA_ACCESS_SINGLE_WORKER=1``（单 worker）或使用共享
    ResourceGovernor（Redis/Postgres）。返回是否强制 single-worker。
    """
    import os

    workers = os.environ.get("DATA_ACCESS_WORKERS", "")
    single = os.environ.get("DATA_ACCESS_SINGLE_WORKER", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if workers and int(workers) > 1 and not single:
        logger.warning(
            "DATA_ACCESS_WORKERS=%s 且未设 DATA_ACCESS_SINGLE_WORKER=1——进程级 "
            "ResourceGovernor 是 single-worker contract；多 worker 并发会被放大。"
            "请强制 single worker 或接共享 limiter（R25 §63）。",
            workers,
        )
        return False
    return True
