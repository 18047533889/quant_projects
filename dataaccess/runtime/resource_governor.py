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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Sequence

from data_access.core.exceptions import ResourceAdmissionError

logger = logging.getLogger("data_access.resource_governor")


def _auto_bound_standalone_memory() -> bool:
    """R38 P0-025：DA standalone 是否自动从 host/cgroup/RLIMIT 派生 safe cap。

    **默认开启**（P0-025 fail-closed）：standalone 下没有上层 coordinator 注入
    上限时，``None`` ≈ 无限是内存安全漏洞，必须自动派生保守 safe cap。
    仅显式 ``DATA_ACCESS_AUTO_BOUND_MEMORY=0/false/no/off`` 才允许无上限
    （测试 / 无上限部署显式选择）。
    """
    import os

    raw = os.environ.get("DATA_ACCESS_AUTO_BOUND_MEMORY", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True  # 默认 fail-closed（含未设置 / "1"/"true"/"yes"）


def _default_safe_memory_bytes() -> int:
    """探测 host RAM / cgroup memory.max / RLIMIT，取最严格并 ×0.5（保守 safe cap）。"""
    import os

    candidates: list[int] = []
    # cgroup v2 memory.max
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if raw and raw not in {"max", "0"}:
                candidates.append(int(raw))
        except (OSError, ValueError):
            pass
    # RLIMIT_AS
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft not in (resource.RLIM_INFINITY, -1) and soft > 0:
            candidates.append(int(soft))
    except Exception:
        pass
    # host RAM
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass
    if not candidates:
        # 无法探测 → 保守固定 4GB safe cap（fail-closed，不是无限）。
        return 4 * 1024**3
    return max(0, int(min(candidates) * 0.5))


@dataclass
class ResourceReservation:
    """一个查询/操作持有的一段全局资源。"""

    query_id: str
    principal_id: str
    estimated_scan_bytes: int = 0
    estimated_memory: int = 0
    remote_requests: int = 0

    released: bool = field(default=False)
    #: R38 P0-020（P0-018）：host-backed 时附加的 HostLeaseRef（FE JobLease child
    #: lease）。release 时一并释放——DA 扫描真正进同一棵 host lease 树。
    host_lease: Any = field(default=None, repr=False)


class _DuckDBSemaphore(threading.BoundedSemaphore):
    """R39 #66：BoundedSemaphore 子类，acquire/release 维护同一份 inflight 计数。

    旧 ``duckdb_inflight()`` 读私有 ``BoundedSemaphore._value``（``max - _value``），
    是依赖 CPython 私有字段的实现细节。这里把计数收进信号量自身：``acquire`` 成功
    即 ``+1``、``release`` 即 ``-1``。engine 直接 acquire 同一信号量（``_exec_sem``
    与 governor ``_duckdb_sem`` 是同一对象）时计数也同步更新——单一权威，不读私有
    字段。
    """

    def __init__(self, value: int) -> None:
        super().__init__(value=value)
        self._inflight_lock = threading.Lock()
        self._inflight = 0

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        ok = super().acquire(*args, **kwargs)
        if ok:
            with self._inflight_lock:
                self._inflight += 1
        return ok

    def release(self) -> None:
        with self._inflight_lock:
            if self._inflight > 0:
                self._inflight -= 1
        super().release()

    def inflight(self) -> int:
        with self._inflight_lock:
            return max(0, self._inflight)


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
        # R38 P0-025（§10）：DA standalone（无 FE coordinator）时，production 不能
        # ``None`` ≈ 无限——检测 host/cgroup/RLIMIT → 保守 safe cap。
        if max_total_reserved_memory is None and _auto_bound_standalone_memory():
            max_total_reserved_memory = _default_safe_memory_bytes()
        if max_total_scan_bytes_inflight is None and _auto_bound_standalone_memory():
            max_total_scan_bytes_inflight = max(0, int(max_total_reserved_memory * 0.5))
        self._max_memory = max_total_reserved_memory
        self._max_scan = max_total_scan_bytes_inflight
        self._max_remote = max_remote_concurrency
        self._max_duckdb = max_duckdb_concurrency
        self._per_principal_active = per_principal_active
        # R40 #58：remote **discovery**（LIST/HEAD/snapshot/manifest 元数据调用）
        # 独立并发治理——与普通 remote 数据请求分开（discovery 无 scan bytes，
        # 但高并发会打爆上游元数据服务）。
        self._max_remote_discovery = max(4, int(max_remote_concurrency // 2))
        self._remote_discovery_inflight = 0
        #: R38 P0-026：动态 setter 与 admit() 同锁；cap 收缩到低于当前 reservations
        #: 时**不 kill incumbents**，只阻止新 admission，telemetry 标 over_current_target。
        self._over_current_target = False
        #: R38 P0-020（P0-018）：FE HostResourceCoordinator 注入的 host child-lease 桥
        #: ``fn(estimated_memory, estimated_scan_bytes) -> HostLeaseRef | None``。
        self._host_lease_request: Any = None

        self._lock = threading.Lock()
        self._active: dict[str, ResourceReservation] = {}
        self._remote_inflight = 0
        # R28-9：远端请求总数（成本/telemetry 维度，admission 不消费——并发由
        # acquire_remote_slot 治理）。QueryBudget 的 remote 预算在 pipeline 层执行。
        self._remote_requests_total = 0
        # R26-P1-016：DuckDB 并发用信号量（blocking acquire，遵守上限但并行任务
        # 排队而非误报）；全局 active-queries 才是 fail-closed admission。
        self._duckdb_sem = _DuckDBSemaphore(
            value=max(1, int(max_duckdb_concurrency))
        )

    # ---- 准入 ----

    def admit(self, reservation: ResourceReservation) -> ResourceReservation:
        """入场前准入（admission）。超限抛 ``ResourceAdmissionError``。

        入场顺序（§26）：``resolve exact objects → 计算 object count/bytes →
        admission → execute``。这里在 execute 前拦截明显超限。

        R26-P1-015：duplicate ``query_id`` 不能覆盖已有 reservation → reject；
        ``remote_requests`` 参与 admission；``released`` 状态受控。
        """
        with self._lock:
            # R38 P0-020（P0-018）：host-backed admission——FE HostResourceCoordinator
            # 注入 child-lease 桥时，DA 扫描先向当前 JobLease 请求 DAScanLease child。
            # 成功 → host 是权威（本 governor 的内存上限由 host 统一治理，不双重计）；
            # 失败/无活跃 job → 回退本地 governor admission（fail-closed 语义保持）。
            if self._host_lease_request is not None:
                try:
                    host_lease = self._host_lease_request(
                        reservation.estimated_memory,
                        reservation.estimated_scan_bytes,
                    )
                except Exception:
                    host_lease = None
                if host_lease is not None:
                    reservation.host_lease = host_lease
                    self._active[reservation.query_id] = reservation
                    self._remote_requests_total += max(0, reservation.remote_requests)
                    return reservation
            if reservation.query_id in self._active:
                raise ResourceAdmissionError(
                    f"resource admission: query_id={reservation.query_id!r} 已存在"
                    "（R26-P1-015：重复 query_id 禁止覆盖已有 reservation）。"
                )
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
            # R28-9：remote 维度拆成两个独立指标，**admission 不再把「远端请求总数」
            # 当成并发上限**。旧实现把 ``reservation.remote_requests``（一次查询要
            # 访问的 COS 对象数，可能 100+）累加后与 ``max_remote_concurrency``
            # 比——一次查 100 个对象并不等于同时发 100 个请求，实际并发可能只有
            # 4，导致系统性过度限流。
            #   - ``remote_requests``：成本/QueryBudget 维度，由 read_pipeline
            #     ``enforce_remote_request_budget``（per-query 对象数预算）治理；
            #   - ``remote_concurrency``：真正的并发 governor，由
            #     ``acquire_remote_slot()`` / ``release_remote_slot()`` 治理
            #     （``_remote_inflight`` 是真实 in-flight 并发数）。
            # 这里只保留一个累积计数器供 telemetry（admission 判定不消费它）。
            self._remote_requests_total += max(0, reservation.remote_requests)
            self._active[reservation.query_id] = reservation
            return reservation

    # ---- R36 P0-017：由 HostResourceCoordinator 派生 envelope 上限 ----

    def set_max_total_reserved_memory(self, memory_bytes: int) -> None:
        """FE HostResourceCoordinator 注入 Safe Envelope（§54：DA 不独立决定全局内存）。

        R38 P0-026（§10）：与 ``admit()`` **同一把锁**。cap 收缩到低于当前
        reservations 时：不 kill incumbents；新 admission 被 ``admit`` 阻止；
        telemetry 标 ``over_current_target``（incumbents release 后恢复）。
        """
        with self._lock:
            self._max_memory = max(0, int(memory_bytes)) if memory_bytes is not None else None
            used = sum(r.estimated_memory for r in self._active.values())
            self._over_current_target = (
                self._max_memory is not None and used > self._max_memory
            )

    def set_max_total_scan_bytes_inflight(self, scan_bytes: int) -> None:
        """协调器注入 scan inflight 上限（§56：scan bytes inflight 成为 host lease 一部分）。

        R38 P0-026：与 ``admit()`` 同锁。
        """
        with self._lock:
            self._max_scan = max(0, int(scan_bytes)) if scan_bytes is not None else None

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

    # ---- 远程 discovery 并发（R40 #58：LIST/HEAD/snapshot/manifest 独立治理）----

    def acquire_remote_discovery_slot(self) -> bool:
        """限制 LIST / HEAD / snapshot / manifest 元数据调用的并发。

        discovery 与一般 remote 数据请求分开治理——批处理反复解析同一数据集时，
        元数据调用在 execution admission 前发生（R40 #57 两相租约的发现相），
        必须有独立 slot 上限防打爆上游。
        """
        with self._lock:
            if self._remote_discovery_inflight >= self._max_remote_discovery:
                return False
            self._remote_discovery_inflight += 1
            return True

    def release_remote_discovery_slot(self) -> None:
        with self._lock:
            if self._remote_discovery_inflight > 0:
                self._remote_discovery_inflight -= 1

    def remote_discovery_inflight(self) -> int:
        with self._lock:
            return self._remote_discovery_inflight

    # ---- DuckDB 并发（R26-P1-016：声明的能力必须执行）----

    def acquire_duckdb_slot(self) -> bool:
        """R26-P1-016：DuckDB 并发 slot（blocking semaphore acquire）。

        遵守 ``max_duckdb_concurrency`` 上限；并发任务排队而非误报 fail（避免
        把「并行读」误判成超限）。返回 True（acquire 后）。
        """
        self._duckdb_sem.acquire()
        return True

    def release_duckdb_slot(self) -> None:
        self._duckdb_sem.release()

    @property
    def duckdb_semaphore(self) -> threading.BoundedSemaphore:
        """R29-P0 #201：DuckDB 并发**单一信号量**（engine ``_exec_sem`` 与
        ``duckdb_slot`` 共用同一个对象）——不再存在「Store slot + Engine sem」
        两把并发闸（每 query 双份计账、有效并发减半）。
        """
        return self._duckdb_sem

    def duckdb_inflight(self) -> int:
        """R39 #66：读原子 inflight 计数（不再读私有 ``BoundedSemaphore._value``）。"""
        return self._duckdb_sem.inflight()

    # ---- 释放 ----

    def set_host_lease_request(self, fn: Any) -> None:
        """R38 P0-020（P0-018）：注入 FE host child-lease 桥。

        ``fn(estimated_memory, estimated_scan_bytes) -> HostLeaseRef | None``；
        None（无活跃 job / 无 headroom）→ admit 回退本地 governor。
        """
        with self._lock:
            self._host_lease_request = fn

    def release(self, query_id: str) -> None:
        with self._lock:
            reservation = self._active.pop(query_id, None)
            # host-backed reservation：一并释放 host child lease（同一棵 lease 树）。
            if reservation is not None and reservation.host_lease is not None:
                try:
                    release_fn = getattr(reservation.host_lease, "release", None)
                    if callable(release_fn):
                        release_fn()
                except Exception:
                    pass
            # R38 P0-026：incumbent release 后重算 over_current_target。
            used = sum(r.estimated_memory for r in self._active.values())
            self._over_current_target = (
                self._max_memory is not None and used > self._max_memory
            )

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
                "remote_requests_total": self._remote_requests_total,
                "remote_discovery_inflight": self._remote_discovery_inflight,
                "max_remote_discovery": self._max_remote_discovery,
                "over_current_target": self._over_current_target,
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
    if workers and workers.isdigit() and int(workers) > 1 and not single:
        logger.warning(
            "DATA_ACCESS_WORKERS=%s 且未设 DATA_ACCESS_SINGLE_WORKER=1——进程级 "
            "ResourceGovernor 是 single-worker contract；多 worker 并发会被放大。"
            "请强制 single worker 或接共享 limiter（R25 §63）。",
            workers,
        )
        return False
    return True


@contextmanager
def duckdb_slot(governor: "GlobalResourceGovernor | None" = None):
    """R26-P1-016：DuckDB 并发 slot（声明的 ``max_duckdb_concurrency`` 真正执行）。

    引擎执行（execute_arrow / execute_reader）持有 slot；acquire 失败 → 拒绝
    （fail-closed，不等排队）。成功/异常都 release exactly once。
    """
    gov = governor if governor is not None else get_global_governor()
    if not gov.acquire_duckdb_slot():
        raise ResourceAdmissionError(
            "duckdb concurrency 达到上限（R26-P1-016：max_duckdb_concurrency 真正执行）"
        )
    try:
        yield
    finally:
        gov.release_duckdb_slot()
