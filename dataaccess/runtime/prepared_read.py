"""R26-P0-003 —— PreparedRead：一个 immutable executable plan。

R25 目标「一个 immutable executable plan」必须成为事实：所有 public read path
**先 ``store.prepare_read()`` 再 ``store.execute_prepared_read()``**，禁止 backend
自己重新解释 registry / COS contract / semantic catalog / mirror spec / predicate。

``PreparedRead`` 携带一次读的全部已决议状态：
    - ``runtime_contract``    ContractIR v2 编译产物（单一事实源）
    - ``security_context``    request-scoped 执行上下文（P0-005）
    - ``effective_filters``   PredicateConstraint IR（P0-012，非 raw Mapping）
    - ``temporal_plan``       谓词时钟 / 分区时钟 / availability
    - ``physical_scope``      **exact** 物理对象（snapshot resolver 输出）
    - ``resolved_source_snapshot``
    - ``query_budget``        QueryBudget v2（含 remote objects/bytes/memory）
    - ``resource_reservation`` GlobalResourceGovernor 已准入（P0-017）
    - ``lineage_seed``        lineage 种子
"""
from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Mapping

from data_access.core.exceptions import DeadlineExceeded, ValidationError


@dataclass(frozen=True)
class PredicateConstraint:
    """一条**已编译**的过滤约束（R26-P0-012）。

    不再是 raw Mapping——``kind`` 显式区分：
        - ``exact``    ：值域精确（Eq/In 字面量）
        - ``unprovable``：无法证明 result ⊆ allowed domain（Ne/NotIn/IsNotNull/Or）
                         —— production 下 equivalent 到「未限制」
    ``scope``：panel / event / dimension / all。
    """

    field: str
    dataset: str
    kind: str
    scope: str = "all"
    values: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"exact", "in", "unprovable"}:
            raise ValidationError(
                f"PredicateConstraint.kind 必须是 exact/in/unprovable，收到 {self.kind!r}"
            )
        if self.scope not in {"panel", "event", "dimension", "all"}:
            raise ValidationError(
                f"PredicateConstraint.scope 必须是 panel/event/dimension/all，"
                f"收到 {self.scope!r}"
            )

    @property
    def proves_restriction(self) -> bool:
        return self.kind in {"exact", "in"} and bool(self.values)


@dataclass(frozen=True)
class TemporalPlan:
    """一次读的时间轴计划（R26-P0-014 PIT floor 消费）。

    ``pit_floor`` 是系统级 floor（contract 编译出的权威 availability），request
    只能 same-or-stricter，不能 loosen（见 ``apply_pit_policy_floor``）。
    """

    predicate_clock: str | None = None
    partition_clock: str | None = None
    availability: Any = None          # AvailabilityResult | None
    pit_floor: str | None = None      # same_day / next_session_open / ...
    request_availability: str | None = None
    pit_fidelity: str | None = None


@dataclass(frozen=True)
class AvailabilityContract:
    """R39 P0 #30：dataset 级 availability 声明（由 RuntimeDatasetContract 派生）。

    从 ``RuntimeDatasetContract`` 提取：
        - ``knowledge``   knowledge/availability 时间列（``PITContract.availability_column``
                          或 temporal_axes["knowledge"].column）
        - ``availability`` availability 策略串（``same_day`` / ``next_trading_day`` / ...，
                          由 ``pit_policy`` 经 ``_PIT_FLOOR_OF`` 映射）
    之后作为 ``compile_available_from_result(knowledge, availability, ...)`` 的两个
    位置参数传入——修复旧代码把 ``dataset``（str）当 knowledge、availability 缺失的
    TypeError 并被 ``except Exception`` 吞掉的问题。
    """

    knowledge: str | None
    availability: str = "same_day"


class EmptyPhysicalScope(list):
    """R39 P0 #44：**合法空**物理读取范围（数据集存在但没有匹配文件）。

    与 ``SourceResolutionError``（解析失败）严格区分：``[]`` 只表示「合法空」，
    绝不静默同时代表「解析失败」。作为 list 子类，``list(EmptyPhysicalScope(...))``
    与下游 ``for p in source_paths`` 全部兼容，同时带类型标记。
    """

    def __init__(self, *, dataset: str, reason: str = "no matching partitions") -> None:
        super().__init__()
        self.dataset = dataset
        self.reason = reason


_deadline_ctx_var: ContextVar["DeadlineContext | None"] = ContextVar(
    "data_access_request_deadline", default=None
)


@dataclass
class DeadlineContext:
    """R39 P0 #32/#33/#37：request-scoped **单一**绝对 deadline。

    prepare_read / 组合读在最外层入口建立，贯穿 resolution → snapshot → schema
    epoch → contract compile → admission → execute 全部子阶段——早期阶段也计入
    同一请求预算（旧实现 deadline 在 prep 之后才建，HEAD/LIST/schema 的耗时
    完全不在预算内）。``deadline_at`` 为 None 表示无 deadline（不限制）。
    """

    deadline_at: float | None
    source: str = "unknown"

    @classmethod
    def start(
        cls,
        budget: Any = None,
        *,
        max_elapsed_ms: float | None = None,
        source: str = "request",
    ) -> "DeadlineContext":
        ms = None
        if budget is not None:
            ms = getattr(budget, "max_elapsed_ms", None)
        if ms is None:
            ms = max_elapsed_ms
        deadline_at = None
        if ms is not None and float(ms) > 0:
            deadline_at = time.monotonic() + float(ms) / 1000.0
        return cls(deadline_at=deadline_at, source=source)

    def check(self, context: str = "请求") -> None:
        """检查当前是否已过 deadline；已过 → DeadlineExceeded（fail-fast）。"""
        if self.deadline_at is not None and time.monotonic() >= self.deadline_at:
            raise DeadlineExceeded(
                f"{context}已超过请求绝对 deadline（R39 P0 #32/#33：早期阶段计入同一预算）。"
            )

    def remaining_ms(self) -> float | None:
        """剩余毫秒；已过 → DeadlineExceeded。None = 无 deadline。"""
        if self.deadline_at is None:
            return None
        remaining = self.deadline_at - time.monotonic()
        if remaining <= 0:
            raise DeadlineExceeded(
                "请求已超过绝对 deadline（R39 P0 #32/#33：prepare 阶段已计入）。"
            )
        return remaining * 1000.0

    def remaining_secs(self) -> float | None:
        """剩余秒；已过 → DeadlineExceeded。None = 无 deadline。"""
        if self.deadline_at is None:
            return None
        remaining = self.deadline_at - time.monotonic()
        if remaining <= 0:
            raise DeadlineExceeded(
                "请求已超过绝对 deadline（R39 P0 #32/#33）。"
            )
        return remaining

    def enter(self) -> Any:
        """把本 deadline 设为当前 request 的权威 deadline（ContextVar），返回 token。"""
        return _deadline_ctx_var.set(self)


def set_deadline_context(ctx: "DeadlineContext | None") -> Any:
    return _deadline_ctx_var.set(ctx)


def reset_deadline_context(token: Any) -> None:
    _deadline_ctx_var.reset(token)


def current_deadline() -> "DeadlineContext | None":
    """当前 request 的权威绝对 deadline（无则 None）。"""
    return _deadline_ctx_var.get()


@dataclass(frozen=True)
class VerifiedPhysicalScope:
    """R27-D：已核验的物理读取范围——只能由 Store 内部（prepare_read /
    ReadPlan pin / SourceSnapshotResolver）构造，public API 拒绝 raw path。

    ``dataset_id``  绑定数据集（authorize 的是它，扫描的也是它——两者不可再分离）
    ``exact_objects`` 已冻结的精确文件/URI 集合（不再二次 glob）
    ``contract_digest`` 数据集 Contract 指纹（校验 scope 与 dataset 语义一致）
    ``source_snapshot_id`` 可选：构造时刻的 source snapshot 身份
    """

    dataset_id: str
    exact_objects: tuple[str, ...]
    contract_digest: str
    source_snapshot_id: str | None = None
    # ReadPlan pin carries the exact identities already checked at plan/execute
    # verification; prepare_read must compare freshly observed identities against
    # these facts before constructing the terminal snapshot. None means that this
    # scope has no pin identity contract; () is an authoritative expected-empty set.
    expected_file_versions: tuple[Any, ...] | None = None
    snapshot_policy: str = "latest"
    publisher_snapshot: Mapping[str, Any] | None = None
    # Both pin and verified_fail_if_changed consume exact_objects as the terminal
    # scan scope. A verified publisher object set must never be followed by live
    # path re-resolution.
    use_exact_objects: bool = True

    @property
    def paths(self) -> list[str]:
        return list(self.exact_objects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "object_count": len(self.exact_objects),
            "contract_digest": self.contract_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "snapshot_policy": self.snapshot_policy,
            "expected_identity_count": (
                len(self.expected_file_versions)
                if self.expected_file_versions is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PreparedRead:
    """一次读的不可变执行计划（R26-P0-003/004）。

    R32-P0-061/062: 深度不可变——所有嵌套字段冻结为 tuple/immutable types，
    禁止持有 mutable request 或 live store capability（backend_plan/lineage_seed
    用 tuple[tuple[str, Any], ...] 代替 dict，确保 plan 可安全跨上下文传递）。
    """

    request_identity: str
    dataset: str
    runtime_contract: Any = None
    security_context: Any = None
    effective_filters: tuple[PredicateConstraint, ...] = ()
    temporal_plan: TemporalPlan | None = None
    physical_scope: tuple[str, ...] = ()
    resolved_source_snapshot: Any = None
    snapshot_policy: str = "latest"
    query_budget: Any = None
    resource_reservation: Any = None
    # R32-P0-061: backend_plan/lineage_seed 改为 immutable tuple of pairs
    backend_plan: tuple[tuple[str, Any], ...] = ()
    lineage_seed: tuple[tuple[str, Any], ...] = ()
    # R29-P0：prepare 时刻固化的安全身份。``security_digest`` =
    # (principal_id + policy.digest + run_mode) 的指纹；``credential_scope_id`` =
    # 生效 credential provider 的 scope。terminal execute 前必须与当前 execution
    # context 重新计算并相等——高权限上下文 prepare 的对象不能被低权限上下文
    # 直接 execute（PreparedRead 不是跨上下文的通行证）。
    security_digest: str | None = None
    credential_scope_id: str | None = None
    # R29-P0 #201：**全请求 absolute deadline**——prepare 开始即建立
    # ``monotonic() + max_elapsed_ms``，execute 只拿剩余时间（prepare 阶段
    # HEAD/LIST/schema 的耗时也计入请求预算，不再给 execute 一个全新完整 deadline）。
    deadline_at: float | None = None
    # R39 P0 #32/#33：携带的 DeadlineContext（单一 authority；execute 只从它
    # 取剩余时间，prepare 全部子阶段共用同一预算）。
    deadline_context: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_identity": self.request_identity,
            "dataset": self.dataset,
            "physical_object_count": len(self.physical_scope),
            "snapshot_policy": self.snapshot_policy,
            "snapshot_digest": (
                getattr(self.resolved_source_snapshot, "content_digest", None)
                if self.resolved_source_snapshot is not None
                else None
            ),
            "filters": [
                {
                    "field": f.field,
                    "dataset": f.dataset,
                    "kind": f.kind,
                    "scope": f.scope,
                    "values": list(f.values),
                }
                for f in self.effective_filters
            ],
            "temporal_plan": (
                self.temporal_plan.__dict__ if self.temporal_plan is not None else None
            ),
            # R32-P0-061: tuple of pairs → dict for serialization
            "backend_plan": dict(self.backend_plan) if self.backend_plan else {},
            "lineage_seed": dict(self.lineage_seed) if self.lineage_seed else {},
        }
