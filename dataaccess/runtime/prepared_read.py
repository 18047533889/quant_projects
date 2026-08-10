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

from dataclasses import dataclass, field
from typing import Any, Mapping

from data_access.core.exceptions import ValidationError


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

    @property
    def paths(self) -> list[str]:
        return list(self.exact_objects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "object_count": len(self.exact_objects),
            "contract_digest": self.contract_digest,
            "source_snapshot_id": self.source_snapshot_id,
        }


@dataclass(frozen=True)
class PreparedRead:
    """一次读的不可变执行计划（R26-P0-003/004）。"""

    request_identity: str
    dataset: str
    runtime_contract: Any = None
    security_context: Any = None
    effective_filters: tuple[PredicateConstraint, ...] = ()
    temporal_plan: TemporalPlan | None = None
    physical_scope: tuple[str, ...] = ()
    resolved_source_snapshot: Any = None
    query_budget: Any = None
    resource_reservation: Any = None
    backend_plan: Mapping[str, Any] = field(default_factory=dict)
    lineage_seed: Mapping[str, Any] = field(default_factory=dict)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_identity": self.request_identity,
            "dataset": self.dataset,
            "physical_object_count": len(self.physical_scope),
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
        }
