# -*- coding: utf-8 -*-
"""R31-P0-025/026 + R33-P0-001..006 + R39-P0-PERF-002/003/004: BatchDataRequest
—— 一批因子合并成一次 DataAccess 规划。

R33 修订
    - **R33-P0-001/002**：不再从 ``engine.data_source`` 推断**唯一** anchor 源。
      从 ``analyses.referenced_columns`` + plan 里的 ``SourceRef`` 列提取
      （dataset / market）构造多个独立 ``SourceScanGroup`` —— secondary
      fundamental / industry / universe / minute 源各自成为独立 source group。
    - **R33-P0-004**：``estimate_scan_cost`` 传入真实 ``time_range`` +
      ``instrument_scope``（不再只 ``fields=ordered`` 估全量）。
    - **R33-P0-003**：估算失败记录 :class:`ScanCostUnavailable`（degraded
      planning），**不再** ``except Exception: scan_cost = None`` 静默吞。
    - **R33-P0-005/006**：source 身份用 typed :class:`SourceScopeId`（字段取值，
      禁止 ``split("::")`` 解析业务身份）。

R39 修订
    - **R39-P0-PERF-002**：secondary SourceRef 分组不再走 ``dict[str, str]``
      （dataset 名/market 混塞进 ``SourceScopeId(dataset=...)`` 是类型混用）。
      改为 :mod:`planner.source_binding` 的 typed :class:`ColumnSourceBinding`，
      每列直接绑定到其真实 dataset/field/market/typed scope。
    - **R39-P0-PERF-003**：每个 :class:`SourceScanGroup` 通过
      :class:`BatchSourceResolver` 解析**自己的** source adapter，调用
      ``adapter.estimate_scan_cost``；secondary 源绝不再复用 anchor 的
      estimator（适配器不可得时记录独立的 :class:`ScanCostUnavailable`）。
    - **R39-P0-PERF-004**：``SourceScanGroup.time_range`` 用 typed
      :class:`TimeRange`，保留 ``(None, end)`` / ``(start, None)`` /
      ``(start, end)`` 三种形态，不再因 ``time_range[0] is None`` 丢弃整窗。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Iterable

from planner.physical_factor_dag import SourceScopeId
from planner.source_binding import TimeRange, discover_column_source_bindings


@dataclass(frozen=True)
class ScanCostUnavailable:
    """R33-P0-003：性能规划失去成本数据时**不无痕降级**。

    correctness 可继续；performance evidence 必须记录 degraded planning
    （dataset / reason / fallback_estimate / confidence）。
    """

    dataset: str
    reason: str
    fallback_estimate: int = 0
    confidence: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "reason": self.reason,
            "fallback_estimate": self.fallback_estimate,
            "confidence": self.confidence,
        }


def _scan_cost_dict(cost: Any) -> dict[str, Any]:
    return {
        "selected_bytes": int(getattr(cost, "selected_bytes", 0) or 0),
        "projection_bytes": int(getattr(cost, "projection_bytes", 0) or 0),
        "estimated_rows": int(getattr(cost, "estimated_rows", 0) or 0),
        "files": int(getattr(cost, "file_count", 0) or 0),
        "remote": bool(getattr(cost, "remote", False)),
        "instrument_count": int(getattr(cost, "instrument_count", 0) or 0),
    }


@dataclass(frozen=True)
class SourceScanGroup:
    """同一 source scope 的合并扫描请求。

    ``source_scope`` 是 typed :class:`SourceScopeId`；``source_scope_key`` 是其
    canonical 字符串（跨系统对齐用，业务身份解析永远读 typed 字段）。

    R39-P0-PERF-003：每个 group 绑定**自己的** ``source_adapter`` /
    ``cost_estimator`` / ``storage_kind`` / ``cost_dataset`` —— secondary 源
    不再复用 anchor 的 cost estimator。
    R39-P0-PERF-004：``time_range`` 是 typed :class:`TimeRange`（允许一端为
    None）。
    """

    group_id: int
    dataset: str
    source_scope: SourceScopeId
    snapshot_id: str
    fields: tuple[str, ...]
    time_range: TimeRange | None = None
    instrument_scope: tuple[str, ...] | None = None
    scan_cost: Any | None = None
    scan_cost_unavailable: ScanCostUnavailable | None = None
    # R39-P0-PERF-003：per-scope 解析出的 adapter / estimator / 存储形态。
    source_adapter: Any | None = None
    cost_estimator: Any | None = None
    storage_kind: str = ""
    cost_dataset: str = ""

    @property
    def source_scope_key(self) -> str:
        return self.source_scope.key()

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "dataset": self.dataset,
            "source_scope": self.source_scope.to_dict(),
            "snapshot_id": self.snapshot_id,
            "fields": list(self.fields),
            "n_fields": len(self.fields),
            # PERF-004：time_range 保留 None 端（list 形态向后兼容旧 JSON）。
            "time_range": (
                list(self.time_range.as_tuple())
                if self.time_range is not None
                else None
            ),
            "time_range_typed": (
                self.time_range.to_dict() if self.time_range is not None else None
            ),
            "scan_cost": _scan_cost_dict(self.scan_cost) if self.scan_cost is not None else None,
            "scan_cost_unavailable": (
                self.scan_cost_unavailable.to_dict()
                if self.scan_cost_unavailable is not None
                else None
            ),
            "storage_kind": self.storage_kind,
            "cost_dataset": self.cost_dataset or self.dataset,
        }


@dataclass
class BatchDataRequest:
    """整批因子合并后的 DataAccess 数据需求（R31 §29 / R33 multi-source）。

    R39-P0-PERF-002 hard-gate 计数器放在 ``hard_gate_counters``：
    ``SOURCE_REF_WITHOUT_TYPED_SOURCE_BINDING == 0`` 表示每个 SourceRef 列都
    产生了 typed :class:`ColumnSourceBinding`。
    """

    fields: tuple[str, ...] = ()
    groups: list[SourceScanGroup] = field(default_factory=list)
    degraded_planning: list[ScanCostUnavailable] = field(default_factory=list)
    hard_gate_counters: dict[str, int] = field(default_factory=dict)

    @property
    def scan_cost_map(self) -> dict[str, Any]:
        """``source_scope_key -> ScanCost``（供 read wave / IO token / admission）。"""
        return {
            g.source_scope_key: g.scan_cost
            for g in self.groups
            if g.scan_cost is not None
        }

    @property
    def total_selected_bytes(self) -> int:
        return sum(
            int(getattr(g.scan_cost, "selected_bytes", 0) or 0) for g in self.groups
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": list(self.fields),
            "n_fields": len(self.fields),
            "groups": [g.to_dict() for g in self.groups],
            "total_selected_bytes": self.total_selected_bytes,
            "n_groups": len(self.groups),
            "degraded_planning": [d.to_dict() for d in self.degraded_planning],
            "hard_gate_counters": dict(self.hard_gate_counters),
        }


class BatchSourceResolver:
    """R39-P0-PERF-003：scope → 真实 DataSourceAdapter 的解析器。

    anchor scope 永远解析到 ``anchor_source``；secondary scope 依次尝试：
        1. ``anchor_source.sources`` 注册表里 ``dataset == scope.dataset`` 的子源；
        2. ``anchor_source._child(dataset)`` 子源工厂（LQTP logical source）；
        3. ``get_dataset_source`` / ``source_for`` / ``resolve_source_for_dataset``；
    全部不可得时返回 ``None``（调用方记录独立 ``ScanCostUnavailable``，**绝不**
    静默复用 anchor 的 estimator）。
    """

    def __init__(self, anchor_source: Any, *, market: str = ""):
        self.anchor_source = anchor_source
        self.market = str(market or "")
        self.anchor_scope = _anchor_source_scope(anchor_source, market=self.market)
        self._cache: dict[str, Any] = {}

    def resolve_source(self, scope: SourceScopeId) -> Any:
        """返回 scope 对应的 DataSourceAdapter；不可得返回 ``None``。"""
        key = scope.key()
        if key in self._cache:
            return self._cache[key]
        adapter = self._resolve(scope)
        self._cache[key] = adapter
        return adapter

    def _resolve(self, scope: SourceScopeId) -> Any:
        if scope.dataset == self.anchor_scope.dataset:
            return self.anchor_source
        src = self.anchor_source
        if src is None:
            return None
        # 1) composite-like 注册表：按 dataset 匹配子源。
        registry = getattr(src, "sources", None)
        if isinstance(registry, dict):
            for sub in registry.values():
                if str(getattr(sub, "dataset", "") or "") == scope.dataset:
                    return sub
            if scope.dataset in registry:
                return registry[scope.dataset]
        # 2) 子源工厂（LQTP logical source._child(dataset) 等）。
        child_factory = getattr(src, "_child", None)
        if callable(child_factory):
            try:
                child = child_factory(scope.dataset)
            except Exception:  # noqa: BLE001
                child = None
            if child is not None:
                return child
        # 3) 通用 dataset → source 解析接口。
        for method_name in (
            "get_dataset_source",
            "source_for",
            "resolve_source_for_dataset",
        ):
            fn = getattr(src, method_name, None)
            if callable(fn):
                try:
                    found = fn(scope.dataset)
                except Exception:  # noqa: BLE001
                    found = None
                if found is not None:
                    return found
        return None


def _anchor_source_scope(source: Any, *, market: str = "") -> SourceScopeId:
    """从 anchor 源构造 typed source scope（R33-P0-005）。"""
    dataset = str(getattr(source, "dataset", "") or "")
    snapshot_id = ""
    for key in ("_manifest_token", "_data_snapshot_id", "snapshot_id"):
        value = getattr(source, key, None)
        if value:
            snapshot_id = str(value)
            break
    return SourceScopeId(dataset=dataset, snapshot_id=snapshot_id, market=market)


def _walk_columns(plan: Any, out: set[str]) -> None:
    op = str(getattr(plan, "op", "") or "")
    if op == "column":
        name = str((getattr(plan, "attrs", None) or {}).get("name") or "")
        if name:
            out.add(name)
    for child in getattr(plan, "inputs", ()) or ():
        _walk_columns(child, out)


def _looks_like_source_ref(name: str) -> bool:
    try:
        from api.source_ref import looks_like_source_ref
    except Exception:  # pragma: no cover
        return False
    return looks_like_source_ref(name)


def _infer_storage_kind(adapter: Any) -> str:
    """R39-P0-PERF-003：best-effort 存储形态推断（无既有 ``storage_kind`` 属性）。"""
    if adapter is None:
        return "unknown"
    kind = getattr(adapter, "storage_kind", None)
    if kind:
        return str(kind)
    capabilities = getattr(adapter, "capabilities", None)
    engine_kind = getattr(capabilities, "engine_kind", None)
    if engine_kind:
        return str(engine_kind)
    name = type(adapter).__name__.lower()
    if "access" in name:
        return "data_access"
    if "composite" in name:
        return "composite"
    if "memory" in name or "mock" in name:
        return "memory"
    return "unknown"


def _estimator_accepts(estimator: Any, param: str) -> bool:
    """estimate_scan_cost 是否接受某关键字参数（真实 DataAccessSource 不接受
    ``dataset=``，mock 常带 ``**kw`` 接受一切）。"""
    try:
        sig = inspect.signature(estimator)
    except (TypeError, ValueError):
        return True
    for name, p in sig.parameters.items():
        if name == param:
            return True
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _call_scan_estimator(
    estimator: Any,
    *,
    dataset: str,
    fields: tuple[str, ...],
    time_range: tuple[Any, Any] | None,
    instruments: tuple[str, ...] | None,
) -> Any:
    """按 estimator 实际签名调用 ``estimate_scan_cost``。

    ``dataset`` 仅在 estimator 接受该参数时才传入（适配器已经 per-scope 解析，
    dataset 是证据语义，不是 DataAccessSource 的必需参数）。
    """
    kwargs: dict[str, Any] = {
        "fields": fields,
        "time_range": time_range,
        "instruments": instruments,
    }
    if _estimator_accepts(estimator, "dataset"):
        kwargs["dataset"] = dataset
    return estimator(**kwargs)


def build_batch_data_request(
    engine_or_source: Any,
    *,
    analyses: dict[str, Any] | None = None,
    dag: Any | None = None,
    fields: Iterable[str] | None = None,
    ctx: Any | None = None,
) -> BatchDataRequest:
    """合并整批因子依赖，按 source scope 生成**一个或多个** DataRequest 规划。

    - anchor 源 = ``engine.data_source``；secondary 源 = 计划中的 SourceRef
      数据集（R33-P0-001）。
    - R39-P0-PERF-002：secondary 源经 :class:`ColumnSourceBinding` 直接由列
      解析（typed dataset/field/market/scope），不再 ``dict[str,str]`` 混塞。
    - R39-P0-PERF-003：每 source scope 一个独立 adapter 的
      ``estimate_scan_cost``（真实 time_range + instruments）。
    - R39-P0-PERF-004：time_range 用 typed :class:`TimeRange` 保留 None 端。
    - 估算失败记录 :class:`ScanCostUnavailable`（R33-P0-003），不再静默吞。
    - source 身份用 typed :class:`SourceScopeId`（R33-P0-005/006）。
    """
    source = getattr(engine_or_source, "data_source", None) or engine_or_source
    # 1) 字段并集。
    all_fields: set[str] = set()
    for a in (analyses or {}).values():
        refs = getattr(a, "referenced_columns", None) or ()
        all_fields |= set(refs)
    if fields:
        all_fields |= set(fields)
    plans: list[Any] = []
    if dag is not None:
        for fp in getattr(dag, "roots", ()) or ():
            root = getattr(fp, "root", fp)
            plans.append(root)
            _walk_columns(root, all_fields)
        for sub in (getattr(dag, "shared_nodes", None) or {}).values():
            plans.append(sub)
            _walk_columns(sub, all_fields)
    ordered = tuple(sorted(all_fields))

    market = str(getattr(ctx, "market", "") or "")
    anchor_scope = _anchor_source_scope(source, market=market)

    # 2) R39-P0-PERF-002：typed ColumnSourceBinding（从列直接解析）。
    discovery = discover_column_source_bindings(plans)
    bindings = discovery.bindings
    group_by_scope: dict[SourceScopeId, set[str]] = {}
    anchor_cols: set[str] = set()
    unbound_sref = 0
    for name in ordered:
        binding = bindings.get(name)
        if binding is not None:
            # 与 anchor 同源（同 dataset + 同 market）的 ref 交给 anchor 源
            # 处理（anchor 源就是能读 ref 的 composite/lqtp 源），避免同一
            # dataset 出现两份 scope 相同的 group。
            if (
                binding.dataset == anchor_scope.dataset
                and (binding.market or market) == market
            ):
                anchor_cols.add(name)
            else:
                group_by_scope.setdefault(binding.source_scope, set()).add(
                    binding.field
                )
        else:
            if _looks_like_source_ref(name):
                unbound_sref += 1
            anchor_cols.add(name)
    # 说明：``ordered`` 已覆盖计划内全部 column（``_walk_columns`` 与
    # ``discover_column_source_bindings`` 走同一批 plans），所以上面循环对
    # unbound 的计数就是完整的 hard-gate 值；不再叠加 seen-bound 差（避免
    # 同一列被数两次）。

    # anchor group 总是存在（兜底）。
    anchor_fields: tuple[str, ...] = tuple(sorted(anchor_cols))
    if not anchor_fields and not group_by_scope:
        anchor_fields = ordered  # 纯兜底：无任何 SourceRef 时全部进 anchor。
    group_specs: list[tuple[SourceScopeId, set[str]]] = []
    if anchor_scope.dataset or anchor_cols or not group_by_scope:
        group_specs.append((anchor_scope, anchor_cols))
    for scope, cols in group_by_scope.items():
        group_specs.append((scope, cols))

    # 3) R39-P0-PERF-003：每 source scope 独立 adapter + cost estimator。
    resolver = BatchSourceResolver(source, market=market)
    requests: list[SourceScanGroup] = []
    degraded: list[ScanCostUnavailable] = []
    time_range = TimeRange.from_source(source)
    instrument_scope = tuple(
        getattr(source, "instrument_filter", None) or ()
    ) or None
    for gid, (scope, cols) in enumerate(group_specs):
        adapter = resolver.resolve_source(scope)
        estimator = (
            getattr(adapter, "estimate_scan_cost", None)
            if adapter is not None
            else None
        )
        storage_kind = _infer_storage_kind(adapter)
        cost_dataset = scope.dataset
        cost = None
        unavail = None
        group_fields = tuple(sorted(cols)) or anchor_fields
        if callable(estimator):
            try:
                cost = _call_scan_estimator(
                    estimator,
                    dataset=cost_dataset,
                    fields=group_fields,
                    time_range=time_range.as_tuple() if time_range is not None else None,
                    instruments=instrument_scope,
                )
            except Exception as exc:  # noqa: BLE001
                unavail = ScanCostUnavailable(
                    dataset=scope.dataset,
                    reason=f"{type(exc).__name__}: {exc}",
                    fallback_estimate=0,
                    confidence="none",
                )
        else:
            if adapter is None:
                if scope.dataset == anchor_scope.dataset:
                    reason = "no estimate_scan_cost on source"
                else:
                    reason = f"no secondary adapter for dataset {scope.dataset!r}"
            else:
                reason = "no estimate_scan_cost on resolved source"
            unavail = ScanCostUnavailable(
                dataset=scope.dataset,
                reason=reason,
                fallback_estimate=0,
                confidence="none",
            )
        if unavail is not None:
            degraded.append(unavail)
        requests.append(
            SourceScanGroup(
                group_id=gid,
                dataset=scope.dataset,
                source_scope=scope,
                snapshot_id=scope.snapshot_id,
                fields=group_fields,
                time_range=time_range,
                instrument_scope=instrument_scope,
                scan_cost=cost,
                scan_cost_unavailable=unavail,
                source_adapter=adapter,
                cost_estimator=estimator,
                storage_kind=storage_kind,
                cost_dataset=cost_dataset,
            )
        )
    return BatchDataRequest(
        fields=ordered,
        groups=requests,
        degraded_planning=degraded,
        hard_gate_counters={
            "SOURCE_REF_WITHOUT_TYPED_SOURCE_BINDING": unbound_sref,
        },
    )
