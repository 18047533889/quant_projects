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
import operator
from typing import Any, Iterable

from factor_engine.planner.physical_factor_dag import SourceScopeId
from factor_engine.planner.source_binding import TimeRange, discover_column_source_bindings


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


class CompositeExecutionScanCostUnavailable(ValueError):
    """Raw execution scope cannot be admitted without every component cost."""


@dataclass(frozen=True)
class CompositeExecutionScanCost:
    """Conservative raw-read cost for one composite logical-source wave.

    Every byte/row field is the sum of independent physical source scopes.  In
    particular, ``estimated_rows`` is *raw scan work* and is not the row count
    of any as-of aligned logical output.
    """

    dataset: str
    file_count: int
    total_bytes: int
    estimated_rows: int
    projected_columns: int
    total_columns: None
    remote: bool
    selected_files: int
    selected_bytes: int
    projection_bytes: int
    cost_basis: str = "composite_execution_raw_scope_sum"
    component_scope_keys: tuple[str, ...] = ()


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
    anchor_source_scope: SourceScopeId | None = None
    groups: list[SourceScanGroup] = field(default_factory=list)
    degraded_planning: list[ScanCostUnavailable] = field(default_factory=list)
    hard_gate_counters: dict[str, int] = field(default_factory=dict)
    column_source_scope_keys: dict[str, str] = field(default_factory=dict)
    column_scan_costs: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def scan_cost_map(self) -> dict[str, Any]:
        """``source_scope_key -> ScanCost``（供 read wave / IO token / admission）。"""
        return {
            g.source_scope_key: g.scan_cost
            for g in self.groups
            if g.scan_cost is not None
        }

    @property
    def anchor_source_scope_key(self) -> str:
        return self.anchor_source_scope.key() if self.anchor_source_scope is not None else ""

    def execution_scan_cost_map(self) -> dict[str, Any]:
        """Return the cost map consumed by the current physical lowerer.

        The lowerer creates one SOURCE_SCAN/read wave under the anchor scope,
        while a composite logical source may synchronously read several typed
        secondary scopes.  Bind the sum of those raw physical costs to that
        exact anchor key.  This map is intentionally separate from
        :attr:`scan_cost_map` and ``column_scan_costs``; neither per-scope CBO
        facts nor logical-column attribution is mutated.

        Missing/unknown component evidence fails closed.  A known empty scope
        is accepted only when its rows, files, selected bytes and projection
        bytes are all zero.
        """
        anchor_scope_key = self.anchor_source_scope_key
        if not anchor_scope_key:
            raise ValueError("anchor_scope_key must be non-empty")
        if not self.groups:
            raise CompositeExecutionScanCostUnavailable(
                "composite execution scan cost has no typed source groups"
            )
        scope_keys = tuple(group.source_scope_key for group in self.groups)
        if len(set(scope_keys)) != len(scope_keys):
            raise CompositeExecutionScanCostUnavailable(
                "composite execution scan cost contains duplicate typed source scopes"
            )
        anchor_groups = [
            group for group in self.groups
            if group.source_scope_key == anchor_scope_key
        ]
        if len(anchor_groups) != 1:
            raise CompositeExecutionScanCostUnavailable(
                "lowerer anchor scope is not represented exactly once in execution groups: "
                f"{anchor_scope_key!r}"
            )
        if len(self.groups) == 1:
            group = self.groups[0]
            if group.source_scope_key != anchor_scope_key:
                raise CompositeExecutionScanCostUnavailable(
                    "single execution source scope does not match lowerer anchor: "
                    f"{group.source_scope_key!r} != {anchor_scope_key!r}"
                )
            self._validated_raw_component(group)
            # Preserve the established one-source contract exactly.
            return {anchor_scope_key: group.scan_cost}

        components = [self._validated_raw_component(group) for group in self.groups]
        return {
            anchor_scope_key: CompositeExecutionScanCost(
                dataset=anchor_groups[0].dataset,
                file_count=sum(item["file_count"] for item in components),
                total_bytes=sum(item["total_bytes"] for item in components),
                estimated_rows=sum(item["estimated_rows"] for item in components),
                projected_columns=sum(item["projected_columns"] for item in components),
                total_columns=None,
                remote=any(item["remote"] for item in components),
                selected_files=sum(item["selected_files"] for item in components),
                selected_bytes=sum(item["selected_bytes"] for item in components),
                projection_bytes=sum(item["projection_bytes"] for item in components),
                component_scope_keys=scope_keys,
            )
        }

    @staticmethod
    def _validated_raw_component(group: SourceScanGroup) -> dict[str, Any]:
        cost = group.scan_cost
        scope = group.source_scope_key
        if group.scan_cost_unavailable is not None or cost is None:
            reason = (
                group.scan_cost_unavailable.reason
                if group.scan_cost_unavailable is not None
                else "missing scan cost"
            )
            raise CompositeExecutionScanCostUnavailable(
                f"raw execution cost unavailable for scope {scope!r}: {reason}"
            )
        basis = str(getattr(cost, "cost_basis", "") or "")
        if basis in {"", "unknown", "cost_unknown_conservative"}:
            raise CompositeExecutionScanCostUnavailable(
                f"raw execution cost has non-exact basis for scope {scope!r}: {basis!r}"
            )

        def integer(name: str, *, nullable: bool = False) -> int:
            value = getattr(cost, name, None)
            if value is None and nullable:
                raise CompositeExecutionScanCostUnavailable(
                    f"raw execution cost lacks {name} for scope {scope!r}"
                )
            if isinstance(value, bool):
                raise CompositeExecutionScanCostUnavailable(
                    f"raw execution cost has invalid {name} for scope {scope!r}"
                )
            try:
                result = operator.index(value)
            except (TypeError, ValueError) as exc:
                raise CompositeExecutionScanCostUnavailable(
                    f"raw execution cost has invalid {name} for scope {scope!r}"
                ) from exc
            if result < 0:
                raise CompositeExecutionScanCostUnavailable(
                    f"raw execution cost has negative {name} for scope {scope!r}"
                )
            return result

        rows = integer("estimated_rows")
        files = integer("file_count")
        selected_files = integer("selected_files")
        selected = integer("selected_bytes", nullable=True)
        projection = integer("projection_bytes", nullable=True)
        projected_columns = integer("projected_columns")
        total = integer("total_bytes", nullable=True)
        known_empty = not any((rows, files, selected_files, selected, projection))
        # A frozen physical scope may contain files while exact footer
        # predicates prove that no row group can match the actual query.
        # None is unknown evidence, not a zero-row-group proof.
        if basis == "physical_scope" and getattr(cost, "selected_rowgroups", None) is not None:
            selected_rowgroups = integer("selected_rowgroups")
            known_empty = known_empty or (
                selected_rowgroups == 0
                and getattr(cost, "empty_result_proven", False) is True
                and getattr(cost, "rowgroup_pruning_basis", "") == "parquet_footer_min_max_closed_predicates"
                and not any((rows, selected_files, selected, projection))
            )
        if not known_empty and (projection <= 0 or selected <= 0):
            raise CompositeExecutionScanCostUnavailable(
                f"raw execution cost is non-empty but has no positive byte bound for scope {scope!r}"
            )
        return {
            "estimated_rows": rows,
            "file_count": files,
            "selected_files": selected_files,
            "selected_bytes": selected,
            "projection_bytes": projection,
            "projected_columns": projected_columns,
            "total_bytes": total,
            "remote": bool(getattr(cost, "remote", False)),
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

    def resolve_source(
        self,
        scope: SourceScopeId,
        *,
        semantic_filters: dict[str, Any] | None = None,
    ) -> Any:
        """返回 scope 对应的 DataSourceAdapter；不可得返回 ``None``。"""
        key = scope.key()
        if key in self._cache:
            return self._cache[key]
        adapter = self._resolve(scope, semantic_filters=semantic_filters)
        self._cache[key] = adapter
        return adapter

    def _resolve(
        self,
        scope: SourceScopeId,
        *,
        semantic_filters: dict[str, Any] | None = None,
    ) -> Any:
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
                    return self._scope_filtered_adapter(sub, semantic_filters)
            if scope.dataset in registry:
                return self._scope_filtered_adapter(
                    registry[scope.dataset], semantic_filters
                )
        # 2) 子源工厂（LQTP logical source._child(dataset) 等）。
        child_factory = getattr(src, "_child", None)
        if callable(child_factory):
            try:
                if semantic_filters:
                    child = child_factory(
                        scope.dataset,
                        semantic_filters=dict(semantic_filters),
                    )
                else:
                    child = child_factory(scope.dataset)
            except TypeError:
                # Older generic child factories may not expose the keyword.
                # They are only usable when no explicit SourceRef identity is
                # required; silently dropping rank/index/industry is forbidden.
                child = None if semantic_filters else child_factory(scope.dataset)
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
                    if semantic_filters:
                        found = fn(
                            scope.dataset,
                            semantic_filters=dict(semantic_filters),
                        )
                    else:
                        found = fn(scope.dataset)
                except TypeError:
                    found = None if semantic_filters else fn(scope.dataset)
                except Exception:  # noqa: BLE001
                    found = None
                if found is not None:
                    return self._scope_filtered_adapter(found, semantic_filters)
        # A strict DataAccess anchor can create a same-window adapter only for
        # a dataset registered to the already resolved market. Arbitrary
        # dataset strings and cross-market scopes remain unresolved.
        try:
            from factor_engine.storage.sources.data_access_source import DataAccessSource
            from factor_engine.storage.sources.logical_tables import (
                ASHARE_LOGICAL_TABLES,
                US_LOGICAL_TABLES,
            )

            registered = {
                "ashare": {
                    contract.dataset for contract in ASHARE_LOGICAL_TABLES.values()
                    if contract.dataset
                },
                "us": {
                    contract.dataset for contract in US_LOGICAL_TABLES.values()
                    if contract.dataset
                },
            }
            from factor_engine.fields.market_registry import MultiMarketFieldRegistry

            field_registries = MultiMarketFieldRegistry()
            for market_name in tuple(registered):
                registered[market_name].update(
                    str(spec.dataset)
                    for spec in field_registries.table_specs(market_name)
                    if getattr(spec, "dataset", None)
                )
            if (
                isinstance(src, DataAccessSource)
                and self.market in registered
                and scope.market == self.market
                and scope.dataset in registered[self.market]
            ):
                from data_access.cos_contract import get_cos_contract

                contract = get_cos_contract(scope.dataset)
                temporal_model = str(
                    getattr(contract, "temporal_model", "") or ""
                ).upper()
                secondary_read_mode = (
                    "event" if temporal_model in {"E1", "E2"} else
                    str(getattr(src, "read_mode", "panel") or "panel")
                )
                return DataAccessSource(
                    dataset=scope.dataset,
                    start_date=src.start_date,
                    end_date=src.end_date,
                    instrument_filter=list(src.instrument_filter or ()),
                    read_auto=getattr(src, "read_auto", None),
                    params=dict(getattr(src, "params", None) or {}),
                    semantic_filters={
                        **dict(getattr(src, "semantic_filters", None) or {}),
                        **dict(semantic_filters or {}),
                    },
                    read_mode=secondary_read_mode,
                    strict_unknown_fields=src.strict_unknown_fields,
                    run_mode=src.run_mode,
                    production=src.production,
                )
        except Exception:
            return None
        return None

    @staticmethod
    def _scope_filtered_adapter(
        adapter: Any,
        semantic_filters: dict[str, Any] | None,
    ) -> Any:
        """Return an identity-specific adapter without mutating shared sources.

        A pre-registered ``DataAccessSource`` is often unfiltered.  Reusing it
        for ``ShareholderRank=1`` and ``ShareholderRank=2`` would collapse two
        distinct SourceRef scopes into the same physical estimate/read identity.
        Clone only this public adapter type; opaque adapters fail closed unless
        they already carry the exact requested semantic filters.
        """
        requested = dict(semantic_filters or {})
        if not requested:
            return adapter
        existing = dict(getattr(adapter, "semantic_filters", None) or {})
        merged = {**existing, **requested}
        if existing == merged:
            return adapter
        try:
            from factor_engine.storage.sources.data_access_source import DataAccessSource

            if isinstance(adapter, DataAccessSource):
                return DataAccessSource(
                    dataset=adapter.dataset,
                    fields=dict(getattr(adapter, "fields", None) or {}),
                    start_date=adapter.start_date,
                    end_date=adapter.end_date,
                    instrument_filter=list(adapter.instrument_filter or ()),
                    normalize_timestamp=adapter.normalize_timestamp,
                    timestamp_unit=adapter.timestamp_unit,
                    read_auto=getattr(adapter, "read_auto", None),
                    params=dict(getattr(adapter, "params", None) or {}),
                    semantic_filters=merged,
                    read_mode=str(getattr(adapter, "read_mode", "panel") or "panel"),
                    strict_unknown_fields=adapter.strict_unknown_fields,
                    run_mode=adapter.run_mode,
                    production=adapter.production,
                    pit_enforce=bool(getattr(adapter, "pit_enforce", False)),
                )
        except Exception:  # noqa: BLE001
            return None
        return None


def _binding_semantic_filters(binding: Any) -> dict[str, Any]:
    """Recover the exact SourceRef row identity carried by an encoded column.

    ``params_digest`` separates physical scopes but is intentionally not parsed
    back into business values.  The encoded SourceRef remains the authoritative
    reversible carrier.  Transform parameters are execution semantics and are
    not row filters.
    """
    try:
        from factor_engine.api.source_ref import decode_source_ref

        ref = decode_source_ref(binding.encoded_column)
        return dict(ref.params_dict())
    except Exception:  # noqa: BLE001
        return {}


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
    visited: set[int] = set()
    active: set[int] = set()
    stack = [(plan, False)]
    while stack:
        node, exiting = stack.pop()
        node_key = id(node)
        if exiting:
            active.remove(node_key)
            visited.add(node_key)
            continue
        if node_key in active:
            raise ValueError("cyclic plan dependency while collecting batch columns")
        if node_key in visited:
            continue
        active.add(node_key)
        if str(getattr(node, "op", "") or "") == "column":
            name = str((getattr(node, "attrs", None) or {}).get("name") or "")
            if name:
                out.add(name)
        stack.append((node, True))
        stack.extend(
            (child, False)
            for child in reversed(tuple(getattr(node, "inputs", ()) or ()))
        )


def _looks_like_source_ref(name: str) -> bool:
    try:
        from factor_engine.api.source_ref import looks_like_source_ref
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
    discovery = discover_column_source_bindings(
        plans,
        market=market,
        anchor_dataset=anchor_scope.dataset,
    )
    bindings = discovery.bindings
    binding_markets = {
        binding.market for binding in bindings.values() if binding.market
    }
    if not market and len(binding_markets) == 1:
        market = next(iter(binding_markets))
    group_by_scope: dict[SourceScopeId, set[str]] = {}
    filters_by_scope: dict[SourceScopeId, dict[str, Any]] = {}
    anchor_cols: set[str] = set()
    effective_scope_keys: dict[str, str] = {}
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
                # The anchor owns the actual snapshot, time window and
                # instrument filter. A legacy SourceRef's dialect-version
                # digest is parser identity, not a distinct physical scan.
                # Bind same-dataset/same-market columns to the exact anchor
                # group so they receive that group's evidenced ScanCost.
                effective_scope_keys[name] = anchor_scope.key()
            else:
                group_by_scope.setdefault(binding.source_scope, set()).add(
                    binding.field
                )
                filters = _binding_semantic_filters(binding)
                prior = filters_by_scope.setdefault(binding.source_scope, filters)
                if prior != filters:
                    raise ValueError(
                        "SourceRef scope digest collision with different semantic filters"
                    )
                effective_scope_keys[name] = binding.source_scope.key()
        else:
            if _looks_like_source_ref(name):
                unbound_sref += 1
            else:
                # A plain (unbound) column is read from the anchor source, so it
                # carries the anchor group's evidenced ScanCost.  The bound
                # branch above already records this; without the same mapping
                # here the per-column cost map stays empty for every batch made
                # only of bare columns, and the physical readiness gate then
                # sees no row-count evidence and refuses the plan with
                # "row-count estimate unavailable".
                effective_scope_keys[name] = anchor_scope.key()
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
        adapter = resolver.resolve_source(
            scope,
            semantic_filters=filters_by_scope.get(scope),
        )
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
                if cost is None:
                    unavail = ScanCostUnavailable(
                        dataset=scope.dataset,
                        reason="estimate_scan_cost returned None",
                        fallback_estimate=0,
                        confidence="none",
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
    costs_by_scope = {
        group.source_scope_key: group.scan_cost
        for group in requests
        if group.scan_cost is not None
    }
    column_scope_keys = dict(effective_scope_keys)
    return BatchDataRequest(
        fields=ordered,
        anchor_source_scope=anchor_scope,
        groups=requests,
        degraded_planning=degraded,
        hard_gate_counters={
            "SOURCE_REF_WITHOUT_TYPED_SOURCE_BINDING": unbound_sref,
        },
        column_source_scope_keys=column_scope_keys,
        column_scan_costs={
            name: costs_by_scope[scope_key]
            for name, scope_key in column_scope_keys.items()
            if scope_key in costs_by_scope
        },
    )
