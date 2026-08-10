# -*- coding: utf-8 -*-
"""R31-P0-025/026 + R33-P0-001..006: BatchDataRequest —— 一批因子合并成一次
DataAccess 规划。

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from planner.physical_factor_dag import SourceScopeId


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
    """

    group_id: int
    dataset: str
    source_scope: SourceScopeId
    snapshot_id: str
    fields: tuple[str, ...]
    time_range: tuple[str, str] | None = None
    instrument_scope: tuple[str, ...] | None = None
    scan_cost: Any | None = None
    scan_cost_unavailable: ScanCostUnavailable | None = None

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
            "time_range": list(self.time_range) if self.time_range else None,
            "scan_cost": _scan_cost_dict(self.scan_cost) if self.scan_cost is not None else None,
            "scan_cost_unavailable": (
                self.scan_cost_unavailable.to_dict()
                if self.scan_cost_unavailable is not None
                else None
            ),
        }


@dataclass
class BatchDataRequest:
    """整批因子合并后的 DataAccess 数据需求（R31 §29 / R33 multi-source）。"""

    fields: tuple[str, ...] = ()
    groups: list[SourceScanGroup] = field(default_factory=list)
    degraded_planning: list[ScanCostUnavailable] = field(default_factory=list)

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
        }


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


def _source_ref_datasets(plans: Iterable[Any]) -> dict[str, str]:
    """扫描计划中的 SourceRef 列，返回 ``{ref_dataset: source_scope_key}``。

    R33-P0-001：secondary SourceRef（fundamental / industry / universe / minute）
    成为独立 source group。提取失败不致命——仅无法识别时跳过（已有 anchor 源
    兜底），source 身份解析仍走 typed 字段。
    """
    out: dict[str, str] = {}
    try:
        from api.source_ref import decode_source_ref, looks_like_source_ref
    except Exception:
        return out

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op == "column":
            name = str((getattr(node, "attrs", None) or {}).get("name") or "")
            if name and looks_like_source_ref(name):
                try:
                    ref = decode_source_ref(name)
                except Exception:
                    ref = None
                if ref is not None and getattr(ref, "dataset", None):
                    out[str(ref.dataset)] = str(ref.market or "")
        for child in getattr(node, "inputs", ()) or ():
            walk(child)

    for plan in plans:
        walk(plan)
    return out


def _walk_columns(plan: Any, out: set[str]) -> None:
    op = str(getattr(plan, "op", "") or "")
    if op == "column":
        name = str((getattr(plan, "attrs", None) or {}).get("name") or "")
        if name:
            out.add(name)
    for child in getattr(plan, "inputs", ()) or ():
        _walk_columns(child, out)


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
    - 每 source scope 一次 ``estimate_scan_cost``，传真实 ``time_range`` +
      ``instrument_scope``（R33-P0-004）。
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

    # 2) secondary SourceRef datasets（R33-P0-001）。
    ref_datasets = _source_ref_datasets(plans)
    anchor_dataset = anchor_scope.dataset
    group_specs: list[tuple[SourceScopeId, set[str]]] = []
    seen: set[str] = set()
    # anchor group 总是存在（兜底）。
    anchor_cols: set[str] = set()
    for name in ordered:
        ref_ds = _source_ref_dataset_for(name, ref_datasets)
        if ref_ds is not None and ref_ds != anchor_dataset:
            if ref_ds not in seen:
                seen.add(ref_ds)
                group_specs.append((SourceScopeId(dataset=ref_ds, market=market), set()))
            group_specs[-1][1].add(name)
        else:
            anchor_cols.add(name)
    if anchor_dataset or anchor_cols:
        group_specs.insert(0, (anchor_scope, anchor_cols))

    # 3) 每 source scope 一次 ScanCost（真实 time_range + instruments）。
    requests: list[SourceScanGroup] = []
    degraded: list[ScanCostUnavailable] = []
    estimator = getattr(source, "estimate_scan_cost", None)
    time_range = _source_time_range(source)
    instrument_scope = tuple(
        getattr(source, "instrument_filter", None) or ()
    ) or None
    for gid, (scope, cols) in enumerate(group_specs):
        cost = None
        unavail = None
        if callable(estimator):
            try:
                cost = estimator(
                    fields=tuple(sorted(cols)) or ordered,
                    time_range=time_range,
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
            unavail = ScanCostUnavailable(
                dataset=scope.dataset,
                reason="no estimate_scan_cost on source",
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
                fields=tuple(sorted(cols)) or ordered,
                time_range=(
                    (str(time_range[0]), str(time_range[1]))
                    if time_range is not None and time_range[0] is not None
                    else None
                ),
                instrument_scope=instrument_scope,
                scan_cost=cost,
                scan_cost_unavailable=unavail,
            )
        )
    return BatchDataRequest(fields=ordered, groups=requests, degraded_planning=degraded)


def _source_ref_dataset_for(name: str, ref_datasets: dict[str, str]) -> str | None:
    """单个列名是否命中 SourceRef 数据集（精确：ref key 是完整 column 名）。"""
    for ref_key, ds in ref_datasets.items():
        if name == ref_key:
            return ds
    return None


def _source_time_range(source: Any) -> tuple[Any, Any] | None:
    start = getattr(source, "start_date", None)
    end = getattr(source, "end_date", None)
    if start is None and end is None:
        return None
    return (start, end)
