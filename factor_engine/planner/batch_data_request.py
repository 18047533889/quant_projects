# -*- coding: utf-8 -*-
"""R31-P0-025/026: BatchDataRequest —— 一批因子合并成一次 DataAccess 规划。

目标（R31 §29/30/31）
    - 1000 因子 → union source dependencies → group by source scope → one/few
      DataRequests → one scan per physical source wave。**不是**每个 factor/column
      驱动数据读取。
    - ``estimate_scan_cost`` 每 source scope **一次**（selected_bytes /
      projection_bytes / estimated_rows / files / remote），喂给 read wave /
      IO token / memory admission / backend selection（R31-P0-026）。
    - 输出 ``scan_cost_map``（source_scope -> ScanCost），scheduler 的
      ``build_waves_from_dag`` 直接消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class SourceScanGroup:
    """同一 source scope 的合并扫描请求。"""

    group_id: int
    dataset: str
    source_scope: str
    snapshot_id: str
    fields: tuple[str, ...]
    scan_cost: Any | None = None  # DataAccess ScanCost（不可用时 None）

    def to_dict(self) -> dict[str, Any]:
        cost = self.scan_cost
        return {
            "group_id": self.group_id,
            "dataset": self.dataset,
            "source_scope": self.source_scope,
            "snapshot_id": self.snapshot_id,
            "fields": list(self.fields),
            "n_fields": len(self.fields),
            "scan_cost": {
                "selected_bytes": int(getattr(cost, "selected_bytes", 0) or 0),
                "projection_bytes": int(getattr(cost, "projection_bytes", 0) or 0),
                "estimated_rows": int(getattr(cost, "estimated_rows", 0) or 0),
                "files": int(getattr(cost, "files", 0) or 0),
                "remote": bool(getattr(cost, "remote", False)),
            }
            if cost is not None
            else None,
        }


@dataclass
class BatchDataRequest:
    """整批因子合并后的 DataAccess 数据需求（R31 §29）。"""

    fields: tuple[str, ...] = ()
    groups: list[SourceScanGroup] = field(default_factory=list)

    @property
    def scan_cost_map(self) -> dict[str, Any]:
        """``source_scope -> ScanCost``（供 read wave / IO token / admission）。"""
        return {g.source_scope: g.scan_cost for g in self.groups if g.scan_cost is not None}

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
        }


def _extract_dataset(source_scope: str) -> str:
    if "::" in source_scope:
        return source_scope.split("::", 1)[0]
    if source_scope.startswith("dataset:"):
        return source_scope[len("dataset:") :]
    return source_scope


def _source_scope_for(
    source: Any,
    *,
    dataset: str | None = None,
    snapshot_id: str | None = None,
    market: str = "",
) -> str:
    """构造 source scope 身份（dataset + snapshot + market）。"""
    parts = []
    if dataset:
        parts.append(f"dataset:{dataset}")
    if snapshot_id:
        parts.append(f"snapshot:{snapshot_id}")
    if market:
        parts.append(f"market:{market}")
    return "::".join(parts)


def build_batch_data_request(
    engine_or_source: Any,
    *,
    analyses: dict[str, Any] | None = None,
    dag: Any | None = None,
    fields: Iterable[str] | None = None,
    ctx: Any | None = None,
) -> BatchDataRequest:
    """合并整批因子依赖，按 source scope 生成一次 DataRequest 规划。

    - 字段集合 = 所有 factor analysis 的 ``referenced_columns`` 并集
      （R31 §30：union source dependencies → group by source scope）。
    - 每 source scope 调用一次 ``estimate_scan_cost``（R31-P0-026/031）。
    - 返回 ``BatchDataRequest``（含 ``scan_cost_map``）。
    """
    source = getattr(engine_or_source, "data_source", None) or engine_or_source
    # 1) 字段并集（不逐字段 resolve；R31-P0-027 批量一次）。
    all_fields: set[str] = set()
    for a in (analyses or {}).values():
        refs = getattr(a, "referenced_columns", None) or ()
        all_fields |= set(refs)
    if fields:
        all_fields |= set(fields)
    if dag is not None:
        for fp in getattr(dag, "roots", ()) or ():
            root = getattr(fp, "root", fp)
            _walk_columns(root, all_fields)
        for sub in (getattr(dag, "shared_nodes", None) or {}).values():
            _walk_columns(sub, all_fields)
    ordered = tuple(sorted(all_fields))

    dataset = getattr(source, "dataset", None)
    market = str(getattr(ctx, "market", "") or "")
    snapshot_id = None
    for key in ("_manifest_token", "_data_snapshot_id", "snapshot_id"):
        value = getattr(source, key, None)
        if value:
            snapshot_id = str(value)
            break
    source_scope = _source_scope_for(
        source, dataset=dataset or "", snapshot_id=snapshot_id, market=market
    )
    # 2) 每 source scope 一次 ScanCost。
    scan_cost = None
    estimator = getattr(source, "estimate_scan_cost", None)
    if callable(estimator):
        try:
            scan_cost = estimator(fields=ordered)
        except Exception:
            scan_cost = None
    group = SourceScanGroup(
        group_id=0,
        dataset=_extract_dataset(source_scope),
        source_scope=source_scope,
        snapshot_id=snapshot_id or "",
        fields=ordered,
        scan_cost=scan_cost,
    )
    return BatchDataRequest(fields=ordered, groups=[group])


def _walk_columns(plan: Any, out: set[str]) -> None:
    op = str(getattr(plan, "op", "") or "")
    if op == "column":
        name = str((getattr(plan, "attrs", None) or {}).get("name") or "")
        if name:
            out.add(name)
    for child in getattr(plan, "inputs", ()) or ():
        _walk_columns(child, out)
