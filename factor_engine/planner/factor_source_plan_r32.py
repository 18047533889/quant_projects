# -*- coding: utf-8 -*-
"""R32-P0-087/089: FactorSourcePlan typed bindings + fail-closed dependency extraction.

R32-P0-087
    FactorSourcePlan 必须保存 Concept/Column→Dataset typed binding，不只是分离的
    concepts/datasets 元组。新增 ``column_bindings: tuple[ColumnSourceBinding, ...]``
    字段，向后兼容保留 ``leaf_concepts`` / ``source_datasets``。

R32-P0-089
    Dependency extraction production fail-closed。``_build_manifest()`` 失败时：
    - production / automated_research → raise DependencyExtractionError
    - interactive → 返回 None（显式降级）
    - 不再静默 ``except Exception: return None``
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Mapping

__all__ = [
    "FactorSourcePlan",
    "stable_digest",
    "DependencyExtractionError",
]


class DependencyExtractionError(Exception):
    """R32-P0-089: dependency extraction 失败（production/automated_research fail-closed）。"""
    pass


def _json_scalar(value: Any) -> Any:
    """把不可 JSON 序列化的值折叠成可序列化标量/列表。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_scalar(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_scalar(v) for k, v in value.items()}
    # R32-P0-087: ColumnSourceBinding 有 to_dict()
    to_dict_fn = getattr(value, "to_dict", None)
    if callable(to_dict_fn):
        try:
            return to_dict_fn()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _normalize_dict(d: Mapping[str, Any]) -> dict[str, Any]:
    """dict 折叠为排序键、JSON 安全的值视图（identity/digest 用）。"""
    return {
        str(k): _json_scalar(v)
        for k, v in sorted((d or {}).items(), key=lambda kv: str(kv[0]))
    }


def _normalize_grain(grain: Any) -> Any:
    """grain 归一化为可哈希/可序列化形态。"""
    if grain is None:
        return None
    if isinstance(grain, (tuple, list, set, frozenset)):
        return tuple(str(g) for g in grain)
    return str(grain)


def stable_digest(payload: Any) -> str:
    """确定性 sha256 前缀 digest（24 hex）。不依赖对象 id / 时间。"""
    raw = json.dumps(
        _json_scalar(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _detect_runtime_mode() -> str:
    """R32-P0-089: 探测当前 runtime mode（production / automated_research / interactive）。
    
    production → 严格 fail-closed
    automated_research → fail-closed（自动挖掘不允许依赖缺失）
    interactive → 显式降级允许（用户主动探索）
    """
    mode = os.environ.get("RUNTIME_MODE", "").strip().lower()
    if mode in ("production", "prod", "strict"):
        return "production"
    if mode in ("automated_research", "automated", "auto_research", "research_auto"):
        return "automated_research"
    if mode in ("interactive", "dev", "development", "notebook"):
        return "interactive"
    # 默认 fail-closed（R32-P0-089 安全默认）
    return "production"


def _build_manifest(expression_plan: Any, *, allow_degraded: bool = False) -> tuple[str, ...] | None:
    """R32-P0-089: 调用 build_source_dependency_manifest，production fail-closed。
    
    Args:
        expression_plan: 表达式计划
        allow_degraded: 是否允许降级（interactive 模式）
        
    Returns:
        manifest tuple 或 None（仅 allow_degraded=True 时）
        
    Raises:
        DependencyExtractionError: production/automated_research 模式下提取失败
    """
    runtime_mode = _detect_runtime_mode()
    fail_closed = runtime_mode in ("production", "automated_research")
    
    try:
        from planner.source_dependencies import build_source_dependency_manifest
    except Exception as exc:
        if fail_closed and not allow_degraded:
            raise DependencyExtractionError(
                f"Cannot import build_source_dependency_manifest in {runtime_mode} mode: {exc}"
            ) from exc
        return None
    
    try:
        manifest = build_source_dependency_manifest(expression_plan)
    except Exception as exc:
        if fail_closed and not allow_degraded:
            raise DependencyExtractionError(
                f"Dependency extraction failed in {runtime_mode} mode: {exc}"
            ) from exc
        return None
    
    return tuple(manifest or ())


def _parse_manifest(
    manifest: Iterable[Any],
) -> tuple[tuple[str, ...], tuple[str, ...], list[Any]]:
    """R32-P0-087/089: 从 manifest 抽取 (concepts, datasets, raw_bindings)。
    
    raw_bindings 保存原始 manifest 项（供 ColumnSourceBinding 构造）。
    """
    concepts: set[str] = set()
    datasets: set[str] = set()
    raw_bindings: list[Any] = []
    
    for item in manifest or ():
        payload: Any = None
        if isinstance(item, str):
            try:
                payload = json.loads(item)
            except Exception:  # noqa: BLE001
                continue
        elif isinstance(item, Mapping):
            payload = item
        else:
            payload = getattr(item, "to_dict", lambda: None)()
            if payload is None and item is not None:
                payload = {
                    "table": getattr(item, "table", None),
                    "field": getattr(item, "field", None),
                }
        
        if not isinstance(payload, Mapping):
            continue
        
        table = payload.get("table") or payload.get("dataset")
        field = payload.get("field")
        
        if table:
            datasets.add(str(table))
        if field:
            concepts.add(str(field))
        
        raw_bindings.append(payload)
    
    return tuple(sorted(concepts)), tuple(sorted(datasets)), raw_bindings


def _build_column_bindings(raw_bindings: list[Any], *, market: str = "") -> tuple[Any, ...]:
    """R32-P0-087: 从 manifest raw bindings 构造 typed ColumnSourceBinding。
    
    防御性 import ColumnSourceBinding / SourceScopeId；不可得时返回空 tuple。
    """
    try:
        from planner.source_binding import ColumnSourceBinding
        from planner.physical_factor_dag import SourceScopeId
    except Exception:  # pragma: no cover
        return ()
    
    bindings: list[ColumnSourceBinding] = []
    for item in raw_bindings:
        if not isinstance(item, Mapping):
            continue
        
        dataset = str(item.get("table") or item.get("dataset") or "")
        field = str(item.get("field") or "")
        if not dataset or not field:
            continue
        
        # 构造 SourceScopeId
        scope = SourceScopeId(dataset=dataset, market=market)
        
        # 构造 ColumnSourceBinding
        bindings.append(ColumnSourceBinding(
            encoded_column=field,  # manifest 里的 field 作为列名
            dataset=dataset,
            field=field,
            market=market,
            source_scope=scope,
        ))
    
    return tuple(bindings)


class FactorSourcePlan:
    """R32-P0-087/089: 单个因子的 source 依赖 IR（typed bindings + fail-closed）。
    
    R32-P0-087 新增：
        - ``column_bindings``: typed ColumnSourceBinding 元组
        - ``to_dict()`` 序列化 bindings
        
    向后兼容：
        - ``leaf_concepts`` / ``source_datasets`` 保留（旧调用方仍可读）
    """

    def __init__(
        self,
        factor_id: str,
        market: str,
        leaf_concepts: Iterable[str] = (),
        source_datasets: Iterable[str] = (),
        column_bindings: Iterable[Any] = (),  # R32-P0-087
        required_frequency: str | None = None,
        required_grain: Any = None,
        pit_requirements: Mapping[str, Any] | None = None,
        price_basis: str | None = None,
        aggregations: Iterable[Any] = (),
        joins: Iterable[Any] = (),
        coverage_requirements: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self.factor_id = str(factor_id)
        self.market = str(market or "")
        self.leaf_concepts = tuple(sorted(str(c) for c in (leaf_concepts or ())))
        self.source_datasets = tuple(sorted(str(d) for d in (source_datasets or ())))
        # R32-P0-087: typed bindings
        self.column_bindings = tuple(column_bindings or ())
        self.required_frequency = str(required_frequency) if required_frequency else None
        self.required_grain = _normalize_grain(required_grain)
        self.pit_requirements = dict(pit_requirements or {})
        self.price_basis = str(price_basis) if price_basis else None
        self.aggregations = tuple(aggregations or ())
        self.joins = tuple(joins or ())
        self.coverage_requirements = dict(coverage_requirements or {})
        self._extra = dict(extra or {})

    def to_dict(self) -> dict[str, Any]:
        """R32-P0-087: 序列化包含 column_bindings。"""
        return {
            "factor_id": self.factor_id,
            "market": self.market,
            "leaf_concepts": list(self.leaf_concepts),
            "source_datasets": list(self.source_datasets),
            # R32-P0-087: 序列化 typed bindings
            "column_bindings": [_json_scalar(b) for b in self.column_bindings],
            "required_frequency": self.required_frequency,
            "required_grain": _json_scalar(self.required_grain),
            "pit_requirements": _normalize_dict(self.pit_requirements),
            "price_basis": self.price_basis,
            "aggregations": _json_scalar(list(self.aggregations)),
            "joins": _json_scalar(list(self.joins)),
            "coverage_requirements": _normalize_dict(self.coverage_requirements),
            "extra": _normalize_dict(self._extra),
        }

    def identity(self) -> str:
        """稳定 digest：同一 factor 编译多次 → 同一 digest。"""
        return stable_digest(self.to_dict())

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FactorSourcePlan(factor_id={self.factor_id!r}, market={self.market!r}, "
            f"bindings={len(self.column_bindings)}, identity={self.identity()[:8]}…)"
        )

    @classmethod
    def extract(
        cls,
        factor_id: str,
        market: str,
        expression_plan: Any = None,
        source_manifest: Iterable[Any] | None = None,
        allow_degraded: bool = False,  # R32-P0-089
        **overrides: Any,
    ) -> "FactorSourcePlan":
        """R32-P0-087/089: 从 manifest 提取依赖并构造 typed bindings。
        
        Args:
            factor_id: 因子 ID
            market: 市场
            expression_plan: 表达式计划（用于 build_source_dependency_manifest）
            source_manifest: 显式 manifest（优先使用）
            allow_degraded: 是否允许降级（interactive 模式）
            **overrides: 覆盖字段
            
        Raises:
            DependencyExtractionError: production/automated_research 模式下提取失败
        """
        leaf_concepts = tuple(str(c) for c in (overrides.pop("leaf_concepts", ()) or ()))
        source_datasets = tuple(str(d) for d in (overrides.pop("source_datasets", ()) or ()))
        column_bindings = tuple(overrides.pop("column_bindings", ()) or ())
        
        if source_manifest is None and expression_plan is not None:
            # R32-P0-089: fail-closed dependency extraction
            source_manifest = _build_manifest(expression_plan, allow_degraded=allow_degraded)
        
        if source_manifest is not None:
            concepts, datasets, raw_bindings = _parse_manifest(source_manifest)
            if not leaf_concepts:
                leaf_concepts = concepts
            if not source_datasets:
                source_datasets = datasets
            # R32-P0-087: 构造 typed bindings
            if not column_bindings:
                column_bindings = _build_column_bindings(raw_bindings, market=market)
        
        return cls(
            factor_id=factor_id,
            market=market,
            leaf_concepts=leaf_concepts,
            source_datasets=source_datasets,
            column_bindings=column_bindings,
            **overrides,
        )
