# -*- coding: utf-8 -*-
"""数据更新事件 → 受影响因子 → 最小增量重算窗口。

R10 #51: the scheduler must NOT reuse one engine for all affected factors.
Each factor is rebuilt from the catalog's FULL spec (data sources, filters,
composite config, universe) and materialized on a fresh ``FactorEngine`` whose
data-source config matches the catalog — so two factors with different source
configs never share an engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cleaned_operators.operator_policy import effective_lookback
from runtime.incremental import build_incremental_plan
from storage.catalog import FactorCatalog


@dataclass(frozen=True)
class DataEvent:
    """源数据字段更新事件 (R10 #53 extended).

    Keeps the original ``(dataset, column, updated_date)`` positional
    constructor working; the R10 fields (``field_id``, revision window,
    snapshot before/after, ``revision_kind``, ``deleted_keys``) are optional and
    default to ``None``.
    """

    dataset: str
    column: str
    updated_date: str
    field_id: str | None = None
    affected_start: str | None = None
    affected_end: str | None = None
    snapshot_before: Any = None
    snapshot_after: Any = None
    revision_kind: str | None = None
    deleted_keys: tuple[str, ...] | None = None

    @property
    def field(self) -> str:
        """Canonical field the event touches (``field_id`` falls back to ``column``)."""
        return self.field_id or self.column

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "column": self.column,
            "updated_date": self.updated_date,
            "field_id": self.field_id or self.column,
            "affected_start": self.affected_start,
            "affected_end": self.affected_end,
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after,
            "revision_kind": self.revision_kind,
            "deleted_keys": list(self.deleted_keys) if self.deleted_keys else None,
        }


def normalize_data_event(event: DataEvent | dict[str, Any]) -> DataEvent:
    """Coerce a dict (or ``DataEvent``) to a :class:`DataEvent` (R10 #53)."""
    if isinstance(event, DataEvent):
        return event
    raw = dict(event)
    return DataEvent(
        dataset=str(raw["dataset"]),
        column=str(raw["column"]),
        updated_date=str(raw["updated_date"]),
        field_id=str(raw["field_id"]) if raw.get("field_id") else None,
        affected_start=str(raw["affected_start"]) if raw.get("affected_start") else None,
        affected_end=str(raw["affected_end"]) if raw.get("affected_end") else None,
        snapshot_before=raw.get("snapshot_before"),
        snapshot_after=raw.get("snapshot_after"),
        revision_kind=str(raw["revision_kind"]) if raw.get("revision_kind") else None,
        deleted_keys=tuple(raw["deleted_keys"]) if raw.get("deleted_keys") else None,
    )


@dataclass(frozen=True)
class FactorUpdatePlan:
    """单因子增量重算计划。"""

    factor_id: str
    column: str
    lookback_bars: int
    since: str | None
    end_date: str | None
    load_start: str | None
    load_end: str | None
    source_dataset: str | None
    is_full_run: bool
    field_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        return {
            "factor_id": self.factor_id,
            "column": self.column,
            "field_id": self.field_id or self.column,
            "lookback_bars": self.lookback_bars,
            "since": self.since,
            "end_date": self.end_date,
            "load_start": self.load_start,
            "load_end": self.load_end,
            "source_dataset": self.source_dataset,
            "is_full_run": self.is_full_run,
        }


class FactorSemanticIdentityMismatch(ValueError):
    """Rebuilt factor / data source does not match the catalog definition.

    ``runtime/factor_identity.py`` does not exist yet (R10); the check uses the
    identity fields available today (``source_expr`` vs catalog ``expression``,
    best-effort IR hash vs catalog ``ast_hash``).
    """


def _resolve_source_dataset(data_source: Any) -> str | None:
    return getattr(data_source, "dataset", None)


def _dependency_rows_for_event(
    catalog: Any,
    event: DataEvent,
) -> list[dict[str, Any]]:
    """Dependency rows for an event — edges first, legacy column index fallback.

    ``catalog`` may be a ``runtime.dependency_catalog.DependencyCatalog`` (R10
    edge-aware) or a raw ``storage.catalog.FactorCatalog`` (legacy rows).
    """
    if hasattr(catalog, "factors_for_event"):
        return catalog.factors_for_event(event)
    return catalog.list_factors_for_column(event.field_id or event.column)


def plan_updates_from_data_event(
    catalog: Any,
    event: DataEvent,
    *,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
) -> list[FactorUpdatePlan]:
    """``column``/``field_id`` 更新后，查询 catalog 中受影响因子并计算 lookback 窗口。"""
    deps = _dependency_rows_for_event(catalog, event)
    plans: list[FactorUpdatePlan] = []
    for dep in deps:
        source_dataset = dep.get("source_dataset")
        if source_dataset and str(source_dataset) != str(event.dataset):
            continue
        factor_id = str(dep["factor_id"])
        lookback = int(dep.get("lookback") or 0)
        inc = build_incremental_plan(
            factor_id=factor_id,
            analysis_lookback=lookback,
            watermark={"end_date": event.updated_date},
            since=event.updated_date,
            end_date=end_date,
            lookback_extra=lookback_extra,
            market=market,
            factor_freq=dep.get("frequency"),
        )
        plans.append(
            FactorUpdatePlan(
                factor_id=factor_id,
                column=event.column,
                field_id=event.field_id or event.column,
                lookback_bars=int(inc.lookback_bars),
                since=event.updated_date,
                end_date=end_date,
                load_start=None if inc.load_start is None else inc.load_start.isoformat(),
                load_end=None if inc.load_end is None else inc.load_end.isoformat(),
                source_dataset=source_dataset,
                is_full_run=bool(inc.is_full_run),
            )
        )
    return plans


def _edges_from_analysis(
    factor_id: str,
    analysis: Any,
    data_source: Any,
) -> list[Any]:
    """Build rich dependency edges from ``analysis.referenced_fields`` (R10 #50)."""
    from runtime.dependency_catalog import FactorDependencyEdge

    source_dataset = _resolve_source_dataset(data_source)
    ref_fields = getattr(analysis, "referenced_fields", {}) or {}
    edges: list[FactorDependencyEdge] = []
    for name, spec in ref_fields.items():
        table = getattr(spec, "table", None) or source_dataset
        ds = source_dataset or table
        if ds is None:
            continue
        field_id = str(getattr(spec, "field_id", None) or name)
        edges.append(
            FactorDependencyEdge(
                factor_id=factor_id,
                source_dataset=str(ds),
                field_id=field_id,
                physical_field=str(
                    getattr(spec, "source_name", None) or field_id
                ),
                transform=None,
                snapshot_semantics="point",
                lookback=0,
            )
        )
    return edges


def record_factor_dependency_from_analysis(
    catalog: FactorCatalog,
    *,
    factor_id: str,
    analysis: Any,
    data_source: Any,
    frequency: str | None = None,
) -> None:
    """从 ``AnalysisResult`` 写入依赖 catalog（legacy 行 + R10 edges）。"""
    from runtime.dependency_catalog import DependencyCatalog

    referenced_columns = getattr(analysis, "referenced_columns", set()) or set()
    lookback = int(getattr(analysis, "lookback", 0))
    source_dataset = _resolve_source_dataset(data_source)
    catalog.record_factor_dependency(
        factor_id,
        referenced_columns=referenced_columns,
        lookback=lookback,
        frequency=frequency,
        source_dataset=source_dataset,
    )
    edges = _edges_from_analysis(factor_id, analysis, data_source)
    if edges:
        dep_catalog = (
            catalog if isinstance(catalog, DependencyCatalog) else DependencyCatalog(catalog)
        )
        dep_catalog.record_factor_edges(
            factor_id,
            edges=edges,
            lookback=lookback,
            frequency=frequency,
            source_dataset=source_dataset,
        )


def factor_from_catalog_info(info: dict) -> Any:
    """catalog 行 → ``Factor``（需含 ``expression``；R10 支持 surface/dialect/universe）。"""
    from api.dsl_parser import parse_factor

    expression = info.get("expression")
    if not expression or not str(expression).strip():
        raise ValueError(f"因子 {info.get('factor_id')!r} 缺少 expression，无法重建")
    factor_id = str(info.get("factor_id") or info.get("name") or "unknown")
    return parse_factor(
        str(expression),
        name=factor_id,
        freq=str(info.get("frequency") or "1d"),
        universe=info.get("universe"),
        description=info.get("description"),
        surface=str(info.get("surface") or "daily"),
        dialect=str(info.get("dialect") or "native"),
        dialect_version=info.get("dialect_version"),
    )


# ---------------------------------------------------------------------------
# R10 #51: per-factor engine reconstruction
# ---------------------------------------------------------------------------

def _default_engine_factory(
    full_def: dict[str, Any],
    *,
    lake_root: str | Path | None = None,
    market: str | None = None,
    run_mode: str | None = None,
) -> Any:
    """Build a fresh ``FactorEngine`` for one factor's full definition.

    The data source is reconstructed from the catalog's ``data_source_config``,
    so two factors with different source configs are guaranteed to run on their
    own engine.  Overridable via ``engine_factory=`` for tests / alternate
    backends.
    """
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from storage.factory import DataSourceBuildContext, build_data_source

    ds_config = dict(full_def.get("data_source_config") or {})
    if not ds_config:
        raise ValueError(
            f"因子 {full_def.get('factor_id')!r} 缺少 data_source_config，无法重建引擎"
        )
    if "type" not in ds_config:
        ds_config["type"] = "parquet"
    mode = run_mode or full_def.get("run_mode")
    build_context = DataSourceBuildContext(
        run_mode=mode,
        market=market or full_def.get("market"),
        calendar_id=full_def.get("calendar"),
        pit_enforce=bool(full_def.get("pit_enforce")),
    )
    data_source = build_data_source(ds_config, build_context=build_context)
    backend = build_backend(str(full_def.get("backend") or "pandas"))
    return FactorEngine(backend=backend, data_source=data_source, run_mode=mode)


def _build_engine_for_factor(
    full_def: dict[str, Any],
    *,
    lake_root: str | Path | None = None,
    market: str | None = None,
    engine_factory: Any | None = None,
) -> Any:
    factory = engine_factory or _default_engine_factory
    return factory(
        full_def,
        lake_root=lake_root,
        market=market,
        run_mode=full_def.get("run_mode"),
    )


def _verify_factor_semantic_identity(factor: Any, full_def: dict[str, Any]) -> None:
    """Best-effort check that a rebuilt factor matches the catalog identity.

    Uses the identity fields available today (no ``factor_identity.py`` yet):
    ``source_expr`` vs catalog ``expression``, plus a best-effort IR structural
    hash vs catalog ``ast_hash``.
    """
    catalog_expr = full_def.get("expression")
    if catalog_expr and str(catalog_expr).strip():
        actual_expr = str(getattr(factor, "source_expr", "") or "").strip()
        if actual_expr and actual_expr != str(catalog_expr).strip():
            raise FactorSemanticIdentityMismatch(
                f"因子 {full_def.get('factor_id')!r} 重建表达式与 catalog 不一致: "
                f"{actual_expr!r} != {catalog_expr!r}"
            )
    ast_hash = full_def.get("ast_hash")
    if ast_hash and str(ast_hash).strip():
        from storage.catalog import compute_ir_hash

        try:
            actual_hash = compute_ir_hash(factor.expr)
        except Exception:  # Expr is not always a serializable IR node
            actual_hash = None
        if (
            actual_hash is not None
            and str(actual_hash) != str(ast_hash).strip()
        ):
            raise FactorSemanticIdentityMismatch(
                f"因子 {full_def.get('factor_id')!r} IR hash 与 catalog 不一致: "
                f"{str(actual_hash)[:12]} != {str(ast_hash)[:12]}"
            )


def execute_incremental_updates_from_event(
    engine: Any,
    event: DataEvent | dict[str, Any],
    *,
    lake_root: str | None = None,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
    dry_run: bool = False,
    materialize_kwargs: dict[str, Any] | None = None,
    engine_factory: Any | None = None,
) -> dict[str, Any]:
    """DataEvent → 计划 → 逐因子物化（每个因子使用自己的重建引擎，R10 #51）。"""
    from storage.materializer import ParquetMaterializer
    from runtime.dependency_catalog import DependencyCatalog

    event = normalize_data_event(event)
    catalog = ParquetMaterializer(lake_root=lake_root).catalog
    dep_catalog = DependencyCatalog(catalog)
    plans = plan_updates_from_data_event(
        dep_catalog,
        event,
        end_date=end_date,
        lookback_extra=lookback_extra,
        market=market,
    )
    out: dict[str, Any] = {
        "event": event.to_dict(),
        "plans": [p.to_dict() for p in plans],
        "factor_count": len(plans),
        "dry_run": dry_run,
        "materializations": {},
    }
    if dry_run or not plans:
        return out

    mat_kwargs = dict(materialize_kwargs or {})
    mat_kwargs.setdefault("lake_root", lake_root)
    mat_kwargs.setdefault("lookback_extra", lookback_extra)
    mat_kwargs.setdefault("market", market)

    failures: list[dict[str, str]] = []
    for plan in plans:
        info = dep_catalog.full_factor_definition(plan.factor_id)
        if info is None:
            failures.append(
                {
                    "factor_id": plan.factor_id,
                    "error": "catalog 中无 factor_registry 记录",
                }
            )
            continue
        try:
            factor = factor_from_catalog_info({**info, "factor_id": plan.factor_id})
        except ValueError as exc:
            failures.append({"factor_id": plan.factor_id, "error": str(exc)})
            continue
        try:
            # R10 #51: rebuild a dedicated engine from the catalog's full spec.
            ds_config = info.get("data_source_config") or {}
            if ds_config:
                target_engine = _build_engine_for_factor(
                    info,
                    lake_root=lake_root,
                    market=market,
                    engine_factory=engine_factory,
                )
                _verify_factor_semantic_identity(factor, info)
            else:
                # Legacy rows carry no source config — fall back to the caller's
                # engine so existing catalog content keeps materializing.
                target_engine = engine
            result = target_engine.materialize_incremental(
                factor,
                factor_id=plan.factor_id,
                since=plan.since,
                end_date=plan.end_date or end_date,
                expression=info.get("expression"),
                frequency=info.get("frequency"),
                description=info.get("description"),
                **mat_kwargs,
            )
            out["materializations"][plan.factor_id] = result.get(
                "materialization", result
            )
        except Exception as exc:
            failures.append({"factor_id": plan.factor_id, "error": str(exc)})

    if failures:
        out["failures"] = failures
    out["succeeded"] = len(out["materializations"])
    out["failed"] = len(failures)
    return out
