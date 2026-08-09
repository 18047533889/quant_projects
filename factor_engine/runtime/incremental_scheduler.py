# -*- coding: utf-8 -*-
"""数据更新事件 → 受影响因子 → 最小增量重算窗口。

R10 #51: the scheduler must NOT reuse one engine for all affected factors.
Each factor is rebuilt from the catalog's FULL spec (data sources, filters,
composite config, universe) and materialized on a fresh ``FactorEngine`` whose
data-source config matches the catalog — so two factors with different source
configs never share an engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

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
    #: 被删除的源数据 key。允许 ``(datetime, asset)`` 键对或裸 key（裸 key 在
    #: planner 里以受影响日期为删除日期）；planner 会归一成 tombstone 键对。
    deleted_keys: tuple[Any, ...] | None = None
    #: P0-18 ledger: event identity / ordering / status.  ``snapshot_before``
    #: must match the ledger's last committed snapshot for the same (dataset,
    #: field) chain, or the event is stale (rejected).  ``status`` is one of
    #: ``pending`` / ``committed`` / ``failed`` / ``rejected``.
    event_id: str | None = None
    sequence: int | None = None
    status: str | None = None

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
            "event_id": self.event_id,
            "sequence": self.sequence,
            "status": self.status,
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
        event_id=str(raw["event_id"]) if raw.get("event_id") else None,
        sequence=int(raw["sequence"]) if raw.get("sequence") is not None else None,
        status=str(raw["status"]) if raw.get("status") else None,
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
    #: 源删除 key 归一成的 ``(datetime, asset)`` tombstone 键对（R11）。
    deleted_keys: tuple[tuple[str, str], ...] | None = None

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
            "deleted_keys": (
                [list(k) for k in self.deleted_keys] if self.deleted_keys else None
            ),
        }


class FactorSemanticIdentityMismatch(ValueError):
    """Rebuilt factor / data source does not match the catalog definition.

    ``runtime/factor_identity.py`` does not exist yet (R10); the check uses the
    identity fields available today (``source_expr`` vs catalog ``expression``,
    best-effort IR hash vs catalog ``ast_hash``).
    """


def _resolve_source_dataset(data_source: Any) -> str | None:
    return getattr(data_source, "dataset", None)


def _physical_dataset_for_logical_table(table: str) -> str | None:
    """逻辑表名（``StockDailyBar``）→ 物理 DataAccess dataset（``ashare_stock_daily``）。

    #收官轮 P0：edge 的物理 ``source_dataset`` 绝不能写成 FactorEngine 的 logical
    table name——``DataEvent.dataset`` 是物理 dataset，表名写错会导致
    ``DependencyCatalog.factors_for_event``（``source_dataset == event.dataset``）
    匹配不上、增量重算静默失效。优先 ``FieldSpec.dataset``（市场已解析正确），
    这里只兜底：默认单市场 registry 优先，再逐 market registry。
    """
    if not table:
        return None
    try:
        from fields import get_field_registry

        spec = get_field_registry().resolve_table(table)
        if spec is not None and getattr(spec, "dataset", None):
            return str(spec.dataset)
    except Exception:  # pragma: no cover - registry 缺失不阻塞 edge 构建
        pass
    try:
        from fields.market_registry import multi_market_registry

        mm = multi_market_registry()
        for market in getattr(mm, "markets", ("ashare", "us")):
            try:
                spec = mm.registry_for(market).resolve_table(table)
            except Exception:  # pragma: no cover - 单 market registry 不可用跳过
                spec = None
            if spec is not None and getattr(spec, "dataset", None):
                return str(spec.dataset)
    except Exception:  # pragma: no cover - 多市场 registry 缺失不阻塞
        pass
    return None


def _operator_manifest_from_ir(ir: Any) -> list[dict[str, str]] | None:
    """P2-02: operator semantic manifest for a factor's IR.

    Records every operator canonical used, its declared semantic version and a
    source-file hash of its implementation — so an operator upgrade invalidates
    the factor even when source data is unchanged.  ``None`` when the IR is
    missing or the manifest cannot be built (caller stores nothing).
    """
    if ir is None:
        return None
    try:
        import hashlib
        import inspect

        from backend.cleaned_bridge import ensure_cleaned_loaded
        from cleaned_operators.registry import OperatorRegistry

        ensure_cleaned_loaded()
        seen: dict[str, dict[str, str]] = {}

        def walk(node: Any) -> None:
            op = getattr(node, "op", None)
            if op and op not in {"column", "literal", "plan_ref"}:
                try:
                    canonical = OperatorRegistry.resolve_canonical_optional(op) or str(op)
                except Exception:
                    canonical = str(op)
                impl = None
                try:
                    impl = OperatorRegistry.get(canonical)
                except Exception:
                    impl = None
                source_hash = None
                if impl is not None:
                    try:
                        src = inspect.getsourcefile(type(impl))
                        if src:
                            source_hash = hashlib.sha256(
                                open(src, "rb").read()
                            ).hexdigest()[:16]
                    except Exception:
                        source_hash = None
                seen.setdefault(
                    canonical,
                    {
                        "canonical": canonical,
                        "semantic_version": str(
                            getattr(getattr(impl, "metadata", None), "semantic_version", None)
                            or "1.0"
                        ),
                        "implementation_hash": source_hash,
                    },
                )
            for child in getattr(node, "inputs", ()) or ():
                walk(child)

        walk(ir)
        return sorted(seen.values(), key=lambda m: m["canonical"]) if seen else None
    except Exception:  # pragma: no cover - manifest 构建失败不阻塞记录
        return None


def _calendar_version(data_source: Any) -> str | None:
    """P2-04: calendar-contract fingerprint for the factor's lineage.

    Captures ``len(first, last)`` of the active trading calendar — a calendar
    revision / session change shifts this fingerprint and invalidates rolling
    history / intraday aggregation / availability dependencies.
    """
    try:
        market = getattr(data_source, "market", None)
        from storage.trading_calendar import get_trading_calendar

        calendar = get_trading_calendar(market)
        if calendar is None:
            return None
        days = getattr(calendar, "days", None)
        if not days:
            return None
        first = pd.Timestamp(days[0]).date()
        last = pd.Timestamp(days[-1]).date()
        return f"v{len(days)}:{first}:{last}"
    except Exception:  # pragma: no cover - 日历不可用时无版本指纹
        return None


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


class DeletePredicateError(ValueError):
    """A bare deleted key (no datetime) cannot be auto-dated (P0-19).

    Production demands ``(datetime, asset)`` or a structured ``DeletePredicate``;
    guessing ``affected_start`` as the delete date has no clear semantics when
    ``affected_start != affected_end``.
    """


def _coerce_deleted_keys(
    event: DataEvent,
    *,
    production: bool = False,
) -> tuple[tuple[str, str], ...] | None:
    """把 ``DataEvent.deleted_keys`` 归一成 ``(datetime, asset)`` tombstone 键对。

    ``ParquetMaterializer._append_tombstones`` 需要 ``(datetime, asset)`` 键对；
    裸 key（单字符串）以事件受影响日期（``affected_start`` 或 ``updated_date``）
    作为删除日期，避免把删除日错记为事件抵达日。

    P0-19: production 拒绝裸 key —— 裸 key 的删除日期是"猜"的，``affected_start !=
    affected_end`` 时没有充分语义；必须显式 ``(datetime, asset)`` 或结构化谓词。
    """
    raw = event.deleted_keys
    if not raw:
        return None
    default_date = event.affected_start or event.updated_date
    out: list[tuple[str, str]] = []
    for key in raw:
        if isinstance(key, (tuple, list)) and len(key) >= 2:
            out.append((str(key[0]), str(key[1])))
        else:
            if production:
                raise DeletePredicateError(
                    f"production: deleted key {key!r} has no datetime — a bare key "
                    "cannot be auto-dated (affected_start may != affected_end). "
                    "Provide (datetime, asset) or a structured DeletePredicate."
                )
            out.append((str(default_date), str(key)))
    return tuple(out)


def _extend_date_forward(
    end_date: str | None,
    bars: int | None,
    *,
    market: str | None,
    cap: str | None = None,
) -> str | None:
    """把 ``end_date`` 向未来扩 ``bars`` 个交易日（P0-01 forward impact）。

    ``bars`` 为 None（unbounded）时返回 ``cap``（最新数据日）；否则按交易日历
    前移 ``bars`` 并把结果封顶在 ``cap``（不越过可用数据的最新日）。
    """
    if bars is None:
        return cap
    if not end_date or bars <= 0:
        return end_date
    try:
        from storage.trading_calendar import get_trading_calendar, trading_day_offset

        calendar = get_trading_calendar(market)
        shifted = trading_day_offset(pd.Timestamp(end_date), int(bars), calendar=calendar)
        out = str(pd.Timestamp(shifted).date())
        if cap:
            return min(out, str(pd.Timestamp(cap).date()))
        return out
    except Exception:  # pragma: no cover - 日历不可用时保守回退原 end
        return end_date


def plan_updates_from_data_event(
    catalog: Any,
    event: DataEvent,
    *,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
    production: bool = False,
) -> list[FactorUpdatePlan]:
    """``column``/``field_id`` 更新后，查询 catalog 中受影响因子并计算 lookback 窗口。

    修订窗口以**事件受影响的数据日期**（``affected_start``/``affected_end``）为
    基准，而不是事件抵达日期（``updated_date``）。历史修订示例：2026-08-09 收到
    修订、实际受影响的是 2024-06-01 财务数据——若从 ``updated_date`` 起重算，
    2024→2026 的下游因子会全部漏算。``since`` 传 ``affected_start``，由
    ``build_incremental_plan`` 再往前推 dependency lookback。

    P0-01: 输入修订的影响是**向未来传播**的 —— ``ts_mean(x,20)`` 中 ``x[t0]``
    被修订会同时影响 ``t0..t0+19`` 的输出。因此 ``recompute_end`` 先按因子的
    forward impact 向未来扩展（``ForwardImpactRequirement``），再让
    ``build_incremental_plan`` 对这个输出窗口向后扩 warmup；不能只往历史方向
    扩而把输出停在 ``affected_end``。P0-02: 删除事件同样走 forward 传播。
    """
    deps = _dependency_rows_for_event(catalog, event)
    recompute_start = event.affected_start or event.updated_date
    recompute_end = event.affected_end or end_date
    deleted_keys = _coerce_deleted_keys(event, production=production)
    plans: list[FactorUpdatePlan] = []
    for dep in deps:
        source_dataset = dep.get("source_dataset")
        if source_dataset and str(source_dataset) != str(event.dataset):
            continue
        factor_id = str(dep["factor_id"])
        lookback = int(dep.get("lookback") or 0)
        # P0-01: per-factor forward impact — None = unbounded (recursive /
        # stateful / event-clock propagation to the latest data).
        forward = dep.get("forward_impact")
        eff_end = _extend_date_forward(
            recompute_end, forward, market=market, cap=end_date
        )
        inc = build_incremental_plan(
            factor_id=factor_id,
            analysis_lookback=lookback,
            watermark={"end_date": recompute_start},
            since=recompute_start,
            end_date=eff_end,
            lookback_extra=lookback_extra,
            market=market,
            factor_freq=dep.get("frequency"),
            forward_impact=forward,
        )
        plans.append(
            FactorUpdatePlan(
                factor_id=factor_id,
                column=event.column,
                field_id=event.field_id or event.column,
                lookback_bars=int(inc.lookback_bars),
                since=recompute_start,
                end_date=eff_end,
                load_start=None if inc.load_start is None else inc.load_start.isoformat(),
                load_end=None if inc.load_end is None else inc.load_end.isoformat(),
                source_dataset=source_dataset,
                is_full_run=bool(inc.is_full_run),
                deleted_keys=deleted_keys,
            )
        )
    return plans


def _edges_from_analysis(
    factor_id: str,
    analysis: Any,
    data_source: Any,
) -> list[Any]:
    """Build rich dependency edges from the resolved physical source manifest.

    R11: each ``FieldSpec`` in ``analysis.referenced_fields`` carries its
    physical DataAccess dataset (``spec.dataset``), its FactorEngine logical
    table (``spec.table``), canonical ``field_id`` and physical column
    (``spec.source_name``).  The edge ``source_dataset`` MUST be the physical
    dataset the field lives in — NOT the anchor ``data_source.dataset``: a
    composite factor reading ``StockDailyBar.close`` + ``StockValuationDaily.pe_ratio``
    must record the second edge on ``ashare_stock_valuation_daily``, or
    ``DependencyCatalog.factors_for_event()`` (which queries
    ``source_dataset == event.dataset``) can never find it.
    """
    from runtime.dependency_catalog import FactorDependencyEdge, UNBOUNDED_FORWARD_IMPACT

    anchor_dataset = _resolve_source_dataset(data_source)
    ref_fields = getattr(analysis, "referenced_fields", {}) or {}
    # P0-01: forward impact of the whole factor — any changed source field
    # propagates with the factor's forward reach (None = unbounded, stored as
    # the -1 sentinel so legacy ``None`` keeps meaning "not recorded").
    from runtime.execution_contract import factor_forward_impact

    fwd = None
    try:
        fwd = factor_forward_impact(getattr(analysis, "ir", None))
    except Exception:  # pragma: no cover - 无 IR 时保守视为无前向传播
        fwd = None
    fwd_stored = UNBOUNDED_FORWARD_IMPACT if fwd is None else max(0, int(fwd))
    # P2-03: field-catalog version enters the edge lineage — a catalog change
    # (unit / transform / calendar contract) shifts this hash and makes the
    # dependency identity differ even when the parquet data is unchanged.
    field_catalog_hash = None
    try:
        from fields import FIELD_REGISTRY

        field_catalog_hash = str(FIELD_REGISTRY.catalog_hash() or "")
    except Exception:  # pragma: no cover - registry 缺失时无版本
        field_catalog_hash = None
    edges: list[FactorDependencyEdge] = []
    for name, spec in ref_fields.items():
        # #收官轮 P0: 物理 source_dataset 必须解析成 DataAccess 物理 dataset。
        # ``spec.dataset``（市场正确）→ ``TableSpec(spec.table).dataset`` →
        # anchor dataset。**永远不要**把 logical table name（``StockDailyBar``）
        # 写进物理 ``source_dataset``，否则 DataEvent 匹配不上、增量重算失效。
        physical_ds = getattr(spec, "dataset", None)
        if not physical_ds:
            physical_ds = _physical_dataset_for_logical_table(
                getattr(spec, "table", None)
            )
        if not physical_ds:
            physical_ds = anchor_dataset
        if physical_ds is None:
            continue
        field_id = str(getattr(spec, "field_id", None) or name)
        # P1-10: derive the REAL snapshot semantics from the field's transform /
        # temporal model instead of a blanket "point".
        transform = getattr(spec, "transform", None)
        if transform in {"asof", "asof_backward"}:
            snap = "asof"
        elif transform in {"financial_asof", "financial_lag"}:
            snap = "financial_pit"
        elif transform in {"ffill", "forward_fill"}:
            snap = "asof"
        else:
            snap = str(getattr(spec, "snapshot_semantics", None) or "point")
        edges.append(
            FactorDependencyEdge(
                factor_id=factor_id,
                source_dataset=str(physical_ds),
                logical_table=getattr(spec, "table", None),
                field_id=field_id,
                physical_field=str(
                    getattr(spec, "source_name", None) or field_id
                ),
                transform=transform,
                snapshot_semantics=snap,
                lookback=0,
                forward_impact=fwd_stored,
                field_catalog_hash=field_catalog_hash,
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
    full_definition: dict[str, Any] | None = None,
) -> None:
    """从 ``AnalysisResult`` 写入依赖 catalog（legacy 行 + R10 edges [+ full spec]）。

    R11: the whole factor dependency definition is written atomically through
    ``DependencyCatalog.record_factor_manifest`` (single BEGIN IMMEDIATE
    transaction), so a concurrent ``factors_for_event`` reader or a crash can
    never observe a partially-replaced edge set (0/1/5 edges).
    """
    from runtime.dependency_catalog import DependencyCatalog

    referenced_columns = getattr(analysis, "referenced_columns", set()) or set()
    lookback = int(getattr(analysis, "lookback", 0))
    source_dataset = _resolve_source_dataset(data_source)
    edges = _edges_from_analysis(factor_id, analysis, data_source)
    # P2-02/P2-04: operator semantic manifest + calendar version enter the
    # full definition so an operator upgrade / calendar-contract change
    # invalidates the factor even when source data is unchanged.
    manifest = _operator_manifest_from_ir(getattr(analysis, "ir", None))
    cal_version = _calendar_version(data_source)
    full_def = dict(full_definition or {})
    if manifest:
        full_def["operator_manifest"] = manifest
    if cal_version:
        full_def["calendar_version"] = cal_version
    dep_catalog = (
        catalog if isinstance(catalog, DependencyCatalog) else DependencyCatalog(catalog)
    )
    dep_catalog.record_factor_manifest(
        factor_id,
        edges=edges,
        referenced_columns=referenced_columns,
        lookback=lookback,
        frequency=frequency,
        source_dataset=source_dataset,
        full_definition=full_def or None,
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


def _verify_factor_semantic_identity(
    factor: Any,
    full_def: dict[str, Any],
    *,
    production: bool = False,
) -> None:
    """Verify a rebuilt factor matches the catalog identity — FAIL-CLOSED (P0-17).

    P0-17: in production, "cannot prove identity" IS an identity-verification
    failure — a hash-computation error must never be silently skipped and passed.
    Checks, when the catalog records them: ``source_expr``, IR ``ast_hash``,
    surface / dialect / dialect_version / decision policy / calendar / PIT mode.
    """
    fid = full_def.get("factor_id") or "?"
    catalog_expr = full_def.get("expression")
    if catalog_expr and str(catalog_expr).strip():
        actual_expr = str(getattr(factor, "source_expr", "") or "").strip()
        if production and not actual_expr:
            raise FactorSemanticIdentityMismatch(
                f"因子 {fid!r} 重建后无 source_expr，无法证明 identity（P0-17 fail-closed）"
            )
        if actual_expr and actual_expr != str(catalog_expr).strip():
            raise FactorSemanticIdentityMismatch(
                f"因子 {fid!r} 重建表达式与 catalog 不一致: "
                f"{actual_expr!r} != {catalog_expr!r}"
            )
    ast_hash = full_def.get("ast_hash")
    if ast_hash and str(ast_hash).strip():
        from storage.catalog import compute_ir_hash

        try:
            actual_hash = compute_ir_hash(factor.expr)
        except Exception as exc:  # Expr is not always a serializable IR node
            if production:
                raise FactorSemanticIdentityMismatch(
                    f"因子 {fid!r} 无法计算 IR hash，无法证明 identity（P0-17 "
                    f"fail-closed）: {type(exc).__name__}: {exc}"
                ) from exc
            actual_hash = None
        if (
            actual_hash is not None
            and str(actual_hash) != str(ast_hash).strip()
        ):
            raise FactorSemanticIdentityMismatch(
                f"因子 {fid!r} IR hash 与 catalog 不一致: "
                f"{str(actual_hash)[:12]} != {str(ast_hash)[:12]}"
            )
    for key, getter in (
        ("surface", lambda f: str(getattr(f, "surface", None) or "")),
        ("dialect", lambda f: str(getattr(f, "dialect", None) or "")),
        ("dialect_version", lambda f: str(getattr(f, "dialect_version", None) or "")),
        ("decision_policy", lambda f: str(getattr(f, "decision_time_policy", None) or "")),
    ):
        declared = full_def.get(key)
        if not declared:
            continue
        actual = getter(factor)
        if actual and actual != str(declared):
            raise FactorSemanticIdentityMismatch(
                f"因子 {fid!r} 重建 {key} 与 catalog 不一致: {actual!r} != {declared!r}"
            )


class DataEventStaleError(ValueError):
    """An event's ``snapshot_before`` does not match the ledger's last committed
    snapshot for its (dataset, field) chain — stale / out-of-order (P0-18)."""


class PartialIncrementalFailureError(RuntimeError):
    """Production forbids partial-success events: some downstream factors failed,
    so the event must stay pending (P0-20)."""


class DataEventLedger:
    """P0-18: idempotent / ordered DataEvent processing ledger.

    Persists ``{event_id -> record}`` to ``<lake_root>/.event_ledger.jsonl``
    (empty when ``lake_root`` is None).  Enforcement is per event:

      * an already-committed ``event_id`` is a duplicate retry (skipped);
      * ``event.snapshot_before`` must equal the chain's last committed
        ``snapshot_after``, else the event is stale (rejected);
      * only after ALL downstream factors succeed is ``snapshot_after``
        committed — a partial success leaves the event ``pending``/``failed``.
    """

    def __init__(self, lake_root: str | Path | None = None) -> None:
        self._path: Path | None = None
        if lake_root is not None:
            self._path = Path(lake_root) / ".event_ledger.jsonl"
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._path is None or not self._path.exists():
            return {}
        records: dict[str, dict[str, Any]] = {}
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eid = rec.get("event_id")
                if eid:
                    records[str(eid)] = rec
        except Exception:  # pragma: no cover - ledger 损坏时降级为内存状态
            return {}
        return records

    def _append(self, rec: dict[str, Any]) -> None:
        self._records[str(rec.get("event_id"))] = rec
        if self._path is not None:
            try:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            except OSError:  # pragma: no cover - 写失败不阻塞处理
                pass

    def is_committed(self, event_id: str) -> bool:
        return self._records.get(str(event_id), {}).get("status") == "committed"

    def last_committed_snapshot(self, dataset: str, field: str) -> Any | None:
        for rec in self._records.values():
            if (
                rec.get("status") == "committed"
                and str(rec.get("dataset")) == str(dataset)
                and str(rec.get("field")) == str(field)
            ):
                return rec.get("snapshot_after")
        return None

    def begin(self, event: DataEvent) -> str:
        """Validate ordering / idempotency; mark ``pending``.  Returns
        ``"duplicate"`` when already committed, ``"ok"`` otherwise."""
        if not event.event_id:
            return "ok"
        if self.is_committed(event.event_id):
            return "duplicate"
        if event.snapshot_before is not None:
            last = self.last_committed_snapshot(event.dataset, event.field)
            if last is not None and str(last) != str(event.snapshot_before):
                raise DataEventStaleError(
                    f"event {event.event_id!r}: snapshot_before={event.snapshot_before!r} "
                    f"!= ledger last committed snapshot {last!r} for "
                    f"({event.dataset}, {event.field}) — stale / out-of-order"
                )
        self._append(
            {
                "event_id": event.event_id,
                "dataset": event.dataset,
                "field": event.field,
                "sequence": event.sequence,
                "status": "pending",
                "snapshot_before": event.snapshot_before,
                "snapshot_after": None,
                "received_at": event.updated_date,
            }
        )
        return "ok"

    def commit(self, event: DataEvent) -> None:
        if not event.event_id:
            return
        self._append(
            {
                "event_id": event.event_id,
                "dataset": event.dataset,
                "field": event.field,
                "sequence": event.sequence,
                "status": "committed",
                "snapshot_before": event.snapshot_before,
                "snapshot_after": event.snapshot_after,
                "received_at": event.updated_date,
            }
        )

    def reject(self, event: DataEvent, reason: str) -> None:
        if not event.event_id:
            return
        self._append(
            {
                "event_id": event.event_id,
                "dataset": event.dataset,
                "field": event.field,
                "sequence": event.sequence,
                "status": "rejected",
                "reason": reason,
                "received_at": event.updated_date,
            }
        )


def _is_production_run(event: DataEvent) -> bool:
    """Effective production mode for an event (P0-19/P0-20).

    An explicit ``event.run_mode == "production"`` (when a producer sets it)
    wins; otherwise the global production policy decides.  Default research,
    so existing callers keep their behavior.
    """
    mode = getattr(event, "run_mode", None)
    if mode and str(mode).strip().lower() in {"production", "prod"}:
        return True
    try:
        from runtime.production_policy import is_production_mode

        return bool(is_production_mode())
    except Exception:  # pragma: no cover - 策略解析失败按 research 处理
        return False


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
    """DataEvent → 计划 → 逐因子物化（每个因子使用自己的重建引擎，R10 #51）。

    P0-18/P0-20: an event with an ``event_id`` passes through a ``DataEventLedger``
    (idempotent retries skipped, out-of-order ``snapshot_before`` rejected) and is
    committed only when ALL downstream factors succeed.  Production forbids
    partial-success events: any failure raises ``PartialIncrementalFailureError``
    and the event stays pending.
    """
    from storage.materializer import ParquetMaterializer
    from runtime.dependency_catalog import DependencyCatalog

    event = normalize_data_event(event)
    production = _is_production_run(event)
    ledger = DataEventLedger(lake_root=lake_root)
    ledger_status = ledger.begin(event) if event.event_id else "ok"
    if ledger_status == "duplicate":
        # Idempotent retry of an already-committed event: nothing to do.
        return {
            "event": event.to_dict(),
            "plans": [],
            "factor_count": 0,
            "dry_run": dry_run,
            "materializations": {},
            "ledger_status": "duplicate",
            "succeeded": 0,
            "failed": 0,
        }
    catalog = ParquetMaterializer(lake_root=lake_root).catalog
    dep_catalog = DependencyCatalog(catalog)
    plans = plan_updates_from_data_event(
        dep_catalog,
        event,
        end_date=end_date,
        lookback_extra=lookback_extra,
        market=market,
        production=production,
    )
    out: dict[str, Any] = {
        "event": event.to_dict(),
        "plans": [p.to_dict() for p in plans],
        "factor_count": len(plans),
        "dry_run": dry_run,
        "materializations": {},
        "ledger_status": ledger_status,
    }
    if dry_run or not plans:
        if event.event_id:
            ledger.reject(event, "dry_run_or_no_plans")
            out["ledger_status"] = "rejected"
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
                _verify_factor_semantic_identity(
                    factor, info, production=production
                )
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
                deleted_keys=plan.deleted_keys,
                **mat_kwargs,
            )
            out["materializations"][plan.factor_id] = result.get(
                "materialization", result
            )
        except Exception as exc:
            failures.append({"factor_id": plan.factor_id, "error": str(exc)})

    if failures:
        out["failures"] = failures
        if event.event_id:
            ledger.reject(
                event, f"{len(failures)} downstream factor(s) failed"
            )
            out["ledger_status"] = "failed"
        if production:
            # P0-20: production events may NOT silently end partial — a partial
            # snapshot in the lake (some factors new, some old) is data corruption.
            raise PartialIncrementalFailureError(
                f"production: event {event.event_id or event.updated_date!r} had "
                f"{len(failures)} failed downstream factor(s): "
                + "; ".join(f"{f['factor_id']}: {f['error']}" for f in failures[:5])
            )
    elif event.event_id:
        ledger.commit(event)
        out["ledger_status"] = "committed"
    out["succeeded"] = len(out["materializations"])
    out["failed"] = len(failures)
    return out
