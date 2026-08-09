# -*- coding: utf-8 -*-
"""因子依赖 catalog 查询层：data event → 受影响因子。

R10 #50: the flat ``factor_dependency`` row (single ``source_dataset`` + a flat
``referenced_columns`` list) is complemented by a first-class dependency edge
table.  A factor may depend on MANY ``(dataset, field)`` pairs — e.g.
``StockDailyBar.close`` + ``StockValuationDaily.pe_ttm`` — each with its own
physical column, transform and snapshot semantics.

The edge and full-definition tables live in the same SQLite database managed by
``storage.catalog.FactorCatalog``, but are created lazily from this module so
the storage layer stays untouched.  Existing rows recorded through
``FactorCatalog.record_factor_dependency`` keep working (legacy column index is
still consulted).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from runtime.incremental_scheduler import DataEvent, FactorUpdatePlan, plan_updates_from_data_event
from storage.catalog import FactorCatalog


_EDGE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS factor_dependency_edge (
    factor_id              TEXT NOT NULL,
    dependency_id          TEXT NOT NULL,
    source_dataset         TEXT NOT NULL,
    field_id               TEXT NOT NULL,
    physical_field         TEXT,
    transform              TEXT,
    snapshot_semantics     TEXT,
    lookback               INTEGER NOT NULL DEFAULT 0,
    forward_impact         INTEGER,
    logical_table          TEXT,
    join_policy            TEXT,
    source_ref_params      TEXT,
    semantic_filters       TEXT,
    operator_canonical     TEXT,
    operator_semantic_version TEXT,
    implementation_hash    TEXT,
    calendar_version       TEXT,
    field_catalog_hash     TEXT,
    PRIMARY KEY (factor_id, dependency_id)
)
"""

_FULL_DEF_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS factor_full_definition (
    factor_id  TEXT PRIMARY KEY,
    spec_json  TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _parse_json_field(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


#: Sentinel forward-impact value meaning "unbounded" (a recursive / stateful /
#: event-clock operator propagates an input change to the end of the series).
#: ``None`` in storage means "not recorded" (legacy edge) and is treated as 0 —
#: only this explicit sentinel is unbounded.
UNBOUNDED_FORWARD_IMPACT = -1


def dependency_edge_id(edge: dict[str, Any]) -> str:
    """Stable identity for ONE semantic reference of a source field (P1-11).

    The same ``(factor_id, source_dataset, field_id)`` may be referenced two
    different ways (two transforms / two IndexSymbols / two period selections /
    two semantic filters) — each is a DISTINCT dependency with its own
    invalidation, so the edge PK is ``(factor_id, dependency_id)``.
    """
    import hashlib

    payload = json.dumps(
        {
            "source_dataset": str(edge.get("source_dataset") or ""),
            "field_id": str(edge.get("field_id") or ""),
            "physical_field": str(edge.get("physical_field") or edge.get("field_id") or ""),
            "transform": edge.get("transform"),
            "snapshot_semantics": str(edge.get("snapshot_semantics") or "point"),
            "join_policy": edge.get("join_policy"),
            "source_ref_params": edge.get("source_ref_params"),
            "semantic_filters": edge.get("semantic_filters"),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class FactorDependencyEdge:
    """One ``(dataset, field)`` dependency of a factor on its source data.

    A factor has one :class:`FactorDependencyEdge` per source field it reads
    (R10 #50).  A composite factor reading ``StockDailyBar`` +
    ``StockValuationDaily`` therefore records two edges.

    Attributes
    ----------
    factor_id: owning factor.
    source_dataset: logical data set the field lives in (e.g. ``StockDailyBar``).
    field_id: canonical field identifier the factor reads (e.g. ``close``).
    physical_field: underlying physical column in the source (defaults to ``field_id``).
    transform: optional transform applied to the raw field (``asof`` / ``ffill`` …).
    snapshot_semantics: ``point`` / ``asof`` / ``snapshot`` / ``window``.
    lookback: extra bars of history this edge requires.
    forward_impact: future output bars a change to this source field affects
        (``None`` = unbounded; P0-01).
    """

    factor_id: str
    source_dataset: str
    field_id: str
    physical_field: str | None = None
    transform: str | None = None
    snapshot_semantics: str = "point"
    lookback: int = 0
    # R11 #4: the FactorEngine logical table a field comes from (e.g.
    # ``StockValuationDaily``) — distinct from ``source_dataset`` (the physical
    # DataAccess dataset, e.g. ``ashare_stock_valuation_daily``).
    logical_table: str | None = None
    join_policy: str | None = None
    # P0-01 / P1-11: forward propagation + the semantic reference identity.
    forward_impact: int | None = None
    source_ref_params: str | None = None
    semantic_filters: str | None = None
    operator_canonical: str | None = None
    operator_semantic_version: str | None = None
    implementation_hash: str | None = None
    calendar_version: str | None = None
    field_catalog_hash: str | None = None

    @property
    def dependency_id(self) -> str:
        return dependency_edge_id(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FactorDependencyEdge":
        return cls(
            factor_id=str(raw.get("factor_id", "")),
            source_dataset=str(raw.get("source_dataset", "")),
            field_id=str(raw.get("field_id", "")),
            physical_field=raw.get("physical_field") or raw.get("field_id"),
            transform=raw.get("transform"),
            snapshot_semantics=str(raw.get("snapshot_semantics") or "point"),
            lookback=int(raw.get("lookback") or 0),
            logical_table=raw.get("logical_table"),
            join_policy=raw.get("join_policy"),
            forward_impact=raw.get("forward_impact"),
            source_ref_params=raw.get("source_ref_params"),
            semantic_filters=raw.get("semantic_filters"),
            operator_canonical=raw.get("operator_canonical"),
            operator_semantic_version=raw.get("operator_semantic_version"),
            implementation_hash=raw.get("implementation_hash"),
            calendar_version=raw.get("calendar_version"),
            field_catalog_hash=raw.get("field_catalog_hash"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "source_dataset": self.source_dataset,
            "field_id": self.field_id,
            "physical_field": self.physical_field or self.field_id,
            "transform": self.transform,
            "snapshot_semantics": self.snapshot_semantics,
            "lookback": self.lookback,
            "logical_table": self.logical_table,
            "join_policy": self.join_policy,
            "forward_impact": self.forward_impact,
            "source_ref_params": self.source_ref_params,
            "semantic_filters": self.semantic_filters,
            "operator_canonical": self.operator_canonical,
            "operator_semantic_version": self.operator_semantic_version,
            "implementation_hash": self.implementation_hash,
            "calendar_version": self.calendar_version,
            "field_catalog_hash": self.field_catalog_hash,
        }


@dataclass(frozen=True)
class DependencySummary:
    """列 → 因子反向索引摘要。"""

    column: str
    factor_ids: tuple[str, ...]
    source_datasets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "factor_ids": list(self.factor_ids),
            "source_datasets": list(self.source_datasets),
            "factor_count": len(self.factor_ids),
        }


class DependencyCatalog:
    """``FactorCatalog.factor_dependency`` 表的只读/写入门面（R10 edge + full spec）。"""

    def __init__(self, catalog: FactorCatalog) -> None:
        self._catalog = catalog

    @classmethod
    def from_lake(cls, lake_root: str | Path) -> "DependencyCatalog":
        from storage.materializer import ParquetMaterializer

        cat = ParquetMaterializer(lake_root=lake_root).catalog
        return cls(cat)

    @property
    def catalog(self) -> FactorCatalog:
        return self._catalog

    # ------------------------------------------------------------------
    # R10 #50 rich dependency edges
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Lazily create the supplementary edge / full-definition tables.

        R11 #4: columns added after the initial table definition are migrated
        via ``ALTER TABLE ADD COLUMN`` for existing catalogs.  R13 P1-11: the
        edge PK changed to ``(factor_id, dependency_id)`` — a full table
        rebuild is needed (SQLite cannot alter a PRIMARY KEY).
        """
        self._catalog._exec_commit(_FULL_DEF_TABLE_SQL)
        existing = self._catalog._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='factor_dependency_edge'"
        ).fetchone()
        if existing is not None:
            edge_cols = {
                row[1]
                for row in self._catalog._conn.execute(
                    "PRAGMA table_info(factor_dependency_edge)"
                )
            }
            if "dependency_id" not in edge_cols:
                # R13 P1-11 migration: rebuild under the new (factor_id,
                # dependency_id) PK so one field referenced two ways stays two
                # edges.  Existing rows get a deterministic dependency_id
                # (computed in Python BEFORE insert — a blanket '' would collide
                # on the new PK for multi-edge factors).
                self._catalog._exec_commit(
                    "ALTER TABLE factor_dependency_edge RENAME TO factor_dependency_edge_legacy"
                )
                self._catalog._exec_commit(_EDGE_TABLE_SQL)
                legacy_cols = {
                    row[1]
                    for row in self._catalog._conn.execute(
                        "PRAGMA table_info(factor_dependency_edge_legacy)"
                    )
                }
                select_cols = (
                    "factor_id, source_dataset, field_id, physical_field, transform, "
                    "snapshot_semantics, lookback, logical_table, join_policy"
                )
                present = [c for c in select_cols.split(", ") if c in legacy_cols]
                legacy_rows = self._catalog._conn.execute(
                    f"SELECT {', '.join(present)} FROM factor_dependency_edge_legacy"
                ).fetchall()
                for row in legacy_rows:
                    rec = dict(row)
                    did = dependency_edge_id(
                        {
                            "source_dataset": rec.get("source_dataset"),
                            "field_id": rec.get("field_id"),
                            "physical_field": rec.get("physical_field"),
                            "transform": rec.get("transform"),
                            "snapshot_semantics": rec.get("snapshot_semantics"),
                            "join_policy": rec.get("join_policy"),
                        }
                    )
                    self._catalog._exec_commit(
                        "INSERT OR REPLACE INTO factor_dependency_edge "
                        "(factor_id, dependency_id, source_dataset, field_id, physical_field, "
                        " transform, snapshot_semantics, lookback, forward_impact, logical_table, "
                        " join_policy, source_ref_params, semantic_filters, operator_canonical, "
                        " operator_semantic_version, implementation_hash, calendar_version, "
                        " field_catalog_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(rec.get("factor_id") or ""),
                            did,
                            str(rec.get("source_dataset") or ""),
                            str(rec.get("field_id") or ""),
                            rec.get("physical_field") or rec.get("field_id"),
                            rec.get("transform"),
                            str(rec.get("snapshot_semantics") or "point"),
                            int(rec.get("lookback") or 0),
                            None,
                            rec.get("logical_table"),
                            rec.get("join_policy"),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        ),
                    )
                self._catalog._exec_commit("DROP TABLE factor_dependency_edge_legacy")
            else:
                for col, col_type in (
                    ("forward_impact", "INTEGER"),
                    ("source_ref_params", "TEXT"),
                    ("semantic_filters", "TEXT"),
                    ("operator_canonical", "TEXT"),
                    ("operator_semantic_version", "TEXT"),
                    ("implementation_hash", "TEXT"),
                    ("calendar_version", "TEXT"),
                    ("field_catalog_hash", "TEXT"),
                ):
                    if col not in edge_cols:
                        self._catalog._exec_commit(
                            f"ALTER TABLE factor_dependency_edge ADD COLUMN {col} {col_type}"
                        )
        else:
            self._catalog._exec_commit(_EDGE_TABLE_SQL)

    def record_factor_edges(
        self,
        factor_id: str,
        *,
        edges: Iterable[FactorDependencyEdge],
        referenced_columns: Iterable[str] | None = None,
        lookback: int = 0,
        frequency: str | None = None,
        source_dataset: str | None = None,
    ) -> None:
        """Record rich dependency edges for a factor (replacing previous edges).

        R11 #8: the whole replace (legacy row + delete-old-edges + insert-all-
        new-edges) is a single ``BEGIN IMMEDIATE`` transaction — a concurrent
        ``factors_for_event`` reader or a crash can never observe a partial
        edge set (0 edges / 1-of-5 edges).

        The flat legacy row is mirrored through
        ``FactorCatalog.record_factor_dependency`` so pre-R10 consumers
        (``list_factors_for_column`` / ``plan_updates_from_data_event`` on a raw
        ``FactorCatalog``) keep working.
        """
        self.record_factor_manifest(
            factor_id,
            edges=edges,
            referenced_columns=referenced_columns,
            lookback=lookback,
            frequency=frequency,
            source_dataset=source_dataset,
        )

    # ------------------------------------------------------------------
    # R11 #5/#8: atomic whole-definition mutation
    # ------------------------------------------------------------------

    def _begin_atomic(self) -> None:
        """``BEGIN IMMEDIATE`` with bounded SQLITE_BUSY retry (mirrors _exec_commit)."""
        import time

        conn = self._catalog._conn
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        raise last_error  # type: ignore[misc]

    def _atomic_apply(self, fn) -> None:
        """Run ``fn(conn)`` inside one atomic transaction (COMMIT/ROLLBACK)."""
        conn = self._catalog._conn
        self._begin_atomic()
        try:
            fn(conn)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:  # pragma: no cover - transaction may already be dead
                pass
            raise

    def record_factor_manifest(
        self,
        factor_id: str,
        *,
        edges: Iterable[FactorDependencyEdge],
        referenced_columns: Iterable[str] | None = None,
        lookback: int = 0,
        frequency: str | None = None,
        source_dataset: str | None = None,
        full_definition: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """原子写整个 factor dependency definition（legacy + edges [+ full spec]).

        R11 #8: legacy upsert / old-edge delete / new-edge insert / full-spec
        write 全部在一个 ``BEGIN IMMEDIATE`` 事务内提交，任何中间态（部分 edges、
        edges 与 full spec 不一致）都不会被并发 reader 或 crash 观察到。

        ``full_definition`` 非 None 时额外写 ``factor_full_definition`` 并同步
        ``factor_registry.data_source_json``。
        """
        self._ensure_tables()
        fid = str(factor_id)
        edge_list = [FactorDependencyEdge.from_dict(e.to_dict()) for e in edges]
        if source_dataset is None and edge_list:
            source_dataset = edge_list[0].source_dataset
        cols: set[str] = set()
        for edge in edge_list:
            if edge.field_id:
                cols.add(edge.field_id)
            if edge.physical_field:
                cols.add(edge.physical_field)
        if referenced_columns:
            cols.update(str(c) for c in referenced_columns if c)
        lookback = lookback or max((e.lookback for e in edge_list), default=0)
        now = datetime.now(timezone.utc).isoformat()

        def _write(conn) -> None:
            conn.execute(
                "INSERT INTO factor_dependency "
                "(factor_id, referenced_columns_json, lookback, frequency, source_dataset, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(factor_id) DO UPDATE SET "
                "referenced_columns_json=excluded.referenced_columns_json, "
                "lookback=excluded.lookback, frequency=excluded.frequency, "
                "source_dataset=excluded.source_dataset, updated_at=excluded.updated_at",
                (
                    fid,
                    json.dumps(sorted(cols), ensure_ascii=False),
                    int(lookback),
                    frequency,
                    source_dataset,
                    now,
                ),
            )
            conn.execute(
                "DELETE FROM factor_column_dep WHERE factor_id = ?", (fid,)
            )
            for col in sorted(cols):
                conn.execute(
                    "INSERT OR IGNORE INTO factor_column_dep (column_name, factor_id) "
                    "VALUES (?, ?)",
                    (col, fid),
                )
            conn.execute(
                "DELETE FROM factor_dependency_edge WHERE factor_id = ?", (fid,)
            )
            for edge in edge_list:
                conn.execute(
                    "INSERT OR REPLACE INTO factor_dependency_edge "
                    "(factor_id, dependency_id, source_dataset, field_id, physical_field, "
                    " transform, snapshot_semantics, lookback, forward_impact, logical_table, "
                    " join_policy, source_ref_params, semantic_filters, operator_canonical, "
                    " operator_semantic_version, implementation_hash, calendar_version, "
                    " field_catalog_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        fid,
                        edge.dependency_id,
                        str(edge.source_dataset),
                        str(edge.field_id),
                        edge.physical_field or edge.field_id,
                        edge.transform,
                        edge.snapshot_semantics,
                        int(edge.lookback),
                        edge.forward_impact,
                        edge.logical_table,
                        edge.join_policy,
                        edge.source_ref_params,
                        edge.semantic_filters,
                        edge.operator_canonical,
                        edge.operator_semantic_version,
                        edge.implementation_hash,
                        edge.calendar_version,
                        edge.field_catalog_hash,
                    ),
                )
            if full_definition is not None:
                spec = {
                    k: v
                    for k, v in dict(full_definition).items()
                    if v is not None
                }
                conn.execute(
                    "INSERT INTO factor_full_definition (factor_id, spec_json, updated_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(factor_id) DO UPDATE SET "
                    "spec_json=excluded.spec_json, updated_at=excluded.updated_at",
                    (
                        fid,
                        json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str),
                        now,
                    ),
                )
                ds_config = spec.get("data_source_config")
                if ds_config:
                    conn.execute(
                        "UPDATE factor_registry SET data_source_json = ? "
                        "WHERE factor_id = ?",
                        (
                            json.dumps(
                                ds_config, sort_keys=True, default=str, ensure_ascii=False
                            ),
                            fid,
                        ),
                    )

        self._atomic_apply(_write)
        return {
            "factor_id": fid,
            "edges": [e.to_dict() for e in edge_list],
            "referenced_columns": sorted(cols),
            "lookback": lookback,
            "frequency": frequency,
            "source_dataset": source_dataset,
            "full_definition": bool(full_definition),
        }

    def get_factor_dependency_edges(self, factor_id: str) -> list[FactorDependencyEdge]:
        """Return all rich dependency edges recorded for ``factor_id``."""
        self._ensure_tables()
        rows = self._catalog._conn.execute(
            "SELECT * FROM factor_dependency_edge WHERE factor_id = ? "
            "ORDER BY source_dataset, field_id",
            (str(factor_id),),
        ).fetchall()
        return [FactorDependencyEdge.from_dict(dict(r)) for r in rows]

    def list_edges_for_field(
        self,
        field_id: str,
        *,
        dataset: str | None = None,
    ) -> list[FactorDependencyEdge]:
        """Return all edges whose ``field_id``/``physical_field`` == ``field_id``."""
        self._ensure_tables()
        if dataset is not None:
            rows = self._catalog._conn.execute(
                "SELECT * FROM factor_dependency_edge "
                "WHERE (field_id = ? OR physical_field = ?) AND source_dataset = ? "
                "ORDER BY factor_id",
                (str(field_id), str(field_id), str(dataset)),
            ).fetchall()
        else:
            rows = self._catalog._conn.execute(
                "SELECT * FROM factor_dependency_edge "
                "WHERE field_id = ? OR physical_field = ? ORDER BY factor_id",
                (str(field_id), str(field_id)),
            ).fetchall()
        return [FactorDependencyEdge.from_dict(dict(r)) for r in rows]

    def factors_for_event(self, event: DataEvent) -> list[dict[str, Any]]:
        """Find dependency rows affected by a ``DataEvent``.

        Combines the R10 edge index (``source_dataset`` + ``field_id`` /
        ``physical_field``) with the legacy column index.  A composite factor
        with edges on several datasets is returned once per event, carrying the
        matching ``field_id`` / ``edges`` alongside the legacy columns.
        """
        self._ensure_tables()
        field = event.field_id or event.column
        out: dict[str, dict[str, Any]] = {}
        for edge in self.list_edges_for_field(field, dataset=event.dataset):
            fid = edge.factor_id
            dep = self._catalog.get_factor_dependency(fid) or {}
            cur = out.setdefault(fid, {})
            cur["factor_id"] = fid
            cur["source_dataset"] = edge.source_dataset
            cur["lookback"] = int(dep.get("lookback") or 0) or edge.lookback
            cur["frequency"] = dep.get("frequency")
            cur["referenced_columns"] = dep.get("referenced_columns") or []
            cur["field_id"] = edge.field_id
            cur["edges"] = list(cur.get("edges", [])) + [edge]
            # P0-01: forward impact of THIS semantic reference — the scheduler
            # extends recompute_end forward by it.  The explicit -1 sentinel is
            # unbounded (propagate to the end of the series); a legacy
            # ``None`` (never recorded) is 0 (no forward propagation).
            fwd = edge.forward_impact
            if fwd == UNBOUNDED_FORWARD_IMPACT:
                fwd = None
            elif fwd is None:
                fwd = 0
            if "forward_impact" not in cur:
                cur["forward_impact"] = fwd
            elif cur["forward_impact"] is None or fwd is None:
                # an unbounded reading anywhere makes the whole factor unbounded
                cur["forward_impact"] = None
            else:
                cur["forward_impact"] = max(cur["forward_impact"], fwd)
            spec = self._get_full_spec(fid)
            if spec:
                cur.setdefault("operator_manifest", spec.get("operator_manifest"))
                cur.setdefault("calendar_version", spec.get("calendar_version"))
        for dep in self._catalog.list_factors_for_column(field):
            fid = str(dep["factor_id"])
            ds = dep.get("source_dataset")
            if ds and str(ds) != str(event.dataset):
                continue
            cur = out.setdefault(fid, {})
            cur.setdefault("factor_id", fid)
            cur.setdefault("source_dataset", ds)
            cur["lookback"] = cur.get("lookback") or int(dep.get("lookback") or 0)
            cur["frequency"] = dep.get("frequency") or cur.get("frequency")
            cur["referenced_columns"] = dep.get("referenced_columns") or []
            cur.setdefault("edges", [])
            cur.setdefault("field_id", field)
        return list(out.values())

    # ------------------------------------------------------------------
    # R10 #52 full factor definition
    # ------------------------------------------------------------------

    def record_full_factor_definition(
        self,
        factor_id: str,
        *,
        expression: str | None = None,
        surface: str | None = None,
        dialect: str | None = None,
        dialect_version: str | None = None,
        market: str | None = None,
        universe: str | None = None,
        frequency: str | None = None,
        data_source_config: dict[str, Any] | None = None,
        decision_policy: str | None = None,
        run_mode: str | None = None,
        calendar: str | None = None,
        backend: str | None = None,
        pit_enforce: bool | None = None,
        author: str | None = None,
        description: str | None = None,
        ast_hash: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Save the FULL spec needed to rebuild a factor later (R10 #52).

        Defaults every unrecorded field to ``None``; non-None keys are merged
        into the registry snapshot by :meth:`full_factor_definition`.
        """
        self._ensure_tables()
        fid = str(factor_id)
        spec = {
            "factor_id": fid,
            "expression": expression,
            "surface": surface,
            "dialect": dialect,
            "dialect_version": dialect_version,
            "market": market,
            "universe": universe,
            "frequency": frequency,
            "data_source_config": data_source_config,
            "decision_policy": decision_policy,
            "run_mode": run_mode,
            "calendar": calendar,
            "backend": backend,
            "pit_enforce": pit_enforce,
            "author": author,
            "description": description,
            "ast_hash": ast_hash,
            **extra,
        }
        spec = {k: v for k, v in spec.items() if v is not None}
        now = datetime.now(timezone.utc).isoformat()
        self._catalog._exec_commit(
            "INSERT INTO factor_full_definition (factor_id, spec_json, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(factor_id) DO UPDATE SET "
            "spec_json=excluded.spec_json, updated_at=excluded.updated_at",
            (fid, json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str), now),
        )
        if data_source_config is not None:
            ds_json = json.dumps(
                data_source_config, sort_keys=True, default=str, ensure_ascii=False
            )
            self._catalog._exec_commit(
                "UPDATE factor_registry SET data_source_json = ? WHERE factor_id = ?",
                (ds_json, fid),
            )
        return dict(spec)

    def full_factor_definition(self, factor_id: str) -> dict[str, Any] | None:
        """Return the factor's full rebuild spec (registry + R10 #52 fields).

        Fields not yet recorded default to ``None``.  Returns ``None`` when the
        factor is not registered in ``factor_registry``.
        """
        info = self._catalog.get_factor_info(factor_id)
        if info is None:
            return None
        spec = self._get_full_spec(factor_id) or {}
        ds_config = spec.get("data_source_config")
        if not ds_config:
            ds_config = _parse_json_field(info.get("data_source_json"))
        base: dict[str, Any] = {
            "factor_id": str(info.get("factor_id") or factor_id),
            "author": info.get("author"),
            "frequency": info.get("frequency"),
            "description": info.get("description"),
            "ast_hash": info.get("ast_hash"),
            "expression": info.get("expression"),
            "data_source_config": ds_config or {},
            "created_at": info.get("created_at"),
            "surface": None,
            "dialect": None,
            "dialect_version": None,
            "market": None,
            "universe": None,
            "decision_policy": None,
            "run_mode": None,
            "calendar": None,
            "backend": "pandas",
            "pit_enforce": None,
        }
        merged = {**base, **{k: v for k, v in spec.items() if v is not None}}
        return merged

    def _get_full_spec(self, factor_id: str) -> dict[str, Any] | None:
        self._ensure_tables()
        row = self._catalog._conn.execute(
            "SELECT spec_json FROM factor_full_definition WHERE factor_id = ?",
            (str(factor_id),),
        ).fetchone()
        if row is None:
            return None
        spec = _parse_json_field(row[0])
        return spec or None

    # ------------------------------------------------------------------
    # Legacy query surface (kept for backward compatibility)
    # ------------------------------------------------------------------

    def factors_for_column(
        self,
        column: str,
        *,
        dataset: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._catalog.list_factors_for_column(column)
        by_id = {str(r["factor_id"]): dict(r) for r in rows}
        for edge in self.list_edges_for_field(column, dataset=dataset):
            fid = edge.factor_id
            if fid in by_id:
                continue
            by_id[fid] = {
                "factor_id": fid,
                "source_dataset": edge.source_dataset,
                "lookback": edge.lookback,
                "frequency": None,
                "referenced_columns": [edge.field_id],
            }
        out = list(by_id.values())
        if dataset is None:
            return out
        return [
            r
            for r in out
            if not r.get("source_dataset") or str(r["source_dataset"]) == str(dataset)
        ]

    def factors_for_dataset(self, dataset: str) -> list[dict[str, Any]]:
        if hasattr(self._catalog, "list_factors_for_dataset"):
            rows = self._catalog.list_factors_for_dataset(dataset)
        else:
            rows = self._catalog.list_factors()
            out: list[dict[str, Any]] = []
            for row in rows:
                dep = self._catalog.get_factor_dependency(str(row.get("factor_id", "")))
                if dep is None:
                    continue
                if dep.get("source_dataset") == dataset:
                    out.append(dep)
            return out
        # Merge edge-recorded factors whose primary dataset differs but an edge
        # points at this dataset.
        edge_rows = self._edges_for_dataset(dataset)
        by_id = {str(r["factor_id"]): dict(r) for r in rows}
        for edge in edge_rows:
            if str(edge.factor_id) not in by_id:
                by_id[str(edge.factor_id)] = {
                    "factor_id": edge.factor_id,
                    "source_dataset": edge.source_dataset,
                    "lookback": edge.lookback,
                    "frequency": None,
                    "referenced_columns": [edge.field_id],
                }
        return list(by_id.values())

    def _edges_for_dataset(self, dataset: str) -> list[FactorDependencyEdge]:
        self._ensure_tables()
        rows = self._catalog._conn.execute(
            "SELECT * FROM factor_dependency_edge WHERE source_dataset = ? "
            "ORDER BY factor_id",
            (str(dataset),),
        ).fetchall()
        return [FactorDependencyEdge.from_dict(dict(r)) for r in rows]

    def reverse_index(self, columns: Iterable[str] | None = None) -> list[DependencySummary]:
        """构建列 → 因子列表摘要。"""
        if columns is None:
            cols = self._catalog.list_dependency_columns()
        else:
            cols = sorted(set(columns))
        summaries: list[DependencySummary] = []
        for col in cols:
            deps = self.factors_for_column(col)
            factor_ids = tuple(sorted(d["factor_id"] for d in deps))
            datasets = tuple(
                sorted(
                    {
                        str(d["source_dataset"])
                        for d in deps
                        if d.get("source_dataset")
                    }
                )
            )
            summaries.append(
                DependencySummary(column=col, factor_ids=factor_ids, source_datasets=datasets)
            )
        return summaries

    def plan_for_event(
        self,
        event: DataEvent | dict[str, Any],
        *,
        end_date: str | None = None,
        lookback_extra: int = 5,
        market: str | None = None,
    ) -> list[FactorUpdatePlan]:
        from runtime.incremental_scheduler import normalize_data_event

        if not isinstance(event, DataEvent):
            event = normalize_data_event(event)
        return plan_updates_from_data_event(
            self,
            event,
            end_date=end_date,
            lookback_extra=lookback_extra,
            market=market,
        )
