"""Bounded, SQLite-backed run-level catalogue for compiled factor DAGs.

The catalogue persists identities and dependency metadata only.  It never
serializes evaluated frames, arrays, or other factor values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Mapping

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.plan_hash import structural_key


_SCHEMA_VERSION = "factor_engine.run_dag_catalog.v1"


class CatalogIdentityError(ValueError):
    """The database belongs to another durable run or execution context."""


@dataclass(frozen=True)
class CatalogRoot:
    ordinal: int
    factor_name: str
    root_identity: str
    node_count: int
    consumed: bool


@dataclass(frozen=True)
class CatalogNode:
    identity: str
    op: str
    summary: str
    consumer_count: int
    remaining_consumers: int


def _canonical_scope(scope: Mapping[str, Any]) -> str:
    from data_access.core.identity_encoder import CanonicalIdentityEncoder

    if not isinstance(scope, Mapping):
        raise TypeError("data_scope must be a mapping")
    return CanonicalIdentityEncoder(strict=True).encode(dict(scope))


def _node_identity(node: PlanNode, scope_json: str) -> tuple[str, str]:
    summary = structural_key(node)
    payload = json.dumps(
        {"schema": _SCHEMA_VERSION, "scope": scope_json, "plan": summary},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), summary


class RunDAGCatalog:
    """Run/context-scoped DAG metadata with bounded ingestion and pagination."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        context_identity: Mapping[str, Any],
        create: bool = False,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        self.path = Path(path)
        self.run_id = run_id
        self.context_json = _canonical_scope(context_identity)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        try:
            self._configure()
            self._create_schema()
            self._bind_identity(create=create)
        except BaseException:
            self._conn.close()
            self._closed = True
            raise

    def _configure(self) -> None:
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS catalog_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version TEXT NOT NULL,
                run_id TEXT NOT NULL,
                context_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
                identity TEXT PRIMARY KEY,
                op TEXT NOT NULL,
                summary TEXT NOT NULL,
                consumer_count INTEGER NOT NULL CHECK (consumer_count >= 0),
                remaining_consumers INTEGER NOT NULL CHECK (remaining_consumers >= 0)
            );
            CREATE TABLE IF NOT EXISTS node_edges (
                parent_identity TEXT NOT NULL REFERENCES nodes(identity),
                child_position INTEGER NOT NULL CHECK (child_position >= 0),
                child_identity TEXT NOT NULL REFERENCES nodes(identity),
                PRIMARY KEY (parent_identity, child_position)
            );
            CREATE TABLE IF NOT EXISTS roots (
                ordinal INTEGER PRIMARY KEY CHECK (ordinal >= 0),
                factor_name TEXT NOT NULL,
                root_identity TEXT NOT NULL REFERENCES nodes(identity),
                node_count INTEGER NOT NULL CHECK (node_count > 0),
                consumed INTEGER NOT NULL DEFAULT 0 CHECK (consumed IN (0, 1))
            );
            CREATE TABLE IF NOT EXISTS root_nodes (
                ordinal INTEGER NOT NULL REFERENCES roots(ordinal),
                node_identity TEXT NOT NULL REFERENCES nodes(identity),
                PRIMARY KEY (ordinal, node_identity)
            );
            CREATE INDEX IF NOT EXISTS root_nodes_identity
                ON root_nodes(node_identity, ordinal);
            """
        )

    def _bind_identity(self, *, create: bool) -> None:
        row = self._conn.execute(
            "SELECT schema_version, run_id, context_json FROM catalog_meta WHERE singleton=1"
        ).fetchone()
        if row is None:
            if not create:
                raise CatalogIdentityError("catalog has no bound run identity")
            self._conn.execute(
                "INSERT INTO catalog_meta VALUES (1, ?, ?, ?)",
                (_SCHEMA_VERSION, self.run_id, self.context_json),
            )
            self._conn.commit()
            return
        actual = (row["schema_version"], row["run_id"], row["context_json"])
        expected = (_SCHEMA_VERSION, self.run_id, self.context_json)
        if actual != expected:
            raise CatalogIdentityError("catalog run/context identity mismatch")

    def close(self) -> None:
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "RunDAGCatalog":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("catalog is closed")

    @staticmethod
    def _walk(root: PlanNode, scope_json: str) -> tuple[str, dict[str, tuple[str, str, tuple[str, ...]]]]:
        if not isinstance(root, PlanNode):
            raise TypeError("plan must be a compiled PlanNode")
        pending: list[tuple[PlanNode, bool]] = [(root, False)]
        identities: dict[int, str] = {}
        records: dict[str, tuple[str, str, tuple[str, ...]]] = {}
        while pending:
            node, visited = pending.pop()
            if not visited:
                pending.append((node, True))
                pending.extend((child, False) for child in reversed(node.inputs))
                continue
            identity, summary = _node_identity(node, scope_json)
            children = tuple(identities[id(child)] for child in node.inputs)
            identities[id(node)] = identity
            record = (node.op, summary, children)
            prior = records.get(identity)
            if prior is not None and prior != record:
                raise RuntimeError("node identity collision")
            records[identity] = record
        return identities[id(root)], records

    def add_root(
        self,
        ordinal: int,
        factor_name: str,
        plan: PlanNode,
        *,
        data_scope: Mapping[str, Any],
        commit: bool = True,
    ) -> str:
        self._require_open()
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative int")
        if not isinstance(factor_name, str) or not factor_name:
            raise ValueError("factor_name must be nonempty")
        scope_json = _canonical_scope(data_scope)
        root_identity, records = self._walk(plan, scope_json)
        try:
            if commit:
                self._conn.execute("BEGIN")
            for identity, (op, summary, _children) in records.items():
                self._conn.execute(
                    "INSERT INTO nodes(identity, op, summary, consumer_count, remaining_consumers) "
                    "VALUES (?, ?, ?, 0, 0) ON CONFLICT(identity) DO NOTHING",
                    (identity, op, summary),
                )
            for parent, (_op, _summary, children) in records.items():
                for position, child in enumerate(children):
                    self._conn.execute(
                        "INSERT INTO node_edges VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                        (parent, position, child),
                    )
            self._conn.execute(
                "INSERT INTO roots(ordinal, factor_name, root_identity, node_count) "
                "VALUES (?, ?, ?, ?)",
                (ordinal, factor_name, root_identity, len(records)),
            )
            self._conn.executemany(
                "INSERT INTO root_nodes(ordinal, node_identity) VALUES (?, ?)",
                ((ordinal, identity) for identity in records),
            )
            self._conn.executemany(
                "UPDATE nodes SET consumer_count=consumer_count+1, "
                "remaining_consumers=remaining_consumers+1 WHERE identity=?",
                ((identity,) for identity in records),
            )
            if commit:
                self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return root_identity

    def add_roots(
        self,
        roots: Iterable[tuple[int, str, PlanNode, Mapping[str, Any]]],
        *,
        transaction_size: int = 512,
    ) -> int:
        """Consume an iterable one root at a time; no input collection is made."""
        if transaction_size <= 0:
            raise ValueError("transaction_size must be positive")
        count = 0
        pending = 0
        try:
            self._conn.execute("BEGIN")
            for ordinal, factor_name, plan, scope in roots:
                self.add_root(
                    ordinal, factor_name, plan, data_scope=scope, commit=False
                )
                count += 1
                pending += 1
                if pending >= transaction_size:
                    self._conn.commit()
                    self._conn.execute("BEGIN")
                    pending = 0
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return count

    def page_roots(
        self,
        *,
        after_ordinal: int = -1,
        limit: int = 512,
        max_payload_bytes: int = 1_048_576,
    ) -> list[CatalogRoot]:
        self._require_open()
        if limit <= 0 or max_payload_bytes <= 0:
            raise ValueError("page budgets must be positive")
        rows = self._conn.execute(
            "SELECT ordinal, factor_name, root_identity, node_count, consumed "
            "FROM roots WHERE ordinal>? ORDER BY ordinal LIMIT ?",
            (after_ordinal, limit),
        )
        out: list[CatalogRoot] = []
        used = 0
        for row in rows:
            charge = 96 + len(row["factor_name"].encode()) + len(row["root_identity"])
            if out and used + charge > max_payload_bytes:
                break
            if charge > max_payload_bytes:
                raise ValueError("one root descriptor exceeds max_payload_bytes")
            out.append(CatalogRoot(row["ordinal"], row["factor_name"], row["root_identity"], row["node_count"], bool(row["consumed"])))
            used += charge
        return out

    def page_subgraph(
        self,
        ordinal: int,
        *,
        after_identity: str = "",
        limit: int = 512,
        max_payload_bytes: int = 4_194_304,
    ) -> list[CatalogNode]:
        self._require_open()
        if limit <= 0 or max_payload_bytes <= 0:
            raise ValueError("page budgets must be positive")
        rows = self._conn.execute(
            "SELECT n.identity, n.op, n.summary, n.consumer_count, n.remaining_consumers "
            "FROM root_nodes rn JOIN nodes n ON n.identity=rn.node_identity "
            "WHERE rn.ordinal=? AND n.identity>? ORDER BY n.identity LIMIT ?",
            (ordinal, after_identity, limit),
        )
        out: list[CatalogNode] = []
        used = 0
        for row in rows:
            charge = 128 + len(row["identity"]) + len(row["op"].encode()) + len(row["summary"].encode())
            if out and used + charge > max_payload_bytes:
                break
            if charge > max_payload_bytes:
                raise ValueError("one node descriptor exceeds max_payload_bytes")
            out.append(CatalogNode(row["identity"], row["op"], row["summary"], row["consumer_count"], row["remaining_consumers"]))
            used += charge
        return out

    def consume_root(self, ordinal: int) -> tuple[str, ...]:
        """Atomically consume one root and return node identities reaching zero."""
        self._require_open()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT consumed FROM roots WHERE ordinal=?", (ordinal,)
            ).fetchone()
            if row is None:
                raise KeyError(ordinal)
            if row["consumed"]:
                raise RuntimeError(f"root ordinal {ordinal} already consumed")
            self._conn.execute("UPDATE roots SET consumed=1 WHERE ordinal=?", (ordinal,))
            identities = [row[0] for row in self._conn.execute(
                "SELECT node_identity FROM root_nodes WHERE ordinal=?", (ordinal,)
            )]
            self._conn.executemany(
                "UPDATE nodes SET remaining_consumers=remaining_consumers-1 "
                "WHERE identity=? AND remaining_consumers>0",
                ((identity,) for identity in identities),
            )
            released = tuple(row[0] for row in self._conn.execute(
                "SELECT n.identity FROM nodes n JOIN root_nodes rn "
                "ON rn.node_identity=n.identity WHERE rn.ordinal=? "
                "AND n.remaining_consumers=0 ORDER BY n.identity",
                (ordinal,),
            ))
            self._conn.commit()
            return released
        except BaseException:
            self._conn.rollback()
            raise

    def counts(self) -> dict[str, int]:
        self._require_open()
        roots = self._conn.execute("SELECT COUNT(*) FROM roots").fetchone()[0]
        nodes = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        live = self._conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE remaining_consumers>0"
        ).fetchone()[0]
        return {"roots": roots, "nodes": nodes, "live_nodes": live}

    def summary(self) -> dict[str, int | str]:
        """Return bounded scalar metadata for whole-DAG broker admission."""
        self._require_open()
        row = self._conn.execute(
            "SELECT COUNT(*) AS unique_nodes, "
            "COALESCE(SUM(LENGTH(CAST(summary AS BLOB))), 0) AS plan_bytes, "
            "COALESCE(SUM(CASE WHEN consumer_count > 1 THEN 1 ELSE 0 END), 0) "
            "AS shared_nodes, COALESCE(MAX(consumer_count), 0) AS max_consumers "
            "FROM nodes"
        ).fetchone()
        root_count = int(self._conn.execute("SELECT COUNT(*) FROM roots").fetchone()[0])
        edge_count = int(self._conn.execute("SELECT COUNT(*) FROM node_edges").fetchone()[0])
        return {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "root_count": root_count,
            "unique_nodes": int(row["unique_nodes"]),
            "shared_nodes": int(row["shared_nodes"]),
            "max_node_consumers": int(row["max_consumers"]),
            "node_edges": edge_count,
            "serialized_plan_bytes": int(row["plan_bytes"]),
        }

    def integrity_check(self) -> str:
        self._require_open()
        return str(self._conn.execute("PRAGMA integrity_check").fetchone()[0])
