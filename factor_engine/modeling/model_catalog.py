# -*- coding: utf-8 -*-
"""Durable catalog for frozen model artifacts."""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modeling.artifact import ModelArtifact

__all__ = [
    "ModelArtifactCatalog",
    "ModelArtifactCatalogRecord",
    "parse_catalog_timestamp",
    "validate_artifact_id",
]

_MACHINE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


def validate_artifact_id(value: str) -> str:
    """Validate the strict ASCII machine identifier used in artifact paths."""
    if not isinstance(value, str) or not value or _MACHINE_ID.fullmatch(value) is None:
        raise ValueError(
            "artifact_id must match [A-Za-z0-9_.:-]+; path separators, NUL, "
            "whitespace, and non-ASCII identifiers are forbidden"
        )
    return value


def parse_catalog_timestamp(value: Any) -> datetime:
    """Return a timezone-aware UTC timestamp, rejecting ambiguous values."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 artifact timestamp {value!r}") from exc
    else:
        raise ValueError(f"artifact timestamp is required; got {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _timestamp_text(value: Any) -> str:
    return parse_catalog_timestamp(value).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ModelArtifactCatalogRecord:
    artifact_id: str
    model_name: str
    semantic_version: str
    training_cutoff: str
    available_at: str
    published_at: str
    storage_path: str
    namespace: str = "default"
    tenant: str = "default"
    project: str = "default"
    market: str = "default"
    access_classification: str = "internal"
    promotion_state: str = "CANDIDATE"
    revoked: bool = False

    @property
    def training_cutoff_timestamp(self) -> datetime:
        return parse_catalog_timestamp(self.training_cutoff)

    @property
    def available_at_timestamp(self) -> datetime:
        return parse_catalog_timestamp(self.available_at)


class ModelArtifactCatalog:
    """SQLite-backed transactional artifact catalog with immutable snapshots."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_artifacts (
                artifact_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                semantic_version TEXT NOT NULL,
                training_cutoff TEXT NOT NULL,
                available_at TEXT NOT NULL,
                published_at TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                namespace TEXT NOT NULL,
                tenant TEXT NOT NULL,
                project TEXT NOT NULL,
                market TEXT NOT NULL,
                access_classification TEXT NOT NULL,
                promotion_state TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_artifacts_resolution
            ON model_artifacts (
                tenant, project, market, namespace, model_name,
                revoked, training_cutoff, available_at
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def register(
        self,
        artifact: ModelArtifact,
        storage_path: str,
        *,
        namespace: str = "default",
        tenant: str = "default",
        project: str = "default",
        market: str = "default",
        access_classification: str = "internal",
        promotion_state: str = "CANDIDATE",
        published_at: Any | None = None,
    ) -> ModelArtifactCatalogRecord:
        m = artifact.manifest
        record = ModelArtifactCatalogRecord(
            artifact_id=validate_artifact_id(m.artifact_id),
            model_name=m.model_name,
            semantic_version=m.model_version,
            training_cutoff=_timestamp_text(m.training_cutoff),
            available_at=_timestamp_text(m.available_at),
            published_at=_timestamp_text(published_at or m.available_at),
            storage_path=str(storage_path),
            namespace=namespace,
            tenant=tenant,
            project=project,
            market=market,
            access_classification=access_classification,
            promotion_state=promotion_state,
        )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO model_artifacts (
                        artifact_id, model_name, semantic_version,
                        training_cutoff, available_at, published_at, storage_path,
                        namespace, tenant, project, market,
                        access_classification, promotion_state, revoked
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        record.artifact_id,
                        record.model_name,
                        record.semantic_version,
                        record.training_cutoff,
                        record.available_at,
                        record.published_at,
                        record.storage_path,
                        record.namespace,
                        record.tenant,
                        record.project,
                        record.market,
                        record.access_classification,
                        record.promotion_state,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return record

    def revoke(self, artifact_id: str) -> None:
        validate_artifact_id(artifact_id)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "UPDATE model_artifacts SET revoked=1 WHERE artifact_id=?",
                    (artifact_id,),
                )
                if cur.rowcount != 1:
                    raise KeyError(f"artifact {artifact_id!r} is not cataloged")
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def snapshot(
        self,
        *,
        model_name: str | None = None,
        namespace: str = "default",
        tenant: str = "default",
        project: str = "default",
        market: str = "default",
        include_revoked: bool = False,
    ) -> tuple[ModelArtifactCatalogRecord, ...]:
        clauses = ["namespace=?", "tenant=?", "project=?", "market=?"]
        params: list[Any] = [namespace, tenant, project, market]
        if model_name is not None:
            clauses.append("model_name=?")
            params.append(model_name)
        if not include_revoked:
            clauses.append("revoked=0")
        sql = (
            "SELECT * FROM model_artifacts WHERE "
            + " AND ".join(clauses)
            + " ORDER BY training_cutoff, available_at, semantic_version, artifact_id"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return tuple(
            ModelArtifactCatalogRecord(
                **{**dict(row), "revoked": bool(row["revoked"])}
            )
            for row in rows
        )
