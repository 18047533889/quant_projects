# -*- coding: utf-8 -*-
"""R39 PERF-067: MaterializationIdentityCertificate — compile-time identity cache.

The identity fields a matrix writer needs (semantic digest, source snapshot,
calendar, universe, frequency, storage precision, operator manifest hash) are all
known once the plan/analysis exists.  Building them once per batch as a
``MaterializationIdentityCertificate`` and handing them to the writer lets the
writer avoid reconstructing large metadata dicts per factor.

The *digest* is computed by the same ``_matrix_factor_digest`` mechanism used by
``runtime.matrix_service`` (kept in that module so existing unit mocks on
``matrix_service._matrix_factor_digest`` keep intercepting).  This module only
owns the immutable certificate value object + a small field-extraction helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class MaterializationIdentityCertificate:
    """Immutable compile-time identity snapshot for one factor of a matrix.

    Attributes:
        factor_id: matrix factor id.
        semantic_digest: 16-hex factor semantic digest (P0-23 version binding).
        source_snapshot: resolved data snapshot id (source contract).
        calendar: calendar id from the execution scope / data source.
        universe: universe id from the execution scope.
        frequency: effective frequency.
        storage_precision: value dtype used for durable storage (e.g. "float32").
        operator_manifest_hash: operator catalog manifest hash.
    """

    factor_id: str
    semantic_digest: Optional[str] = None
    source_snapshot: Optional[str] = None
    calendar: Optional[str] = None
    universe: Optional[str] = None
    frequency: Optional[str] = None
    storage_precision: Optional[str] = None
    operator_manifest_hash: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_factor_version(self) -> str | None:
        """The factor version used for P0-23 matrix version binding."""
        return self.semantic_digest

    def to_manifest_entry(self) -> dict[str, Any]:
        """Flat dict to merge into the matrix manifest ``factors[fid]`` entry."""
        out: dict[str, Any] = {}
        if self.semantic_digest:
            out["semantic_digest"] = self.semantic_digest
            out["version"] = self.semantic_digest
        for fld, val in (
            ("source_snapshot", self.source_snapshot),
            ("calendar", self.calendar),
            ("universe", self.universe),
            ("frequency", self.frequency),
            ("storage_precision", self.storage_precision),
            ("operator_manifest_hash", self.operator_manifest_hash),
        ):
            if val is not None:
                out[fld] = val
        return out


def extract_scope_fields(scope: Any, engine: Any, value_dtype: str | None) -> dict[str, Any]:
    """Best-effort extraction of identity fields from an execution scope + engine.

    Never raises: a missing field simply stays ``None`` (the manifest records what
    is genuinely known rather than guessing).
    """
    ds = getattr(engine, "data_source", None)
    calendar = (
        getattr(scope, "calendar_id", None)
        or getattr(ds, "calendar_id", None)
        or getattr(ds, "calendar", None)
    )
    universe = getattr(scope, "universe_id", None)
    frequency = getattr(scope, "frequency", None)
    source_snapshot = getattr(ds, "data_snapshot_id", None)
    if source_snapshot is None:
        cfg = getattr(engine, "data_source_config", None)
        if cfg is not None:
            source_snapshot = str(cfg)  # fallback marker; digest remains authoritative
    return {
        "source_snapshot": str(source_snapshot) if source_snapshot else None,
        "calendar": str(calendar) if calendar else None,
        "universe": str(universe) if universe else None,
        "frequency": str(frequency) if frequency else None,
        "storage_precision": str(value_dtype) if value_dtype else None,
    }
