"""Daily FeatureSet Snapshot — the single-wide-table read path for model
inference / simulation / live trading (spec §17, §18).

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` (spec §17, §18)
for the **daily materialized snapshot** layer: an immutable, per-``TradeDate``
wide table (``InstrumentID`` + ``Feature_001..N``) bound to a ``FeatureSetVersion``
and delivered alongside a ``manifest.json`` for fail-closed verification on read.

PURE stdlib frozen dataclasses + pure functions only (the PURE-DTO rule). No
parquet/pandas/pydantic imports at contract layer — the loader passes already
materialized schema columns / rows into the pure verification helpers. Testable
without a DB or COS.

Key ideas
---------
- **Snapshot identity** (§17): keyed by ``feature_set_version_id`` + ``TradeDate``.
  A snapshot binds every field the model needs: feature-set identity, the
  ordered feature manifest, factor-definition ids, treatment ids, orientation,
  feature names, schema hash, data snapshot id, universe id, decision time,
  signal-available time, created time, and the wide-table artifact.
- **Manifest** (§18): ``manifest.json`` carrying identity + hashes +
  artifact identity. The parquet's own metadata may embed ``manifest_hash`` so a
  reader can prove the manifest was the one that produced the artifact.
- **Fail-closed verification** (§18): a loader must load the manifest, verify the
  expected ``FeatureSetVersion``, verify feature order, verify schema hash,
  verify data hash / artifact identity. **Any** mismatch raises
  ``SnapshotVerificationError`` — never auto-reorder-and-continue.
- **Schema hash**: deterministic hash of the ordered column schema
  (feature identity + dtype + position). **Data content hash**: a hash over the
  table's row/col content.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from ._contenthash import canonical_str, content_hash

__all__ = [
    "DAILY_SNAPSHOT_LAYOUT_DIR",
    "DATA_PARQUET_NAME",
    "MANIFEST_NAME",
    "SnapshotFeatureRef",
    "DailySnapshot",
    "SnapshotManifest",
    "SnapshotVerificationResult",
    "SnapshotVerificationError",
    "compute_schema_hash",
    "compute_data_hash",
    "manifest_hash",
    "snapshot_layout_key",
    "verify_snapshot",
    "verify_snapshot_strict",
]

# Recommended layout (spec §17):
#   <root>/daily_snapshot/feature_set_version_id=FS_xxxx/TradeDate=YYYY-MM-DD/
#       {data.parquet, manifest.json}
DAILY_SNAPSHOT_LAYOUT_DIR = "daily_snapshot"
DATA_PARQUET_NAME = "data.parquet"
MANIFEST_NAME = "manifest.json"

# Ordering-sensitive schema: feature identity + dtype in column order.
_DTYPE_RE = ("int64", "int32", "float64", "float32", "str", "bool")


def _require_non_empty(value: Any, name: str) -> None:
    if value is None or value == "":
        raise ValueError(f"{name} is required")


@dataclass(frozen=True)
class SnapshotFeatureRef:
    """One ordered feature bound into a daily snapshot wide table (§17/§18).

    Binds feature identity + dtype + position so the ordered column schema is
    exactly determinable (schema hash = feature identity + dtype + position).
    """

    position: int
    feature_name: str
    factor_definition_id: str
    treatment_id: str | None = None
    orientation: str | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("position must be >= 0")
        _require_non_empty(self.feature_name, "feature_name")
        _require_non_empty(self.factor_definition_id, "factor_definition_id")


@dataclass(frozen=True)
class DailySnapshot:
    """The materialized daily snapshot identity (spec §17).

    Binds everything the model-inference / simulation / live-trading read path
    needs. ``feature_set_version_id`` + ``trade_date`` are the snapshot identity;
    the ordered feature manifest pins the exact column layout.
    """

    feature_set_version_id: str
    trade_date: str  # ISO-8601 YYYY-MM-DD
    feature_set_content_hash: str
    ordered_features: tuple[SnapshotFeatureRef, ...]
    data_snapshot_id: str
    universe_id: str
    decision_time: datetime
    signal_available_time: datetime
    created_at: datetime
    schema_hash: str = ""
    data_content_hash: str = ""
    factor_definition_ids: tuple[str, ...] = ()
    treatment_ids: tuple[str, ...] = ()
    orientation: str | None = None
    feature_names: tuple[str, ...] = ()
    schema_version: str = "1"

    def __post_init__(self) -> None:
        _require_non_empty(self.feature_set_version_id, "feature_set_version_id")
        _require_non_empty(self.trade_date, "trade_date")
        _require_non_empty(self.feature_set_content_hash, "feature_set_content_hash")
        _require_non_empty(self.data_snapshot_id, "data_snapshot_id")
        _require_non_empty(self.universe_id, "universe_id")
        if not self.ordered_features:
            raise ValueError("ordered_features is required (must be non-empty)")
        if not self.schema_hash:
            object.__setattr__(
                self,
                "schema_hash",
                compute_schema_hash(
                    [
                        (f.feature_name, f.dtype or "float64", f.position)
                        for f in self.ordered_features
                    ]
                ),
            )


@dataclass(frozen=True)
class SnapshotManifest:
    """``manifest.json`` payload (spec §18).

    Must contain at least: feature_set_version_id, feature_set_content_hash,
    ordered_features, schema_hash, data_content_hash, data_snapshot_id,
    universe_id, trade_date, row_count, column_count, created_at. The parquet's
    own metadata may embed ``manifest_hash`` so a reader can prove the manifest
    that produced the artifact.
    """

    feature_set_version_id: str
    feature_set_content_hash: str
    ordered_features: tuple[str, ...]  # ordered feature identities
    schema_hash: str
    data_content_hash: str
    data_snapshot_id: str
    universe_id: str
    trade_date: str
    row_count: int
    column_count: int
    created_at: datetime
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.feature_set_version_id, "feature_set_version_id")
        _require_non_empty(self.feature_set_content_hash, "feature_set_content_hash")
        _require_non_empty(self.schema_hash, "schema_hash")
        _require_non_empty(self.data_content_hash, "data_content_hash")
        _require_non_empty(self.data_snapshot_id, "data_snapshot_id")
        _require_non_empty(self.universe_id, "universe_id")
        _require_non_empty(self.trade_date, "trade_date")
        if self.row_count < 0 or self.column_count < 0:
            raise ValueError("row_count and column_count must be >= 0")
        if not self.ordered_features:
            raise ValueError("ordered_features is required")
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", manifest_hash(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SnapshotManifest":
        """Build a manifest from a parsed ``manifest.json`` dict.

        Fails closed (raises ``SnapshotVerificationError``) on any missing
        required field, so a malformed manifest can never be silently accepted.
        """
        required = {
            "feature_set_version_id",
            "feature_set_content_hash",
            "ordered_features",
            "schema_hash",
            "data_content_hash",
            "data_snapshot_id",
            "universe_id",
            "trade_date",
            "row_count",
            "column_count",
            "created_at",
        }
        missing = required - set(data.keys())
        if missing:
            raise SnapshotVerificationError(
                f"manifest missing required field(s): {sorted(missing)}"
            )
        return cls(
            feature_set_version_id=str(data["feature_set_version_id"]),
            feature_set_content_hash=str(data["feature_set_content_hash"]),
            ordered_features=tuple(str(x) for x in data["ordered_features"]),
            schema_hash=str(data["schema_hash"]),
            data_content_hash=str(data["data_content_hash"]),
            data_snapshot_id=str(data["data_snapshot_id"]),
            universe_id=str(data["universe_id"]),
            trade_date=str(data["trade_date"]),
            row_count=int(data["row_count"]),
            column_count=int(data["column_count"]),
            created_at=(
                data["created_at"]
                if isinstance(data["created_at"], datetime)
                else datetime.fromisoformat(str(data["created_at"]))
            ),
            manifest_hash=str(data.get("manifest_hash", "")),
        )


class SnapshotVerificationError(ValueError):
    """Raised when a snapshot verification check fails (fail-closed)."""


@dataclass(frozen=True)
class SnapshotVerificationResult:
    """Per-check report from :func:`verify_snapshot`.

    ``checks`` maps a check name to ``(passed: bool, detail: str)``.
    ``ok`` is ``True`` iff every check passed. Call ``raise_on_fail()`` to
    fail closed by raising ``SnapshotVerificationError`` on the first failure.
    """

    checks: Mapping[str, tuple[bool, str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(passed for passed, _ in self.checks.values())

    def raise_on_fail(self) -> None:
        for name, (passed, detail) in self.checks.items():
            if not passed:
                raise SnapshotVerificationError(f"verify {name}: {detail}")

    def summary(self) -> str:
        return "; ".join(
            f"{name}={'PASS' if passed else 'FAIL'}"
            for name, (passed, _) in self.checks.items()
        )


def _canonical_cell(value: Any) -> bytes:
    """Deterministic byte form of one table cell for the data content hash.

    Unlike the semantic ``content_hash`` codec (which rejects NaN/Inf), the data
    content hash MUST be able to cover real wide-table values deterministically,
    so NaN/Inf are reduced to stable markers rather than raising.
    """
    if value is None:
        return b"null"
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, float):
        if math.isnan(value):
            return b"nan"
        if math.isinf(value):
            return b"inf" if value > 0 else b"-inf"
        if value == 0.0:
            return b"0.0"
        return repr(value).encode("utf-8")
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            utc = value.replace(tzinfo=timezone.utc)
        else:
            utc = value.astimezone(timezone.utc)
        return utc.isoformat().replace("+00:00", "Z").encode("ascii")
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(
        f"unsupported cell type for data content hash: {type(value).__name__!r}"
    )


def compute_schema_hash(columns: Sequence[Any]) -> str:
    """Deterministic hash of the ordered column schema.

    ``columns`` is an ordered sequence where each element identifies one column:
    either a ``(name, dtype)`` tuple or a ``(name, dtype, position)`` tuple, or a
    ``SnapshotFeatureRef`` (uses its position + feature_name + dtype). Because the
    sequence order participates (length-prefixed, canonical), a **feature-order
    change yields a different schema hash** — exactly the property a fail-closed
    loader needs.
    """
    normalized: list[tuple[str, str, int]] = []
    for idx, col in enumerate(columns):
        if isinstance(col, SnapshotFeatureRef):
            normalized.append(
                (col.feature_name, col.dtype or "float64", col.position)
            )
        elif isinstance(col, (tuple, list)):
            name = str(col[0])
            dtype = str(col[1]) if len(col) > 1 and col[1] is not None else "float64"
            pos = int(col[2]) if len(col) > 2 else idx
            normalized.append((name, dtype, pos))
        else:
            normalized.append((str(col), "float64", idx))
    return content_hash(normalized)


def compute_data_hash(rows: Iterable[Sequence[Any]], columns: Sequence[Any]) -> str:
    """Deterministic streaming hash over the wide-table row/col content.

    ``columns`` gives the ordered column list (only its *length* participates —
    the position/schema is captured by ``compute_schema_hash``); ``rows`` is an
    iterable of sequences aligned to ``columns``. Hashing is streaming so it does
    not buffer the whole table. Any change to a cell value changes the digest.
    """
    digest = hashlib.sha256()
    col_count = len(columns)
    digest.update(str(col_count).encode("ascii"))
    digest.update(b":")
    row_count = 0
    for row in rows:
        if len(row) != col_count:
            raise ValueError(
                f"row width {len(row)} != column count {col_count} — fail closed"
            )
        for cell in row:
            cell_bytes = _canonical_cell(cell)
            digest.update(str(len(cell_bytes)).encode("ascii"))
            digest.update(b":")
            digest.update(cell_bytes)
            digest.update(b",")
        digest.update(b"|")
        row_count += 1
    digest.update(str(row_count).encode("ascii"))
    return digest.hexdigest()


def manifest_hash(manifest: SnapshotManifest) -> str:
    """``manifest_hash``: hash of all manifest fields (excl. itself), embedded in
    the parquet metadata so a reader can prove the artifact's originating manifest."""
    return content_hash(
        manifest.feature_set_version_id,
        manifest.feature_set_content_hash,
        manifest.ordered_features,
        manifest.schema_hash,
        manifest.data_content_hash,
        manifest.data_snapshot_id,
        manifest.universe_id,
        manifest.trade_date,
        manifest.row_count,
        manifest.column_count,
        manifest.created_at,
    )


def snapshot_layout_key(
    feature_set_version_id: str,
    trade_date: str,
    *,
    root: str = DAILY_SNAPSHOT_LAYOUT_DIR,
) -> dict[str, str]:
    """Recommended layout keys (spec §17).

    Returns the directory prefix and the two artifact keys under the recommended
    ``.../daily_snapshot/feature_set_version_id=FS_xxxx/TradeDate=YYYY-MM-DD/``
    layout. ``trade_date`` must be ISO-8601 ``YYYY-MM-DD``.
    """
    _require_non_empty(feature_set_version_id, "feature_set_version_id")
    if len(trade_date) != 10 or trade_date[4] != "-" or trade_date[7] != "-":
        raise ValueError(f"trade_date must be ISO-8601 YYYY-MM-DD, got {trade_date!r}")
    fs = str(feature_set_version_id)
    # Normalize the feature_set_version_id into a safe path segment.
    safe_fs = fs.replace("/", "_").replace("\\", "_").replace("..", "_")
    prefix = (
        f"{root.rstrip('/')}/feature_set_version_id={safe_fs}/TradeDate={trade_date}"
    )
    return {
        "directory": prefix,
        "data_parquet": f"{prefix}/{DATA_PARQUET_NAME}",
        "manifest": f"{prefix}/{MANIFEST_NAME}",
    }


def _coerce_manifest(manifest: SnapshotManifest | Mapping[str, Any]) -> SnapshotManifest:
    if isinstance(manifest, SnapshotManifest):
        return manifest
    if isinstance(manifest, Mapping):
        return SnapshotManifest.from_dict(manifest)
    raise TypeError(
        f"manifest must be SnapshotManifest or mapping, got {type(manifest).__name__}"
    )


def verify_snapshot(
    manifest: SnapshotManifest | Mapping[str, Any],
    parquet_meta_or_data: Any,
    *,
    expected_feature_set_version_id: str | None = None,
    expected_feature_set_content_hash: str | None = None,
    expected_data_snapshot_id: str | None = None,
    expected_feature_order: Sequence[str] | None = None,
    actual_schema_columns: Sequence[Any] | None = None,
    actual_columns: Sequence[Any] | None = None,
    actual_rows: Sequence[Sequence[Any]] | None = None,
    actual_row_count: int | None = None,
    actual_column_count: int | None = None,
) -> SnapshotVerificationResult:
    """Fail-closed snapshot verification (spec §18), pure and testable.

    Checks, in order:
      1. manifest completeness (missing required field → fail),
      2. expected FeatureSetVersion (``feature_set_version_id`` +
         ``feature_set_content_hash``),
      3. artifact identity (``data_snapshot_id`` + universe + row/col counts),
      4. feature order — recomputed from ``ordered_features`` vs
         ``expected_feature_order``,
      5. schema hash — recomputed from ``actual_schema_columns`` (the ordered
         feature columns, feature identity + dtype + position) vs manifest,
      6. data content hash — recomputed over ``actual_columns`` +
         ``actual_rows`` (the full wide table incl. ``InstrumentID``) vs manifest.

    ``parquet_meta_or_data`` is the artifact identity / metadata object the caller
    reads out of the parquet footer (used only for the data-snapshot-id identity
    check). ``actual_schema_columns`` feeds the schema-hash check (feature
    columns); ``actual_columns`` (the full wide-table column list, incl.
    ``InstrumentID``) + ``actual_rows`` feed the data-hash check. Returns a
    :class:`SnapshotVerificationResult` reporting each check. The result NEVER
    silently reorders-and-continues; call ``result.raise_on_fail()`` (or
    :func:`verify_snapshot_strict`) to fail closed.
    """
    m = _coerce_manifest(manifest)
    checks: dict[str, tuple[bool, str]] = {}

    # (2) expected FeatureSetVersion.
    if expected_feature_set_version_id is not None:
        ok = m.feature_set_version_id == expected_feature_set_version_id
        checks["expected_feature_set_version"] = (
            ok,
            f"manifest {m.feature_set_version_id!r} vs expected "
            f"{expected_feature_set_version_id!r}",
        )
    if expected_feature_set_content_hash is not None:
        ok = m.feature_set_content_hash == expected_feature_set_content_hash
        checks["expected_feature_set_content_hash"] = (
            ok,
            f"manifest {m.feature_set_content_hash!r} vs expected "
            f"{expected_feature_set_content_hash!r}",
        )

    # (3) artifact identity / data snapshot id.
    data_snapshot_id = getattr(parquet_meta_or_data, "data_snapshot_id", None)
    if isinstance(parquet_meta_or_data, Mapping):
        data_snapshot_id = parquet_meta_or_data.get("data_snapshot_id", None)
    if expected_data_snapshot_id is not None:
        ok = m.data_snapshot_id == expected_data_snapshot_id
        checks["expected_data_snapshot_id"] = (
            ok,
            f"manifest {m.data_snapshot_id!r} vs expected {expected_data_snapshot_id!r}",
        )
    if data_snapshot_id is not None:
        ok = m.data_snapshot_id == str(data_snapshot_id)
        checks["artifact_identity"] = (
            ok,
            f"manifest {m.data_snapshot_id!r} vs parquet {data_snapshot_id!r}",
        )

    # (4) feature order.
    if expected_feature_order is not None:
        expected = tuple(str(x) for x in expected_feature_order)
        ok = tuple(m.ordered_features) == expected
        checks["feature_order"] = (
            ok,
            f"manifest {tuple(m.ordered_features)!r} vs expected {expected!r}",
        )

    # (5) schema hash — recomputed from the actual ordered feature columns.
    schema_cols = actual_schema_columns
    if schema_cols is None and actual_columns is not None:
        # Wide table's leading InstrumentID is not a feature column; default to
        # dropping it for the schema-hash check unless a caller passes explicit
        # feature columns via ``actual_schema_columns``.
        schema_cols = list(actual_columns)[1:]
    if schema_cols is not None:
        recomputed = compute_schema_hash(schema_cols)
        ok = recomputed == m.schema_hash
        checks["schema_hash"] = (
            ok,
            f"recomputed {recomputed!r} vs manifest {m.schema_hash!r}",
        )

    # (6) data content hash — recomputed over the full wide table.
    if actual_rows is not None and actual_columns is not None:
        recomputed = compute_data_hash(actual_rows, actual_columns)
        ok = recomputed == m.data_content_hash
        checks["data_content_hash"] = (
            ok,
            f"recomputed {recomputed!r} vs manifest {m.data_content_hash!r}",
        )

    # row/col counts from the parquet metadata.
    if actual_row_count is not None:
        ok = actual_row_count == m.row_count
        checks["row_count"] = (
            ok,
            f"parquet {actual_row_count} vs manifest {m.row_count}",
        )
    if actual_column_count is not None:
        ok = actual_column_count == m.column_count
        checks["column_count"] = (
            ok,
            f"parquet {actual_column_count} vs manifest {m.column_count}",
        )

    return SnapshotVerificationResult(checks)


def verify_snapshot_strict(
    manifest: SnapshotManifest | Mapping[str, Any],
    parquet_meta_or_data: Any,
    **kwargs: Any,
) -> SnapshotVerificationResult:
    """Convenience wrapper: :func:`verify_snapshot` then ``raise_on_fail()``.

    The loaders' fail-closed entry point — any mismatch raises
    ``SnapshotVerificationError`` instead of reordering-and-continuing.
    """
    result = verify_snapshot(manifest, parquet_meta_or_data, **kwargs)
    result.raise_on_fail()
    return result
