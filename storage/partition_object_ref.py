# -*- coding: utf-8 -*-
"""R39 PERF-063: COW generation manifest-driven partition inventory.

Problem
-------
The legacy copy-on-write generation switch did a whole-tree discovery on *every*
publish::

    for old_part in old_gen_dir.rglob("data.parquet"):        # legacy
    for old_part in old_gen_dir.rglob("block=*.parquet"):     # block layout

then hardlinked/copied the untouched partitions into the new generation.  For a
matrix with thousands of hive partitions this rescans the entire old generation
tree on every incremental publish.

Fix
---
The manifest maintains the complete partition inventory of the generation it
points at: a list of :class:`PartitionObjectRef` (``rel_path``, ``content_id``,
``rows``, ``schema_hash``).  A new generation materializes *precisely* from that
inventory — exact ``rel_path``, no ``rglob`` discovery.  The inventory is only
built once (counting ``generation_rglob_discovery_count``) when an old generation
has no manifest inventory yet (first encounter of a pre-R39 legacy generation).

Counters (module level, tests / benchmarks):
  * ``generation_rglob_discovery_count``      — incremented only when
    :func:`build_partition_inventory` really falls back to ``rglob``;
  * ``generation_inventory_materialize_count`` — incremented once per inventory
    driven COW materialization (:func:`materialize_generation_from_inventory`).

``content_id`` is a *stable fingerprint* over ``(rel_path, rows, schema_hash)``
(as permitted by the R39 spec: "基于路径+rows+schema_hash 的稳定指纹"), not a
byte-for-byte content hash — deterministic across rebuilds of the same object,
and cheap to compute without reading the parquet data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Both physical layouts are scanned: legacy single-wide ``data.parquet`` and the
# column-factor ``block=NNNN.parquet``.
_PARQUET_PATTERNS = ("data.parquet", "block=*.parquet")

# ---- performance counters -------------------------------------------------
generation_rglob_discovery_count = 0
generation_inventory_materialize_count = 0


def reset_generation_counters() -> None:
    """Reset the R39 generation inventory counters (tests / benchmarks)."""
    global generation_rglob_discovery_count, generation_inventory_materialize_count
    generation_rglob_discovery_count = 0
    generation_inventory_materialize_count = 0


def get_generation_counters() -> dict[str, int]:
    """Return the current values of both generation inventory counters."""
    return {
        "generation_rglob_discovery_count": generation_rglob_discovery_count,
        "generation_inventory_materialize_count": generation_inventory_materialize_count,
    }


@dataclass(frozen=True)
class PartitionObjectRef:
    """One immutable partition object inside a generation.

    Attributes:
        rel_path: POSIX relative path from the generation root, e.g.
            ``"year=2024/month=01/data.parquet"`` or
            ``"year=2024/month=01/block=0001.parquet"``.
        content_id: stable fingerprint (rel_path + rows + schema_hash).
        rows: number of rows in the partition object.
        schema_hash: hash of the ordered column names (identical between a pandas
            DataFrame and the pyarrow footer of the file it writes, so the
            fingerprint is stable across both construction paths).
    """

    rel_path: str
    content_id: str
    rows: int
    schema_hash: str


def _column_schema_hash(columns: Iterable[str]) -> str:
    """Hash the *ordered* column names (schema identity for fingerprinting)."""
    return hashlib.blake2b(
        json.dumps(list(columns), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def content_id_for(rel_path: str, rows: int, schema_hash: str) -> str:
    """Stable content fingerprint: ``(rel_path, rows, schema_hash)``."""
    return hashlib.blake2b(
        f"{rel_path}\n{rows}\n{schema_hash}".encode("utf-8")
    ).hexdigest()


def ref_from_frame(rel_path: str, frame: Any) -> PartitionObjectRef:
    """Build a ref for a freshly written partition from its DataFrame (no I/O)."""
    rows = int(len(frame))
    schema_hash = _column_schema_hash(str(c) for c in frame.columns)
    return PartitionObjectRef(
        rel_path=rel_path,
        content_id=content_id_for(rel_path, rows, schema_hash),
        rows=rows,
        schema_hash=schema_hash,
    )


def _ref_from_parquet(path: Path, rel_path: str) -> PartitionObjectRef:
    """Build a ref by reading only the parquet footer (metadata, no data scan)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    rows = int(pf.metadata.num_rows)
    schema_hash = _column_schema_hash(pf.schema_arrow.names)
    return PartitionObjectRef(
        rel_path=rel_path,
        content_id=content_id_for(rel_path, rows, schema_hash),
        rows=rows,
        schema_hash=schema_hash,
    )


def build_partition_inventory(
    gen_dir: str | Path,
    *,
    patterns: Iterable[str] = _PARQUET_PATTERNS,
) -> list[PartitionObjectRef]:
    """One-time full inventory of a generation directory (rglob discovery).

    This is the *only* place legacy whole-tree discovery happens.  It increments
    ``generation_rglob_discovery_count`` — callers that already have a manifest
    inventory MUST NOT call this (repeated generation switches must not re-rglob).

    ``.staging`` / ``.quarantine`` / dot-file parquet are skipped, mirroring the
    pre-R39 discovery filters.
    """
    global generation_rglob_discovery_count
    generation_rglob_discovery_count += 1
    root = Path(gen_dir)
    refs: list[PartitionObjectRef] = []
    for pat in patterns:
        for path in sorted(root.rglob(pat)):
            if ".staging" in path.parts or ".quarantine" in path.parts:
                continue
            if path.name.startswith("."):
                continue
            rel = path.relative_to(root)
            try:
                refs.append(_ref_from_parquet(path, rel.as_posix()))
            except Exception as exc:  # pragma: no cover - fail loud
                raise ValueError(
                    f"partition inventory 构建失败: 无法读取 {path}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
    return refs


def materialize_generation_from_inventory(
    src_gen_dir: str | Path,
    src_inventory: Iterable[PartitionObjectRef],
    dst_gen_dir: str | Path,
    *,
    hardlink: bool = True,
    skip_rel: Iterable[str] = frozenset(),
) -> int:
    """Materialize ``src_inventory`` refs into ``dst_gen_dir`` by exact rel_path.

    No ``rglob``: each object is hardlinked (``os.link``, copy fallback) from
    ``src_gen_dir/rel_path`` to ``dst_gen_dir/rel_path``.  ``skip_rel`` holds the
    rel_paths already written in this publish (touched partitions), so only
    untouched objects are copied.  Returns the number of objects materialized.

    Fail-closed: a ref whose source file is missing raises ``FileNotFoundError``
    (the pre-R39 rglob loop silently dropped missing files, which could lose a
    partition silently; an immutable generation with a manifest inventory must be
    complete).
    """
    global generation_inventory_materialize_count
    src_root = Path(src_gen_dir)
    dst_root = Path(dst_gen_dir)
    skip = frozenset(skip_rel)
    count = 0
    for ref in src_inventory:
        if ref.rel_path in skip:
            continue
        src = src_root / ref.rel_path
        dst = dst_root / ref.rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(src, dst, hardlink=hardlink)
        count += 1
    generation_inventory_materialize_count += 1
    return count


def _link_or_copy(src: Path, dst: Path, *, hardlink: bool) -> None:
    """``os.link`` with ``shutil.copy2`` fallback on filesystems without links."""
    if hardlink:
        try:
            os.link(str(src), str(dst))
            return
        except OSError:
            shutil.copy2(str(src), str(dst))
            return
    shutil.copy2(str(src), str(dst))


# ---- JSON serialization for the manifest ----------------------------------
def inventory_to_dicts(
    inventory: Iterable[PartitionObjectRef],
) -> list[dict[str, Any]]:
    """Serialize an inventory to manifest-ready JSON dicts."""
    return [
        {
            "rel_path": r.rel_path,
            "content_id": r.content_id,
            "rows": r.rows,
            "schema_hash": r.schema_hash,
        }
        for r in inventory
    ]


def inventory_from_dicts(
    items: Iterable[dict[str, Any]],
) -> list[PartitionObjectRef]:
    """Parse manifest JSON dicts back into :class:`PartitionObjectRef`."""
    out: list[PartitionObjectRef] = []
    for item in items:
        out.append(
            PartitionObjectRef(
                rel_path=str(item["rel_path"]),
                content_id=str(item["content_id"]),
                rows=int(item["rows"]),
                schema_hash=str(item["schema_hash"]),
            )
        )
    return out


def inventory_parquet_paths(
    gen_dir: str | Path,
    inventory: Iterable[PartitionObjectRef],
    *,
    suffix: str,
) -> list[Path]:
    """Ordered file paths from an inventory matching ``suffix`` (read pushdown).

    Read path helper: when a manifest inventory is available, ``load_matrix``
    uses these exact paths instead of ``rglob``.  Staging/quarantine entries are
    never recorded in an inventory, but they are filtered defensively anyway.
    """
    root = Path(gen_dir)
    out: list[Path] = []
    for ref in inventory:
        if ref.rel_path.endswith(suffix):
            p = root / ref.rel_path
            if ".staging" in p.parts or ".quarantine" in p.parts:
                continue
            out.append(p)
    return sorted(out)
