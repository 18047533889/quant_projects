# -*- coding: utf-8 -*-
"""R39 opt-in immutable delta storage for factor partitions.

Layout (per hive partition dir)::

    <factor>/year=2024/
    ├── base/
    │   ├── base_00000.parquet        # immutable compacted base
    │   └── ...
    ├── delta/
    │   ├── gen_00000_w0.parquet      # immutable delta fragments
    │   └── ...
    ├── data.parquet                  # legacy monolithic (readable; becomes base on migrate/compact)
    └── manifest.json                 # atomic generation manifest

Semantics (spec §12/§15/§25):

- Normal incremental writes emit a *sorted* delta fragment of only the changed
  rows — never a full-history read→concat→dedup→sort→rewrite (PERF-043/083).
- Read merges base + deltas with ``(datetime, asset)`` dedup keep-latest
  (equivalent to ``row_number() over (partition by datetime, asset order by
  generation desc) = 1``).  Legacy monolithic partitions without a manifest are
  also readable (PERF-031 compatibility).
- The generation manifest flips once per batch for durability (PERF-052).
- Compaction (PERF-084) is triggered by ``delta_count`` / ``delta_bytes/base``
  ratio and produces one sorted new base; resource-controlled and never run
  inline on the hot path unless the caller opts in.
- ``PartitionLockManager`` reuses lock handles within a generation (PERF-054);
  orphan ``.tmp`` fragments are swept at startup recovery, not per write
  (PERF-053).

DEFAULT OFF: the materializer only uses this module when delta mode is enabled
(``FACTOR_ENGINE_DELTA_STORAGE=1`` or an explicit flag).  When OFF the legacy
monolithic ``data.parquet`` upsert path is used byte-identically.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# DeltaManifest
# ---------------------------------------------------------------------------


@dataclass
class DeltaManifest:
    """Atomic generation manifest mapping base + delta objects.

    ``seq`` is the monotonic next-delta sequence; each delta's ``gen`` column
    value equals its ``seq`` so the read path can order fragments.
    ``merged_rows``/``merged_valid_rows`` are running counters maintained
    incrementally (PERF-049) and reconciled exactly at compaction.
    """

    layout: str = "delta"
    seq: int = 0
    base: str = "data.parquet"
    base_rows: int = 0
    base_valid_rows: int = 0
    deltas: list[dict] = field(default_factory=list)
    compaction_generation: str | None = None
    merged_rows: int = 0
    merged_valid_rows: int = 0

    @staticmethod
    def manifest_path(partition_dir: Path) -> Path:
        return Path(partition_dir) / "manifest.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeltaManifest":
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in (data or {}).items() if k in known}
        return cls(**clean)

    @classmethod
    def load(cls, partition_dir: Path) -> "DeltaManifest | None":
        path = cls.manifest_path(partition_dir)
        try:
            if not path.exists():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(raw)
        except (OSError, ValueError):
            return None

    def save_atomic(self, partition_dir: Path) -> None:
        """Write manifest.json via tmp + fsync + os.replace + dir fsync."""
        path = self.manifest_path(partition_dir)
        tmp = path.with_name(f".manifest.{uuid.uuid4().hex[:12]}.tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(str(tmp), str(path))
        _fsync_dir(partition_dir)

    def delta_bytes(self) -> int:
        return sum(int(d.get("bytes", 0) or 0) for d in self.deltas)

    def base_bytes(self, partition_dir: Path) -> int:
        p = Path(partition_dir) / self.base
        try:
            return int(p.stat().st_size)
        except OSError:
            return 0

    def compaction_debt_bytes(self, partition_dir: Path) -> int:
        """Delta bytes awaiting compaction (PERF-084)."""
        return self.delta_bytes()


def _fsync_dir(path: Path) -> None:
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


# ---------------------------------------------------------------------------
# Delta fragment writes
# ---------------------------------------------------------------------------


def write_delta_fragment(
    partition_dir: Path,
    delta_df: pd.DataFrame,
    *,
    manifest: DeltaManifest,
    run_generation: str = "0",
    writer_id: str = "w0",
) -> dict:
    """Write a sorted immutable delta fragment and update the manifest atomically.

    ``delta_df`` must be the long-table frame (datetime/asset/value/metadata).
    Only changed rows are sorted and written — never the whole annual history
    (PERF-083).  Returns the delta object dict recorded in the manifest.
    """
    delta_dir = Path(partition_dir) / "delta"
    delta_dir.mkdir(parents=True, exist_ok=True)
    seq = int(manifest.seq)
    filename = f"gen_{seq:05d}_{writer_id}.parquet"
    rel = f"delta/{filename}"
    tmp = delta_dir / f".{filename}.{run_generation}.{uuid.uuid4().hex[:8]}.tmp"

    frag = delta_df.sort_values(["asset", "datetime"]).reset_index(drop=True)
    frag = frag.copy()
    frag["gen"] = int(seq)

    frag.to_parquet(tmp, index=False, engine="pyarrow")
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    final = Path(partition_dir) / rel
    os.replace(str(tmp), str(final))
    _fsync_dir(delta_dir)

    file_bytes = int(final.stat().st_size)
    valid_rows = (
        int((frag["is_valid"]) == 1).sum())
        if "is_valid" in frag.columns
        else int(len(frag))
    )
    entry = {
        "file": rel,
        "seq": seq,
        "rows": int(len(frag)),
        "valid_rows": valid_rows,
        "bytes": file_bytes,
        "run_generation": str(run_generation),
        "writer_id": str(writer_id),
    }
    manifest.deltas.append(entry)
    manifest.seq = seq + 1
    manifest.merged_rows += int(len(frag))
    manifest.merged_valid_rows += valid_rows
    manifest.save_atomic(partition_dir)
    return entry


# ---------------------------------------------------------------------------
# Read path: base + deltas merge (legacy monolithic also supported)
# ---------------------------------------------------------------------------


def read_delta_partition(partition_dir: Path) -> pd.DataFrame:
    """Merge base + deltas into a deduped, ``[asset, datetime]``-sorted long frame.

    Dedup is equivalent to ``row_number() over (partition by datetime, asset
    order by generation desc) = 1``.  A legacy monolithic partition (no
    manifest.json) is read directly from ``data.parquet``.
    """
    partition_dir = Path(partition_dir)
    manifest = DeltaManifest.load(partition_dir)
    if manifest is None:
        legacy = partition_dir / "data.parquet"
        if legacy.exists():
            return pd.read_parquet(legacy)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    base_path = partition_dir / manifest.base
    if base_path.exists():
        base = pd.read_parquet(base_path)
        if "gen" not in base.columns:
            base = base.copy()
            base["gen"] = 0
        frames.append(base)
    for entry in sorted(manifest.deltas, key=lambda d: int(d.get("seq", 0))):
        p = partition_dir / entry["file"]
        if p.exists():
            frag = pd.read_parquet(p)
            frames.append(frag)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "gen" in combined.columns:
        combined["gen"] = combined["gen"].astype("int64")
        combined = combined.sort_values(["gen"]).drop_duplicates(
            subset=["datetime", "asset"], keep="last"
        )
        combined = combined.drop(columns=["gen"])
    else:
        combined = combined.drop_duplicates(
            subset=["datetime", "asset"], keep="last"
        )
    combined = combined.sort_values(["asset", "datetime"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Compaction (PERF-084)
# ---------------------------------------------------------------------------


def should_compact(
    manifest: DeltaManifest | None,
    *,
    partition_dir: Path | None = None,
    max_delta_count: int = 8,
    max_delta_ratio: float = 0.5,
    max_read_amplification: int = 6,
) -> bool:
    """Threshold-based compaction trigger (PERF-084).

    Compact when: too many deltas, OR delta_bytes/base_bytes ratio exceeded, OR
    read-amplification (number of files the reader must open) exceeded.
    """
    if manifest is None:
        return False
    if len(manifest.deltas) >= int(max_delta_count):
        return True
    if partition_dir is not None:
        base_bytes = manifest.base_bytes(partition_dir)
        if base_bytes > 0:
            if (manifest.delta_bytes() / base_bytes) >= float(max_delta_ratio):
                return True
    # read-amplification: base + each delta is one file open
    if (1 + len(manifest.deltas) >= int(max_read_amplification):
        return True
    return False


def compact_partition(
    partition_dir: Path,
    *,
    run_generation: str = "compact",
    memory_budget_bytes: int | None = None,
) -> dict:
    """Full merge → one sorted new immutable base (PERF-083/084).

    The merged view (base + deltas, deduped) becomes ``base/base_<seq>.parquet``;
    the manifest flips once (durable) and old delta fragments are removed only
    after the flip.  ``memory_budget_bytes`` is accepted for resource control —
    the caller decides when/where to run; this routine never runs inline on the
    production write path by default.
    """
    partition_dir = Path(partition_dir)
    manifest = DeltaManifest.load(partition_dir)
    if manifest is None:
        return {"compacted": False, "reason": "no_delta_manifest"}

    merged = read_delta_partition(partition_dir)
    if merged.empty:
        manifest.deltas = []
        manifest.merged_rows = 0
        manifest.merged_valid_rows = 0
        manifest.save_atomic(partition_dir)
        return {
            "compacted": True,
            "rows": 0,
            "new_base": manifest.base,
            "removed_deltas": [],
        }

    merged = merged.sort_values(["asset", "datetime"]).reset_index(drop=True)

    base_dir = partition_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    new_gen = int(manifest.seq)
    base_name = f"base/base_{new_gen:05d}.parquet"
    tmp = base_dir / f".base_{new_gen:05d}.{run_generation}.{uuid.uuid4().hex[:8]}.tmp"

    merged.to_parquet(tmp, index=False, engine="pyarrow")
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    final = partition_dir / base_name
    os.replace(str(tmp), str(final))
    _fsync_dir(base_dir)

    removed = [d["file"] for d in manifest.deltas]
    manifest.base = base_name
    manifest.base_rows = int(len(merged))
    manifest.base_valid_rows = (
        int((merged["is_valid"]) == 1).sum())
        if "is_valid" in merged.columns
        else int(len(merged))
    )
    manifest.deltas = []
    manifest.compaction_generation = str(new_gen)
    manifest.merged_rows = int(len(merged))
    manifest.merged_valid_rows = manifest.base_valid_rows
    manifest.save_atomic(partition_dir)

    # Remove old deltas only AFTER the manifest flip (crash-safe; any leftovers
    # are swept by generation-recovery at startup).
    for rel in removed:
        try:
            (partition_dir / rel).unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "compacted": True,
        "rows": int(len(merged)),
        "new_base": base_name,
        "removed_deltas": removed,
        "memory_budget_bytes": memory_budget_bytes,
    }


# ---------------------------------------------------------------------------
# Migration (spec §31.2)
# ---------------------------------------------------------------------------


def migrate_factor_partition_to_delta(partition_dir: Path) -> dict:
    """Migrate a legacy monolithic partition to delta layout (idempotent).

    Reads the legacy ``data.parquet`` once, writes it as immutable
    ``base/base_00000.parquet``, then flips the manifest atomically.  The legacy
    file is left in place so a crash before the flip keeps the partition fully
    readable via the legacy path.
    """
    partition_dir = Path(partition_dir)
    legacy = partition_dir / "data.parquet"
    if DeltaManifest.load(partition_dir) is not None:
        return {"migrated": False, "reason": "already_delta"}
    if not legacy.exists():
        return {"migrated": False, "reason": "no_legacy_data"}

    df = pd.read_parquet(legacy)
    base_dir = partition_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_name = "base/base_00000.parquet"
    tmp = base_dir / f".base_00000.{uuid.uuid4().hex[:8]}.tmp"
    df.to_parquet(tmp, index=False, engine="pyarrow")
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    final = partition_dir / base_name
    os.replace(str(tmp), str(final))
    _fsync_dir(base_dir)

    valid_rows = (
        int((df["is_valid"]) == 1).sum()) if "is_valid" in df.columns else int(len(df))
    )
    manifest = DeltaManifest(
        layout="delta",
        seq=0,
        base=base_name,
        base_rows=int(len(df)),
        base_valid_rows=valid_rows,
        deltas=[],
        merged_rows=int(len(df)),
        merged_valid_rows=valid_rows,
    )
    manifest.save_atomic(partition_dir)
    return {"migrated": True, "base": base_name, "rows": int(len(df))}


# ---------------------------------------------------------------------------
# Generation recovery (PERF-053) + lock reuse (PERF-054)
# ---------------------------------------------------------------------------


def recover_orphan_delta_tmp_files(partition_dir: Path) -> int:
    """Startup/recovery sweep of leftover ``.tmp`` delta/base fragments.

    Run once per factor partition at startup — NOT per-partition-write
    (PERF-053).  Returns the number of files removed.
    """
    partition_dir = Path(partition_dir)
    removed = 0
    for sub in ("delta", "base"):
        d = partition_dir / sub
        if not d.exists():
            continue
        for tmp in d.glob(".*.tmp"):
            try:
                tmp.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    return removed


class _ReuseLock:
    """flock wrapper that keeps the file handle open across re-acquisitions."""

    __slots__ = ("_fh",)

    def __init__(self, fh: Any) -> None:
        self._fh = fh

    def __enter__(self) -> "_ReuseLock":
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        return self

    def __exit__(self, *exc: Any) -> bool:
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        return False


class PartitionLockManager:
    """Reuses per-partition lock handles within a generation (PERF-054).

    One handle per partition dir is opened lazily and reused for every delta
    write in the batch; ``close()`` releases all handles at generation end.
    External concurrency semantics are unchanged (still a real ``flock``).
    """

    def __init__(self) -> None:
        self._handles: dict[str, Any] = {}

    def partition_lock(self, partition_dir: Path) -> _ReuseLock:
        key = str(Path(partition_dir))
        fh = self._handles.get(key)
        if fh is None:
            lock_path = Path(partition_dir) / ".lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(str(lock_path), "a+")  # noqa: SIM115 - handle reused
            self._handles[key] = fh
        return _ReuseLock(fh)

    def close(self) -> None:
        for fh in self._handles.values():
            try:
                fh.close()
            except OSError:
                pass
        self._handles.clear()


__all__ = [
    "DeltaManifest",
    "PartitionLockManager",
    "compact_partition",
    "migrate_factor_partition_to_delta",
    "read_delta_partition",
    "recover_orphan_delta_tmp_files",
    "should_compact",
    "write_delta_fragment",
]
