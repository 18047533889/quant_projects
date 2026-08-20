# -*- coding: utf-8 -*-
"""FactorMatrix R39 column-factor block layout + checksum staging proof.

Owned by R39:
  * PERF-060  — ``build_matrix_block``: replace N-way per-factor pandas outer merge
                with (a) a zero-join direct column-stack when every factor series
                shares the identical axis, or (b) a single canonical row-key index
                + vectorized per-factor reindex when axes differ.  A module level
                ``matrix_join_count`` counter is 0 when all axes are equal.
  * PERF-061  — column-factor block physical layout: ``year=YYYY/month=MM/
                block=NNNN.parquet``; manifest ``factor_id -> {block_id, column}``.
  * PERF-062  — incremental updates never read-merge-rewrite the whole month wide
                table; only touched blocks get new fragments (``_overlay_block``),
                new factors get fresh blocks, untouched blocks are copy-on-write
                hardlinked.  ``matrix_rewrite_amplification`` metric reported.
  * PERF-064  — checksum-based staging proof: writer computes ``key_order_checksum``,
                per-column ``finite_mask_checksum`` / ``numeric_checksum``,
                ``row_count``, ``schema_hash``; read-back uses Arrow (footer/schema
                + row groups) and compares checksums — no pandas full read+sort of
                both matrices + per-column ``np.allclose``.
  * PERF-065  — ``load_matrix`` pushdown helpers (partition range pruning,
                block selection, ``columns=`` pushdown).

Default path (legacy single-wide-file) stays byte-identical to the pre-R39 writer;
the block layout is opt-in via ``FACTOR_ENGINE_MATRIX_BLOCK_LAYOUT=1``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LONG_KEY_COLUMNS = ("datetime", "asset")

# ---- performance counters -------------------------------------------------
matrix_join_count = 0
_matrix_rewrite_amplification_bytes = 0
_matrix_changed_logical_bytes = 0
matrix_rewrite_amplification = 0.0


def reset_perf_counters() -> None:
    """Reset the R39 matrix performance counters (tests / benchmarks)."""
    global matrix_join_count, matrix_rewrite_amplification
    global _matrix_rewrite_amplification_bytes, _matrix_changed_logical_bytes
    matrix_join_count = 0
    matrix_rewrite_amplification = 0.0
    _matrix_rewrite_amplification_bytes = 0
    _matrix_changed_logical_bytes = 0


def record_rewrite_amplification(rewritten_bytes: int, changed_logical_bytes: int) -> None:
    """Accumulate write-amplification evidence for the current run.

    ``rewritten_bytes``  = bytes of *pre-existing* block files that had to be
    rewritten (historical rewrite bytes).  ``changed_logical_bytes`` = bytes of the
    new logical data written this run.
    """
    global matrix_rewrite_amplification
    global _matrix_rewrite_amplification_bytes, _matrix_changed_logical_bytes
    _matrix_rewrite_amplification_bytes += int(rewritten_bytes)
    _matrix_changed_logical_bytes += int(changed_logical_bytes)
    matrix_rewrite_amplification = _matrix_rewrite_amplification_bytes / max(
        _matrix_changed_logical_bytes, 1
    )


def block_columns() -> int:
    """Columns per factor-column block (env-tunable, default 256)."""
    try:
        return max(16, int(os.environ.get("FACTOR_ENGINE_MATRIX_COLUMNS_PER_BLOCK", "256")))
    except ValueError:
        return 256


def block_mode_enabled() -> bool:
    """Opt-in column-factor block layout (default OFF -> legacy single wide file)."""
    return os.environ.get("FACTOR_ENGINE_MATRIX_BLOCK_LAYOUT", "0") == "1"


def checksum_proof_enabled() -> bool:
    """Opt-in checksum staging proof (default OFF -> legacy value-compare)."""
    return os.environ.get("FACTOR_ENGINE_MATRIX_CHECKSUM_PROOF", "0") == "1"


def block_file_name(block_id: str) -> str:
    """Parquet file name for a column-factor block."""
    return f"block={int(block_id):04d}.parquet"


def parse_block_id(filename: str) -> str | None:
    """Parse ``block=NNNN.parquet`` -> ``NNNN`` (or None if not a block file)."""
    if not filename.startswith("block=") or not filename.endswith(".parquet"):
        return None
    raw = filename[len("block="): -len(".parquet")]
    try:
        int(raw)
    except ValueError:
        return None
    return raw


# ---- PERF-060: block assembly ---------------------------------------------
def _hash_sorted_levels(idx: pd.MultiIndex) -> bytes:
    """Order-independent lightweight digest of the two axis levels.

    ``datetime`` level is sorted as int64 nanoseconds; ``asset`` is label-encoded
    with ``pd.factorize(..., sort=True)`` (order-independent codes) then sorted.
    All operations are C-level; no Python-object tuple construction per row.
    """
    dt = idx.get_level_values(0)
    ast = idx.get_level_values(1)
    dt_sorted = np.sort(np.asarray(dt.values).astype("datetime64[ns]").view("int64")).tobytes()
    ast_codes, _ = pd.factorize(np.asarray(ast, dtype=object), sort=True)
    ast_sorted = np.sort(np.asarray(ast_codes, dtype=np.int64)).tobytes()
    h = hashlib.blake2b(digest_size=8)
    h.update(dt_sorted)
    h.update(ast_sorted)
    return h.digest()


def axis_key(series: pd.Series) -> tuple[Any, ...]:
    """Lightweight axis identity key for a factor Series.

    Used by PERF-060 to decide the direct column-stack vs canonical-reindex path.
    The comparison itself short-circuits via ``Index is`` / ``Index.equals`` so the
    common engine case (all results share one axis object) is O(1) per factor.
    """
    idx = series.index
    return (
        len(idx),
        idx.nlevels,
        tuple(idx.names),
        str(idx.get_level_values(0).dtype),
        str(idx.get_level_values(1).dtype),
        _hash_sorted_levels(idx),
    )


def axes_equal(factors: dict[str, pd.Series]) -> bool:
    """True when every factor Series shares the identical row axis.

    Uses O(1) object-identity short-circuit first, then pandas ``Index.equals``
    (C-level label compare) as the correctness fallback.
    """
    if not factors:
        return True
    ref = next(iter(factors.values()))
    ref_idx = ref.index
    return all(
        s.index is ref_idx or s.index.equals(ref_idx) for s in factors.values()
    )


def build_matrix_block(
    factors: dict[str, pd.Series],
) -> tuple[pd.MultiIndex, pd.DataFrame, int]:
    """Assemble a column block of factors WITHOUT an N-way pandas outer merge.

    Args:
        factors: ``fid -> MultiIndex(timestamp, instrument) Series`` (float64).

    Returns:
        ``(axis, wide, joins)``:
          * ``axis`` — shared MultiIndex row axis (equal-axis path) or the canonical
            union axis (different-axis path);
          * ``wide``  — ``DataFrame`` indexed by ``axis``, columns ``fids``, values
            float64 (missing keys -> NaN);
          * ``joins`` — number of joins performed: 0 when all axes are equal
            (pure column stack), 1 for the canonical-reindex path.
    """
    global matrix_join_count
    if not factors:
        raise ValueError("build_matrix_block: factors 不能为空")
    fids = list(factors)
    if axes_equal(factors):
        ref_idx = factors[fids[0]].index
        data = {f: factors[f].to_numpy(dtype="float64") for f in fids}
        wide = pd.DataFrame(data, index=ref_idx)
        joins = 0
    else:
        # ONE canonical row-key index + vectorized per-factor reindex: pandas
        # concat(axis=1) builds a single union index and reindexes each column.
        wide = pd.concat({f: factors[f] for f in fids}, axis=1).astype("float64")
        joins = 1
    matrix_join_count += joins
    return wide.index, wide, joins


def wide_to_merged(
    wide: pd.DataFrame,
    factor_ids: Iterable[str],
    *,
    value_dtype: str = "float32",
) -> pd.DataFrame:
    """Convert a wide column block (MultiIndex rows, factor columns) to the legacy
    long-ish matrix frame: ``[datetime, asset, fid1, ...]`` sorted by key.

    Mirrors the exact schema/dtype/row-order contract the pre-R39 pairwise-merge
    writer produced (datetime64[ns], asset as pandas ``string``, factor columns
    cast to ``value_dtype``, rows sorted by ``[datetime, asset]``).
    """
    fids = [str(f) for f in factor_ids]
    merged = wide.reset_index()
    merged = merged.rename(
        columns={merged.columns[0]: LONG_KEY_COLUMNS[0], merged.columns[1]: LONG_KEY_COLUMNS[1]}
    )
    merged[LONG_KEY_COLUMNS[0]] = pd.to_datetime(merged[LONG_KEY_COLUMNS[0]])
    merged[LONG_KEY_COLUMNS[1]] = merged[LONG_KEY_COLUMNS[1]].astype("string")
    for fid in fids:
        merged[fid] = merged[fid].astype(str(value_dtype or "float32"))
    return (
        merged[[LONG_KEY_COLUMNS[0], LONG_KEY_COLUMNS[1], *fids]]
        .sort_values(list(LONG_KEY_COLUMNS))
        .reset_index(drop=True)
    )


def build_merged_from_series(
    factors: dict[str, pd.Series],
    *,
    value_dtype: str = "float32",
) -> pd.DataFrame:
    """One-shot helper: assemble all factors into the legacy matrix frame."""
    axis, wide, _joins = build_matrix_block(factors)
    fids = list(factors)
    return wide_to_merged(wide, fids, value_dtype=value_dtype)


# ---- PERF-062: block-level overlay (no whole-month read-merge-rewrite) -----
def overlay_block(
    existing: pd.DataFrame | None,
    new: pd.DataFrame,
    *,
    value_dtype: str,
    key_cols: tuple[str, str] = LONG_KEY_COLUMNS,
) -> pd.DataFrame:
    """Overlay ``new`` factor columns onto an existing column-factor block.

    Bypasses ``_merge_matrix_frames`` (the whole-month wide-table read-merge-rewrite
    path): this operates on a single 256-column block, preserving untouched columns
    and untouched rows, and letting ``new`` values win (including explicit
    NaN tombstones) for the touched factor columns.  Semantics identical to
    ``_merge_matrix_frames`` — one outer merge per block, never N pairwise merges.
    """
    if existing is None or existing.empty:
        return new
    if new is None or new.empty:
        return existing
    k0, k1 = key_cols
    existing = existing.copy()
    new = new.copy()
    existing[k0] = pd.to_datetime(existing[k0])
    new[k0] = pd.to_datetime(new[k0])
    existing[k1] = existing[k1].astype("string")
    new[k1] = new[k1].astype("string")
    existing = existing.drop_duplicates(subset=[k0, k1], keep="first")
    new = new.drop_duplicates(subset=[k0, k1], keep="first")
    merged = existing.merge(
        new, on=[k0, k1], how="outer", suffixes=("_old", ""), indicator="_fe_src"
    )
    new_cols = set(new.columns) - {k0, k1}
    src = merged["_fe_src"]
    drop: list[str] = ["_fe_src"]
    for c in sorted(new_cols):
        old_c = f"{c}_old"
        if old_c not in merged.columns:
            continue
        only_old = src == "left_only"
        if only_old.any():
            merged.loc[only_old, c] = merged.loc[only_old, old_c]
        drop.append(old_c)
    if drop:
        merged = merged.drop(columns=drop)
    merged = merged.drop_duplicates(subset=[k0, k1], keep="last")
    for c in sorted(new_cols):
        if c in merged.columns:
            merged[c] = merged[c].astype(str(value_dtype or "float32"))
    cols = [c for c in (k0, k1) if c in merged.columns] + sorted(
        c for c in merged.columns if c not in (k0, k1)
    )
    return merged[cols].sort_values([k0, k1]).reset_index(drop=True)


# ---- PERF-064: checksum staging proof -------------------------------------
def _hash_bytes(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=16).hexdigest()


def _numeric_checksums(series: pd.Series) -> dict[str, str]:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(v)
    hf = hashlib.blake2b(digest_size=16)
    hf.update(np.ascontiguousarray(finite, dtype=np.uint8).tobytes())
    hnum = hashlib.blake2b(digest_size=16)
    hnum.update(
        np.ascontiguousarray(np.where(finite, v, 0.0), dtype="float64").tobytes()
    )
    return {
        "finite_mask_checksum": hf.hexdigest(),
        "numeric_checksum": hnum.hexdigest(),
    }


def compute_matrix_checksums(
    df: pd.DataFrame,
    key_cols: tuple[str, str] = LONG_KEY_COLUMNS,
) -> dict[str, Any]:
    """Compute write-time checksums over a matrix frame.

    Returns:
        ``{"row_count", "schema_hash", "key_order_checksum",
        "columns": {fid: {"finite_mask_checksum", "numeric_checksum"}}}``

    ``key_order_checksum`` is order-sensitive (proves row order); per-column
    checksums prove finite-mask (NaN/Inf positions) and numeric values.  NaN bytes
    are normalised to ``0.0`` before hashing so the checksum is stable across
    parquet round-trips regardless of NaN bit-pattern variance.
    """
    k0, k1 = key_cols
    cols = list(df.columns)
    dt = pd.to_datetime(df[k0]).values.astype("datetime64[ns]").view("int64")
    ast = df[k1].astype("string")
    codes, _ = pd.factorize(np.asarray(ast, dtype=object), sort=True)
    hkey = hashlib.blake2b(digest_size=16)
    hkey.update(np.ascontiguousarray(dt).tobytes())
    hkey.update(np.ascontiguousarray(codes, dtype=np.int64).tobytes())
    per_col: dict[str, dict[str, str]] = {}
    for c in cols:
        if c in key_cols:
            continue
        per_col[c] = _numeric_checksums(df[c])
    schema_hash = _hash_bytes(
        json.dumps(
            {c: str(df[c].dtype) for c in cols}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return {
        "row_count": int(len(df)),
        "schema_hash": schema_hash,
        "key_order_checksum": hkey.hexdigest(),
        "columns": per_col,
    }


def read_parquet_checksums(
    path: str | Path,
    expected_columns: Iterable[str],
    key_cols: tuple[str, str] = LONG_KEY_COLUMNS,
) -> dict[str, Any]:
    """Arrow-backed read-back checksums for a parquet matrix block.

    Uses ``pyarrow.parquet.ParquetFile`` footer/schema + ``read`` of the requested
    columns, then computes the same checksums as ``compute_matrix_checksums``.
    No pandas ``sort_values`` of both matrices, no per-column ``np.allclose`` loop.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    cols_needed = [c for c in expected_columns if c in pf.schema_arrow.names]
    table = pf.read(columns=cols_needed)
    k0, k1 = key_cols
    # Build a minimal frame (Arrow columnar -> numpy) without sorting.
    data: dict[str, Any] = {}
    if k0 in cols_needed:
        data[k0] = pd.to_datetime(table.column(k0).to_pandas())
    if k1 in cols_needed:
        data[k1] = table.column(k1).to_pandas().astype("string")
    for c in cols_needed:
        if c in (k0, k1):
            continue
        data[c] = table.column(c).to_pandas()
    frame = pd.DataFrame(data)
    # The reader may not request key columns via columns= pushdown in every path;
    # re-read them if needed for the checksum.
    missing = [c for c in (k0, k1) if c not in frame.columns]
    if missing:
        extra = pf.read(columns=missing)
        for c in missing:
            frame[c] = extra.column(c).to_pandas()
    return compute_matrix_checksums(frame, key_cols)


def compare_checksums(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Return a list of human-readable mismatch messages (empty == OK)."""
    errors: list[str] = []
    if expected.get("row_count") != actual.get("row_count"):
        errors.append(
            f"row_count {actual.get('row_count')} != 期望 {expected.get('row_count')}"
        )
    if expected.get("schema_hash") != actual.get("schema_hash"):
        errors.append("schema_hash 不一致")
    if expected.get("key_order_checksum") != actual.get("key_order_checksum"):
        errors.append("key_order_checksum 不一致")
    exp_cols = expected.get("columns", {})
    act_cols = actual.get("columns", {})
    if set(exp_cols) != set(act_cols):
        errors.append(
            f"列集合不一致 {sorted(act_cols)} != {sorted(exp_cols)}"
        )
    for c in sorted(exp_cols):
        if c not in act_cols:
            errors.append(f"读回缺少列 {c}")
            continue
        if exp_cols[c]["finite_mask_checksum"] != act_cols[c]["finite_mask_checksum"]:
            errors.append(f"列 {c} finite_mask_checksum 不一致")
        if exp_cols[c]["numeric_checksum"] != act_cols[c]["numeric_checksum"]:
            errors.append(f"列 {c} numeric_checksum 不一致")
    return errors


# ---- PERF-065: block layout read helpers ----------------------------------
def _partition_range(
    path: Path,
) -> dict[str, int]:
    """Parse ``year=YYYY/month=MM`` segments from a directory path."""
    out: dict[str, int] = {}
    for part in path.parts:
        if "=" in part:
            k, v = part.split("=", 1)
            if k in {"year", "month", "day"}:
                try:
                    out[k] = int(v)
                except ValueError:
                    pass
    return out


def partition_overlaps_time_range(
    part_path: Path,
    time_range: tuple[Any, Any] | None,
) -> bool:
    """Prune a hive partition dir by ``(start, end)`` timestamps (inclusive)."""
    if time_range is None:
        return True
    start, end = time_range
    pv = _partition_range(part_path)
    year = pv.get("year")
    if year is None:
        return True
    if "month" in pv:
        month_start = pd.Timestamp(year=year, month=pv["month"], day=1)
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
    elif "day" in pv:
        month_start = pd.Timestamp(year=year, month=1, day=1)
        month_end = pd.Timestamp(
            year=year, month=pv.get("month", 12), day=pv.get("day", 1)
        )
        month_end = (month_end + pd.offsets.Day(1)) - pd.Timedelta(days=1)
    else:
        month_start = pd.Timestamp(year=year, month=1, day=1)
        month_end = pd.Timestamp(year=year, month=12, day=31)
    if start is not None and pd.Timestamp(start) > month_end:
        return False
    if end is not None and pd.Timestamp(end) < month_start:
        return False
    return True


def resolve_factor_blocks(
    manifest: dict[str, Any] | None,
    factor_ids: Iterable[str],
) -> dict[str, str]:
    """Resolve ``fid -> block_id`` from the manifest (existing factors only)."""
    factors = (manifest or {}).get("factors", {}) or {}
    out: dict[str, str] = {}
    for fid in factor_ids:
        entry = factors.get(str(fid)) or {}
        bid = entry.get("block_id")
        if bid is not None:
            out[str(fid)] = str(bid)
    return out
