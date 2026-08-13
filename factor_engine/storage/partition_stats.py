# -*- coding: utf-8 -*-
"""R39 partition commit stats + single-pass write-pass DQ stats + footer row counting.

- ``PartitionCommitStats`` (PERF-049) — returned by each partition commit so the
  normal write path never needs a full-history value-cell re-scan to compute
  watermark / manifest aggregate metrics.
- ``WritePassDQStats`` + ``compute_write_pass_dq`` (PERF-059) — DQ statistics
  computed *during* the emission/write pass (single pass over the array/block),
  not by a separate post-write DataFrame re-scan.  The existing
  ``evaluate_factor_dq`` gate still runs untouched.
- ``count_parquet_rows_from_footer`` (PERF-049) — counts rows via Parquet footer
  metadata (pyarrow ``ParquetFile.metadata``), never loading value cells.  Used
  for legacy partitions that predate incremental stats.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WritePassDQStats:
    """Single-pass DQ statistics collected during the write pass (PERF-059)."""

    row_count: int = 0
    finite_count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    min_value: float | None = None
    max_value: float | None = None
    checksum: str = ""
    date_min: str | None = None
    date_max: str | None = None
    duplicate_key_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "finite_count": self.finite_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "checksum": self.checksum,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "duplicate_key_count": self.duplicate_key_count,
        }


@dataclass(frozen=True)
class PartitionCommitStats:
    """Per-partition commit accounting (R39 PERF-049).

    ``rows_added`` counts new ``[datetime, asset]`` keys introduced by this
    commit; ``rows_replaced`` counts existing keys overwritten by this commit.
    ``valid_*`` count rows with ``is_valid == 1``.
    """

    partition_key: str
    rows_before: int
    rows_after: int
    rows_added: int
    rows_replaced: int
    valid_before: int
    valid_after: int
    min_date: str | None
    max_date: str | None
    file_bytes: int
    dq_stats: WritePassDQStats | None = None


def compute_write_pass_dq(values: Any, index: Any = None) -> WritePassDQStats:
    """Single-pass DQ statistics over a value array (PERF-059).

    ``values`` may be a numpy array / pandas Series / Arrow array (converted to
    float64).  ``index`` is an optional pandas Index / MultiIndex used to derive
    ``date_min``/``date_max`` and the duplicate-key count.  This is deliberately
    a lightweight single pass over the already-materialized buffer — the caller
    threads the result into the writer + ``PartitionCommitStats``.
    """
    arr = np.asarray(values, dtype=float).ravel()
    n = int(arr.size)
    if n == 0:
        return WritePassDQStats(row_count=0)
    finite = np.isfinite(arr)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    finite_count = int(finite.sum())
    if finite_count:
        fv = np.ascontiguousarray(arr[finite])
        min_value = float(np.min(fv))
        max_value = float(np.max(fv))
        checksum = hashlib.sha256(fv.tobytes()).hexdigest()
    else:
        min_value = None
        max_value = None
        checksum = ""

    date_min: str | None = None
    date_max: str | None = None
    dup_keys = 0
    if index is not None:
        if isinstance(index, pd.MultiIndex):
            ts_level = index.get_level_values(0)
            dup_keys = int(index.duplicated().sum())
        else:
            ts_level = pd.Index(index)
            dup_keys = 0
        if len(ts_level):
            dt = pd.to_datetime(ts_level)
            date_min = str(dt.min().isoformat())
            date_max = str(dt.max().isoformat())
    return WritePassDQStats(
        row_count=n,
        finite_count=finite_count,
        nan_count=nan_count,
        inf_count=inf_count,
        min_value=min_value,
        max_value=max_value,
        checksum=checksum,
        date_min=date_min,
        date_max=date_max,
        duplicate_key_count=dup_keys,
    )


def compute_partition_commit_stats(
    partition_key: str,
    *,
    existing_df: pd.DataFrame | None,
    new_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    file_bytes: int,
    dq_stats: WritePassDQStats | None = None,
) -> PartitionCommitStats:
    """Derive ``PartitionCommitStats`` from frames the writer already holds.

    No disk re-read: ``existing_df`` and ``combined_df`` are the frames already
    in memory during the merge.  This is what lets the normal write path avoid a
    full-history value-cell re-scan for watermark / metrics (PERF-049).
    """
    rows_before = int(len(existing_df)) if existing_df is not None else 0
    rows_after = int(len(combined_df))
    if existing_df is None or len(existing_df) == 0:
        rows_added = int(len(new_df))
        rows_replaced = 0
    else:
        new_unique = new_df[["datetime", "asset"]].drop_duplicates()
        existing_unique = existing_df[["datetime", "asset"]].drop_duplicates()
        rows_replaced = int(
            len(
                pd.merge(
                    new_unique,
                    existing_unique,
                    on=["datetime", "asset"],
                    how="inner",
                )
            )
        )
        rows_added = int(len(new_unique)) - rows_replaced

    valid_before = 0
    if existing_df is not None and "is_valid" in existing_df.columns:
        valid_before = int((existing_df["is_valid"]) == 1).sum())
    valid_after = (
        int((combined_df["is_valid"]) == 1).sum())
        if "is_valid" in combined_df.columns
        else rows_after
    )

    min_date: str | None = None
    max_date: str | None = None
    if len(combined_df):
        dt = pd.to_datetime(combined_df["datetime"])
        min_date = str(dt.min().isoformat())
        max_date = str(dt.max().isoformat())

    return PartitionCommitStats(
        partition_key=partition_key,
        rows_before=rows_before,
        rows_after=rows_after,
        rows_added=rows_added,
        rows_replaced=rows_replaced,
        valid_before=valid_before,
        valid_after=valid_after,
        min_date=min_date,
        max_date=max_date,
        file_bytes=int(file_bytes),
        dq_stats=dq_stats,
    )


def count_parquet_rows_from_footer(path: str | Path) -> int:
    """Count rows of a parquet file or a directory of ``*.parquet`` files using
    footer metadata only (PERF-049).  Never loads value cells.

    Falls back to 0 on any read error (callers treat this as "unknown").
    """
    import pyarrow.parquet as pq

    p = Path(path)
    if p.is_dir():
        total = 0
        for f in sorted(p.glob("*.parquet")):
            try:
                total += int(pq.ParquetFile(str(f)).metadata.num_rows)
            except Exception:
                continue
        return total
    if p.is_file():
        try:
            return int(pq.ParquetFile(str(p)).metadata.num_rows)
        except Exception:
            return 0
    return 0
