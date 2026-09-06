"""Canonical factor-matrix discovery and full-window coverage checks.

Reports must never infer availability from the directory a file happened to be
written into.  Every report builder uses this module to select a *raw* matrix
and records the selected source in its manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np


PROJECT = Path("/home/sunhaiwei/quant_projects")
FULL_WINDOW_START = pd.Timestamp("2016-01-04")
FULL_WINDOW_END = pd.Timestamp("2026-08-27")

# The first directory is canonical.  The remaining directories are legacy
# landing locations.  They are accepted only after an actual date-coverage
# check; their names (especially *_2026daily) are not evidence of coverage.
RAW_MATRIX_DIRS = (
    "factor_matrices_all",
    "factor_matrices_all_2026daily",
    "factor_matrices_all_2026r2",
    "factor_matrices_all_2026daily_r2",
)


@dataclass(frozen=True)
class MatrixSource:
    page: str
    path: Path | None
    source: str | None
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    valid_start: pd.Timestamp | None
    valid_end: pd.Timestamp | None
    valid_days: int
    is_full_window: bool
    reason: str | None = None


def _paths(page: str, directories: Iterable[str] = RAW_MATRIX_DIRS):
    base = PROJECT / "weekly_backtest_output"
    for directory in directories:
        for name in (f"{page}.parquet", f"factor_{page}.parquet"):
            path = base / directory / name
            if path.exists():
                yield directory, path


def _coverage(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None,
                                  pd.Timestamp | None, pd.Timestamp | None, int]:
    """Return index and *usable value* coverage.

    A parquet file can have the complete date index while every later row is
    NaN (a legacy training-only landing).  Such a file is not a full-window
    matrix and must never win source selection.
    """
    frame = pd.read_parquet(path)
    index = pd.to_datetime(frame.index, errors="coerce")
    if len(index) == 0 or index.isna().any() or index.has_duplicates:
        return None, None, None, None, 0
    valid = np.isfinite(frame.to_numpy(dtype=float)).sum(axis=1) >= 30
    valid_index = index[valid]
    if len(valid_index) == 0:
        return index.min(), index.max(), None, None, 0
    return index.min(), index.max(), valid_index.min(), valid_index.max(), int(len(valid_index))


def resolve_raw_matrix(page: str) -> MatrixSource:
    """Return the best raw-matrix candidate, preferring verified full coverage."""
    candidates: list[MatrixSource] = []
    for directory, path in _paths(page):
        try:
            start, end, valid_start, valid_end, valid_days = _coverage(path)
        except Exception as exc:
            candidates.append(MatrixSource(page, path, directory, None, None, None, None, 0, False,
                                           f"cannot read parquet index: {type(exc).__name__}"))
            continue
        # Allow only a brief warm-up before the first usable cross-section,
        # but require usable values through the last requested date.
        full = bool(valid_start is not None and valid_end is not None
                    and start <= FULL_WINDOW_START
                    and valid_start <= FULL_WINDOW_START
                    # The t+2 return convention makes the final two to three
                    # signal dates unusable by design; this is not a landing
                    # gap.  Anything older than one calendar week is.
                    and valid_end >= FULL_WINDOW_END - pd.Timedelta(days=7)
                    and valid_days >= 1800)
        reason = None if full else (
            f"requires usable history from {FULL_WINDOW_START.date()}; "
            f"index={start}..{end}, usable={valid_start}..{valid_end}, "
            f"usable_days={valid_days}")
        candidates.append(MatrixSource(page, path, directory, start, end,
                                       valid_start, valid_end, valid_days, full, reason))
    if not candidates:
        return MatrixSource(page, None, None, None, None, None, None, 0, False, "raw matrix not found")
    # Prefer the most complete usable history; directory order is only a tie
    # breaker.  This selects a later full backfill over a stale training copy.
    candidates.sort(key=lambda c: (c.is_full_window, c.valid_days,
                                   c.valid_end or pd.Timestamp.min), reverse=True)
    return candidates[0]


def load_raw_full_window(page: str) -> tuple[pd.DataFrame | None, MatrixSource]:
    source = resolve_raw_matrix(page)
    if source.path is None or not source.is_full_window:
        return None, source
    matrix = pd.read_parquet(source.path)
    matrix.index = pd.to_datetime(matrix.index)
    return matrix.sort_index(), source
