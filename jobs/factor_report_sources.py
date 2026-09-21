"""Canonical factor-matrix discovery and full-window coverage checks.

Reports must never infer availability from the directory a file happened to be
written into.  Every report builder uses this module to select a *raw* matrix
and records the selected source in its manifest.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np


PROJECT = Path("/home/sunhaiwei/quant_projects")
FULL_WINDOW_START = pd.Timestamp("2016-01-04")
FULL_WINDOW_END = pd.Timestamp("2026-08-27")

# There is **no warm-up requirement on the left edge at all**.  A factor is
# judged from its own first usable value onwards: whatever lookback a windowed
# operator needs (``ts_delta(x, 1)`` needs one bar, ``ts_std(x, 60)`` needs
# sixty) simply delays the first usable cross-section, and that delay is never
# a defect.  The earlier rule required a value on FULL_WINDOW_START itself —
# the very first day of the panel — which is impossible for any factor with a
# lookback, and it mislabelled 26 candidates of the 2026-09-16 new-mining batch
# as "unavailable".  A fixed allowance (260d) was a stopgap; this replaces it
# with the actual rule: no left-edge requirement whatsoever.
#
# Two things still make a matrix unusable, and neither is about warm-up:
#   * it stops early  — a genuine landing gap.  The t+2 return convention makes
#     the final sessions unusable by design, so one calendar week of slack.
#   * too few usable sessions — the RankIC would be noise.  This is a
#     statistical floor, not a warm-up bound, and the default is deliberately
#     low (about one quarter).  Measured on the 2026-09-16 new-mining batch:
#     every one of the 83 matrices this pipeline landed itself has 2363-2588
#     usable sessions, so the floor blocks nothing real and moving it from 250
#     to 60 changed the verdict of exactly zero factors.  It exists only to
#     stop a landing that produced almost nothing from being evaluated as if it
#     had.  Set FACTOR_REPORT_MIN_USABLE_DAYS=0 to disable it entirely and
#     accept any factor with a usable value through the end of the window.
MIN_USABLE_DAYS = int(os.environ.get("FACTOR_REPORT_MIN_USABLE_DAYS", "60"))

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
        # No left-edge (warm-up) requirement: the factor's usable history starts
        # wherever its own operators allow, and that is fine.
        #
        # A *leading gap* is a different failure and is still caught: if the
        # matrix itself begins later than the panel (a landing that did not cover
        # the whole window) and then stays empty for a long stretch, the landing
        # is broken, not "warming up".  Landing from the factor's own first
        # usable value is explicitly allowed (gap ≈ 0).
        leading_gap = bool(
            start is not None and valid_start is not None
            and start > FULL_WINDOW_START
            and (valid_start - start) > pd.Timedelta(days=200))
        full = bool(valid_end is not None
                    and not leading_gap
                    and valid_end >= FULL_WINDOW_END - pd.Timedelta(days=7)
                    and valid_days >= MIN_USABLE_DAYS)
        reason = None if full else (
            (f"index starts {start} (panel starts {FULL_WINDOW_START.date()}) "
             f"but the first usable value is {valid_start} — broken landing, not warm-up; "
             if leading_gap else "")
            + f"usable={valid_start}..{valid_end}, usable_days={valid_days}; "
            f"requires usable values through {FULL_WINDOW_END.date()} (-7d t+2 slack) "
            f"and >= {MIN_USABLE_DAYS} usable sessions (no warm-up requirement)")
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
