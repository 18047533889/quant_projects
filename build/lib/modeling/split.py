# -*- coding: utf-8 -*-
"""Date-authoritative split helpers (Model Layer Major Redesign taskbook §4.2 /
§4.3).

All splits are DATE-authoritative: every row of a date belongs to exactly one
split.  These helpers build explicit train / validation / test slices and verify
that no date appears in two splits.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from modeling.dataset import PanelDataset

__all__ = [
    "date_bounded_split",
    "split_by_date_cutoff",
    "assert_date_authoritative",
]


def date_bounded_split(
    ds: PanelDataset,
    *,
    train_start: Any = None,
    train_end: Any = None,
    validation_start: Any = None,
    validation_end: Any = None,
    test_start: Any = None,
    test_end: Any = None,
) -> tuple[PanelDataset, PanelDataset, PanelDataset]:
    """Three date-authoritative slices (both bounds inclusive per slice)."""
    train = ds.filter_dates(start=train_start, end=train_end)
    validation = ds.filter_dates(start=validation_start, end=validation_end)
    test = ds.filter_dates(start=test_start, end=test_end)
    violations = assert_date_authoritative(train, validation, test)
    if violations:
        raise ValueError("date-bounded split is not date-authoritative: " + "; ".join(violations))
    return train, validation, test


def split_by_date_cutoff(
    ds: PanelDataset, cutoff: Any, *, date_col: str = "date"
) -> tuple[PanelDataset, PanelDataset]:
    """Split into (before-or-at cutoff, strictly-after cutoff)."""
    col = date_col if date_col in ds.frame.columns else ds.date_col
    before = ds.filter_dates(end=cutoff)
    after_mask = ds.frame[col] > cutoff
    after = PanelDataset(
        frame=ds.frame.loc[after_mask].reset_index(drop=True),
        date_col=ds.date_col,
        stock_col=ds.stock_col,
        feature_cols=list(ds.feature_cols),
        label_col=ds.label_col,
    )
    return before, after


def assert_date_authoritative(
    train_ds: PanelDataset | None,
    validation_ds: PanelDataset | None,
    test_ds: PanelDataset | None,
    *,
    date_col: str = "date",
) -> list[str]:
    """Return a list of violations when a date appears in two splits (empty when
    the splits are cleanly date-authoritative)."""
    violations: list[str] = []

    def _date_set(ds: PanelDataset | None) -> set:
        if ds is None or len(ds.frame) == 0:
            return set()
        col = date_col if date_col in ds.frame.columns else ds.date_col
        return set(ds.frame[col].tolist())

    tr, va, te = _date_set(train_ds), _date_set(validation_ds), _date_set(test_ds)
    if tr & va:
        violations.append(f"train/validation date overlap: {len(tr & va)} dates")
    if tr & te:
        violations.append(f"train/test date overlap: {len(tr & te)} dates")
    if va & te:
        violations.append(f"validation/test date overlap: {len(va & te)} dates")
    return violations
