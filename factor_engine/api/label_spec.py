# -*- coding: utf-8 -*-
"""Forward-label maturity and purged/embargoed time-split contracts (P0-K/P0-L).

A supervised panel model must never fit a label that is not yet *mature* at the
model-fit time.  For a forward-H label ``label_s = P[s+H]/P[s] - 1`` the label is
only knowable once ``s + H <= t``; calling the caller "just shift the label" is
forbidden — the LabelSpec carries the maturity rule and the split helpers
enforce it.

``purged_time_split`` is the P0-L contract: train/validation are split by date
(never by instrument), every instrument on the same date belongs to the same
fold, the boundary is purged by ``horizon`` observations, and an optional
``embargo`` of ``n`` dates is applied after the purge so overlapping labels can
never leak across the split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    """Forward-H label availability contract.

    ``horizon`` is the number of periods the label looks ahead; ``mature`` is
    True exactly when the label observed at ``s`` is knowable at decision time
    ``t`` (``s + horizon <= t``).
    """

    horizon: int

    def mature(self, anchor: int, fit_time: int) -> bool:
        """True when a label anchored at ``anchor`` is knowable at ``fit_time``."""
        return int(anchor) + int(self.horizon) <= int(fit_time)

    def mature_mask(self, anchors: Sequence[int], fit_time: int) -> np.ndarray:
        a = np.asarray(list(anchors), dtype=int)
        return a + int(self.horizon) <= int(fit_time)


def label_available_mask(
    dates: Sequence[pd.Timestamp],
    horizon: int,
    fit_date: pd.Timestamp,
) -> np.ndarray:
    """Boolean mask over ``dates``: is the forward-``horizon`` label at each date
    mature by ``fit_date`` (i.e. ``date + horizon <= fit_date``)?"""
    dt = pd.DatetimeIndex(list(dates))
    fit = pd.Timestamp(fit_date)
    lookback = dt + pd.tseries.offsets.BDay(int(horizon))
    return np.asarray(lookback <= fit, dtype=bool)


def purged_time_split(
    dates: Sequence[pd.Timestamp],
    *,
    train_frac: float = 0.7,
    horizon: int = 5,
    embargo: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Purged / embargoed chronological split by DATE.

    Returns ``(train_idx, valid_idx)``.  All instruments on a date share the
    same fold; the split point is purged by ``horizon`` observations and an
    additional ``embargo`` dates are dropped after the boundary so a forward-H
    label can never straddle train and validation (audit P0-L).
    """
    dt = pd.DatetimeIndex(list(dates))
    if len(dt) < 3:
        raise ValueError("purged_time_split needs at least 3 dates")
    if not 0.0 < float(train_frac) < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    h = max(0, int(horizon))
    e = max(0, int(embargo))
    split_at = int(round(len(dt) * float(train_frac)))
    # Purge h observations on each side of the boundary, then the embargo.
    train_end = max(0, split_at - h)
    valid_start = min(len(dt), split_at + h + e)
    if valid_start <= train_end or train_end < 1 or valid_start >= len(dt):
        raise ValueError(
            "purged_time_split: horizon/embargo leave no train or validation "
            "window"
        )
    train_idx = np.arange(0, train_end, dtype=int)
    valid_idx = np.arange(valid_start, len(dt), dtype=int)
    return train_idx, valid_idx


def assert_no_label_overlap(
    train_idx: Iterable[int],
    valid_idx: Iterable[int],
    dates: Sequence[pd.Timestamp],
    *,
    horizon: int,
) -> None:
    """Assert no forward-H label anchored in train reaches into validation."""
    dt = pd.DatetimeIndex(list(dates))
    h = int(horizon)
    tr = np.asarray(sorted(train_idx), dtype=int)
    va = np.asarray(sorted(valid_idx), dtype=int)
    if tr.size == 0 or va.size == 0:
        return
    last_train = dt[tr[-1]]
    first_valid = dt[va[0]]
    if (first_valid - last_train).days <= 0:
        raise AssertionError("validation is not after training")
    lookback = first_valid - pd.tseries.offsets.BDay(h)
    overlapping = dt[tr] > lookback
    if np.any(overlapping):
        raise AssertionError(
            "forward-H label anchored in train reaches into validation: "
            f"train dates <= {dt[tr[overlapping][-1]]} overlap with first valid "
            f"{first_valid} minus horizon {h}"
        )
