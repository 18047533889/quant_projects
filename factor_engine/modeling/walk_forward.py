# -*- coding: utf-8 -*-
"""Date-authoritative walk-forward splitting, purge and embargo (Model Layer
Major Redesign taskbook §3.4 / §4.2 / §9 / §10 / §20 / §73).

The walk-forward split is DATE-authoritative (§4.2): every row of a date belongs
to exactly one split.  Purge (§9) drops training rows whose label interval
``[t, t+H]`` overlaps the validation window; embargo (§10) drops the last
``embargo_bars`` dates of training so a validation boundary is never leaked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from modeling.contracts import LabelContract
from modeling.dataset import PanelDataset

__all__ = [
    "WalkForwardSpec",
    "WalkForwardFold",
    "make_walk_forward_splits",
    "check_fold_order",
    "purge_overlap",
    "purge_before_boundary",
    "apply_embargo",
    "purge_and_embargo",
    "nested_splits",
    "decay_weights",
]


@dataclass(frozen=True)
class WalkForwardSpec:
    """Walk-forward training specification (§3.4 / §68)."""

    train_lookback_bars: int | None = None
    train_expanding: bool = False
    validation_bars: int = 0
    test_bars: int = 126
    step_bars: int = 21
    retrain_every_bars: int = 21
    purge_policy: str = "label_interval"  # "label_interval" | "purge_bars"
    embargo_bars: int = 0
    min_train_dates: int = 252
    min_train_stocks: int = 30
    min_train_obs: int = 10000
    decay_half_life_bars: int | None = None


@dataclass
class WalkForwardFold:
    """One walk-forward fold.  ``train_ds`` / ``validation_ds`` / ``test_ds``
    are date-authoritative slices of the input panel."""

    fold_id: int
    train_start: Any
    train_end: Any
    validation_start: Any
    validation_end: Any
    test_start: Any
    test_end: Any
    train_ds: PanelDataset
    validation_ds: PanelDataset | None = None
    test_ds: PanelDataset | None = None


def _date_col_of(ds: PanelDataset, date_col: str) -> str:
    return date_col if date_col in ds.frame.columns else ds.date_col


def _filter_by_dates(ds: PanelDataset, start: Any, end: Any, col: str) -> PanelDataset:
    if start is None and end is None:
        return ds
    mask = pd.Series(True, index=ds.frame.index)
    if start is not None:
        mask &= ds.frame[col] >= start
    if end is not None:
        mask &= ds.frame[col] <= end
    return PanelDataset(
        frame=ds.frame.loc[mask].reset_index(drop=True),
        date_col=ds.date_col,
        stock_col=ds.stock_col,
        feature_cols=list(ds.feature_cols),
        label_col=ds.label_col,
    )


def _empty_like(ds: PanelDataset) -> PanelDataset:
    return PanelDataset(
        frame=ds.frame.iloc[0:0].reset_index(drop=True),
        date_col=ds.date_col,
        stock_col=ds.stock_col,
        feature_cols=list(ds.feature_cols),
        label_col=ds.label_col,
    )


def make_walk_forward_splits(
    ds: PanelDataset, spec: WalkForwardSpec, *, date_col: str = "date"
) -> list[WalkForwardFold]:
    """Build date-authoritative sliding / expanding walk-forward folds.

    Each fold: training = last ``train_lookback_bars`` dates before validation
    (rolling) or all dates from the start (expanding); validation = the next
    ``validation_bars`` dates; test = next ``test_bars`` dates.  Fold emission
    advances by ``retrain_every_bars`` (the §3.4 retrain cadence) when set
    (``> 0``), else by ``step_bars`` — one fold is emitted per retrain point.
    A fold is skipped when its training window fails the ``min_train_dates`` /
    ``min_train_stocks`` / ``min_train_obs`` guards.
    """
    col = _date_col_of(ds, date_col)
    dates = np.sort(ds.frame[col].unique())
    n = len(dates)
    folds: list[WalkForwardFold] = []
    stride = spec.retrain_every_bars if spec.retrain_every_bars > 0 else spec.step_bars

    def _slice(start_pos: int | None, end_pos: int | None) -> PanelDataset:
        if start_pos is None:
            start_pos = 0
        return _filter_by_dates(ds, dates[start_pos], dates[end_pos], col)

    expanding = spec.train_expanding or spec.train_lookback_bars is None
    lookback = spec.train_lookback_bars if spec.train_lookback_bars is not None else 0
    # First validation position: enough train history to satisfy the guard.
    if expanding:
        pos = spec.min_train_dates
    else:
        pos = max(spec.min_train_dates, lookback)

    fold_id = 0
    while True:
        train_end_pos = pos - 1
        if expanding:
            train_start_pos = 0
        else:
            train_start_pos = pos - lookback
        if validation_bars := spec.validation_bars:
            val_start_pos = pos
            val_end_pos = pos + validation_bars - 1
            test_start_pos = val_end_pos + 1
        else:
            val_start_pos = val_end_pos = None
            test_start_pos = pos
        test_end_pos = test_start_pos + spec.test_bars - 1
        if test_end_pos >= n:
            break
        if train_start_pos < 0:
            pos += stride
            continue

        train_ds = _slice(train_start_pos, train_end_pos)
        tele = train_ds.telemetry()
        if tele["n_unique_dates"] < spec.min_train_dates:
            pos += stride
            continue
        if tele["median_stocks_per_date"] < spec.min_train_stocks:
            pos += stride
            continue
        if tele["raw_obs"] < spec.min_train_obs:
            pos += stride
            continue

        val_ds = _slice(val_start_pos, val_end_pos) if val_start_pos is not None else None
        test_ds = _slice(test_start_pos, test_end_pos)
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=dates[train_start_pos],
                train_end=dates[train_end_pos],
                validation_start=dates[val_start_pos] if val_start_pos is not None else None,
                validation_end=dates[val_end_pos] if val_end_pos is not None else None,
                test_start=dates[test_start_pos],
                test_end=dates[test_end_pos],
                train_ds=train_ds,
                validation_ds=val_ds,
                test_ds=test_ds,
            )
        )
        fold_id += 1
        pos += stride
    return folds


def check_fold_order(folds: list[WalkForwardFold]) -> list[str]:
    """Report §59 order violations; empty list when all folds are clean."""
    violations: list[str] = []
    for fold in folds:
        if fold.validation_start is not None and fold.train_end is not None:
            if fold.train_end >= fold.validation_start:
                violations.append(
                    f"fold {fold.fold_id}: train_end >= validation_start "
                    f"({fold.train_end} >= {fold.validation_start})"
                )
        if fold.validation_end is not None and fold.test_start is not None:
            if fold.validation_end >= fold.test_start:
                violations.append(
                    f"fold {fold.fold_id}: validation_end >= test_start "
                    f"({fold.validation_end} >= {fold.test_start})"
                )
        if fold.train_ds is not None and fold.test_ds is not None:
            tr = set(fold.train_ds.frame[fold.train_ds.date_col].tolist())
            te = set(fold.test_ds.frame[fold.test_ds.date_col].tolist())
            if tr & te:
                violations.append(
                    f"fold {fold.fold_id}: train/test date overlap "
                    f"({len(tr & te)} shared dates)"
                )
    return violations


def purge_overlap(
    train_ds: PanelDataset,
    validation_ds: PanelDataset | None,
    label_contract: LabelContract,
    *,
    date_col: str = "date",
) -> PanelDataset:
    """§9 label-interval purge.

    Drop training rows whose label interval ``[t, t+H]`` overlaps the
    validation start — i.e. keep rows where ``date < validation_start`` AND
    ``date + horizon_bars < validation_start`` (strict).  Implemented in bar
    positions so string / date / datetime dates are handled uniformly.
    """
    if validation_ds is None or len(validation_ds.frame) == 0 or train_ds.n_rows == 0:
        return train_ds
    col = _date_col_of(train_ds, date_col)
    vcol = _date_col_of(validation_ds, date_col)
    val_start = validation_ds.frame[vcol].min()
    dates = np.sort(train_ds.frame[col].unique())
    p = int(pd.Index(dates).searchsorted(val_start, side="left"))
    horizon = int(getattr(label_contract, "horizon_bars", 0))
    cutoff_pos = p - horizon - 1
    if cutoff_pos < 0:
        return _empty_like(train_ds)
    return _filter_by_dates(train_ds, None, dates[cutoff_pos], col)


def apply_embargo(
    train_ds: PanelDataset, embargo_bars: int, *, date_col: str = "date"
) -> PanelDataset:
    """§10 — drop the last ``embargo_bars`` dates of training."""
    if embargo_bars <= 0 or train_ds.n_rows == 0:
        return train_ds
    col = _date_col_of(train_ds, date_col)
    dates = np.sort(train_ds.frame[col].unique())
    if embargo_bars >= len(dates):
        return _empty_like(train_ds)
    return _filter_by_dates(train_ds, None, dates[-embargo_bars - 1], col)


def purge_before_boundary(
    train_ds: PanelDataset,
    boundary: Any,
    label_contract: LabelContract,
    *,
    date_col: str = "date",
) -> PanelDataset:
    """Purge training rows whose label interval ``[t, t+H]`` overlaps a held-out
    evaluation boundary (the start of the test window when there is no
    validation window).

    Keeps rows where ``date + horizon_bars < boundary`` (strict).  This closes
    the §9 gap where ``validation_ds is None``: without a validation window the
    label of a late training row (e.g. ``VWAP_{t+20}/VWAP_t`` with the test
    window starting at ``t+5``) would leak test-period information into the fit.
    """
    if boundary is None or train_ds.n_rows == 0:
        return train_ds
    col = _date_col_of(train_ds, date_col)
    dates = np.sort(train_ds.frame[col].unique())
    p = int(pd.Index(dates).searchsorted(boundary, side="left"))
    horizon = int(getattr(label_contract, "horizon_bars", 0))
    cutoff_pos = p - horizon - 1
    if cutoff_pos < 0:
        return _empty_like(train_ds)
    return _filter_by_dates(train_ds, None, dates[cutoff_pos], col)


def purge_and_embargo(
    train_ds: PanelDataset,
    validation_ds: PanelDataset | None,
    label_contract: LabelContract,
    spec: WalkForwardSpec | None = None,
    *,
    date_col: str = "date",
) -> PanelDataset:
    """Combine §9 purge and §10 embargo.  ``spec`` supplies the embargo and the
    purge policy; when absent, ``label_contract.embargo_bars`` is used.

    ``purge_policy="label_interval"`` (default) purges training rows whose label
    interval ``[t, t+H]`` overlaps the validation window.  ``purge_policy="purge_bars"``
    purges a fixed ``embargo_bars``-bar buffer off the training tail regardless
    of label horizon.
    """
    if spec is not None:
        embargo = spec.embargo_bars
        if spec.purge_policy == "purge_bars":
            return apply_embargo(train_ds, embargo, date_col=date_col)
    elif label_contract is not None:
        embargo = int(getattr(label_contract, "embargo_bars", 0))
    else:
        embargo = 0
    purged = purge_overlap(train_ds, validation_ds, label_contract, date_col=date_col)
    return apply_embargo(purged, embargo, date_col=date_col)


def nested_splits(
    ds: PanelDataset,
    outer_spec: WalkForwardSpec,
    inner_spec: WalkForwardSpec,
    *,
    date_col: str = "date",
) -> list[tuple[WalkForwardFold, list[WalkForwardFold]]]:
    """§20 nested walk-forward: for each outer fold build inner folds over the
    outer train+validation range (inner validation selects hyperparameters)."""
    outer_folds = make_walk_forward_splits(ds, outer_spec, date_col=date_col)
    out: list[tuple[WalkForwardFold, list[WalkForwardFold]]] = []
    for fold in outer_folds:
        frames = [fold.train_ds.frame]
        if fold.validation_ds is not None:
            frames.append(fold.validation_ds.frame)
        combined = pd.concat(frames, ignore_index=True)
        combined_ds = PanelDataset(
            frame=combined,
            date_col=fold.train_ds.date_col,
            stock_col=fold.train_ds.stock_col,
            feature_cols=list(fold.train_ds.feature_cols),
            label_col=fold.train_ds.label_col,
        )
        inner_folds = make_walk_forward_splits(combined_ds, inner_spec, date_col=date_col)
        out.append((fold, inner_folds))
    return out


def decay_weights(ds: PanelDataset, half_life_bars: int, *, date_col: str = "date") -> np.ndarray:
    """§3.4 / §73 per-row decay weights: ``0.5 ** ((max_pos - pos)/half_life)``
    in bar units, so the most recent date carries weight 1.0."""
    if ds.n_rows == 0:
        return np.zeros(0, dtype=np.float64)
    if half_life_bars is None or half_life_bars <= 0:
        return np.ones(ds.n_rows, dtype=np.float64)
    col = _date_col_of(ds, date_col)
    dates = np.sort(ds.frame[col].unique())
    max_pos = len(dates) - 1
    pos = pd.Index(dates).searchsorted(ds.frame[col].to_numpy(), side="right") - 1
    return np.power(0.5, (max_pos - pos) / float(half_life_bars)).astype(np.float64)
