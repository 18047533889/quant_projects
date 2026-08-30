# -*- coding: utf-8 -*-
"""Date-authoritative walk-forward splitting, purge and embargo (Model Layer
Major Redesign taskbook §3.4 / §4.2 / §9 / §10 / §20 / §73).

The walk-forward split is DATE-authoritative (§4.2): every row of a date belongs
to exactly one split.  Purge (§9) drops training rows whose label interval
``[t, t+H]`` overlaps the validation window; embargo (§10) drops the last
``embargo_bars`` dates of training so a validation boundary is never leaked.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np
import pandas as pd

from modeling.contracts import LabelContract
from modeling.dataset import PanelDataset

# R55 #97: ``factor_engine`` is an OPTIONAL runtime dependency (thin adapter
# boundary, not a vendored copy).  The calendar is duck-typed — only
# ``shift_session`` is called — so the type import is deferred and
# `import modeling.walk_forward` works in a wheel without factor_engine
# installed.  Never reimplement calendar math here.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from factor_engine.market.exchange_session_calendar import ExchangeSessionCalendar

_log = logging.getLogger(__name__)

__all__ = [
    "WalkForwardSpec",
    "WalkForwardFold",
    "OOSStitchPolicy",
    "OOSPredictionWindow",
    "stitch_oos_windows",
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
    """Walk-forward training specification (§3.4 / §68).

    ``gap_days`` enforces temporal separation between train_end and
    validation_start to prevent feature leakage. gap_days=5 means 5 full
    calendar days between the last training date and first validation date.
    Distinct from label-interval purge (which prevents label leakage).
    """

    train_lookback_bars: int | None = None
    train_expanding: bool = False
    validation_bars: int = 0
    test_bars: int = 126
    step_bars: int = 21
    retrain_every_bars: int = 21
    purge_policy: str = "label_interval"  # "label_interval" | "purge_bars"
    embargo_bars: int = 0
    gap_days: int = 0
    min_train_dates: int = 252
    min_train_stocks: int = 30
    min_train_obs: int = 10000
    min_validation_dates: int = 0
    min_validation_effective_dates: int = 0
    min_validation_median_stocks: float = 0.0
    min_validation_label_coverage: float = 0.0
    min_test_dates: int = 0
    min_test_effective_dates: int = 0
    min_test_median_stocks: float = 0.0
    min_test_label_coverage: float = 0.0
    min_valid_daily_ic_dates: int = 0
    decay_half_life_bars: int | None = None

    def __post_init__(self) -> None:
        if self.gap_days < 0:
            raise ValueError("gap_days must be >= 0")


class OOSStitchPolicy(str, Enum):
    """How overlapping fold test windows become one simulated-live history."""

    ACTIVE_UNTIL_NEXT_RETRAIN = "active_until_next_retrain"
    LATEST_LEGAL_ARTIFACT = "latest_legal_artifact"


@dataclass(frozen=True)
class OOSPredictionWindow:
    artifact_id: str
    activation: Any
    predictions: pd.DataFrame
    date_col: str = "date"


def stitch_oos_windows(
    windows: Iterable[OOSPredictionWindow],
    policy: OOSStitchPolicy = OOSStitchPolicy.ACTIVE_UNTIL_NEXT_RETRAIN,
) -> pd.DataFrame:
    """Stitch overlapping test predictions to exactly one legal artifact/date.

    Both policies choose the most recently activated legal artifact.  The
    explicit enum preserves the deployment intent while ensuring that an old
    artifact stops contributing as soon as its successor becomes active.
    """
    ordered = sorted(list(windows), key=lambda item: pd.Timestamp(item.activation))
    policy = OOSStitchPolicy(policy)
    activation_owners: dict[pd.Timestamp, str] = {}
    for item in ordered:
        activation = pd.Timestamp(item.activation)
        owner = activation_owners.setdefault(activation, item.artifact_id)
        if owner != item.artifact_id:
            raise ValueError(
                f"activation {activation} maps to multiple artifacts: "
                f"{owner!r}, {item.artifact_id!r}"
            )
    pieces: list[pd.DataFrame] = []
    for index, item in enumerate(ordered):
        frame = item.predictions.copy()
        if item.date_col not in frame:
            raise ValueError(f"prediction window missing {item.date_col!r}")
        frame["artifact_id"] = item.artifact_id
        frame["artifact_activation"] = pd.Timestamp(item.activation)
        legal = pd.to_datetime(frame[item.date_col]) >= pd.Timestamp(item.activation)
        if policy is OOSStitchPolicy.ACTIVE_UNTIL_NEXT_RETRAIN and index + 1 < len(ordered):
            legal &= pd.to_datetime(frame[item.date_col]) < pd.Timestamp(ordered[index + 1].activation)
        pieces.append(frame.loc[legal])
    if not pieces:
        return pd.DataFrame(columns=["artifact_id", "artifact_activation"])
    candidates = pd.concat(pieces, ignore_index=True)
    date_col = ordered[0].date_col
    if any(item.date_col != date_col for item in ordered):
        raise ValueError("all OOS windows must use the same date column")
    selected = candidates.groupby(date_col, observed=True)["artifact_activation"].transform("max")
    stitched = candidates.loc[candidates["artifact_activation"].eq(selected)].copy()
    mapping_width = stitched.groupby(date_col, observed=True)["artifact_id"].nunique()
    if mapping_width.gt(1).any():
        raise AssertionError("OOS history does not have unique date/artifact mapping")
    return stitched.sort_values([date_col, "artifact_id"]).reset_index(drop=True)


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


def _fold_sample_adequate(ds: PanelDataset | None, spec: WalkForwardSpec, prefix: str) -> bool:
    if ds is None or ds.n_rows == 0:
        return not any(
            getattr(spec, f"min_{prefix}_{suffix}") > 0
            for suffix in ("dates", "effective_dates", "median_stocks", "label_coverage")
        )
    telemetry = ds.telemetry()
    finite = ds.as_matrix()[4]
    effective_dates = int(ds.frame.loc[finite, ds.date_col].nunique())
    return (
        telemetry["n_unique_dates"] >= getattr(spec, f"min_{prefix}_dates")
        and effective_dates >= getattr(spec, f"min_{prefix}_effective_dates")
        and telemetry["median_stocks_per_date"] >= getattr(spec, f"min_{prefix}_median_stocks")
        and telemetry["label_coverage"] >= getattr(spec, f"min_{prefix}_label_coverage")
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
            # Apply gap_days: skip gap_days positions between train_end and val_start
            val_start_pos = pos + spec.gap_days
            val_end_pos = val_start_pos + validation_bars - 1
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
        if not _fold_sample_adequate(val_ds, spec, "validation"):
            pos += spec.step_bars
            continue
        if not _fold_sample_adequate(test_ds, spec, "test"):
            pos += spec.step_bars
            continue
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


def check_fold_order(folds: list[WalkForwardFold], *, gap_days: int = 0) -> list[str]:
    """Report §59 order violations; empty list when all folds are clean.

    When ``gap_days > 0``, validates that each fold has at least gap_days
    full calendar days between train_end and validation_start.
    """
    violations: list[str] = []
    for fold in folds:
        if fold.validation_start is not None and fold.train_end is not None:
            if fold.train_end >= fold.validation_start:
                violations.append(
                    f"fold {fold.fold_id}: train_end >= validation_start "
                    f"({fold.train_end} >= {fold.validation_start})"
                )
            # Validate gap_days requirement
            if gap_days > 0:
                train_end_ts = pd.Timestamp(fold.train_end)
                val_start_ts = pd.Timestamp(fold.validation_start)
                actual_gap = (val_start_ts - train_end_ts).days
                required_gap = gap_days + 1  # gap_days=5 means 6 calendar days apart
                if actual_gap < required_gap:
                    violations.append(
                        f"fold {fold.fold_id}: gap between train_end and validation_start "
                        f"is {actual_gap} days but gap_days={gap_days} requires >= {required_gap} days "
                        f"(train_end={fold.train_end}, validation_start={fold.validation_start})"
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
    calendar: ExchangeSessionCalendar | None = None,
    validate: bool = True,
) -> PanelDataset:
    """§9 label-interval purge.

    Drop training rows whose label interval ``[t, t+H]`` overlaps the
    validation start — i.e. keep rows where ``date < validation_start`` AND
    ``date + horizon_bars < validation_start`` (strict).  Implemented in bar
    positions so string / date / datetime dates are handled uniformly.

    When ``validate=True`` (default), verifies that rows were actually purged
    when horizon_bars > 0 and validation exists (detects purge bypass).
    """
    if validation_ds is None or len(validation_ds.frame) == 0 or train_ds.n_rows == 0:
        return train_ds

    col = _date_col_of(train_ds, date_col)
    vcol = _date_col_of(validation_ds, date_col)
    val_start = validation_ds.frame[vcol].min()
    horizon = int(getattr(label_contract, "horizon_bars", 0))

    # Capture pre-purge state for validation
    pre_purge_rows = train_ds.n_rows
    pre_purge_dates = set(train_ds.frame[col].unique())

    # Perform purge
    if calendar is not None:
        cutoff = calendar.shift_session(val_start, -(horizon + 1))
        result = _filter_by_dates(train_ds, None, cutoff, col)
    else:
        train_dates = train_ds.frame[col].unique()
        validation_dates = validation_ds.frame[vcol].unique()
        # Build bar positions from both sides of the boundary.  Using only dates
        # present in train makes a missing session collapse the gap and can purge a
        # label that actually matures before validation begins.
        dates = np.sort(np.concatenate([train_dates, validation_dates]))
        p = int(pd.Index(dates).searchsorted(val_start, side="left"))
        cutoff_pos = p - horizon - 1
        if cutoff_pos < 0:
            result = _empty_like(train_ds)
        else:
            result = _filter_by_dates(train_ds, None, dates[cutoff_pos], col)

    # Validate and log purge statistics
    post_purge_rows = result.n_rows
    post_purge_dates = set(result.frame[col].unique()) if post_purge_rows > 0 else set()
    dropped_rows = pre_purge_rows - post_purge_rows
    dropped_dates = pre_purge_dates - post_purge_dates

    if validate and horizon > 0:
        # When horizon > 0, purge MUST drop at least some rows (label overlap exists)
        if dropped_rows == 0:
            _log.warning(
                f"purge_overlap: horizon_bars={horizon} but no rows were purged. "
                f"This may indicate purge bypass. "
                f"val_start={val_start}, train_end={max(pre_purge_dates) if pre_purge_dates else None}"
            )

    if dropped_rows > 0:
        cutoff_date = max(post_purge_dates) if post_purge_dates else None
        _log.debug(
            f"purge_overlap: dropped {dropped_rows} rows ({len(dropped_dates)} dates) "
            f"with horizon_bars={horizon}. Cutoff date: {cutoff_date}, val_start: {val_start}"
        )

    return result


def apply_embargo(
    train_ds: PanelDataset,
    embargo_bars: int,
    *,
    date_col: str = "date",
    calendar: ExchangeSessionCalendar | None = None,
) -> PanelDataset:
    """§10 — drop the last ``embargo_bars`` dates of training."""
    if embargo_bars <= 0 or train_ds.n_rows == 0:
        return train_ds
    col = _date_col_of(train_ds, date_col)
    dates = np.sort(train_ds.frame[col].unique())
    if calendar is not None:
        cutoff = calendar.shift_session(pd.Timestamp(dates[-1]), -embargo_bars)
        return _filter_by_dates(train_ds, None, cutoff, col)
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
    calendar: ExchangeSessionCalendar | None = None,
) -> PanelDataset:
    """Combine §9 purge and §10 embargo using the optional calendar authority.

    ``spec`` supplies the embargo and purge policy; when absent,
    ``label_contract.embargo_bars`` is used.

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
    purged = purge_overlap(
        train_ds,
        validation_ds,
        label_contract,
        date_col=date_col,
        calendar=calendar,
    )
    return apply_embargo(
        purged,
        embargo,
        date_col=date_col,
        calendar=calendar,
    )


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
