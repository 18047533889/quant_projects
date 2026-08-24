# -*- coding: utf-8 -*-
"""Audit date_bounded_split for overlapping train/validation/test date ranges and
purge_overlap for missing-session gaps and label-maturity correctness.

This module adds focused regression coverage for concrete PIT defects:

1. date_bounded_split: overlapping train/validation/test date ranges (§4.2).
2. purge_overlap: missing-session gaps causing incorrect purge cutoff (walk_forward.py:340).
3. purge_overlap: label-maturity correctness with calendar authority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.market.exchange_session_calendar import ExchangeSessionCalendar
from modeling.contracts import LabelContract
from modeling.dataset import PanelDataset
from modeling.split import date_bounded_split, assert_date_authoritative
from modeling.walk_forward import purge_overlap, purge_before_boundary


def _panel(dates: list, n_stocks: int = 20, seed: int = 0) -> PanelDataset:
    """Build a panel with explicit dates."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for s in range(n_stocks):
            x = rng.normal(size=4)
            rows.append([d, f"S{s:03d}", *x.tolist(), float(0.5 * x[0] + rng.normal(0, 0.1))])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    return PanelDataset(
        frame=frame, date_col="date", stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"], label_col="label",
    )


# --------------------------------------------------------------------------- #
# date_bounded_split: overlapping date ranges
# --------------------------------------------------------------------------- #
def test_date_bounded_split_overlapping_train_validation_raises():
    """§4.2: train and validation date ranges must not overlap."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    ds = _panel(dates.tolist())
    with pytest.raises(ValueError, match="train/validation date overlap"):
        date_bounded_split(
            ds,
            train_start=dates[0], train_end=dates[5],
            validation_start=dates[4], validation_end=dates[7],  # overlap: dates[4], dates[5]
            test_start=dates[8], test_end=dates[9],
        )


def test_date_bounded_split_overlapping_validation_test_raises():
    """§4.2: validation and test date ranges must not overlap."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    ds = _panel(dates.tolist())
    with pytest.raises(ValueError, match="validation/test date overlap"):
        date_bounded_split(
            ds,
            train_start=dates[0], train_end=dates[3],
            validation_start=dates[4], validation_end=dates[7],
            test_start=dates[6], test_end=dates[9],  # overlap: dates[6], dates[7]
        )


def test_date_bounded_split_overlapping_train_test_raises():
    """§4.2: train and test date ranges must not overlap."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    ds = _panel(dates.tolist())
    with pytest.raises(ValueError, match="train/test date overlap"):
        date_bounded_split(
            ds,
            train_start=dates[0], train_end=dates[5],
            validation_start=None, validation_end=None,
            test_start=dates[4], test_end=dates[7],  # overlap: dates[4], dates[5]
        )


def test_date_bounded_split_non_overlapping_succeeds():
    """Clean date-authoritative split succeeds."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    ds = _panel(dates.tolist())
    train, val, test = date_bounded_split(
        ds,
        train_start=dates[0], train_end=dates[3],
        validation_start=dates[4], validation_end=dates[6],
        test_start=dates[7], test_end=dates[9],
    )
    assert len(train.frame) == 4 * 20  # 4 dates * 20 stocks
    assert len(val.frame) == 3 * 20
    assert len(test.frame) == 3 * 20
    violations = assert_date_authoritative(train, val, test)
    assert violations == []


# --------------------------------------------------------------------------- #
# purge_overlap: missing-session gaps
# --------------------------------------------------------------------------- #
def test_purge_overlap_missing_session_gap_without_calendar():
    """walk_forward.py:336-341 fix: concatenate train and validation dates to
    build bar positions, preserving gaps between them.

    Example: train ends 2020-01-03, validation starts 2020-01-06 (missing
    2020-01-04, 2020-01-05). With horizon_bars=1, we need to determine if
    2020-01-03's label (which matures 1 bar later) overlaps validation.

    WITHOUT calendar: bar arithmetic uses available dates only. The next bar
    after 2020-01-03 is 2020-01-06 (the validation start), so the label DOES
    overlap. The OLD code used only train_dates, making p=3 in a 3-element array,
    leading to cutoff_pos=1 and purging 2020-01-02 and 2020-01-03.

    The FIX concatenates train+validation dates: combined array has 5 elements,
    p=3 (position of 2020-01-06), cutoff_pos = 3-1-1 = 1 → cutoff=2020-01-02.
    This correctly purges 2020-01-03 (whose label matures on the validation start).
    """
    train_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    val_dates = [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07")]  # gap
    train_ds = _panel(train_dates, n_stocks=10)
    val_ds = _panel(val_dates, n_stocks=10)
    contract = LabelContract(label_name="ret_1", horizon_bars=1)

    # After fix: concatenate dates, preserving the gap structure.
    purged = purge_overlap(train_ds, val_ds, contract, calendar=None)
    purged_dates = sorted(purged.frame["date"].unique())

    # Correct behavior: 2020-01-03 + 1 bar = 2020-01-06 (validation start).
    # Keep rows where date + horizon < val_start in bar positions.
    # Combined: [01-01(0), 01-02(1), 01-03(2), 01-06(3), 01-07(4)]
    # 01-03 is at pos 2, +1 bar = pos 3 = 01-06 (validation start) → overlap.
    # Keep 01-01, 01-02. Purge 01-03.
    assert purged_dates == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_purge_overlap_with_calendar_authority():
    """With calendar authority, missing sessions are handled correctly via
    shift_session (the single source of truth for bar arithmetic)."""
    train_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    val_dates = [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07")]
    train_ds = _panel(train_dates, n_stocks=10)
    val_ds = _panel(val_dates, n_stocks=10)
    contract = LabelContract(label_name="ret_1", horizon_bars=1)

    # Calendar with weekend awareness (2020-01-04, 2020-01-05 are Sat/Sun)
    calendar = ExchangeSessionCalendar(
        market="US", exchange="NYSE", timezone="America/New_York",
        segments=(("09:30", "16:00"),), holidays=frozenset(),
    )

    # With calendar: cutoff = calendar.shift_session(val_start, -(horizon + 1))
    # val_start = 2020-01-06 (Monday), shift by -(1+1) = -2 trading sessions.
    # Skip weekend: 2020-01-06 → 2020-01-03 (Fri) → 2020-01-02 (Thu).
    # Keep rows where date <= 2020-01-02 → [2020-01-01, 2020-01-02].
    purged = purge_overlap(train_ds, val_ds, contract, calendar=calendar)
    purged_dates = sorted(purged.frame["date"].unique())
    assert purged_dates == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_purge_overlap_label_matures_inside_validation():
    """When the label interval actually overlaps validation, rows are purged."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    train_ds = _panel(dates[:7].tolist(), n_stocks=10)  # 2020-01-01 to 2020-01-07
    val_ds = _panel(dates[7:9].tolist(), n_stocks=10)   # 2020-01-08 to 2020-01-09
    contract = LabelContract(label_name="ret_3", horizon_bars=3)

    # validation starts 2020-01-08. Keep rows where date + 3 < 2020-01-08.
    # date + 3 < 2020-01-08 → date < 2020-01-05.
    # Keep: 2020-01-01, 02, 03, 04. Purge: 2020-01-05, 06, 07.
    purged = purge_overlap(train_ds, val_ds, contract, calendar=None)
    purged_dates = sorted(purged.frame["date"].unique())
    assert purged_dates == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"),
                            pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-04")]


def test_purge_overlap_horizon_one_with_no_actual_overlap():
    """When label interval does not overlap validation, no rows are purged."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    train_ds = _panel(dates[:5].tolist())  # 2020-01-01 to 2020-01-05
    val_ds = _panel(dates[7:9].tolist())   # 2020-01-08 to 2020-01-09 (gap of 2 days)
    contract = LabelContract(label_name="ret_1", horizon_bars=1)
    purged = purge_overlap(train_ds, val_ds, contract, calendar=None)
    # Train ends 2020-01-05. With horizon=1, last label matures on next bar.
    # Combined dates: [01-01, 01-02, 01-03, 01-04, 01-05, 01-08, 01-09]
    # Val start at pos 5 (2020-01-08), cutoff_pos = 5 - 1 - 1 = 3 → 2020-01-04.
    # Keep [2020-01-01, 01-02, 01-03, 01-04]. Purge 01-05.
    purged_dates = sorted(purged.frame["date"].unique())
    assert purged_dates == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"),
                            pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-04")]


def test_purge_overlap_empty_validation_no_purge():
    """validation_ds=None or empty → no purge."""
    dates = pd.date_range("2020-01-01", periods=7, freq="D")
    train_ds = _panel(dates.tolist())
    contract = LabelContract(label_name="ret_1", horizon_bars=1)
    purged = purge_overlap(train_ds, None, contract, calendar=None)
    assert len(purged.frame) == len(train_ds.frame)


# --------------------------------------------------------------------------- #
# purge_before_boundary: test boundary without validation
# --------------------------------------------------------------------------- #
def test_purge_before_boundary_drops_label_overlap():
    """purge_before_boundary closes the gap when validation_ds is None: any
    training row whose label interval overlaps the test start is purged."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    train_ds = _panel(dates[:7].tolist())  # 2020-01-01 to 2020-01-07
    test_start = dates[7]  # 2020-01-08
    contract = LabelContract(label_name="ret_2", horizon_bars=2)

    # Keep rows where date + 2 < 2020-01-08 → date < 2020-01-06.
    # Keep: 2020-01-01 to 2020-01-05. Purge: 2020-01-06, 2020-01-07.
    purged = purge_before_boundary(train_ds, test_start, contract)
    purged_dates = sorted(purged.frame["date"].unique())
    expected = [pd.Timestamp(f"2020-01-0{i}") for i in range(1, 6)]
    assert purged_dates == expected


def test_purge_before_boundary_none_is_noop():
    """boundary=None is a no-op (backward compatible when no test window)."""
    dates = pd.date_range("2020-01-01", periods=7, freq="D")
    train_ds = _panel(dates.tolist())
    contract = LabelContract(label_name="ret_1", horizon_bars=1)
    purged = purge_before_boundary(train_ds, None, contract)
    assert purged is train_ds


def test_purge_before_boundary_all_rows_purged_when_horizon_exceeds_gap():
    """When horizon is large enough that all train rows' labels overlap the
    boundary, all rows are purged (empty dataset)."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    train_ds = _panel(dates[:3].tolist())  # 2020-01-01 to 2020-01-03
    test_start = dates[3]  # 2020-01-04
    contract = LabelContract(label_name="ret_10", horizon_bars=10)

    # Keep rows where date + 10 < 2020-01-04. No date satisfies this.
    purged = purge_before_boundary(train_ds, test_start, contract)
    assert len(purged.frame) == 0


# --------------------------------------------------------------------------- #
# Integration: combined purge + split correctness
# --------------------------------------------------------------------------- #
def test_purge_overlap_then_split_no_leakage():
    """After purging, the resulting train/validation split must be date-authoritative."""
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    ds = _panel(dates.tolist())
    contract = LabelContract(label_name="ret_3", horizon_bars=3)

    # Train: 2020-01-01 to 2020-01-10
    # Validation: 2020-01-11 to 2020-01-15
    train_ds = ds.filter_dates(start=dates[0], end=dates[9])
    val_ds = ds.filter_dates(start=dates[10], end=dates[14])

    # Purge train rows whose label overlaps validation.
    purged_train = purge_overlap(train_ds, val_ds, contract, calendar=None)

    # Verify date-authoritative: no shared dates between purged train and validation.
    violations = assert_date_authoritative(purged_train, val_ds, None)
    assert violations == []

    # Verify purge correctness: keep rows where date + 3 < 2020-01-11.
    # date + 3 < 2020-01-11 → date < 2020-01-08 → keep 2020-01-01 to 2020-01-07.
    purged_dates = sorted(purged_train.frame["date"].unique())
    expected = [pd.Timestamp(f"2020-01-0{i}") for i in range(1, 8)]
    assert purged_dates == expected
