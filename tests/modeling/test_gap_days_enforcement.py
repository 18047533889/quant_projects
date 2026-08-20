# -*- coding: utf-8 -*-
"""MODEL-P0-001: gap_days enforcement tests.

This module tests that WalkForwardSpec.gap_days creates mandatory temporal
separation between train_end and validation_start to prevent feature leakage.

Gap semantics: gap_days=5 means 5 full calendar days between train_end and
val_start. If train_end=2020-01-10, val_start must be >= 2020-01-16.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.dataset import PanelDataset
from modeling.walk_forward import WalkForwardSpec, make_walk_forward_splits, check_fold_order


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
# gap_days=0 (default, backward compatible)
# --------------------------------------------------------------------------- #
def test_gap_days_zero_allows_adjacent_dates():
    """gap_days=0 (default) allows train_end and val_start to be adjacent."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    ds = _panel(dates.tolist())
    spec = WalkForwardSpec(
        train_lookback_bars=20,
        validation_bars=10,
        test_bars=10,
        step_bars=10,
        retrain_every_bars=10,
        gap_days=0,  # explicit zero
        min_train_dates=20,
        min_train_stocks=20,
        min_train_obs=1,
    )
    folds = make_walk_forward_splits(ds, spec)
    assert len(folds) > 0
    # First fold should have adjacent train_end and val_start
    fold = folds[0]
    train_dates = pd.to_datetime(fold.train_ds.frame["date"].unique())
    val_dates = pd.to_datetime(fold.validation_ds.frame["date"].unique())
    train_end = train_dates.max()
    val_start = val_dates.min()
    # Adjacent means val_start == train_end + 1 day
    assert (val_start - train_end).days == 1


# --------------------------------------------------------------------------- #
# gap_days > 0 enforcement
# --------------------------------------------------------------------------- #
def test_gap_days_five_creates_five_day_gap():
    """gap_days=5 ensures at least 5 full days between train_end and val_start."""
    dates = pd.date_range("2020-01-01", periods=150, freq="D")
    ds = _panel(dates.tolist())
    spec = WalkForwardSpec(
        train_lookback_bars=30,
        validation_bars=10,
        test_bars=10,
        step_bars=20,
        retrain_every_bars=20,
        gap_days=5,
        min_train_dates=30,
        min_train_stocks=20,
        min_train_obs=1,
    )
    folds = make_walk_forward_splits(ds, spec)
    assert len(folds) > 0

    for fold in folds:
        train_dates = pd.to_datetime(fold.train_ds.frame["date"].unique())
        val_dates = pd.to_datetime(fold.validation_ds.frame["date"].unique())
        train_end = train_dates.max()
        val_start = val_dates.min()
        # gap_days=5 means val_start >= train_end + 6 days (5 full days between)
        actual_gap = (val_start - train_end).days
        assert actual_gap >= 6, (
            f"fold {fold.fold_id}: gap {actual_gap} days < required 6 "
            f"(train_end={train_end.date()}, val_start={val_start.date()})"
        )


def test_gap_days_ten_creates_ten_day_gap():
    """gap_days=10 ensures at least 10 full days between train_end and val_start."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    ds = _panel(dates.tolist())
    spec = WalkForwardSpec(
        train_lookback_bars=40,
        validation_bars=15,
        test_bars=15,
        step_bars=30,
        retrain_every_bars=30,
        gap_days=10,
        min_train_dates=40,
        min_train_stocks=20,
        min_train_obs=1,
    )
    folds = make_walk_forward_splits(ds, spec)
    assert len(folds) > 0

    for fold in folds:
        train_dates = pd.to_datetime(fold.train_ds.frame["date"].unique())
        val_dates = pd.to_datetime(fold.validation_ds.frame["date"].unique())
        train_end = train_dates.max()
        val_start = val_dates.min()
        actual_gap = (val_start - train_end).days
        assert actual_gap >= 11, (
            f"fold {fold.fold_id}: gap {actual_gap} days < required 11 "
            f"(train_end={train_end.date()}, val_start={val_start.date()})"
        )


# --------------------------------------------------------------------------- #
# check_fold_order validation
# --------------------------------------------------------------------------- #
def test_check_fold_order_validates_gap_days():
    """check_fold_order() detects gap_days violations."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    ds = _panel(dates.tolist())

    # Create spec with gap_days=5
    spec = WalkForwardSpec(
        train_lookback_bars=20,
        validation_bars=10,
        test_bars=10,
        step_bars=10,
        retrain_every_bars=10,
        gap_days=5,
        min_train_dates=20,
        min_train_stocks=20,
        min_train_obs=1,
    )
    folds = make_walk_forward_splits(ds, spec)
    assert len(folds) > 0

    # check_fold_order should validate gap (returns violations list)
    violations = check_fold_order(folds, gap_days=5)
    # All folds should respect gap_days, so no violations
    assert violations == [], f"Unexpected violations: {violations}"


def test_check_fold_order_detects_insufficient_gap():
    """check_fold_order() detects when actual gap < required gap_days."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    ds = _panel(dates.tolist())

    # Manually create a fold with insufficient gap
    from modeling.walk_forward import WalkForwardFold

    train_ds = ds.filter_dates(start=dates[0], end=dates[19])  # 2020-01-01 to 2020-01-20
    val_ds = ds.filter_dates(start=dates[22], end=dates[29])    # 2020-01-23 to 2020-01-30 (gap=2 days)
    test_ds = ds.filter_dates(start=dates[30], end=dates[39])

    fold = WalkForwardFold(
        fold_id=0,
        train_start=dates[0],
        train_end=dates[19],
        validation_start=dates[22],
        validation_end=dates[29],
        test_start=dates[30],
        test_end=dates[39],
        train_ds=train_ds,
        validation_ds=val_ds,
        test_ds=test_ds,
    )

    # Check with gap_days=5 (requires 5 full days, but fold only has 2)
    violations = check_fold_order([fold], gap_days=5)
    assert len(violations) > 0
    assert "gap" in violations[0].lower()
    assert "fold 0" in violations[0]


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_gap_days_larger_than_step_reduces_fold_count():
    """Large gap_days reduces available folds when data is limited."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    ds = _panel(dates.tolist())

    spec_no_gap = WalkForwardSpec(
        train_lookback_bars=20,
        validation_bars=10,
        test_bars=10,
        step_bars=10,
        retrain_every_bars=10,
        gap_days=0,
        min_train_dates=20,
        min_train_stocks=20,
        min_train_obs=1,
    )
    folds_no_gap = make_walk_forward_splits(ds, spec_no_gap)

    spec_large_gap = WalkForwardSpec(
        train_lookback_bars=20,
        validation_bars=10,
        test_bars=10,
        step_bars=10,
        retrain_every_bars=10,
        gap_days=15,  # large gap
        min_train_dates=20,
        min_train_stocks=20,
        min_train_obs=1,
    )
    folds_large_gap = make_walk_forward_splits(ds, spec_large_gap)

    # Large gap consumes more dates per fold, reducing total fold count
    assert len(folds_large_gap) < len(folds_no_gap)


def test_gap_days_with_expanding_window():
    """gap_days works with expanding training window."""
    dates = pd.date_range("2020-01-01", periods=150, freq="D")
    ds = _panel(dates.tolist())

    spec = WalkForwardSpec(
        train_expanding=True,
        validation_bars=10,
        test_bars=10,
        step_bars=20,
        retrain_every_bars=20,
        gap_days=7,
        min_train_dates=30,
        min_train_stocks=20,
        min_train_obs=1,
    )
    folds = make_walk_forward_splits(ds, spec)
    assert len(folds) > 0

    for fold in folds:
        train_dates = pd.to_datetime(fold.train_ds.frame["date"].unique())
        val_dates = pd.to_datetime(fold.validation_ds.frame["date"].unique())
        train_end = train_dates.max()
        val_start = val_dates.min()
        actual_gap = (val_start - train_end).days
        assert actual_gap >= 8, f"fold {fold.fold_id}: gap {actual_gap} < 8"


def test_gap_days_negative_raises_validation_error():
    """gap_days must be >= 0."""
    with pytest.raises(ValueError, match="gap_days"):
        WalkForwardSpec(
            train_lookback_bars=20,
            validation_bars=10,
            test_bars=10,
            gap_days=-5,  # invalid
        )
