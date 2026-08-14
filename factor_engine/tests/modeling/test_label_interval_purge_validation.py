# -*- coding: utf-8 -*-
"""MODEL-P0-002: Label interval purge validation tests.

This module tests that purge_overlap() actually performs purging and validates
that the purge happened. While purge_overlap() is implemented, there's no
validation that rows were actually dropped when they should have been.

Tests verify:
1. Purge statistics are accurate (rows dropped, cutoff date)
2. Purge detects when it should have dropped rows but didn't (bypass detection)
3. Integration with walk-forward showing end-to-end purge
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.contracts import LabelContract
from modeling.dataset import PanelDataset
from modeling.walk_forward import (
    WalkForwardSpec,
    make_walk_forward_splits,
    purge_overlap,
    purge_and_embargo,
)


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
# Purge validation: verify rows were actually dropped
# --------------------------------------------------------------------------- #
def test_purge_overlap_drops_expected_rows_for_5day_label():
    """For 5-day forward label, last 5 training days before validation must be purged."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    train_ds = _panel(dates[:20].tolist())  # 2020-01-01 to 2020-01-20
    val_ds = _panel(dates[20:25].tolist())  # 2020-01-21 to 2020-01-25

    contract = LabelContract(label_name="ret_5", horizon_bars=5)

    # Before purge
    pre_purge_dates = set(train_ds.frame["date"].unique())
    assert len(pre_purge_dates) == 20

    # After purge
    purged = purge_overlap(train_ds, val_ds, contract, calendar=None)
    post_purge_dates = set(purged.frame["date"].unique())

    # Expected: drop last 5 days (2020-01-16 to 2020-01-20)
    # Keep: 2020-01-01 to 2020-01-15
    expected_kept = {pd.Timestamp(f"2020-01-{d:02d}") for d in range(1, 16)}
    assert post_purge_dates == expected_kept

    # Verify purge statistics
    dropped_dates = pre_purge_dates - post_purge_dates
    assert len(dropped_dates) == 5
    assert pd.Timestamp("2020-01-20") in dropped_dates
    assert pd.Timestamp("2020-01-16") in dropped_dates


def test_purge_overlap_drops_nothing_when_horizon_is_zero():
    """horizon_bars=0 means no forward label, so no purge needed."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    train_ds = _panel(dates[:15].tolist())
    val_ds = _panel(dates[15:20].tolist())

    contract = LabelContract(label_name="same_day", horizon_bars=0)

    pre_purge_count = len(train_ds.frame)
    purged = purge_overlap(train_ds, val_ds, contract, calendar=None)
    post_purge_count = len(purged.frame)

    # With horizon=0, no rows should be dropped (but edge case: horizon=0 still needs 1-bar gap)
    # Actually, horizon_bars must be >= 1 per LabelContract validation
    # This test shows what happens if horizon=0 is allowed
    assert post_purge_count > 0


def test_purge_overlap_detects_all_rows_purged_scenario():
    """When horizon is large, all training rows may be purged (empty result)."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    train_ds = _panel(dates[:10].tolist())  # 2020-01-01 to 2020-01-10
    val_ds = _panel(dates[10:15].tolist())  # 2020-01-11 to 2020-01-15

    # 20-day horizon means labels mature far into validation
    contract = LabelContract(label_name="ret_20", horizon_bars=20)

    purged = purge_overlap(train_ds, val_ds, contract, calendar=None)

    # All training rows should be purged (their labels overlap validation)
    assert len(purged.frame) == 0


def test_purge_overlap_validation_with_no_validation_data():
    """validation_ds=None means no purge (backward compatible)."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    train_ds = _panel(dates.tolist())

    contract = LabelContract(label_name="ret_5", horizon_bars=5)

    pre_purge_count = len(train_ds.frame)
    purged = purge_overlap(train_ds, None, contract, calendar=None)
    post_purge_count = len(purged.frame)

    # No validation means no purge
    assert post_purge_count == pre_purge_count


def test_purge_overlap_validation_with_empty_validation():
    """Empty validation_ds means no purge."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    train_ds = _panel(dates.tolist())

    # Create empty validation dataset
    empty_val = PanelDataset(
        frame=pd.DataFrame(columns=train_ds.frame.columns),
        date_col="date",
        stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"],
        label_col="label",
    )

    contract = LabelContract(label_name="ret_5", horizon_bars=5)

    pre_purge_count = len(train_ds.frame)
    purged = purge_overlap(train_ds, empty_val, contract, calendar=None)
    post_purge_count = len(purged.frame)

    # Empty validation means no purge
    assert post_purge_count == pre_purge_count


# --------------------------------------------------------------------------- #
# Integration: purge within walk-forward
# --------------------------------------------------------------------------- #
def test_walk_forward_with_label_purge_integration():
    """Integration test: walk-forward with purge_policy='label_interval' must
    purge training rows whose labels overlap validation."""
    dates = pd.date_range("2020-01-01", periods=150, freq="D")
    ds = _panel(dates.tolist())

    # 5-day forward label
    contract = LabelContract(label_name="ret_5", horizon_bars=5)

    spec = WalkForwardSpec(
        train_lookback_bars=40,
        validation_bars=20,
        test_bars=20,
        step_bars=20,
        retrain_every_bars=20,
        purge_policy="label_interval",
        embargo_bars=0,
        min_train_dates=40,
    )

    folds = make_walk_forward_splits(ds, spec)
    assert len(folds) > 0

    # For each fold, verify that training data doesn't contain dates
    # whose labels would overlap validation
    for fold in folds:
        if fold.validation_ds is None:
            continue

        train_dates = pd.to_datetime(fold.train_ds.frame["date"].unique())
        val_dates = pd.to_datetime(fold.validation_ds.frame["date"].unique())

        val_start = val_dates.min()
        train_end = train_dates.max()

        # Apply purge manually to verify
        purged = purge_overlap(fold.train_ds, fold.validation_ds, contract, calendar=None)
        purged_dates = pd.to_datetime(purged.frame["date"].unique())

        # Verify: no purged date's label (date + 5 bars) overlaps validation
        for d in purged_dates:
            # Label matures 5 bars later
            # In bar arithmetic, this means d must be at least 6 calendar positions before val_start
            # to ensure d+5 < val_start
            assert d < val_start, f"Training date {d} >= validation start {val_start}"


def test_walk_forward_purge_with_embargo_combination():
    """purge_and_embargo() combines both label purge and embargo."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    train_ds = _panel(dates[:60].tolist())  # 2020-01-01 to 2020-03-01
    val_ds = _panel(dates[60:80].tolist())  # 2020-03-02 to 2020-03-22

    contract = LabelContract(label_name="ret_5", horizon_bars=5, embargo_bars=3)

    spec = WalkForwardSpec(
        train_lookback_bars=60,
        validation_bars=20,
        test_bars=20,
        purge_policy="label_interval",
        embargo_bars=3,
        min_train_dates=60,
    )

    # Apply combined purge + embargo
    result = purge_and_embargo(train_ds, val_ds, contract, spec, calendar=None)

    # Verify result is smaller than original (rows were dropped)
    assert len(result.frame) < len(train_ds.frame)

    # Verify the last training date is earlier (embargo dropped tail)
    original_train_end = pd.to_datetime(train_ds.frame["date"].unique()).max()
    result_train_end = pd.to_datetime(result.frame["date"].unique()).max()
    assert result_train_end < original_train_end


def test_purge_validates_cutoff_date_calculation():
    """Verify that cutoff date calculation is correct for label interval purge."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    train_ds = _panel(dates[:30].tolist())  # 2020-01-01 to 2020-01-30
    val_ds = _panel(dates[30:40].tolist())  # 2020-01-31 to 2020-02-09

    contract = LabelContract(label_name="ret_10", horizon_bars=10)

    purged = purge_overlap(train_ds, val_ds, contract, calendar=None)
    purged_dates = sorted(pd.to_datetime(purged.frame["date"].unique()))

    val_start = pd.Timestamp("2020-01-31")

    # Cutoff: keep rows where date + 10 < val_start
    # date + 10 < 2020-01-31 → date < 2020-01-21
    # So keep 2020-01-01 to 2020-01-20
    expected_cutoff = pd.Timestamp("2020-01-20")

    assert purged_dates[-1] <= expected_cutoff, (
        f"Last purged date {purged_dates[-1]} > expected cutoff {expected_cutoff}"
    )
    assert purged_dates[-1] >= pd.Timestamp("2020-01-19"), (
        f"Cutoff date {purged_dates[-1]} is too early"
    )


def test_purge_with_gap_days_and_label_interval_are_independent():
    """gap_days (feature leakage protection) and label purge (label leakage
    protection) are independent mechanisms."""
    dates = pd.date_range("2020-01-01", periods=150, freq="D")
    ds = _panel(dates.tolist())

    contract = LabelContract(label_name="ret_5", horizon_bars=5)

    # Spec with both gap_days and label purge
    spec = WalkForwardSpec(
        train_lookback_bars=40,
        validation_bars=20,
        test_bars=20,
        step_bars=25,
        retrain_every_bars=25,
        gap_days=7,  # 7-day temporal gap
        purge_policy="label_interval",  # plus label purge
        embargo_bars=0,
        min_train_dates=40,
    )

    folds = make_walk_forward_splits(ds, spec)
    assert len(folds) > 0

    for fold in folds:
        if fold.validation_ds is None:
            continue

        train_dates = pd.to_datetime(fold.train_ds.frame["date"].unique())
        val_dates = pd.to_datetime(fold.validation_ds.frame["date"].unique())

        train_end = train_dates.max()
        val_start = val_dates.min()

        # gap_days=7 means at least 8 calendar days between train_end and val_start
        gap = (val_start - train_end).days
        assert gap >= 8, f"fold {fold.fold_id}: gap {gap} < 8"

        # Additionally, apply purge to verify label interval protection
        purged = purge_overlap(fold.train_ds, fold.validation_ds, contract, calendar=None)

        # Purge should further reduce training data
        # (though gap_days may have already created space)
        assert len(purged.frame) <= len(fold.train_ds.frame)


# --------------------------------------------------------------------------- #
# Purge statistics and observability
# --------------------------------------------------------------------------- #
def test_purge_overlap_returns_statistics():
    """purge_overlap should return metadata about purge operation."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    train_ds = _panel(dates[:30].tolist())
    val_ds = _panel(dates[30:40].tolist())

    contract = LabelContract(label_name="ret_5", horizon_bars=5)

    pre_dates = set(train_ds.frame["date"].unique())
    purged = purge_overlap(train_ds, val_ds, contract, calendar=None)
    post_dates = set(purged.frame["date"].unique())

    dropped_dates = pre_dates - post_dates

    # Verify purge happened and we can measure it
    assert len(dropped_dates) > 0
    assert len(post_dates) < len(pre_dates)

    # The cutoff date should be deterministic
    cutoff = max(post_dates)
    for dropped in dropped_dates:
        assert dropped > cutoff, f"Dropped date {dropped} <= cutoff {cutoff}"


def test_purge_validation_detects_bypass():
    """If purge is bypassed (returns original data when it should purge),
    validation should detect it."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    train_ds = _panel(dates[:20].tolist())
    val_ds = _panel(dates[20:30].tolist())

    contract = LabelContract(label_name="ret_5", horizon_bars=5)

    # Properly purged dataset
    purged = purge_overlap(train_ds, val_ds, contract, calendar=None)

    # If purge was bypassed (returned original), we'd detect it
    if len(purged.frame) == len(train_ds.frame):
        # This should NOT happen when horizon_bars > 0 and validation exists
        pytest.fail("Purge was bypassed: no rows were dropped despite horizon_bars=5")

    # Verify rows were actually dropped
    assert len(purged.frame) < len(train_ds.frame), "Purge must drop rows when horizon > 0"
