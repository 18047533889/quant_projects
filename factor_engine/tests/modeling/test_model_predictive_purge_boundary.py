# -*- coding: utf-8 -*-
"""P0 fix — Purge boundary without a validation window.

``_purged_obs`` in sample_policy only purges when a validation window exists.
When training directly into a test window (``validation_ds is None``) the
train→test boundary MUST still be purged: any training row whose label interval
``[t, t+H]`` overlaps the test start would otherwise be matured using
test-period bars (a leak).  ``purge_before_boundary`` (in modeling.walk_forward)
closes that gap and the trainer applies it when ``evaluation_boundary`` is
provided.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.contracts import (
    AFTER_CLOSE_TO_NEXT_VWAP,
    LabelContract,
    SampleAdequacyContract,
    ashare_decision_clock,
)
from modeling.dataset import PanelDataset
from modeling.learners import PCRLearner
from modeling.trainer import PreprocessingSpec, train_model
from modeling.walk_forward import purge_before_boundary


def _panel(n_dates=8, n_stocks=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    rows = []
    for d in dates:
        for s in range(n_stocks):
            x = rng.normal(size=4)
            rows.append([d, f"S{s:03d}", *x.tolist(), float(0.5 * x[0] - 0.2 * x[1] + rng.normal(0, 0.1))])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    return PanelDataset(
        frame=frame, date_col="date", stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"], label_col="label",
    )


def _lenient() -> SampleAdequacyContract:
    return SampleAdequacyContract(
        min_raw_obs=50, min_effective_obs=20, min_unique_dates=3,
        min_unique_stocks=10, min_obs_per_parameter=2,
    )


def test_purge_before_boundary_drops_label_overlap_rows():
    # integer dates 0..6, horizon 2, boundary 4 -> keep dates where d + 2 < 4 -> 0,1
    rows = []
    for d in range(7):
        for s in (0, 1):
            rows.append([d, f"S{s}", 1.0, 1.0, 1.0, 1.0, 0.0])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    ds = PanelDataset(
        frame=frame, date_col="date", stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"], label_col="label",
    )
    contract = LabelContract(label_name="ret_2", horizon_bars=2)
    purged = purge_before_boundary(ds, 4, contract)
    assert sorted(purged.frame["date"].unique()) == [0, 1]
    assert purged.n_rows == 4  # 2 dates * 2 stocks


def test_purge_before_boundary_none_boundary_is_noop():
    ds = _panel()
    c = LabelContract(label_name="ret_1", horizon_bars=1)
    out = purge_before_boundary(ds, None, c)
    assert out is ds


def test_train_model_no_validation_purges_test_boundary():
    """validation_ds=None + evaluation_boundary=test_start must purge the train
    tail whose labels would mature inside the test window."""
    ds = _panel(n_dates=10, n_stocks=100, seed=3)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[6])   # 7 dates
    test_start = dates[7]
    contract = LabelContract(label_name="ret_1", horizon_bars=1)

    # With evaluation_boundary, the last train date (dates[6]) is purged:
    # d + 1 >= test_start(dates[7]).
    result = train_model(
        PCRLearner, train_ds, None,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=contract,
        decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
        sample_contract=_lenient(),
        evaluation_boundary=test_start,
    )
    m = result.artifact.manifest
    # selection_train_end advanced to the last row that does NOT leak into test
    assert m.selection_train_end < str(dates[6])
    assert m.selection_train_end <= str(dates[5])


def test_train_model_no_validation_without_boundary_no_purge():
    """Backward compatible: no evaluation_boundary -> no purge of the tail."""
    ds = _panel(n_dates=10, n_stocks=100, seed=4)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[6])
    contract = LabelContract(label_name="ret_1", horizon_bars=1)
    result = train_model(
        PCRLearner, train_ds, None,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=contract,
        decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
        sample_contract=_lenient(),
    )
    m = result.artifact.manifest
    assert m.selection_train_end == str(dates[6])


def test_validation_path_still_purges_by_validation_start():
    """When validation exists, the existing purge_and_embargo path is used."""
    ds = _panel(n_dates=10, n_stocks=100, seed=5)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[5])
    val_ds = ds.filter_dates(start=dates[6], end=dates[7])
    contract = LabelContract(label_name="ret_1", horizon_bars=1)
    result = train_model(
        PCRLearner, train_ds, val_ds,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=contract,
        decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
        sample_contract=_lenient(),
        evaluation_boundary=dates[8],
    )
    m = result.artifact.manifest
    assert m.validation_start == str(dates[6])
    # selection_train_end purged against validation start (dates[6]):
    # keep d + 1 < dates[6] -> d <= dates[4]
    assert m.selection_train_end <= str(dates[4])
