# -*- coding: utf-8 -*-
"""P0 fix — WalkForwardSpec fields that were declared but never executed.

* ``retrain_every_bars`` — the fold emission stride.  Previously only
  ``step_bars`` advanced the walk-forward loop and ``retrain_every_bars`` was
  dead.  Now a fold is emitted per retrain point (stride =
  ``retrain_every_bars`` when > 0, else ``step_bars``).
* ``purge_policy`` — ``"label_interval"`` (default) purges rows whose label
  interval overlaps the validation window; ``"purge_bars"`` purges a fixed
  ``embargo_bars``-bar buffer off the training tail.  Previously only the
  label-interval path existed.
* ``decay_half_life_bars`` — §3.4 / §73 per-row half-life decay sample weights,
  wired into the trainer as an opt-in ``decay_half_life_bars`` parameter
  (weights are passed to ``learner.fit``).
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
from modeling.trainer import PreprocessingSpec, train_model, _extract_matrices
from modeling.walk_forward import (
    WalkForwardSpec,
    apply_embargo,
    decay_weights,
    make_walk_forward_splits,
    purge_and_embargo,
)


def _panel(n_dates=12, n_stocks=40, seed=0):
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


# --------------------------------------------------------------------------- #
# retrain_every_bars controls fold emission
# --------------------------------------------------------------------------- #
def test_retrain_every_bars_controls_fold_stride():
    ds = _panel(n_dates=14)
    base = dict(
        train_lookback_bars=3, validation_bars=2, test_bars=2,
        step_bars=1, min_train_dates=3, min_train_stocks=3, min_train_obs=20,
    )
    spec_e1 = WalkForwardSpec(**base, retrain_every_bars=1)
    spec_e2 = WalkForwardSpec(**base, retrain_every_bars=2)
    folds1 = make_walk_forward_splits(ds, spec_e1)
    folds2 = make_walk_forward_splits(ds, spec_e2)
    # stride 1 emits ~2x the folds of stride 2
    assert len(folds1) > len(folds2)
    assert len(folds1) >= 4
    assert len(folds2) >= 2
    starts2 = [pd.Timestamp(f.test_start) for f in folds2]
    gaps = {int(np.busday_count(a.date(), b.date()) / 7) * 7 for a, b in zip(starts2[:-1], starts2[1:])}
    # test starts advance by 2 bars (the retrain cadence)
    deltas = [(b - a).days for a, b in zip(starts2[:-1], starts2[1:])]
    assert all(d == 2 for d in deltas)


def test_step_bars_fallback_when_retrain_zero():
    ds = _panel(n_dates=12)
    spec = WalkForwardSpec(
        train_lookback_bars=3, validation_bars=2, test_bars=2,
        step_bars=1, retrain_every_bars=0,
        min_train_dates=3, min_train_stocks=3, min_train_obs=20,
    )
    folds = make_walk_forward_splits(ds, spec)
    assert len(folds) >= 3  # stride falls back to step_bars=1


# --------------------------------------------------------------------------- #
# purge_policy
# --------------------------------------------------------------------------- #
def test_purge_policy_label_interval_default():
    rows = []
    for d in range(7):
        for s in (0, 1):
            rows.append([d, f"S{s}", 1.0, 1.0, 1.0, 1.0, 0.0])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    ds = PanelDataset(frame=frame, date_col="date", stock_col="stock",
                      feature_cols=["f0", "f1", "f2", "f3"], label_col="label")
    val = ds.filter_dates(start=4, end=5)
    c = LabelContract(label_name="ret_2", horizon_bars=2)
    # label_interval: keep d + 2 < 4 -> d <= 1
    purged = purge_and_embargo(ds, val, c, WalkForwardSpec(purge_policy="label_interval"))
    assert sorted(purged.frame["date"].unique()) == [0, 1]


def test_purge_policy_purge_bars_uses_embargo_buffer():
    rows = []
    for d in range(7):
        for s in (0, 1):
            rows.append([d, f"S{s}", 1.0, 1.0, 1.0, 1.0, 0.0])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    ds = PanelDataset(frame=frame, date_col="date", stock_col="stock",
                      feature_cols=["f0", "f1", "f2", "f3"], label_col="label")
    val = ds.filter_dates(start=4, end=5)
    c = LabelContract(label_name="ret_2", horizon_bars=2)
    # purge_bars: drop the last embargo_bars=2 dates -> keep [0..4]
    purged = purge_and_embargo(ds, val, c, WalkForwardSpec(purge_policy="purge_bars", embargo_bars=2))
    assert sorted(purged.frame["date"].unique()) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# decay_half_life_bars wired into the trainer
# --------------------------------------------------------------------------- #
def test_decay_weights_latest_bar_weight_one():
    ds = _panel(n_dates=5, n_stocks=3)
    w = decay_weights(ds, 2)
    last_mask = (ds.frame["date"] == ds.frame["date"].max()).to_numpy()
    assert np.allclose(w[last_mask], 1.0)
    assert w.shape == (ds.n_rows,)
    assert np.all(w > 0)


def test_extract_matrices_aligns_decay_weights_to_finite_rows():
    from modeling.artifact import FrozenPreprocessing

    ds = _panel(n_dates=6, n_stocks=5)
    pre = FrozenPreprocessing([])
    X, y, aux, w = _extract_matrices(ds, pre, None, decay_half_life_bars=2)
    assert w is not None
    assert len(w) == len(y)
    # weights are per-row, positive, latest finite-label row ~ 1.0
    assert np.all(w > 0)
    assert np.allclose(w.max(), 1.0)


def test_train_model_with_decay_half_life_bars_works():
    ds = _panel(n_dates=10, n_stocks=100, seed=9)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[6])
    val_ds = ds.filter_dates(start=dates[7], end=dates[8])
    contract = LabelContract(label_name="ret_1", horizon_bars=1)
    result = train_model(
        PCRLearner, train_ds, val_ds,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=contract,
        decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
        sample_contract=_lenient(),
        decay_half_life_bars=2,
    )
    assert result.artifact is not None
    assert np.isfinite(result.artifact.predict(train_ds.as_matrix()[0][:10])).all()
