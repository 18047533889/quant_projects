# -*- coding: utf-8 -*-
"""Training-orchestration tests: walk-forward splits, purge/embargo, trainer,
predictor, registry, selection, hyperparameter governance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.artifact import ModelArtifact
from modeling.contracts import (
    AFTER_CLOSE_TO_NEXT_VWAP,
    ModelExecutionClass,
    ParamRole,
    SampleAdequacyContract,
    ashare_decision_clock,
)
from modeling.dataset import PanelDataset
from modeling.hyperparams import APPROVED_SEARCH_SPACES, param_search_policy
from modeling.learners import PCRLearner
from modeling.predictor import Predictor, batch_predict, predict_panel
from modeling.registry import MODEL_REGISTRY, ModelRegistry, register_default_learners
from modeling.selection import neighborhood_stability, select_best_validation
from modeling.split import (
    assert_date_authoritative,
    date_bounded_split,
    split_by_date_cutoff,
)
from modeling.timing import vwap_to_vwap_label
from modeling.trainer import PreprocessingSpec, TrainResult, train_model
from modeling.walk_forward import (
    WalkForwardFold,
    WalkForwardSpec,
    apply_embargo,
    check_fold_order,
    decay_weights,
    make_walk_forward_splits,
    nested_splits,
    purge_and_embargo,
    purge_overlap,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_panel(n_dates=8, n_stocks=40, n_features=4, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_dates, freq="D")
    rows = []
    for d in dates:
        for s in range(n_stocks):
            x = rng.normal(size=n_features)
            y = float(0.5 * x[0] - 0.2 * x[1] + rng.normal(0, 0.1))
            rows.append([d, f"S{s:03d}", *x.tolist(), y])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    return PanelDataset(
        frame=frame,
        date_col="date",
        stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"],
        label_col="label",
    )


def _lenient_contract() -> SampleAdequacyContract:
    return SampleAdequacyContract(
        min_raw_obs=500,
        min_effective_obs=300,
        min_unique_dates=3,
        min_unique_stocks=30,
        min_obs_per_parameter=10,
    )


# --------------------------------------------------------------------------- #
# walk-forward splits
# --------------------------------------------------------------------------- #
def test_make_walk_forward_splits_date_authoritative():
    ds = make_panel(n_dates=8, n_stocks=10)
    spec = WalkForwardSpec(
        train_lookback_bars=3,
        validation_bars=2,
        test_bars=2,
        step_bars=1,
        retrain_every_bars=1,
        min_train_dates=3,
        min_train_stocks=3,
        min_train_obs=20,
    )
    folds = make_walk_forward_splits(ds, spec)
    assert len(folds) == 2
    assert check_fold_order(folds) == []
    for f in folds:
        tr = set(f.train_ds.frame["date"].tolist())
        va = set(f.validation_ds.frame["date"].tolist())
        te = set(f.test_ds.frame["date"].tolist())
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)
        assert f.train_end < f.validation_start < f.test_start


def test_check_fold_order_reports_violations():
    ds = make_panel(n_dates=8, n_stocks=5)
    dates = sorted(ds.frame["date"].unique())
    # deliberately overlapping: train_end == validation_start
    fold = WalkForwardFold(
        fold_id=0,
        train_start=dates[0],
        train_end=dates[3],
        validation_start=dates[2],
        validation_end=dates[4],
        test_start=dates[5],
        test_end=dates[6],
        train_ds=ds.filter_dates(start=dates[0], end=dates[3]),
        validation_ds=ds.filter_dates(start=dates[2], end=dates[4]),
        test_ds=ds.filter_dates(start=dates[5], end=dates[6]),
    )
    violations = check_fold_order([fold])
    assert any("train_end >= validation_start" in v for v in violations)

    # train/test date overlap (validation absent, test starts inside train)
    fold2 = WalkForwardFold(
        fold_id=1,
        train_start=dates[0],
        train_end=dates[4],
        validation_start=None,
        validation_end=None,
        test_start=dates[3],
        test_end=dates[6],
        train_ds=ds.filter_dates(start=dates[0], end=dates[4]),
        validation_ds=None,
        test_ds=ds.filter_dates(start=dates[3], end=dates[6]),
    )
    violations2 = check_fold_order([fold2])
    assert any("train/test date overlap" in v for v in violations2)


# --------------------------------------------------------------------------- #
# purge / embargo
# --------------------------------------------------------------------------- #
def test_purge_overlap_exact_row_counts():
    # integer dates 0..6, 2 stocks, horizon 2, validation starts at 4.
    rows = []
    for d in range(7):
        for s in (0, 1):
            rows.append([d, f"S{s}", 1.0, 1.0, 1.0, 1.0, 0.0])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    ds = PanelDataset(
        frame=frame, date_col="date", stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"], label_col="label",
    )
    val_ds = ds.filter_dates(start=4, end=5)
    contract = vwap_to_vwap_label("ret_2", horizon_bars=2)
    purged = purge_overlap(ds, val_ds, contract)
    kept_dates = sorted(purged.frame["date"].unique())
    # anchor + horizon must be strictly before validation start (4) -> dates <= 1
    assert kept_dates == [0, 1]
    assert len(purged.frame) == 4  # 2 dates * 2 stocks


def test_apply_embargo_drops_last_dates():
    ds = make_panel(n_dates=8, n_stocks=5)
    emb = apply_embargo(ds, 2)
    dates = sorted(ds.frame["date"].unique())
    assert sorted(emb.frame["date"].unique()) == dates[:6]


def test_purge_and_embargo_combines():
    ds = make_panel(n_dates=8, n_stocks=5)
    dates = sorted(ds.frame["date"].unique())
    val_ds = ds.filter_dates(start=dates[4], end=dates[5])
    contract = vwap_to_vwap_label("ret_1", horizon_bars=1)
    spec = WalkForwardSpec(embargo_bars=1)
    purged = purge_and_embargo(ds, val_ds, contract, spec)
    # purge keeps dates <= 2 (i + 1 < 4); embargo drops last 1 -> dates 0..1
    assert sorted(purged.frame["date"].unique()) == [dates[0], dates[1]]


def test_decay_weights_latest_date_is_one():
    ds = make_panel(n_dates=5, n_stocks=3)
    w = decay_weights(ds, 2)
    assert w.shape == (ds.n_rows,)
    assert np.all(w > 0)
    last_mask = (ds.frame["date"] == ds.frame["date"].max()).to_numpy()
    assert np.allclose(w[last_mask], 1.0)


def test_nested_splits():
    ds = make_panel(n_dates=12, n_stocks=8)
    outer = WalkForwardSpec(
        train_lookback_bars=4, validation_bars=2, test_bars=2, step_bars=2,
        retrain_every_bars=2, min_train_dates=4, min_train_stocks=3,
        min_train_obs=20,
    )
    inner = WalkForwardSpec(
        train_lookback_bars=3, validation_bars=1, test_bars=1, step_bars=1,
        retrain_every_bars=1, min_train_dates=3, min_train_stocks=3,
        min_train_obs=10,
    )
    out = nested_splits(ds, outer, inner)
    assert out
    for outer_fold, inner_folds in out:
        assert isinstance(outer_fold, WalkForwardFold)
        assert all(isinstance(f, WalkForwardFold) for f in inner_folds)
        assert inner_folds  # at least one inner fold per outer range


# --------------------------------------------------------------------------- #
# split helpers
# --------------------------------------------------------------------------- #
def test_split_helpers_date_authoritative():
    ds = make_panel(n_dates=8, n_stocks=5)
    dates = sorted(ds.frame["date"].unique())
    before, after = split_by_date_cutoff(ds, dates[3])
    assert set(before.frame["date"].unique()) == set(dates[:4])
    assert set(after.frame["date"].unique()) == set(dates[4:])

    train, val, test = date_bounded_split(
        ds,
        train_start=dates[0], train_end=dates[2],
        validation_start=dates[3], validation_end=dates[4],
        test_start=dates[5], test_end=dates[7],
    )
    assert assert_date_authoritative(train, val, test) == []

    bad_train = ds.filter_dates(start=dates[0], end=dates[3])
    bad_val = ds.filter_dates(start=dates[2], end=dates[4])
    assert assert_date_authoritative(bad_train, bad_val, test) != []


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
def test_train_model_end_to_end():
    ds = make_panel(n_dates=10, n_stocks=200, seed=1)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[5])
    val_ds = ds.filter_dates(start=dates[6], end=dates[7])
    test_ds = ds.filter_dates(start=dates[8], end=dates[9])
    contract = vwap_to_vwap_label("ret_1", horizon_bars=1)
    clock = ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP)
    grid = [{"n_components": 2}, {"n_components": 3}]
    result = train_model(
        PCRLearner, train_ds, val_ds,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=grid,
        label_contract=contract,
        decision_clock=clock,
        feature_schema_hash="abc",
        universe_hash="u1",
        data_source_hash="d1",
        model_version="1.0",
        random_seed=7,
        sample_contract=_lenient_contract(),
    )
    assert isinstance(result, TrainResult)
    assert result.artifact is not None
    assert result.artifact.model_name == PCRLearner.name
    assert result.selected_hyperparams["n_components"] in (2, 3)
    assert len(result.validation_scores) == 2
    X_test = test_ds.as_matrix()[0]
    pred = result.artifact.predict(X_test)
    assert np.isfinite(pred).all()
    assert result.artifact.manifest.preprocessing_state_hash != ""
    assert isinstance(result.artifact, ModelArtifact)


def test_train_model_train_only_mode():
    """validation_ds=None (train→test only) still produces an artifact using an
    in-sample selection proxy."""
    ds = make_panel(n_dates=8, n_stocks=200, seed=5)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[5])
    contract = vwap_to_vwap_label("ret_1", horizon_bars=1)
    clock = ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP)
    result = train_model(
        PCRLearner, train_ds, None,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}, {"n_components": 3}],
        label_contract=contract,
        decision_clock=clock,
        sample_contract=_lenient_contract(),
    )
    assert result.artifact is not None
    assert result.selected_hyperparams["n_components"] in (2, 3)
    assert np.isfinite(result.artifact.predict(ds.as_matrix()[0][:10])).all()


def test_train_model_fails_closed_when_all_candidates_fail_adequacy():
    tiny = make_panel(n_dates=4, n_stocks=25, seed=2)  # 100 rows
    dates = sorted(tiny.frame["date"].unique())
    train_ds = tiny.filter_dates(start=dates[0], end=dates[2])  # 75 rows
    contract = vwap_to_vwap_label("ret_1", horizon_bars=1)
    clock = ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP)
    with pytest.raises(ValueError):
        train_model(
            PCRLearner, train_ds, None,
            preprocessing_spec=PreprocessingSpec(),
            hyperparam_grid=[{"n_components": 2}],
            label_contract=contract,
            decision_clock=clock,
            sample_contract=_lenient_contract(),
        )


class SpyPCRLearner(PCRLearner):
    def __init__(self, spec):
        super().__init__(spec)
        self.fit_calls = 0

    def fit(self, X, y, *, weights=None):
        self.fit_calls += 1
        return super().fit(X, y, weights=weights)


def test_predictor_never_calls_fit():
    ds = make_panel(n_dates=8, n_stocks=200, seed=3)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[4])
    val_ds = ds.filter_dates(start=dates[5], end=dates[6])
    contract = vwap_to_vwap_label("ret_1", horizon_bars=1)
    clock = ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP)
    result = train_model(
        SpyPCRLearner, train_ds, val_ds,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=contract,
        decision_clock=clock,
        sample_contract=_lenient_contract(),
    )
    spy = result.artifact.learner
    assert spy.fit_calls >= 1
    calls_before = spy.fit_calls
    X = ds.as_matrix()[0][:20]
    pred = Predictor().predict(result.artifact, X)
    assert np.isfinite(pred).all()
    assert spy.fit_calls == calls_before

    # panel + batch parity
    ds_slice = PanelDataset(
        frame=ds.frame.iloc[:40].reset_index(drop=True),
        date_col="date", stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"], label_col=None,
    )
    series = predict_panel(result.artifact, ds_slice)
    assert len(series) == 40
    batches = [ds_slice.as_matrix()[0][:20], ds_slice.as_matrix()[0][20:]]
    out = batch_predict(result.artifact, batches)
    assert len(out) == 2
    assert np.allclose(np.concatenate(out), series.to_numpy())


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_register_get_has_duplicate_reject():
    reg = ModelRegistry()
    reg.register("test_linear", ModelExecutionClass.PREDICTIVE_SUPERVISED, PCRLearner)
    assert reg.has("test_linear")
    rec = reg.get("test_linear")
    assert rec["learner_cls"] is PCRLearner
    assert rec["execution_class"] == ModelExecutionClass.PREDICTIVE_SUPERVISED
    assert "test_linear" in reg.all()
    with pytest.raises(ValueError):
        reg.register("test_linear", ModelExecutionClass.PREDICTIVE_SUPERVISED, PCRLearner)
    assert reg.registered_predictive_learners() == ["test_linear"]


def test_register_default_learners_registers_five_predictive():
    register_default_learners()
    learners = MODEL_REGISTRY.registered_predictive_learners()
    assert len(learners) == 5
    assert "predictive_pcr" in learners
    assert "predictive_pls" in learners
    assert "predictive_elastic_net" in learners
    assert "predictive_regime" in learners
    assert "predictive_mixture_of_experts" in learners
    rec = MODEL_REGISTRY.get("predictive_pcr")
    assert rec["training_spec"] is not None
    assert rec["parameter_policies"].get("n_components") is not None


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_select_best_validation():
    scores = [
        {"hyperparams": {"n_components": 2}, "rank_ic": 0.02, "mse": 0.5},
        {"hyperparams": {"n_components": 3}, "rank_ic": 0.05, "mse": 0.4},
        {"hyperparams": {"n_components": 5}, "rank_ic": 0.01, "mse": 0.6},
    ]
    best, diag = select_best_validation(scores)
    assert best == {"n_components": 3}
    assert diag["best_score"] == pytest.approx(0.05)


def test_neighborhood_stability():
    candidates = [
        {"hyperparams": {"n_components": 2}, "rank_ic": 0.04},
        {"hyperparams": {"n_components": 3}, "rank_ic": 0.05},
        {"hyperparams": {"n_components": 4}, "rank_ic": 0.049},
    ]
    stable, _ = neighborhood_stability(candidates, {"n_components": 3})
    assert stable is True

    razor = [
        {"hyperparams": {"n_components": 2}, "rank_ic": 0.01},
        {"hyperparams": {"n_components": 3}, "rank_ic": 0.10},
        {"hyperparams": {"n_components": 4}, "rank_ic": 0.05},
    ]
    stable2, details = neighborhood_stability(razor, {"n_components": 3})
    assert stable2 is False
    assert details["neighbors"]


# --------------------------------------------------------------------------- #
# hyperparameter governance
# --------------------------------------------------------------------------- #
def test_approved_search_spaces():
    assert APPROVED_SEARCH_SPACES["pcr"][0] == {"n_components": 2}
    assert len(APPROVED_SEARCH_SPACES["pls"]) == 4
    assert len(APPROVED_SEARCH_SPACES["elastic_net"]) == 20
    assert APPROVED_SEARCH_SPACES["regime"][0] == {"n_regimes": 2}
    assert APPROVED_SEARCH_SPACES["moe"][0] == {"n_experts": 2}


def test_param_search_policy():
    pol = param_search_policy("pcr", "n_components")
    assert pol is not None
    assert pol.role == ParamRole.MODEL_COMPLEXITY
    assert pol.searchable is True

    pol2 = param_search_policy("elastic_net", "alpha")
    assert pol2 is not None
    assert pol2.searchable is False

    assert param_search_policy("pcr", "bogus_param") is None
