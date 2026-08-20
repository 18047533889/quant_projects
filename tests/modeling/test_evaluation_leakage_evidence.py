# -*- coding: utf-8 -*-
"""Tests for modeling/evaluation, leakage_guard, evidence (§26/§27/§58/§71/§75).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.artifact import FrozenPreprocessing
from modeling.contracts import (
    DecisionClock,
    LabelContract,
    ashare_decision_clock,
)
from modeling.evaluation import (
    EvaluationReport,
    block_aware_ic,
    evaluate_predictions,
    per_date_rank_ic,
)
from modeling.evidence import WalkForwardEvidence, report_hard_gate_set, run_walk_forward_evidence
from modeling.leakage_guard import (
    TrainOnlyFitGuard,
    assert_frozen_preprocessing,
    default_dataset_fn,
    default_trainer_fn,
    future_poison,
    hyperparam_poison,
    label_poison,
    run_all_negative_controls,
    scaler_poison,
    synthetic_panel,
)
from modeling.learners import PCRLearner
from modeling.learners.base import LearnerSpec


# --------------------------------------------------------------------------- #
# evaluation (§27)
# --------------------------------------------------------------------------- #
def test_per_date_rank_ic_positive_when_y_matches_pred():
    rng = np.random.default_rng(1)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=5), 20)
    pred = rng.normal(size=len(dates))
    y = pred + rng.normal(0, 0.1, size=len(dates))
    dates_sorted, ics = per_date_rank_ic(pred, y, dates)
    assert len(dates_sorted) == 5
    assert len(ics) == 5
    assert np.all(ics > 0.5)


def test_per_date_rank_ic_drops_small_dates():
    # two pairs per date is below the >= 3 floor -> date is dropped.
    pred = np.array([1.0, 2.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 2.0, 1.0])
    dates = np.array(["d0", "d0", "d1", "d1"])
    dates_sorted, ics = per_date_rank_ic(pred, y, dates)
    assert len(dates_sorted) == 0
    assert len(ics) == 0


def test_evaluate_predictions_finite_on_synthetic_panel():
    ds = synthetic_panel(n_dates=8, n_stocks=15, n_features=3, seed=2)
    X, y, dates, stocks, finite = ds.as_matrix()
    mask = finite & np.isfinite(y)
    # noise model with real signal so rank_ic is meaningful
    rng = np.random.default_rng(3)
    pred = 0.7 * X[mask, 0] + 0.2 * X[mask, 1] + rng.normal(0, 0.05, size=mask.sum())
    rep = evaluate_predictions(pred, y[mask], dates[mask])
    assert np.isfinite(rep.mse)
    assert np.isfinite(rep.mae)
    assert np.isfinite(rep.rank_ic)
    assert 0.0 <= rep.coverage <= 1.0
    assert isinstance(rep, EvaluationReport)
    d = rep.to_dict()
    assert "rank_ic" in d and np.isfinite(d["rank_ic"])


def test_block_aware_ic_reduces_blocks():
    rng = np.random.default_rng(4)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=12), 10)
    pred = rng.normal(size=len(dates))
    y = pred + rng.normal(0, 0.1, size=len(dates))
    d_full, ics_full = per_date_rank_ic(pred, y, dates)
    d_block, ics_block = block_aware_ic(pred, y, dates, overlap_horizon=4)
    assert len(d_block) < len(d_full)
    assert len(ics_block) == len(d_block)
    assert abs(float(np.mean(ics_block)) - float(np.mean(ics_full))) < 0.2


# --------------------------------------------------------------------------- #
# leakage_guard (§13 / §58)
# --------------------------------------------------------------------------- #
def test_train_only_fit_guard_raises_on_fit():
    learner = PCRLearner(
        LearnerSpec(learner_name="predictive_pcr", family="pcr", hyperparams={"n_components": 3})
    )
    with TrainOnlyFitGuard(learner):
        try:
            learner.fit(np.random.rand(20, 3), np.random.rand(20))
            raised = False
        except RuntimeError as exc:
            raised = "fit called during scoring" in str(exc)
    assert raised


def test_assert_frozen_preprocessing_rowwise_deterministic():
    X = np.random.default_rng(0).normal(size=(10, 3))
    preproc = FrozenPreprocessing(
        [
            {
                "kind": "standardize",
                "mean": X.mean(axis=0).tolist(),
                "scale": X.std(axis=0).tolist(),
            }
        ]
    )
    X_future = np.vstack([X, np.ones((1, 3)) * 999.0])
    assert assert_frozen_preprocessing(preproc, X_future, X)


def test_future_poison_control_passes():
    res = future_poison(poison_row=8)
    assert res["future_poison"] is True


def test_label_poison_control_passes():
    lc = LabelContract(label_name="y", horizon_bars=1)
    res = label_poison(lc, horizon=1)
    assert res["label_poison"] is True


def test_scaler_poison_control_passes():
    res = scaler_poison(poison_after=8)
    assert res["scaler_poison"] is True


def test_hyperparam_poison_control_passes():
    res = hyperparam_poison()
    assert res["hyperparam_poison"] is True


def test_run_all_negative_controls_returns_bools():
    res = run_all_negative_controls()
    for name, outcome in res.items():
        assert isinstance(outcome, dict)
        assert isinstance(outcome.get(name), bool)


# --------------------------------------------------------------------------- #
# evidence (§71 / §83)
# --------------------------------------------------------------------------- #
def _wf_spec():
    return {
        "train_lookback_bars": 4,
        "validation_bars": 2,
        "test_bars": 2,
        "step_bars": 2,
        "embargo_bars": 0,
    }


def _lenient_contract():
    from modeling.contracts import SampleAdequacyContract

    return SampleAdequacyContract(
        min_raw_obs=30, min_effective_obs=15, min_unique_dates=2,
        min_unique_stocks=5, min_obs_per_parameter=2,
    )


def test_run_walk_forward_evidence_min_folds_2():
    ds = synthetic_panel(n_dates=12, n_stocks=20, n_features=3, seed=7)
    lc = LabelContract(label_name="y", horizon_bars=1)
    clock = ashare_decision_clock()
    evidence = run_walk_forward_evidence(
        PCRLearner,
        ds,
        _wf_spec(),
        label_contract=lc,
        decision_clock=clock,
        hyperparam_grid=[{"n_components": 2}, {"n_components": 3}],
        model_version="1.0",
        random_seed=0,
        min_folds=2,
        # synthetic panel is below the production 50k-obs floor — relax the
        # adequacy contract for the demo, as the production runner would for a
        # real DataAccess panel.
        sample_contract=_lenient_contract(),
    )
    assert isinstance(evidence, WalkForwardEvidence)
    assert len(evidence.folds) >= 2
    assert np.isfinite(evidence.oos_evaluation["rank_ic"])
    assert evidence.oos_evaluation["coverage"] > 0
    # §56 ledger rows present
    assert len(evidence.leadger_rows) == len(evidence.folds)
    row = evidence.leadger_rows[0]
    for key in (
        "candidate_id",
        "model_family",
        "artifact_family",
        "feature_set",
        "parameter_set",
        "train_range",
        "validation_range",
        "OOS_range",
        "number_of_search_attempts_before_selection",
        "selection_metric",
    ):
        assert key in row


def test_run_walk_forward_evidence_to_parquet(tmp_path):
    ds = synthetic_panel(n_dates=12, n_stocks=15, n_features=3, seed=8)
    lc = LabelContract(label_name="y", horizon_bars=1)
    evidence = run_walk_forward_evidence(
        PCRLearner,
        ds,
        _wf_spec(),
        label_contract=lc,
        decision_clock=ashare_decision_clock(),
        hyperparam_grid=[{"n_components": 2}],
        min_folds=2,
        sample_contract=_lenient_contract(),
    )
    out = tmp_path / "wf.parquet"
    evidence.to_parquet(str(out))
    assert out.exists()
    df = pd.read_parquet(out)
    assert list(df.columns) == ["fold_id", "date", "stock", "pred", "y"]


def test_report_hard_gate_set_all_present_and_bool():
    gates = report_hard_gate_set()
    expected = {
        "MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH",
        "MODEL_ALL_PREDICTIVE_LEARNERS_HAVE_WALK_FORWARD_SPEC",
        "MODEL_ALL_PREDICTIVE_LEARNERS_HAVE_VALIDATION",
        "MODEL_ALL_PREDICTIVE_LEARNERS_PURGE_LABEL_OVERLAP",
        "MODEL_ZERO_RANDOM_TIME_SPLIT",
        "MODEL_ZERO_FULL_SAMPLE_SCALER",
        "MODEL_ZERO_FULL_SAMPLE_PCA_PREPROCESS",
        "MODEL_ZERO_FULL_SAMPLE_FEATURE_SELECTION",
        "MODEL_ZERO_TEST_DRIVEN_HYPERPARAM_SELECTION",
        "MODEL_ALL_ARTIFACTS_ASOF_RESOLVED",
        "MODEL_ALL_MODELS_HAVE_SAMPLE_ADEQUACY_CONTRACT",
        "MODEL_ALL_SEARCHABLE_PARAMS_HAVE_SEARCH_POLICY",
        "MODEL_ZERO_NUMERICAL_POLICY_SEARCHABLE",
        "MODEL_ZERO_DATA_POLICY_SEARCHABLE_BY_FACTOR_MINER",
        "MODEL_REGIME_ALL_COMPONENTS_HAVE_SUPPORT",
        "MODEL_MOE_ALL_ACTIVE_EXPERTS_HAVE_SUPPORT",
        "MODEL_FINAL_HOLDOUT_NOT_EXPOSED_TO_SEARCH",
        "MODEL_CURRENT_HEAD_EVIDENCE_FRESH",
        "MODEL_ALL_DIRECT_USE_HAVE_BEHAVIORAL_CERTIFICATION",
    }
    assert set(gates.keys()) == expected
    for name, entry in gates.items():
        assert isinstance(entry, dict)
        assert "value" in entry and "check" in entry
        assert isinstance(entry["value"], bool)
