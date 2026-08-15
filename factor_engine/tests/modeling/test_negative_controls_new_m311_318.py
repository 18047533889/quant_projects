# -*- coding: utf-8 -*-
"""NEW-M-311~318 negative-control contract and known-bad mutation tests."""
from __future__ import annotations

import numpy as np

from modeling.artifact import PredictionContext
from modeling.contracts import ApplicationWindow

from modeling.diagnostics import feature_ablation, label_shuffle_control
from modeling.leakage_guard import future_poison, scaler_poison, synthetic_panel
from modeling.learners import PCRLearner
from modeling.learners.base import LearnerSpec


def _learner(seed):
    return PCRLearner(
        LearnerSpec(
            learner_name=PCRLearner.name,
            family=PCRLearner.family,
            hyperparams={"n_components": 2},
            random_seed=seed,
        )
    )


def _corr(pred, y):
    return float(np.corrcoef(np.asarray(pred), np.asarray(y))[0, 1])


def test_future_and_scaler_poison_rerun_pipeline_and_verify_mutation():
    calls = []

    def trainer(ds, **kwargs):
        from modeling.leakage_guard import default_trainer_fn

        calls.append((float(ds.frame[ds.feature_cols].max().max()), kwargs))
        return default_trainer_fn(ds, **kwargs)

    future = future_poison(trainer_fn=trainer, poison_row=8)
    scaler = scaler_poison(train_and_eval_fn=trainer, poison_after=8)
    for result in (future, scaler):
        assert result["status"] == "PASS"
        assert result["exercised"] is True
        assert result["mutation_effect_verified"] is True
        assert result["pass"] is True
        assert "authoritative_pipeline_rerun=True" in result["detail"]
    assert len(calls) == 4
    assert any(maximum == 1.0e9 for maximum, _ in calls)


def test_vacuous_poison_controls_are_invalid_not_pass():
    future = future_poison(poison_row=999)
    scaler = scaler_poison(poison_after=999)
    for result in (future, scaler):
        assert result["status"] == "INVALID_FIXTURE"
        assert result["exercised"] is False
        assert result["mutation_effect_verified"] is False
        assert result["pass"] is False


def test_label_shuffle_is_strict_oos_and_supports_structured_shuffles():
    rng = np.random.default_rng(7)
    dates = np.repeat(np.arange(20), 12)
    X = rng.normal(size=(len(dates), 3))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0, 0.1, size=len(X))
    for mode, block_size in (("within_date", None), ("block", 12)):
        result = label_shuffle_control(
            _learner, X, y, _corr, n_shuffles=5, dates=dates,
            shuffle_mode=mode, block_size=block_size, split_index=180,
        )
        assert result["evaluation_scope"] == "strict_oos"
        assert result["shuffle_mode"] == mode
        assert result["exercised"] is True
        assert result["mutation_effect_verified"] is True
        assert result["status"] in {"PASS", "FAIL"}


def test_label_shuffle_known_bad_learner_fails_null_control():
    class LabelEchoLearner:
        def fit(self, X, y):
            return np.asarray(y)

        def predict(self, frozen, X):
            # Known-bad oracle: ignores fitted state and reconstructs OOS labels.
            return 2.0 * X[:, 0] - X[:, 1]

    rng = np.random.default_rng(9)
    X = rng.normal(size=(200, 2))
    y = 2.0 * X[:, 0] - X[:, 1]
    result = label_shuffle_control(
        lambda seed: LabelEchoLearner(), X[:140], y[:140], _corr,
        X_oos=X[140:], y_oos=y[140:], n_shuffles=3,
    )
    assert result["status"] == "FAIL"
    assert result["pass"] is False
    assert result["leakage_suspected"] is True


def test_feature_diagnostics_distinguish_frozen_perturbation_and_retraining():
    ds = synthetic_panel(n_dates=10, n_stocks=12, seed=3)
    X, y, dates, _, finite = ds.as_matrix()
    X, y, dates = X[finite], y[finite], dates[finite]
    base = _learner(0)
    frozen = base.fit(X, y)

    class Artifact:
        manifest = type("Manifest", (), {"feature_schema_hash": "test"})()

        def predict(self, values, *, context):
            return base.predict(frozen, values)

    retrain_calls = []

    class Retrained:
        def __init__(self, removed):
            self.removed = removed

        def predict(self, values):
            retrain_calls.append(self.removed)
            return values[:, 0]

    context = PredictionContext(
        application_window=ApplicationWindow(start=dates.min(), end=dates.max()),
        dates=dates,
        asof=dates.max(),
        feature_schema_hash="test",
    )
    result = feature_ablation(
        Artifact(), X, y, _corr, dates=dates, shuffle_mode="within_date",
        retrain_without_feature_fn=Retrained, context=context,
    )
    assert result["retrained"] is True
    assert set(result["occlusion"]) == set(result["permutation"])
    assert set(result["retrained_feature_ablation"]) == {"0", "1", "2"}
    assert sorted(retrain_calls) == [0, 1, 2]
