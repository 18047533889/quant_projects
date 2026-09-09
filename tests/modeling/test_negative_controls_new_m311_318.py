# -*- coding: utf-8 -*-
"""NEW-M-311~318 negative-control contract and known-bad mutation tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

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


def test_feature_diagnostics_distinguish_frozen_perturbation_and_retraining(tmp_path):
    from factor_optimizer.contracts.campaign_store import DurableBudgetTracker, SQLiteCampaignStore
    from factor_optimizer.contracts.search_budget import SearchBudget
    from modeling.trainer_governance import FeatureExperimentSpec
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
        retrain_without_feature_fn=lambda j: (Retrained(j), f"qe:evidence:{j}"),
        experiment_spec=FeatureExperimentSpec(
            "model-ablation", "feature-set-v1", ("fold-1",), 0,
            "cpu-small-v1", "qe:baseline:1", "pcr-recipe-v1", 1, 1.0,
        ),
        budget_tracker=DurableBudgetTracker(
            SQLiteCampaignStore(tmp_path / "campaign.sqlite3"), "model-ablation",
            SearchBudget(max_trials=3, max_evaluations=3, max_cost_units=3.0),
        ),
        context=context,
    )
    assert result["retrained"] is True
    assert set(result["occlusion"]) == set(result["permutation"])
    assert set(result["retrained_feature_ablation"]) == {"0", "1", "2"}
    assert sorted(retrain_calls) == [0, 1, 2]
    assert result["retrained_evidence_refs"]["0"] == "qe:evidence:0"


def test_retrained_ablation_stops_at_durable_campaign_budget(tmp_path):
    import pytest
    from factor_optimizer.contracts.campaign_store import DurableBudgetTracker, SQLiteCampaignStore
    from factor_optimizer.contracts.search_budget import SearchBudget
    from modeling.trainer_governance import FeatureExperimentSpec

    class Artifact:
        def predict(self, values, *, context):
            return values[:, 0]

    class Retrained:
        def predict(self, values):
            return values[:, 0]

    X = np.arange(24.0).reshape(8, 3)
    y = X[:, 0]
    dates = np.asarray(pd.date_range("2026-01-01", periods=8), dtype=object)
    context = PredictionContext(ApplicationWindow(dates[0], dates[-1]), dates, dates[-1], "schema")
    tracker = DurableBudgetTracker(
        SQLiteCampaignStore(tmp_path / "one.sqlite3"), "bounded-ablation",
        SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
    )
    spec = FeatureExperimentSpec(
        "bounded-ablation", "features-v1", ("fold-1",), 7,
        "cpu-v1", "qe:baseline", "model-recipe-v1", 1, 1.0,
    )
    with pytest.raises(ValueError, match="budget exhausted"):
        feature_ablation(
            Artifact(), X, y, lambda pred, truth: float(np.mean(pred == truth)),
            dates=dates, retrain_without_feature_fn=lambda j: (Retrained(), f"qe:{j}"),
            experiment_spec=spec, budget_tracker=tracker, context=context,
        )
    state = SQLiteCampaignStore(tmp_path / "one.sqlite3").budget_state("bounded-ablation")
    assert state["evaluations_used"] == 1
