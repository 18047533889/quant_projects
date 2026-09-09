from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.artifact import ModelArtifact, ReplayContract
from modeling.contracts import (
    AFTER_CLOSE_TO_NEXT_VWAP, LabelContract, SampleAdequacyContract,
    ashare_decision_clock,
)
from modeling.dataset import FeatureField, FeatureSchema, PanelDataset
from modeling.learners import PCRLearner
from modeling.monitoring import (
    DRIFT_DIMENSIONS, CampaignRequestPolicy, DriftContract, DriftThreshold,
    MonitorAction, RealisedMetric,
)
from modeling.trainer import PreprocessingSpec, fit_preprocessing, train_model


def _schema(state: str = "state-1", *, order=("f0", "f1")) -> FeatureSchema:
    fields = {
        name: FeatureField(name, f"value:{name}", f"recipe:{name}", state, "float64",
                           f"mask:{name}", "cluster-set:v1")
        for name in ("f0", "f1")
    }
    return FeatureSchema(tuple(order), tuple(fields[name] for name in order), "linear-v1",
                         "feature-set:v1")


def _panels(schema: FeatureSchema):
    rows = []
    for d in pd.date_range("2026-01-01", periods=7):
        for stock in range(30):
            x0 = float(stock + d.day)
            x1 = float(stock - d.day)
            rows.append((d, f"S{stock:03d}", x0, x1, .3 * x0 - .1 * x1))
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "label"])
    kwargs = dict(date_col="date", stock_col="stock", feature_cols=list(schema.columns),
                  label_col="label", feature_schema=schema)
    return (PanelDataset(frame[frame.date <= "2026-01-04"].copy(), **kwargs),
            PanelDataset(frame[frame.date >= "2026-01-05"].copy(), **kwargs))


def _preprocessing() -> PreprocessingSpec:
    return PreprocessingSpec(
        consumer_profile="linear-v1", policy_version="linear-preprocess-v1",
        missing_policy="impute_with_mask", scale_policy="train_zscore",
        categorical_policy="none",
    )


def _replay() -> ReplayContract:
    return ReplayContract("data-v1", "universe-v1", "execution-v1", "numeric-v1",
                          "registry-v1", "dependencies-v1", "feature-state-v1")


def test_full_feature_manifest_hash_detects_semantic_change_and_order():
    base = _schema("state-1")
    assert base.fingerprint().startswith("full:")
    assert base.fingerprint() != _schema("state-2").fingerprint()
    assert base.fingerprint() != _schema("state-1", order=("f1", "f0")).fingerprint()
    assert FeatureSchema(("f0", "f1")).fingerprint().startswith("legacy-columns:")


def test_heldout_mutation_does_not_change_selection_preprocessing_state():
    train, heldout = _panels(_schema())
    spec = _preprocessing()
    before = fit_preprocessing(train, spec).state_hash()
    heldout.frame.loc[:, "f0"] = heldout.frame["f0"] * 10000
    assert fit_preprocessing(train, spec).state_hash() == before


def test_final_refit_preprocessing_matches_declared_train_plus_validation_cohort():
    train, validation = _panels(_schema())
    contract = SampleAdequacyContract(30, 30, 2, 10, 1)
    result = train_model(
        PCRLearner, train, validation,
        preprocessing_spec=_preprocessing(), hyperparam_grid=[{"n_components": 2}],
        label_contract=LabelContract("same_bar", 0),
        decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
        sample_contract=contract, retrain_policy="train_plus_validation",
        evaluation_boundary=pd.Timestamp("2026-01-09"),
        replay_contract=_replay(),
    )
    purged_train = train.filter_dates(end=pd.Timestamp("2026-01-03"))
    purged_validation = validation.filter_dates(end=pd.Timestamp("2026-01-06"))
    combined = PanelDataset(
        pd.concat([purged_train.frame, purged_validation.frame], ignore_index=True),
        feature_cols=list(train.feature_cols), label_col="label", feature_schema=train.feature_schema,
    )
    assert result.preprocessing.state_hash() == fit_preprocessing(combined, _preprocessing()).state_hash()
    assert result.artifact.manifest.feature_manifest["consumer_profile"] == "linear-v1"
    restored = ModelArtifact.from_dict(result.artifact.to_dict(), result.artifact.learner)
    assert restored.manifest.feature_manifest == result.artifact.manifest.feature_manifest
    assert restored.manifest.lineage_hash() == result.artifact.manifest.lineage_hash()
    replayed = ModelArtifact.from_dict_for_replay(
        result.artifact.to_dict(), result.artifact.learner, expected=_replay()
    )
    assert replayed.manifest.replay_contract["execution_spec_ref"] == "execution-v1"
    wrong = ReplayContract("data-v2", "universe-v1", "execution-v1", "numeric-v1",
                           "registry-v1", "dependencies-v1", "feature-state-v1")
    with pytest.raises(ValueError, match="dependency identity mismatch"):
        ModelArtifact.from_dict_for_replay(
            result.artifact.to_dict(), result.artifact.learner, expected=wrong
        )


def test_same_columns_with_changed_state_ref_are_rejected_between_fold_cohorts():
    train, _ = _panels(_schema("state-1"))
    _, validation = _panels(_schema("state-2"))
    with pytest.raises(ValueError, match="feature manifests differ"):
        train_model(
            PCRLearner, train, validation,
            preprocessing_spec=_preprocessing(), hyperparam_grid=[{"n_components": 2}],
            label_contract=LabelContract("same_bar", 0),
            decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
            sample_contract=SampleAdequacyContract(30, 30, 2, 10, 1),
            replay_contract=_replay(),
        )


def test_generic_preprocessing_default_cannot_certify_complete_production_manifest():
    train, validation = _panels(_schema())
    with pytest.raises(ValueError, match="explicit consumer preprocessing policy"):
        train_model(
            PCRLearner, train, validation,
            preprocessing_spec=PreprocessingSpec(), hyperparam_grid=[{"n_components": 2}],
            label_contract=LabelContract("same_bar", 0),
            decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
            sample_contract=SampleAdequacyContract(30, 30, 2, 10, 1),
            replay_contract=_replay(),
        )


def _drift_contract() -> DriftContract:
    return DriftContract({name: DriftThreshold(.05, retrain=.2, block=.8)
                          for name in DRIFT_DIMENSIONS})


def _metrics(**overrides):
    values = {name: .0 for name in DRIFT_DIMENSIONS}
    values.update(overrides)
    return values


def _policy():
    return CampaignRequestPolicy("campaign-1", 2, "artifact-1", "features-1",
                                 "artifact-0", "features-0")


def test_drift_cannot_trigger_unbounded_retrain():
    with pytest.raises(ValueError, match="bounded CampaignRequestPolicy"):
        _drift_contract().assess(_metrics(ks=.3))
    actions = _drift_contract().assess(_metrics(ks=.3), campaign_policy=_policy())
    assert actions["ks"] is MonitorAction.CAMPAIGN_REQUEST


def test_predictive_drift_requires_mature_metric():
    immature = RealisedMetric("p1", "ic_decay", .1, 20, "2026-01-01",
                              "2026-02-01", "2026-01-20")
    with pytest.raises(ValueError, match="not PIT-available"):
        _drift_contract().assess(
            _metrics(ic_decay=.3), campaign_policy=_policy(),
            realised_metrics={"ic_decay": immature}, asof="2026-01-31",
        )


def test_rollback_binding_cannot_mix_model_and_feature_versions():
    with pytest.raises(ValueError, match="matching artifact and feature version"):
        CampaignRequestPolicy("c", 1, "same", "feature-new", "same", "feature-old")
