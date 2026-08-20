# -*- coding: utf-8 -*-
"""Regime / MoE predictive learners + DSL as-of artifact bridge
(Model Layer Major Redesign taskbook §6.1 / §6.2 / §6.3 / §39 / §40 / §49-§51).

Covers:

* RegimeLearner — train-only boundaries, per-regime support fail-closed,
  hard/soft predict, support_report, zero-variance rejection.
* MixtureOfExpertsLearner — per-expert support fail-closed, explicit
  ``single_best`` fallback recording, softmax gating, zero-variance rejection.
* DSL bridge — ArtifactStore round-trip, ArtifactResolver as-of resolution,
  score_asof LookupError, replay_historical_scores per-date legality.
"""
from __future__ import annotations

import numpy as np
import pytest

from modeling.artifact import FrozenPreprocessing, ModelArtifact, ModelArtifactManifest, PredictionContext
from modeling.contracts import ApplicationWindow, SampleAdequacyContract
from modeling.dsl_bridge import (
    ArtifactResolver,
    ArtifactStore,
    identity_of,
    replay_historical_scores,
    score_asof,
)
from modeling.learners.base import LearnerSpec, get_learner

_RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _regime_contract(min_regime_obs: int = 100) -> SampleAdequacyContract:
    return SampleAdequacyContract(
        min_raw_obs=10,
        min_effective_obs=10,
        min_unique_dates=1,
        min_unique_stocks=1,
        min_obs_per_parameter=1.0,
        min_regime_obs=min_regime_obs,
    )


def _moe_contract(min_expert_obs: int = 100) -> SampleAdequacyContract:
    return SampleAdequacyContract(
        min_raw_obs=10,
        min_effective_obs=10,
        min_unique_dates=1,
        min_unique_stocks=1,
        min_obs_per_parameter=1.0,
        min_expert_obs=min_expert_obs,
    )


def _regime_learner(n_regimes: int = 2, min_regime_obs: int = 100, **kw):
    hyperparams = {"n_regimes": n_regimes}
    hyperparams.update(kw)
    spec = LearnerSpec(
        learner_name="predictive_regime",
        family="regime",
        hyperparams=hyperparams,
        sample_contract=_regime_contract(min_regime_obs),
    )
    return get_learner("predictive_regime")(spec)


def _moe_learner(n_experts: int = 2, min_expert_obs: int = 100, **kw):
    hyperparams = {"n_experts": n_experts}
    hyperparams.update(kw)
    spec = LearnerSpec(
        learner_name="predictive_mixture_of_experts",
        family="moe",
        hyperparams=hyperparams,
        sample_contract=_moe_contract(min_expert_obs),
    )
    return get_learner("predictive_mixture_of_experts")(spec)


def _regime_data(n: int = 3000, seed: int = 0):
    r = np.random.default_rng(seed)
    half = n // 2
    gate = np.concatenate([-r.uniform(1.0, 2.0, half), r.uniform(1.0, 2.0, n - half)])
    X = np.column_stack([gate + r.normal(0.0, 0.05, n), r.normal(0.0, 1.0, n)])
    y = np.where(
        gate < 0,
        -1.5 * X[:, 0] + 1.0,
        -0.5 * X[:, 0] - 2.0,
    ) + r.normal(0.0, 0.1, n)
    return X, y, gate


def _moe_data(n: int = 4000, seed: int = 0):
    r = np.random.default_rng(seed)
    half = n // 2
    gate = np.concatenate([-r.uniform(1.0, 3.0, half), r.uniform(1.0, 3.0, n - half)])
    X = np.column_stack([gate + r.normal(0.0, 0.05, n), r.normal(0.0, 1.0, n)])
    y = np.where(
        gate < 0,
        -2.0 * X[:, 0] + 0.5 * X[:, 1] + 1.0,
        -0.5 * X[:, 0] + 1.5 * X[:, 1] - 2.0,
    ) + r.normal(0.0, 0.1, n)
    return X, y, gate


# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #
def test_regime_fit_predict_captures_split():
    X, y, gate = _regime_data(n=3000, seed=0)
    learner = _regime_learner(n_regimes=2)
    frozen = learner.fit(X, y, aux={"regime_state": gate})

    assert frozen.params["regime_mode"] == "hard"
    assert len(frozen.params["per_regime"]) == 2
    assert frozen.metadata["n_regimes_fitted"] == 2

    pred = learner.predict(frozen, X)
    assert np.isfinite(pred).all()
    # The two regimes have opposite signed mean predictions (intercept +1 vs -2).
    assert pred[gate < 0].mean() > 0.0
    assert pred[gate >= 0].mean() < 0.0

    report = learner.support_report(frozen)
    assert report["n_regimes"] == 2
    assert abs(sum(report["regime_occupancy"].values()) - 1.0) < 1e-9
    assert report["per_regime_obs"] == frozen.metadata["per_regime_obs"]


def test_regime_soft_mode_predict_is_finite():
    X, y, gate = _regime_data(n=2000, seed=1)
    learner = _regime_learner(n_regimes=2, regime_mode="soft")
    frozen = learner.fit(X, y, aux={"regime_state": gate})
    assert frozen.params["regime_mode"] == "soft"
    pred = learner.predict(frozen, X)
    assert np.isfinite(pred).all()
    assert pred[gate < 0].mean() > 0.0


def test_regime_fails_closed_when_regime_unsupported():
    # gate has only two distinct values -> n_regimes=3 leaves one empty bin.
    gate = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)])
    X = np.column_stack([gate, _RNG.normal(0.0, 1.0, 100)])
    y = gate * 0.5 + _RNG.normal(0.0, 0.1, 100)
    learner = _regime_learner(n_regimes=3, min_regime_obs=10)
    with pytest.raises(ValueError, match="unsupported regime"):
        learner.fit(X, y, aux={"regime_state": gate})


def test_regime_fails_closed_on_capacity_floor():
    n = 100
    gate = _RNG.normal(0.0, 1.0, n)
    X = np.column_stack([gate, _RNG.normal(0.0, 1.0, n)])
    y = _RNG.normal(0.0, 1.0, n)
    learner = _regime_learner(n_regimes=5, min_regime_obs=30)
    with pytest.raises(ValueError, match="sample floor violated"):
        learner.fit(X, y, aux={"regime_state": gate})


def test_regime_requires_aux():
    X, y, _ = _regime_data(n=200, seed=2)
    learner = _regime_learner(n_regimes=2, min_regime_obs=10)
    with pytest.raises(ValueError, match="regime_state"):
        learner.fit(X, y)


def test_regime_rejects_zero_variance_feature():
    n = 200
    X = np.column_stack([np.ones(n), _RNG.normal(0.0, 1.0, n)])
    y = _RNG.normal(0.0, 1.0, n)
    gate = _RNG.normal(0.0, 1.0, n)
    learner = _regime_learner(n_regimes=2, min_regime_obs=10)
    with pytest.raises(ValueError, match="zero-variance"):
        learner.fit(X, y, aux={"regime_state": gate})


# --------------------------------------------------------------------------- #
# MoE
# --------------------------------------------------------------------------- #
def test_moe_fit_predict():
    X, y, gate = _moe_data(n=4000, seed=3)
    learner = _moe_learner(n_experts=2)
    frozen = learner.fit(X, y, aux={"regime_state": gate})

    assert len(frozen.params["per_expert"]) == 2
    assert frozen.metadata["n_experts_fitted"] == 2
    assert frozen.metadata["dominant_expert_ratio"] >= 0.5

    pred = learner.predict(frozen, X)
    assert np.isfinite(pred).all()
    assert pred[gate < 0].mean() > 0.0
    assert pred[gate >= 0].mean() < 0.0

    report = learner.support_report(frozen)
    assert len(report["per_expert_obs"]) == 2
    assert all(o > 0 for o in report["per_expert_obs"])
    assert all(report["full_rank"])


def test_moe_fails_closed_when_expert_unsupported():
    gate = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)])
    X = np.column_stack([gate, _RNG.normal(0.0, 1.0, 100)])
    y = gate * 0.5 + _RNG.normal(0.0, 0.1, 100)
    learner = _moe_learner(n_experts=3, min_expert_obs=10)
    with pytest.raises(ValueError, match="unsupported expert"):
        learner.fit(X, y, aux={"regime_state": gate})


def test_moe_explicit_fallback_single_best():
    gate = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)])
    X = np.column_stack([gate, _RNG.normal(0.0, 1.0, 100)])
    y = gate * 0.5 + _RNG.normal(0.0, 0.1, 100)
    learner = _moe_learner(n_experts=3, min_expert_obs=10, explicit_fallback="single_best")
    frozen = learner.fit(X, y, aux={"regime_state": gate})

    assert frozen.metadata.get("fallback") == "single_best"
    assert frozen.metadata["n_experts_fitted"] == 1
    pred = learner.predict(frozen, X)
    assert np.isfinite(pred).all()
    report = learner.support_report(frozen)
    assert report.get("fallback") == "single_best"


def test_moe_requires_aux():
    X, y, _ = _moe_data(n=200, seed=4)
    learner = _moe_learner(n_experts=2, min_expert_obs=10)
    with pytest.raises(ValueError, match="regime_state"):
        learner.fit(X, y)


def test_moe_rejects_zero_variance_feature():
    n = 200
    X = np.column_stack([np.ones(n), _RNG.normal(0.0, 1.0, n)])
    y = _RNG.normal(0.0, 1.0, n)
    gate = _RNG.normal(0.0, 1.0, n)
    learner = _moe_learner(n_experts=2, min_expert_obs=10)
    with pytest.raises(ValueError, match="zero-variance"):
        learner.fit(X, y, aux={"regime_state": gate})


# --------------------------------------------------------------------------- #
# DSL bridge
# --------------------------------------------------------------------------- #
def _make_artifact(
    model_name: str,
    training_cutoff: str,
    available_at: str,
    model_version: str = "1.0",
    seed: int = 0,
) -> ModelArtifact:
    r = np.random.default_rng(seed)
    n = 300
    X = r.normal(0.0, 1.0, (n, 2))
    y = 2.0 * X[:, 0] - 1.0 + r.normal(0.0, 0.1, n)
    learner_cls = get_learner("predictive_elastic_net")
    spec = LearnerSpec(
        learner_name="predictive_elastic_net",
        family="elastic_net",
        hyperparams={"alpha": 1e-4, "l1_ratio": 0.0},
        sample_contract=SampleAdequacyContract(
            min_raw_obs=10,
            min_effective_obs=10,
            min_unique_dates=1,
            min_unique_stocks=1,
            min_obs_per_parameter=1.0,
        ),
    )
    frozen = learner_cls(spec).fit(X, y)
    manifest = ModelArtifactManifest(
        model_name=model_name,
        model_version=model_version,
        artifact_id=f"{model_name}-{training_cutoff}-{model_version}",
        train_start="2019-01-01",
        train_end=training_cutoff,
        training_cutoff=training_cutoff,
        available_at=available_at,
        feature_schema_hash="test-schema",
    )
    return ModelArtifact(
        manifest=manifest,
        learner=learner_cls(spec),
        frozen=frozen,
        preprocessing=FrozenPreprocessing([]),
    )


def _prediction_context(artifact, dates, asof):
    return PredictionContext(
        application_window=ApplicationWindow(start=min(dates), end=max(dates)),
        dates=np.asarray(dates, dtype=object),
        asof=asof,
        feature_schema_hash=artifact.manifest.feature_schema_hash,
    )


    store = ArtifactStore(base_dir=str(tmp_path / "store"))
    art = _make_artifact("alpha", "2020-01-01", "2020-01-01", seed=1)
    aid = store.put(art)
    assert aid == art.artifact_id

    loaded = store.get(aid)
    assert loaded is not None
    X = np.random.default_rng(3).normal(0.0, 1.0, (50, 2))
    ctx = _prediction_context(art, ["2020-01-02"] * len(X), "2020-06-01")
    np.testing.assert_allclose(loaded.predict(X, context=ctx), art.predict(X, context=ctx))
    assert store.list() == [aid]
    assert store.get("missing-id") is None


def test_artifact_store_requires_base_dir():
    store = ArtifactStore()
    art = _make_artifact("alpha", "2020-01-01", "2020-01-01", seed=1)
    with pytest.raises(ValueError, match="base_dir"):
        store.put(art)


def test_resolver_picks_correct_asof_artifact():
    old = _make_artifact("alpha", "2020-01-01", "2020-01-01", model_version="1.0", seed=1)
    new = _make_artifact("alpha", "2021-01-01", "2021-01-01", model_version="2.0", seed=2)
    resolver = ArtifactResolver(store=ArtifactStore())
    resolver.register(old)
    resolver.register(new)

    # Before the older cutoff: no artifact exists.
    assert resolver.resolve("alpha", asof="2019-12-01") is None
    # After old cutoff but before new cutoff: only the older artifact is legal.
    assert resolver.resolve("alpha", asof="2020-06-01").artifact_id == old.artifact_id
    # After the new cutoff: the newer artifact wins (LATEST).
    assert resolver.resolve("alpha", asof="2021-06-01").artifact_id == new.artifact_id
    # Unknown model: nothing.
    assert resolver.resolve("nope", asof="2021-06-01") is None


def test_score_asof_scores_legal_artifact():
    art = _make_artifact("alpha", "2020-01-01", "2020-01-01", seed=1)
    resolver = ArtifactResolver(store=ArtifactStore())
    resolver.register(art)
    X = np.random.default_rng(4).normal(0.0, 1.0, (20, 2))
    out = score_asof("alpha", X, asof="2021-01-01", resolver=resolver)
    ctx = _prediction_context(art, ["2020-01-02"] * len(X), "2021-01-01")
    np.testing.assert_allclose(out, art.predict(X, context=ctx))


def test_score_asof_raises_lookup_error_with_no_legal_artifact():
    art = _make_artifact("alpha", "2020-01-01", "2020-01-01", seed=1)
    resolver = ArtifactResolver(store=ArtifactStore())
    resolver.register(art)
    with pytest.raises(LookupError):
        score_asof("alpha", np.zeros((3, 2)), asof="2019-01-01", resolver=resolver)
    with pytest.raises(LookupError):
        score_asof("beta", np.zeros((3, 2)), asof="2021-01-01", resolver=resolver)


def test_replay_historical_scores_uses_per_date_legal_artifacts():
    old = _make_artifact("alpha", "2020-01-01", "2020-01-01", model_version="1.0", seed=1)
    new = _make_artifact("alpha", "2021-01-01", "2021-01-01", model_version="2.0", seed=2)
    resolver = ArtifactResolver(store=ArtifactStore())
    resolver.register(old)
    resolver.register(new)

    X_old = np.random.default_rng(5).normal(0.0, 1.0, (10, 2))
    X_new = np.random.default_rng(6).normal(0.0, 1.0, (10, 2))
    out = replay_historical_scores(
        "alpha",
        {"2020-06-01": X_old, "2021-06-01": X_new},
        resolver,
    )
    old_ctx = _prediction_context(old, ["2020-06-01"] * len(X_old), "2020-06-01")
    new_ctx = _prediction_context(new, ["2021-06-01"] * len(X_new), "2021-06-01")
    np.testing.assert_allclose(out["2020-06-01"], old.predict(X_old, context=old_ctx))
    np.testing.assert_allclose(out["2021-06-01"], new.predict(X_new, context=new_ctx))


def test_replay_historical_scores_raises_on_missing_date():
    art = _make_artifact("alpha", "2020-01-01", "2020-01-01", seed=1)
    resolver = ArtifactResolver(store=ArtifactStore())
    resolver.register(art)
    with pytest.raises(LookupError):
        replay_historical_scores(
            "alpha", {"2019-01-01": np.zeros((2, 2))}, resolver
        )


def test_identity_of_includes_lineage_facts():
    art = _make_artifact("alpha", "2020-01-01", "2020-01-01")
    ident = identity_of(art)
    assert ident["model_name"] == "alpha"
    assert ident["artifact_id"] == art.artifact_id
    assert set(ident) >= {
        "model_name",
        "version",
        "artifact_id",
        "decision_clock_id",
        "label_contract_id",
        "feature_schema_hash",
        "data_source_hash",
        "universe_hash",
        "training_cutoff",
    }
