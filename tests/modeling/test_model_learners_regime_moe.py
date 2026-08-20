# -*- coding: utf-8 -*-
"""Regime / MoE gate identity + unique-date support (audit item 6, P1).

Audit item: confirm the fail-closed double-gates really execute and strengthen
the unique-date support.  Verified here:

* regime: capacity floor AND per-regime support both fail closed with
  ``ValueError``; ``aux["regime_state"]`` missing or length-mismatched fails;
* MoE: capacity floor AND per-expert full-rank/condition gate fail closed (or
  the explicit ``single_best`` fallback is recorded);
* unique dates: when the trainer supplies ``aux["date"]`` the sample's pooled-
  panel independence is checked — length mismatch fails closed, and a sample
  spanning fewer unique dates than the contract floor is rejected; the frozen
  metadata records ``n_unique_dates`` and per-regime/per-expert unique counts.
"""
from __future__ import annotations

import numpy as np
import pytest

from modeling.contracts import SampleAdequacyContract
from modeling.learners.base import LearnerSpec
from modeling.learners.mixture_of_experts import MixtureOfExpertsLearner
from modeling.learners.regime import RegimeLearner


def _contract(**kw) -> SampleAdequacyContract:
    base = dict(
        min_raw_obs=10,
        min_effective_obs=10,
        min_unique_dates=1,
        min_unique_stocks=1,
        min_obs_per_parameter=1.0,
    )
    base.update(kw)
    return SampleAdequacyContract(**base)


def _regime(n_regimes=2, min_regime_obs=50, **kw):
    hp = {"n_regimes": n_regimes}
    hp.update(kw)
    return RegimeLearner(
        LearnerSpec(
            learner_name="predictive_regime",
            family="regime",
            hyperparams=hp,
            sample_contract=_contract(min_regime_obs=min_regime_obs),
        )
    )


def _moe(n_experts=2, min_expert_obs=50, **kw):
    hp = {"n_experts": n_experts}
    hp.update(kw)
    return MixtureOfExpertsLearner(
        LearnerSpec(
            learner_name="predictive_mixture_of_experts",
            family="moe",
            hyperparams=hp,
            sample_contract=_contract(min_expert_obs=min_expert_obs),
        )
    )


def _split_data(n=600, seed=0):
    r = np.random.default_rng(seed)
    half = n // 2
    gate = np.concatenate([-r.uniform(1.0, 2.0, half), r.uniform(1.0, 2.0, n - half)])
    X = np.column_stack([gate + r.normal(0.0, 0.05, n), r.normal(0.0, 1.0, n)])
    y = np.where(gate < 0, -1.5 * X[:, 0] + 1.0, -0.5 * X[:, 0] - 2.0) + r.normal(0.0, 0.1, n)
    return X, y, gate


# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #
def test_regime_capacity_double_gate_executes():
    X, y, gate = _split_data(n=120)
    # 5 regimes * min_regime_obs=30 = 150 > 120 rows -> capacity gate fires.
    with pytest.raises(ValueError, match="sample floor violated"):
        _regime(n_regimes=5, min_regime_obs=30).fit(X, y, aux={"regime_state": gate})


def test_regime_per_regime_support_gate_executes():
    gate = np.concatenate([np.full(80, -1.0), np.full(80, 1.0)])
    X = np.column_stack([gate, np.random.default_rng(1).normal(0.0, 1.0, 160)])
    y = gate * 0.5 + np.random.default_rng(1).normal(0.0, 0.1, 160)
    # 3 regimes but the gate has only 2 distinct values -> one empty bin.
    with pytest.raises(ValueError, match="unsupported regime"):
        _regime(n_regimes=3, min_regime_obs=10).fit(X, y, aux={"regime_state": gate})


def test_regime_aux_regime_state_missing_fails_closed():
    X, y, gate = _split_data(n=300)
    with pytest.raises(ValueError, match="regime_state"):
        _regime().fit(X, y)
    with pytest.raises(ValueError, match="row counts differ"):
        _regime().fit(X, y, aux={"regime_state": gate[:-1]})


def test_regime_unique_dates_recorded_and_enforced():
    X, y, gate = _split_data(n=300, seed=2)
    # 6 unique dates, contract requires 4 -> pass, metadata records the count.
    dates = np.repeat(np.arange(6), 50)
    frozen = _regime(min_regime_obs=40).fit(X, y, aux={"regime_state": gate, "date": dates})
    assert frozen.metadata["n_unique_dates"] == 6
    assert len(frozen.metadata["per_regime_unique_dates"]) == 2
    assert sum(frozen.metadata["per_regime_unique_dates"]) >= 6

    # Contract requires 30 unique dates but the sample only spans 6 -> fail closed.
    learner = RegimeLearner(
        LearnerSpec(
            learner_name="predictive_regime",
            family="regime",
            hyperparams={"n_regimes": 2},
            sample_contract=_contract(min_regime_obs=40, min_unique_dates=30),
        )
    )
    with pytest.raises(ValueError, match="unique dates"):
        learner.fit(X, y, aux={"regime_state": gate, "date": dates})


def test_regime_date_length_mismatch_fails_closed():
    X, y, gate = _split_data(n=300, seed=3)
    with pytest.raises(ValueError, match="date aux and y row counts differ"):
        _regime().fit(X, y, aux={"regime_state": gate, "date": np.arange(10)})


# --------------------------------------------------------------------------- #
# MoE
# --------------------------------------------------------------------------- #
def test_moe_capacity_double_gate_executes():
    X, y, gate = _split_data(n=120)
    with pytest.raises(ValueError, match="sample floor violated"):
        _moe(n_experts=5, min_expert_obs=30).fit(X, y, aux={"regime_state": gate})


def test_moe_expert_support_gate_executes_and_fallback():
    gate = np.concatenate([np.full(80, -1.0), np.full(80, 1.0)])
    X = np.column_stack([gate, np.random.default_rng(2).normal(0.0, 1.0, 160)])
    y = gate * 0.5 + np.random.default_rng(2).normal(0.0, 0.1, 160)
    aux = {"regime_state": gate}
    with pytest.raises(ValueError, match="unsupported expert"):
        _moe(n_experts=3, min_expert_obs=10).fit(X, y, aux=aux)
    # Explicit single_best fallback records the degraded fit.
    frozen = _moe(n_experts=3, min_expert_obs=10, explicit_fallback="single_best").fit(X, y, aux=aux)
    assert frozen.metadata["fallback"] == "single_best"
    assert frozen.metadata["n_experts_fitted"] == 1


def test_moe_aux_regime_state_missing_fails_closed():
    X, y, gate = _split_data(n=400)
    with pytest.raises(ValueError, match="regime_state"):
        _moe().fit(X, y)
    with pytest.raises(ValueError, match="row counts differ"):
        _moe().fit(X, y, aux={"regime_state": gate[:-1]})


def test_moe_unique_dates_recorded_and_enforced():
    X, y, gate = _split_data(n=600, seed=4)
    dates = np.repeat(np.arange(12), 50)
    frozen = _moe(min_expert_obs=100).fit(X, y, aux={"regime_state": gate, "date": dates})
    assert frozen.metadata["n_unique_dates"] == 12
    assert len(frozen.metadata["per_expert_unique_dates"]) == 2

    learner = MixtureOfExpertsLearner(
        LearnerSpec(
            learner_name="predictive_mixture_of_experts",
            family="moe",
            hyperparams={"n_experts": 2},
            sample_contract=_contract(min_expert_obs=100, min_unique_dates=30),
        )
    )
    with pytest.raises(ValueError, match="unique dates"):
        learner.fit(X, y, aux={"regime_state": gate, "date": dates})
