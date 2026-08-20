# -*- coding: utf-8 -*-
"""effective_parameter_count — real degrees of freedom (audit item 4, P0/P1).

Audit item: the pre-fix estimate used ``n_components / n_regimes /
len(hyperparams)`` as a proxy, so the ``obs/parameter`` gate denominator was not
trustworthy.  Each learner now overrides ``effective_parameter_count`` with its
true complexity, and the base default is a conservative ``min(d+1, max(1, N//10))``.

Formulas asserted here:
* PCR:      ``k + 1`` (k OLS betas on k retained PCA scores + intercept);
* PLS:      ``k*(d+1) + k`` (k latent directions + k y-loadings);
* ElasticNet: active (non-zero) coefficients + intercept (sparse solution counts
  only its support); pre-fit worst case ``d + 1``;
* Regime:   ``r*(d+1) + (r-1)`` (per-regime intercept-OLS + quantile boundaries);
* MoE:      ``e*(d+1) + (e-1)`` (per-expert intercept-OLS + gate boundaries),
  ``d + 1`` under the explicit ``single_best`` fallback;
* base default: ``min(d+1, max(1, N//10))``.
"""
from __future__ import annotations

import numpy as np
import pytest

from modeling.learners.base import BaseLearner, LearnerSpec
from modeling.learners.elastic_net import ElasticNetLearner
from modeling.learners.mixture_of_experts import MixtureOfExpertsLearner
from modeling.learners.pcr import PCRLearner
from modeling.learners.pls import PLSLearner
from modeling.learners.regime import RegimeLearner


def _spec(learner_name: str, family: str, **hp) -> LearnerSpec:
    return LearnerSpec(learner_name=learner_name, family=family, hyperparams=hp)


def test_base_default_is_conservative_linear():
    class _L(BaseLearner):
        name = "l"
        family = "l"

        def fit(self, X, y, *, weights=None, aux=None):
            raise NotImplementedError

        def predict(self, frozen, X):
            raise NotImplementedError

    L = _L(_spec("l", "l"))
    # Without n_rows: saturated linear model d+1.
    assert L.effective_parameter_count({}, 100) == 101
    # With n_rows: capped at ~1 parameter per 10 rows.
    assert L.effective_parameter_count({}, 100, n_rows=1000) == 100
    assert L.effective_parameter_count({}, 100, n_rows=50) == 5  # max(1, 50//10)


def test_pcr_parameter_count():
    L = PCRLearner(_spec("predictive_pcr", "pcr", n_components=3))
    assert L.effective_parameter_count({"n_components": 3}, 6) == 4
    # Capped by feature dimension (never more betas than features).
    assert L.effective_parameter_count({"n_components": 10}, 4) == 5


def test_pls_parameter_count():
    L = PLSLearner(_spec("predictive_pls", "pls", n_components=3))
    # k*(d+1)+k = 3*7+3
    assert L.effective_parameter_count({"n_components": 3}, 6) == 24
    assert L.effective_parameter_count({"n_components": 1}, 4) == 6


def test_enet_parameter_count_counts_active_support():
    L = ElasticNetLearner(_spec("predictive_elastic_net", "elastic_net"))
    # Sparse solution: only 2 of 5 coefficients are non-zero -> 2 + 1.
    sparse = np.array([1.0, 0.0, 0.0, -0.5, 0.0])
    assert L.effective_parameter_count({}, 5, coef=sparse) == 3
    # Pre-fit (no fitted coef): worst case d+1.
    assert L.effective_parameter_count({"alpha": 0.01, "l1_ratio": 0.5}, 5) == 6
    # n_active from fit metadata is also honoured.
    assert L.effective_parameter_count({}, 5, n_active=2) == 3


def test_regime_parameter_count():
    L = RegimeLearner(_spec("predictive_regime", "regime", n_regimes=3))
    # r*(d+1) + (r-1) = 3*7 + 2
    assert L.effective_parameter_count({"n_regimes": 3}, 6) == 23
    assert L.effective_parameter_count({"n_regimes": 2}, 3) == 9


def test_moe_parameter_count():
    L = MixtureOfExpertsLearner(_spec("predictive_moe", "moe", n_experts=3))
    # e*(d+1) + (e-1) = 3*7 + 2
    assert L.effective_parameter_count({"n_experts": 3}, 6) == 23
    # explicit single_best fallback degrades to one pooled intercept-OLS.
    assert L.effective_parameter_count({"n_experts": 3}, 6, fallback="single_best") == 7
