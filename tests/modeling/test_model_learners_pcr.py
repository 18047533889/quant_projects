# -*- coding: utf-8 -*-
"""Predictive PCR learner — silent-downgrade fix (audit item 3, P0).

Audit item: ``pcr.py`` used ``k = min(n_components, d)``, so
``n_components=5 / 10 / 20`` on ``d=5`` silently became the SAME model (false
parameter variants).  Fixed to bind-time rejection: ``n_components`` exceeding
the observable rank ``min(n_rows, n_features)`` raises ``ValueError`` instead of
silently downgrading.

Also asserts the fitted model uses the full requested component count and that
the unsupervised PCA projection complexity is reported separately as
``pca_complexity`` (not counted in the supervised free parameters).
"""
from __future__ import annotations

import numpy as np
import pytest

from modeling.learners.base import LearnerSpec
from modeling.learners.pcr import PCRLearner


def _spec(n_components: int) -> LearnerSpec:
    return LearnerSpec(
        learner_name="predictive_pcr",
        family="pcr",
        hyperparams={"n_components": n_components},
    )


def _data(n: int = 400, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ np.array([1.0, -0.5, 0.3, 2.0, -1.0]) + 0.1 * rng.normal(size=n)
    return X, y


def test_pcr_rejects_n_components_greater_than_features():
    """n_components > d must raise at fit — no silent downgrade to d."""
    X, y = _data(n=400, d=5)
    learner = PCRLearner(_spec(6))  # 6 > d=5
    with pytest.raises(ValueError, match="n_components"):
        learner.fit(X, y)
    # Distinct values that previously collapsed onto the same d=5 model must
    # NOT both fit — 5 is the largest legal request here.
    PCRLearner(_spec(5)).fit(X, y)


def test_pcr_rejects_n_components_greater_than_rows():
    """n_components > min(n_rows, d) must fail closed (rank ceiling)."""
    X, y = _data(n=10, d=5)
    learner = PCRLearner(_spec(12))  # 12 > min(10, 5)=5
    with pytest.raises(ValueError, match="n_components"):
        learner.fit(X, y)


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_pcr_uses_requested_component_count(k):
    """n_components <= rank uses the full requested count in metadata."""
    X, y = _data(n=400, d=5)
    frozen = PCRLearner(_spec(k)).fit(X, y)
    assert frozen.metadata["n_components_used"] == k
    assert frozen.params["n_components"] == k
    assert frozen.params["components"].shape == (k, 5)
    assert frozen.params["beta_pca"].shape == (k,)
    # Unsupervised PCA projection reported separately, never in the param count.
    assert frozen.metadata["pca_complexity"] == 5 * k


def test_pcr_predict_reconstructs_signal():
    """PCR with enough components should track the underlying linear signal."""
    rng = np.random.default_rng(1)
    n, d = 500, 6
    X = rng.normal(size=(n, d))
    true_b = np.array([1.0, -0.5, 0.3, 2.0, -1.0, 0.5])
    y = X @ true_b + 0.05 * rng.normal(size=n)
    learner = PCRLearner(_spec(d))
    frozen = learner.fit(X, y)
    pred = learner.predict(frozen, X)
    # PCA with all d components == OLS: correlation with truth is near 1.
    assert np.corrcoef(pred, y)[0, 1] > 0.99
