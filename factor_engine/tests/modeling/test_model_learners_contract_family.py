# -*- coding: utf-8 -*-
"""sample_contract_family explicit declaration (audit item 5).

Audit item: the trainer resolved ``default_sample_contracts().get(learner.family)``
while the contract dict is keyed ``linear/regime/moe`` — PCR/PLS/ElasticNet
(family ``pcr/pls/elastic_net``) matched ``None``.  Each learner now declares
``sample_contract_family`` explicitly (linear/regime/moe) and exposes the
resolved key via ``contract_family``.

Also guards the class-level fallback in ``modeling.sample_policy.resolve_sample_contract``:
``contract_family`` is a dual-context descriptor so class access stays falsy and
the resolver falls through to the bare ``family`` key for legacy learners.
"""
from __future__ import annotations

import pytest

from modeling.learners.base import (
    BaseLearner,
    LearnerSpec,
    default_sample_contracts,
)
from modeling.learners.elastic_net import ElasticNetLearner
from modeling.learners.mixture_of_experts import MixtureOfExpertsLearner
from modeling.learners.pcr import PCRLearner
from modeling.learners.pls import PLSLearner
from modeling.learners.regime import RegimeLearner


def _spec(name: str, family: str, **hp) -> LearnerSpec:
    return LearnerSpec(learner_name=name, family=family, hyperparams=hp)


@pytest.mark.parametrize(
    "learner_cls,family,expected",
    [
        (PCRLearner, "pcr", "linear"),
        (PLSLearner, "pls", "linear"),
        (ElasticNetLearner, "elastic_net", "linear"),
        (RegimeLearner, "regime", "regime"),
        (MixtureOfExpertsLearner, "moe", "moe"),
    ],
)
def test_contract_family_resolves(learner_cls, family, expected):
    spec = _spec(learner_cls.name, family, **{"n_components": 2, "n_regimes": 2, "n_experts": 2})
    learner = learner_cls(spec)
    assert learner.sample_contract_family == expected
    assert learner.contract_family == expected
    # The declared family must resolve to a concrete default contract.
    assert default_sample_contracts().get(expected) is not None


def test_contract_family_class_access_is_falsy_for_legacy_fallback():
    """Class-level ``getattr(cls, 'contract_family', None)`` must be falsy so
    ``resolve_sample_contract`` falls through to the bare family key."""

    class _FamilyOnly(BaseLearner):
        name = "family_only"
        family = "moe"

        def fit(self, X, y, *, weights=None, aux=None):
            raise NotImplementedError

        def predict(self, frozen, X):
            raise NotImplementedError

    from modeling.sample_policy import resolve_sample_contract

    # class-level fallback to family="moe" still resolves.
    assert resolve_sample_contract(_FamilyOnly) is not None
    assert resolve_sample_contract(_FamilyOnly).min_expert_obs == 1_000
    # instance-level attribute access gives the resolved string.
    assert _FamilyOnly(_spec("family_only", "moe")).contract_family == "moe"
