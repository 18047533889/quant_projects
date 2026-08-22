# -*- coding: utf-8 -*-
"""Predictive supervised learners (Model Layer Major Redesign taskbook
§36–§40).  Each learner is artifact-backed: ``fit()`` freezes a
:class:`modeling.learners.base.FrozenModel`, and production scoring only ever
calls ``predict()`` on a frozen model — never ``fit()`` (§22).
"""
from __future__ import annotations

from modeling.learners.base import (
    BaseLearner,
    FrozenModel,
    LearnerSpec,
    learner_param_policy,
    register_learner,
)
from modeling.learners.elastic_net import ElasticNetLearner
from modeling.learners.mixture_of_experts import MixtureOfExpertsLearner
from modeling.learners.pcr import PCRLearner
from modeling.learners.pls import PLSLearner
from modeling.learners.regime import RegimeLearner

__all__ = [
    "BaseLearner",
    "FrozenModel",
    "LearnerSpec",
    "register_learner",
    "learner_param_policy",
    "PCRLearner",
    "PLSLearner",
    "ElasticNetLearner",
    "RegimeLearner",
    "MixtureOfExpertsLearner",
]
