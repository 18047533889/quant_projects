# -*- coding: utf-8 -*-
"""FactorEngine model-layer redesign package (Model Layer Major Redesign taskbook).

**AUTHORITY DECLARATION (MODEL2-P0-006)**:
This is the SINGLE SOURCE OF TRUTH for model training, walk-forward splitting,
label-interval purge, embargo enforcement, and temporal contracts. The standalone
``modeling/`` package at ``/home/shw/quant_projects/modeling/`` is for minimal
adapter contracts only and MUST NOT reimplement enforcement logic.

Splits the model layer into five execution classes:

    A. LOCAL_ROLLING_ESTIMATOR      — rolling/statistical local estimators
    B. RECURSIVE_STATE_ESTIMATOR    — recursive stateful estimators
    C. SAME_TIME_CROSS_SECTIONAL    — same-time cross-sectional models
    D. PREDICTIVE_SUPERVISED        — artifact-backed supervised learners
    E. RESEARCH_STRUCTURAL          — research structural estimators

Class D owns an independent fit / validate / freeze / predict / retrain /
artifact lifecycle (see :mod:`modeling.trainer`, :mod:`modeling.artifact`,
:mod:`modeling.predictor`).  All contracts live in :mod:`modeling.contracts`.

This package is fully additive: it imports from (but never edits) the existing
``cleaned_operators`` model-timing / model-contract / model-lane modules and
keeps every new type in its own namespace.
"""
from __future__ import annotations

from modeling.contracts import (
    AFTER_CLOSE_TO_NEXT_VWAP,
    BEFORE_SAME_DAY_VWAP,
    DecisionClock,
    LabelContract,
    LegacyLocalPredictive,
    ModelExecutionClass,
    ModelOperatorSpec,
    ParamRole,
    ParameterSearchPolicy,
    RichModelTiming,
    SampleAdequacyContract,
    TimingKind,
)
from modeling.model_semantic_registry import (
    KNOWN_NOT_CLOSED_CANONICALS,
    MODEL_SEMANTIC_REGISTRY,
    PRODUCTION_LANES,
    ModelSemanticEntry,
    ModelSemanticRegistry,
)

__all__ = [
    "AFTER_CLOSE_TO_NEXT_VWAP",
    "BEFORE_SAME_DAY_VWAP",
    "DecisionClock",
    "LabelContract",
    "LegacyLocalPredictive",
    "ModelExecutionClass",
    "ModelOperatorSpec",
    "ParamRole",
    "ParameterSearchPolicy",
    "RichModelTiming",
    "SampleAdequacyContract",
    "TimingKind",
    "MODEL_SEMANTIC_REGISTRY",
    "ModelSemanticEntry",
    "ModelSemanticRegistry",
    "PRODUCTION_LANES",
    "KNOWN_NOT_CLOSED_CANONICALS",
]
