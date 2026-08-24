# -*- coding: utf-8 -*-
"""R35 Phase E: model readiness lane assignment (taskbook §179 / §185).

Every model-like canonical gets exactly one lane; zero unclassified.  The lane
inventory is the machine-readable starting point for
``R35_OPERATOR_READINESS``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.cleaned_operators.model_lane import (  # noqa: E402
    MODEL_LANES,
    assign_model_lane,
    model_lane_errors,
    model_lane_inventory,
)


def _load():
    load_all()


def test_zero_unclassified_model_lanes():
    _load()
    canonicals = sorted(OperatorRegistry.list_canonical())
    errs = model_lane_errors(canonicals)
    assert errs == [], f"unclassified model canonicals: {errs}"


def test_all_lanes_are_valid_labels():
    _load()
    canonicals = sorted(OperatorRegistry.list_canonical())
    inv = model_lane_inventory(canonicals)
    for name, lane in inv.items():
        assert lane in MODEL_LANES, f"{name}: invalid lane {lane}"


def test_inventory_covers_all_model_like():
    """The lane inventory must cover exactly the model-like canonicals (282,
    matching the R28 model causality audit count)."""
    from factor_engine.cleaned_operators.model_timing import is_model_like_name
    from factor_engine.cleaned_operators.model_lane import _category_of

    _load()
    canonicals = sorted(OperatorRegistry.list_canonical())
    model_like = [c for c in canonicals if is_model_like_name(c, _category_of(c))]
    inv = model_lane_inventory(canonicals)
    assert set(inv) == set(model_like)
    assert len(model_like) >= 200, f"model-like count collapsed: {len(model_like)}"


def test_legacy_panels_are_legacy_local_predictive():
    """P0 lane/timing governance: the five legacy short-window panel operators are
    LOCAL_ROLLING_ESTIMATOR + default_searchable=False + research_only (modeling.legacy
    is the authority).  They must NOT resolve to the production MODEL_FEATURE_SCORE
    lane — that was a machine semantic conflict (legacy says research, lane said
    model score).  modeling.legacy.classification_of() wins in assign_model_lane."""
    _load()
    for name in (
        "panel_rolling_pcr_forecast",
        "panel_rolling_pls_forecast",
        "panel_rolling_elastic_net_forecast",
        "panel_regime_conditioned_forecast",
        "panel_mixture_of_experts_score",
    ):
        assert assign_model_lane(name) == "LEGACY_LOCAL_PREDICTIVE", name


def test_dmd_hankel_matrix_profile_are_expensive_research_certified():
    """P0 lane/timing governance: mathematically-certified structural models
    (DMD / Hankel / matrix-profile / motif / signature) are research-only — NOT
    production alpha candidates.  GARCH stays EXPENSIVE_CERTIFIED_ALPHA."""
    _load()
    for name in (
        "ts_dmd_dominant_growth_rate",
        "ts_dmd_level_dominant_frequency",
        "ts_hankel_effective_rank",
        "ts_matrix_profile_discord_score",
        "ts_motif_recurrence_count",
        "ts_signature_mahalanobis_anomaly",
    ):
        assert assign_model_lane(name) == "EXPENSIVE_RESEARCH_CERTIFIED", name


def test_ts_ar_coefficient_is_diagnostic_descriptive():
    """P0 lane/timing governance: ts_ar_coefficient is an IN-SAMPLE descriptive
    coefficient (fit includes the current row) — DIAGNOSTIC_DESCRIPTIVE, not an
    alpha candidate.  The strict-prior variant ts_ar_prior_coeff stays FAST_NATIVE_ALPHA."""
    _load()
    assert assign_model_lane("ts_ar_coefficient") == "DIAGNOSTIC_DESCRIPTIVE"
    assert assign_model_lane("ts_ar_prior_coeff") == "FAST_NATIVE_ALPHA"


def test_garch_is_expensive_certified():
    _load()
    for name in (
        "ts_garch_standardized_shock",
        "ts_garch_next_vol_forecast",
        "ts_garch_persistence",
        "ts_garch_vol_surprise",
        "ts_gjr_garch_vol_forecast",
    ):
        assert assign_model_lane(name) == "EXPENSIVE_CERTIFIED_ALPHA", name


def test_kalman_is_fast_native_alpha():
    _load()
    for name in (
        "ts_kalman_level",
        "ts_kalman_trend",
        "ts_kalman_beta",
        "ts_kalman_innovation_z",
    ):
        assert assign_model_lane(name) == "FAST_NATIVE_ALPHA", name


def test_in_sample_is_diagnostic():
    _load()
    for name in (
        "ts_ar_fitted_value",
        "ts_ar_in_sample_resid",
        "ts_regression_resid",
    ):
        assert assign_model_lane(name) == "DIAGNOSTIC_RESEARCH", name
