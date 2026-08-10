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

from cleaned_operators import load_all  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402
from cleaned_operators.model_lane import (  # noqa: E402
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
    from cleaned_operators.model_timing import is_model_like_name
    from cleaned_operators.model_lane import _category_of

    _load()
    canonicals = sorted(OperatorRegistry.list_canonical())
    model_like = [c for c in canonicals if is_model_like_name(c, _category_of(c))]
    inv = model_lane_inventory(canonicals)
    assert set(inv) == set(model_like)
    assert len(model_like) >= 200, f"model-like count collapsed: {len(model_like)}"


def test_supervised_panels_are_model_feature_score():
    _load()
    for name in (
        "panel_rolling_pcr_forecast",
        "panel_rolling_pls_forecast",
        "panel_rolling_elastic_net_forecast",
        "panel_regime_conditioned_forecast",
        "panel_mixture_of_experts_score",
    ):
        assert assign_model_lane(name) == "MODEL_FEATURE_SCORE", name


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
