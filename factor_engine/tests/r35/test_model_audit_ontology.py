# -*- coding: utf-8 -*-
"""Model-operators full audit — Phase-1 ontology hard gates (M-002..M-007).

Each test proves one global-governance fix at current HEAD:
- M-002/M-003: every direct-production model-like canonical has an explicit
  authored ModelTimingContract (zero generated timing for production lanes).
- M-004: TimingKind ontology resolves for every model-like canonical and the
  reference-query / same-time / filter / matured families map to the right kind.
- M-005: is_model_like_name is token-boundary aware (no "ar_" substring false
  positives in dollar_volume / calendar_day_diff).
- M-006: MODEL_LANE_EXPLICIT / MODEL_TIMING_CONTRACTS have no dead keys.
- M-007: in-sample regression diagnostics are NOT in alpha lanes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from cleaned_operators import load_all  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402
from cleaned_operators.model_timing import (  # noqa: E402
    MODEL_TIMING_CONTRACTS,
    MODEL_LIKE_HINTS,
    TimingKind,
    is_model_like_name,
    model_timing_production_errors,
    timing_kind_for,
)
from cleaned_operators.model_lane import (  # noqa: E402
    MODEL_LANE_EXPLICIT,
    assign_model_lane,
    model_lane_dead_key_errors,
    model_lane_inventory,
)


def _load():
    load_all()


def _category_of(canonical: str) -> str:
    try:
        ops = OperatorRegistry._operators.get(canonical, {}) or {}
        if ops:
            return str(getattr(next(iter(ops.values())).metadata, "category", "") or "")
    except Exception:
        pass
    return ""


# --------------------------------------------------------------------------
# M-002 / M-003: production lanes carry explicit authored timing
# --------------------------------------------------------------------------

def test_m002_zero_generated_timing_for_production_lanes():
    _load()
    canonicals = sorted(OperatorRegistry.list_canonical())
    errs = model_timing_production_errors(canonicals)
    assert errs == [], f"direct-production model-like without explicit timing: {errs}"


def test_m002_alpha_lanes_have_explicit_timing():
    _load()
    canonicals = sorted(OperatorRegistry.list_canonical())
    lanes = model_lane_inventory(canonicals)
    prod_lanes = ("FAST_NATIVE_ALPHA", "EXPENSIVE_CERTIFIED_ALPHA",
                  "MODEL_FEATURE_SCORE", "STATE_CONDITION_EVENT")
    missing = [
        c for c in canonicals
        if lanes.get(c) in prod_lanes and c not in MODEL_TIMING_CONTRACTS
    ]
    assert missing == [], f"production-lane model-like without explicit timing: {missing}"


# --------------------------------------------------------------------------
# M-004: TimingKind ontology
# --------------------------------------------------------------------------

def test_m004_timing_kind_resolves_for_all_model_like():
    _load()
    canonicals = sorted(OperatorRegistry.list_canonical())
    model_like = [c for c in canonicals if is_model_like_name(c, _category_of(c))]
    assert len(model_like) >= 200
    for c in model_like:
        kind = timing_kind_for(c)
        assert isinstance(kind, TimingKind), f"{c}: timing_kind_for returned {kind!r}"


def test_m004_reference_query_family_is_prior_reference_current_query():
    for c in (
        "ts_matrix_profile_discord_score",
        "ts_signature_mahalanobis_anomaly",
        "ts_motif_recurrence_count",
    ):
        assert timing_kind_for(c) == TimingKind.PRIOR_REFERENCE_CURRENT_QUERY, c


def test_m004_knn_is_same_time_cross_sectional():
    assert timing_kind_for("cs_knn_local_linear_residual") == TimingKind.SAME_TIME_CROSS_SECTIONAL


def test_m004_kalman_is_recursive_causal_filter():
    for c in ("ts_kalman_level", "ts_kalman_beta", "ts_kalman_innovation_z"):
        assert timing_kind_for(c) == TimingKind.RECURSIVE_CAUSAL_FILTER, c


def test_m004_first_passage_is_matured_historical_outcome():
    for c in ("ts_first_passage_bias", "ts_first_passage_hit_probability"):
        assert timing_kind_for(c) == TimingKind.MATURED_HISTORICAL_OUTCOME, c


def test_m004_prior_forecast_is_prior_fit_predictive():
    for c in ("ts_ar_prior_forecast", "ts_garch_next_vol_forecast",
              "ts_multi_regression_coeff_prior"):
        assert timing_kind_for(c) == TimingKind.PRIOR_FIT_PREDICTIVE, c


def test_m004_in_sample_is_self_fit_descriptive():
    for c in ("ts_ar_fitted_value", "ts_huber_regression_coeff",
              "ts_ssa_reconstruction_residual"):
        assert timing_kind_for(c) == TimingKind.SELF_FIT_DESCRIPTIVE, c


# --------------------------------------------------------------------------
# M-005: model-like recall is token-boundary aware
# --------------------------------------------------------------------------

def test_m005_no_ar_substring_false_positives():
    for c in ("calendar_day_diff", "dollar_volume", "dollar_volume_zscore",
              "signed_dollar_volume", "intra_bar_range_deviation"):
        assert not is_model_like_name(c), f"{c} falsely model-like (ar_ substring)"


def test_m005_legit_ar_and_har_tokens_kept():
    for c in ("ts_ar_coefficient", "ts_ar_prior_forecast", "ts_har_rv_next_vol_forecast",
              "ts_first_passage_bias", "ts_transfer_entropy",
              "ts_matrix_profile_discord_score"):
        assert is_model_like_name(c), f"{c} dropped from model-like recall"


def test_m005_hints_are_recall_only_not_semantics():
    # the hint "ar_" must NOT be a bare substring anymore
    assert "ar_" in MODEL_LIKE_HINTS
    # and classification must be category-aware for e.g. mean-reversion
    assert is_model_like_name("ts_mean_reversion_half_life", "time_series_regression")


# --------------------------------------------------------------------------
# M-006: no dead contract keys
# --------------------------------------------------------------------------

def test_m006_no_dead_lane_keys():
    _load()
    canonicals = sorted(OperatorRegistry.list_canonical())
    errs = model_lane_dead_key_errors(canonicals)
    assert errs == [], f"dead MODEL_LANE_EXPLICIT keys: {errs}"


def test_m006_timing_keys_map_live_or_alias():
    _load()
    live = set(OperatorRegistry.list_canonical())
    aliases = set(getattr(OperatorRegistry, "_aliases", {}).keys())
    dead = sorted(set(MODEL_TIMING_CONTRACTS) - live - aliases)
    assert dead == [], f"dead MODEL_TIMING_CONTRACTS keys: {dead}"


def test_m006_no_wildcards_in_effective_lane_map():
    from cleaned_operators.model_lane import _MODEL_LANE_EXPLICIT
    # M-006: no wildcard placeholders survive into the effective lane map —
    # every key must be a concrete live canonical (or alias/tombstone).
    assert all("*" not in k for k in _MODEL_LANE_EXPLICIT)
    assert _MODEL_LANE_EXPLICIT, "effective lane map must be non-empty"


# --------------------------------------------------------------------------
# M-007: in-sample regression diagnostics are not alpha candidates
# --------------------------------------------------------------------------

def test_m007_in_sample_regressions_not_in_alpha_lane():
    _load()
    for c in ("ts_huber_regression_coeff", "ts_quantile_regression_coeff",
              "ts_quantile_regression_slope", "ts_expectile_regression_coeff",
              "ts_gjr_leverage", "ts_ssa_reconstruction_residual",
              "ts_mean_reversion_half_life"):
        lane = assign_model_lane(c)
        assert lane in ("DIAGNOSTIC_RESEARCH", "DELETE_TOMBSTONE"), (
            f"{c}: in-sample diagnostic in lane {lane}"
        )


def test_m007_prior_variants_are_the_alpha_candidates():
    _load()
    for c in ("ts_multi_regression_coeff_prior", "ts_ridge_regression_coeff_prior",
              "ts_huber_regression_forecast_error", "ts_quantile_regression_coeff_prior"):
        assert assign_model_lane(c) == "FAST_NATIVE_ALPHA", c
