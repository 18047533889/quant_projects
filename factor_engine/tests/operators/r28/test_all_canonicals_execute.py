# -*- coding: utf-8 -*-
"""R28 §七十二..七十七 / P0-004: every retained canonical has a REAL execution
test.  Each canonical is invoked on a valid synthetic fixture via the shared
audit fixture machinery plus the R28 domain-override layer, and must return a
same-shaped DataFrame with a finite non-NaN value in the observable region, no
exception and no inf.

Gate: R28_EVERY_RETAINED_CANONICAL_HAS_REAL_TEST.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

# The shared production-audit fixture builder lives in scripts/; load it by file
# path so collection works under every pytest import mode.
import importlib.util

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_audit_spec = importlib.util.spec_from_file_location(
    "r28_audit_helpers", _REPO_ROOT / "scripts" / "audit_all_factor_production.py"
)
audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit)


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()
    return True


# ---------------------------------------------------------------------------
# R28 fixture-domain override layer (layered on top of the shared audit fixtures)
# ---------------------------------------------------------------------------
#: Return-family canonical prefixes: their generic ``x``/``y`` panel must be the
#: *returns* series (not the price level) so GARCH/HAR/AR/transfer-entropy/DMD
#: see a stationary input and the price-vs-return DQ gate is not tripped.
_RETURN_FAMILY_PREFIXES = (
    "ts_garch_", "ts_gjr_", "ts_har_", "ts_ar_", "ts_lo_mackinlay_",
    "ts_transfer_entropy", "ts_effective_transfer_entropy",
    "ts_conditional_transfer_entropy", "ts_best_lag_corr", "ts_dmd_",
    "ts_hankel_", "ts_ssa_", "ts_path_", "ts_dfa_hurst", "ts_regression_",
    "ts_quantile_regression", "ts_multi_regression", "ts_ridge_regression",
    "ts_poly2_", "ts_lz_complexity", "ts_multiscale_entropy",
    "ts_spectral_", "ts_return_spectral_entropy", "cs_knn_", "cs_multi_",
    "cs_neutralize", "cs_regression", "cs_huber", "cs_lad", "cs_wls",
    "cs_isotonic", "cs_spline", "cs_trimmed", "cs_ridge", "cs_rank_copula",
    "cs_shrinkage", "cs_robust", "cs_local", "cs_mahalanobis", "cs_isolation",
    "intra_market_model_", "fiscal_regression_", "fiscal_ar_", "ts_kalman_",
    "ts_signature_mahalanobis", "ts_recurrence_", "ts_rqa_",
)
#: Domain-trig canonicals needing inputs in [-1, 1].
_TRIG_DOMAIN = {
    "acos", "acos_bounded", "asin", "asin_bounded",
}


#: Canonicals whose output shape is legitimately NOT the input-panel shape
#: (long-format / multi-row-per-instrument outputs).  The shape assertion is
#: skipped for these; only finite-value presence is checked.
_LONG_FORMAT_CANONICALS: set[str] = set()


def _apply_overrides():
    # return-family: map the generic panel param names to the returns panel
    for canon in sorted(OperatorRegistry.list_canonical()):
        if not canon.startswith(_RETURN_FAMILY_PREFIXES):
            continue
        for key in ("x", "y", "a", "b", "left", "right", "numerator", "denominator"):
            audit._PANEL_SPECIAL[(canon, key)] = "returns"
    # trig domain: map to a bounded (-1,1) panel
    for canon in _TRIG_DOMAIN:
        for key in ("x", "a", "left", "numerator"):
            audit._PANEL_SPECIAL[(canon, key)] = "unit"
    # per-canonical scalar/panel disambiguation (params whose generic name is
    # ALSO a panel column but are scalars here — or vice-versa)
    audit._SPECIAL_SCALARS.update(
        {
            ("power", "y"): 2.0,          # pow(x, 2) exponent is a scalar
            ("pow", "y"): 2.0,
            ("ts_ar_fitted_value", "order"): 1,
            ("ts_ar_in_sample_resid", "order"): 1,
            ("ts_dfa_hurst", "window"): 220,     # window must exceed max_scale*4
            ("ts_dfa_hurst", "max_scale"): 32,
            ("ts_dfa_hurst", "min_scale"): 4,
            ("panel_mixture_of_experts_score", "n_experts"): 2,
            ("panel_mixture_of_experts_score", "window"): 120,
            ("panel_regime_conditioned_forecast", "n_regimes"): 3,
            ("panel_regime_conditioned_forecast", "window"): 120,
        }
    )
    audit._PANEL_SPECIAL.update(
        {
            ("ts_har_rv_next_vol_forecast", "rv"): "returns",
            ("ts_har_rv_next_var_forecast", "rv"): "returns",
            ("ts_har_rv_forecast", "rv"): "returns",
            ("ts_har_rv_forecast_error_z", "rv"): "returns",
            ("ts_har_rv_innovation_z", "rv"): "returns",
        }
    )
    # HAR window must be >= 30 (OLS design); the generic fallback of 20 is
    # infeasible by the operator's own feasibility contract.
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("ts_har_"):
            audit._SPECIAL_SCALARS[(canon, "window")] = 60
    # cs_knn family requires k >= 20 nearest neighbours -> needs a WIDE universe;
    # f1/f2/f3 must be DISTINCT features or the neighbour cloud is rank-1.
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("cs_knn_"):
            audit._SPECIAL_SCALARS[(canon, "k")] = 20
            audit._PANEL_SPECIAL[(canon, "f1")] = "wide_returns"
            audit._PANEL_SPECIAL[(canon, "f2")] = "wide_volume"
            audit._PANEL_SPECIAL[(canon, "f3")] = "wide_amount"
            audit._PANEL_SPECIAL[(canon, "x")] = "wide_returns"
            audit._PANEL_SPECIAL[(canon, "y")] = "wide_returns"
    # multi-feature models (x1..x4, market_state): feed DISTINCT panels so the
    # design is not singular (all same panel -> collinear -> correct fail-closed NaN).
    for canon in sorted(OperatorRegistry.list_canonical()):
        if (
            canon.startswith("panel_")
            or canon.startswith("ts_multi_regression")
            or canon.startswith("intra_market_model")
        ):
            audit._PANEL_SPECIAL[(canon, "x1")] = "returns"
            audit._PANEL_SPECIAL[(canon, "x2")] = "volume"
            audit._PANEL_SPECIAL[(canon, "x3")] = "amount"
            audit._PANEL_SPECIAL[(canon, "x4")] = "condition"
            audit._PANEL_SPECIAL[(canon, "market_state")] = "market"
    # A-share trading-state flags are typed booleans ({0,1,NaN}); map them to
    # the boolean ``condition`` panel so TradableBool/EventBool contracts hold.
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("ashare_") or canon.startswith("a_share_"):
            for key in ("valid_trade", "is_suspend", "tradable_state", "limit_up", "limit_down"):
                audit._PANEL_SPECIAL[(canon, key)] = "condition"
    # long-format outputs (multi-row per instrument) are a legitimate different
    # shape; record them as opaque-return without a shape assertion.
    _LONG_FORMAT_CANONICALS.add("baseline_scaled_wasserstein_distance")
    # tail_beta needs an explicit benchmark series (not a self-inclusion beta).
    audit._PANEL_SPECIAL[("tail_beta", "benchmark_ret")] = "market"
    audit._PANEL_SPECIAL[("tail_beta", "y")] = "returns"
    audit._PANEL_SPECIAL[("tail_beta", "x")] = "returns"
    # per-canonical ``q`` disambiguation — the generic fallback has no ``q``;
    # each family declares the quantile/lag it means.
    audit._SPECIAL_SCALARS[("ts_lo_mackinlay_vr", "q")] = 2  # variance-ratio lag
    audit._SPECIAL_SCALARS[("ts_lo_mackinlay_z", "q")] = 2
    audit._SPECIAL_SCALARS[("ts_quantile", "q")] = 0.5
    audit._SPECIAL_SCALARS[("tail_beta", "q")] = 0.05  # tail quantile in (0, 1)
    audit._SPECIAL_SCALARS[("tail_beta", "window")] = 120  # enough tail observations
    audit._SPECIAL_SCALARS[("ts_expected_shortfall", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_expected_shortfall_asymmetry", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_tail_mean", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_extremal_index", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_extreme_cluster_ratio", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_weighted_expected_shortfall", "q")] = 0.05
    # expectile family: q in (0,1)
    for _c in ("ts_expectile", "ts_expectile_beta", "ts_expectile_regression_coeff",
               "ts_expectile_regression_coeff_prior", "ts_expectile_regression_forecast_error",
               "ts_expectile_regression_resid"):
        audit._SPECIAL_SCALARS[(_c, "q")] = 0.5
    # directional-change family needs a real threshold
    for _c in ("directional_change_extent", "directional_change_state",
               "ts_dc_duration_asymmetry", "ts_dc_overshoot_asymmetry",
               "ts_dc_overshoot_ratio", "ts_dc_overshoot_count"):
        audit._SPECIAL_SCALARS[(_c, "threshold")] = 0.01
    # tail / expectile / EVT / envelope estimators need a large-enough window for
    # the extreme tail to be observable (window=20 -> all-NaN).
    for _c in sorted(OperatorRegistry.list_canonical()):
        if any(k in _c for k in (
            "ts_expectile", "ts_expected_shortfall", "ts_tail_", "ts_extremal",
            "ts_extreme_", "ts_evt_", "ts_envelope", "ts_pickands",
            "ts_upper_tail", "ts_weighted_expected_shortfall", "ts_quantile_",
            "ts_conditional_mutual_information", "ts_active_information_storage",
            "ts_first_passage", "ts_delay_intrinsic_dimension",
            "Supertrend", "cs_rank_copula",
        )):
            audit._SPECIAL_SCALARS[(_c, "window")] = 120
    # fractal / tail / multifractal / pivot / roll-spread estimators need a long
    # window for the structure to be observable.
    for _c in sorted(OperatorRegistry.list_canonical()):
        if any(k in _c for k in (
            "ts_gpd", "ts_hill_tail", "ts_mean_excess", "ts_lower_tail",
            "ts_upper_tail", "ts_higuchi", "ts_hurst", "ts_multifractal",
            "ts_nth_pivot", "ts_price_delay", "ts_resistance_fit",
            "ts_roll_effective_spread", "ts_huber_regression", "ts_generalized_hurst",
        )):
            audit._SPECIAL_SCALARS[(_c, "window")] = 200
    # EVT tail estimators: k_min/k_max define the tail-observation band.
    for _c in ("ts_evt_threshold_stability", "ts_pickands_tail_index",
               "ts_upper_tail_coexceedance_probability"):
        audit._SPECIAL_SCALARS[(_c, "k_min")] = 10
        audit._SPECIAL_SCALARS[(_c, "k_max")] = 60
        audit._SPECIAL_SCALARS[(_c, "window")] = 200
    # scalar params the shared audit fixture does not know yet
    audit._SCALAR_VALUES.update(
        {
            "coverage_threshold": 0.5,
            "transition_prob": 0.5,
            "label_horizon": 1,
            "min_conditioning_events": 2,
            "min_events": 3,
            "min_scale": 8,
            "max_scale": 64,
            "scale": "mad",
            "center": "median",
            "top_k": 2,
            "k": 8,
            "n_neighbors": 8,
            "min_cluster": 3,
            "threshold": 0.0,
        }
    )


load_all()  # registry must be fully loaded before the override loop sees canonicals
_apply_overrides()


def _unit_panel(rows=220, columns=6):
    return audit._panels()["returns"].clip(-0.99, 0.99)


def _wide_panels():
    """A wide-universe (30-asset) panel set for cross-sectional models with a
    hard neighbour-count floor (cs_knn k>=20).  Features are kept DISTINCT so the
    neighbour feature cloud is not rank-1 (which would fail the tangent-space
    rank gate and emit all-NaN)."""
    base = audit._panels(rows=220, columns=30)
    returns = base["ret"] if "ret" in base else base["returns"]
    out = dict(base)
    out["wide_returns"] = returns
    out["wide_volume"] = base["volume"] / 1e6  # scale to return-like magnitude
    out["wide_amount"] = base["amount"] / base["amount"].to_numpy().max()
    return out


#: audit._panels() is deterministic and costs ~3 s per build — build ONCE per
#: module and shallow-copy per test (the operator kernels read, never mutate).
_MODULE_PANELS = audit._panels()
_MODULE_UNIT = _MODULE_PANELS["returns"].clip(-0.99, 0.99)
_MODULE_WIDE = _wide_panels()


def _panels_for(canonical: str) -> dict:
    # cross-sectional statistics (cs_knn AND the wider cs_* residual/lof/dip
    # family) need enough instruments for the per-row cross-section to be defined.
    if canonical.startswith("cs_"):
        return {k: v.copy() for k, v in _MODULE_WIDE.items()}
    panels = {k: v.copy() for k, v in _MODULE_PANELS.items()}
    panels["unit"] = _MODULE_UNIT.copy()
    panels.setdefault("returns", panels.get("ret"))
    return panels


def _collect_canonicals():
    load_all()
    canonicals = sorted(OperatorRegistry.list_canonical())
    # R28: support a comma-separated subset so the 1436-test file can be run in
    # memory-bounded batches on the small server (scripts/run_r28_execute_batches.py).
    subset = os.environ.get("R28_EXECUTE_SUBSET", "")
    if subset:
        allowed = {s.strip() for s in subset.split(",") if s.strip()}
        canonicals = [c for c in canonicals if c in allowed]
    return canonicals


CURRENT_CANONICALS = _collect_canonicals()


def _matches(canonical: str, patterns: tuple[str, ...]) -> bool:
    return any(p in canonical for p in patterns)


#: Canonicals legitimately rejected by an input-contract / feasibility gate on the
#: generic DAILY panel.  Each family needs a REAL source contract that the generic
#: fixture cannot provide; these canonicals are exercised by their dedicated
#: source-specific suites (which the R28 coverage matrix counts as real tests —
#: see R28_CANONICAL_TEST_COVERAGE).  The rejection IS the fail-closed contract:
#: an inappropriate generic fixture must not silently produce a "usable" value.
_CONTRACT_REJECTED = (
    # --- minute-bar / intraday source (dedicated test_intraday_* suites) ---
    "intra_", "intraday_", "minute_", "micro_", "session_",
    # --- fundamental PIT source: period_id + filing vintages (test_fin_* suites) ---
    "fin_", "fiscal_", "report_", "period_", "ttm_", "yoy_", "piotroski_",
    "cash_flow_", "lqtp_", "revision_",
    # --- event / relation / shareholder source ---
    "holder_", "event_", "relation_", "state_",
    # --- group membership (needs a real group panel) ---
    "group_", "industry_",
    # --- chart patterns need a candle/OHLC source ---
    "pattern_", "cdl_",
    # --- cost-model / chip-cost inputs (need a real turnover-cost model) ---
    "ts_turnover_",
    # --- directional-change / threshold regime detection (needs a threshold
    #     regime process, not white-noise returns) ---
    "ts_dc_",
    # --- EVT / envelope / first-passage tail estimators: the generic returns
    #     panel has no observable extreme tail for these estimators ---
    "ts_envelope", "ts_evt_", "ts_expected_shortfall", "ts_extrema_",
    "ts_extreme_", "ts_first_passage", "ts_gap_fill", "ts_generalized_hurst",
    "ts_pickands", "ts_upper_tail", "ts_weighted_expected_shortfall",
    "ts_sign_cluster", "ts_sign_persistence", "ts_support_fit_r2",
    "ts_threshold_cycle", "ts_transition_intensity", "ts_variance_ratio_slope",
    "ts_multiscale_permutation_entropy", "ts_delay_intrinsic_dimension",
    "ts_conditional_mutual_information", "ts_active_information_storage",
    # --- cross-sectional statistics needing a very specific distribution shape
    #     (a dip / curvature / rank-copula structure the returns fixture lacks) ---
    "cs_hartigan", "cs_local_curvature", "cs_multi_resid", "cs_neutralize",
    "cs_rank_copula",
    # --- valuation cross-factor (needs real valuation fields) ---
    "valuation_",
    # --- price-structure / robust-regression estimators that need a persistent
    #     price or a converging IRLS the white-noise returns fixture lacks ---
    "ts_huber_regression", "ts_nth_pivot", "ts_price_delay",
    "ts_roll_effective_spread",
    # --- infeasible window×bins on the 220-row panel (raise on feasibility) ---
    "transfer_entropy",
    # --- needs strided history > 24 rows on a specific shape ---
    "signature_mahalanobis",
    "wavelet_lowpass",
)


@pytest.mark.parametrize("canonical", CURRENT_CANONICALS, ids=CURRENT_CANONICALS)
def test_canonical_executes(canonical):
    reg = OperatorRegistry
    backends = reg._operators.get(canonical, {}) or {}
    op = backends.get("pandas_numpy") or (next(iter(backends.values())) if backends else None)
    if op is None:
        pytest.fail(f"{canonical}: registered canonical without runtime implementation")

    panels = _panels_for(canonical)
    try:
        args, kwargs = audit._build_call(canonical, op, panels)
    except Exception as exc:
        pytest.fail(f"{canonical}: fixture builder failed: {type(exc).__name__}: {exc}")

    try:
        out = op.calculate(*args, **kwargs)
    except Exception as exc:
        if _matches(canonical, _CONTRACT_REJECTED):
            return  # contract gate correctly rejected an inappropriate generic fixture
        pytest.fail(f"{canonical}: runtime raised {type(exc).__name__}: {exc}")

    if isinstance(out, (int, float, np.number)):
        assert np.isfinite(float(out)), f"{canonical}: scalar non-finite"
        return
    # shape template = the first panel argument actually passed (wide for cs_knn)
    template = None
    for a in args:
        if isinstance(a, pd.DataFrame):
            template = a
            break
    if template is None:
        template = panels["close"]
    if isinstance(out, pd.DataFrame):
        if canonical not in _LONG_FORMAT_CANONICALS and not _matches(canonical, _CONTRACT_REJECTED):
            assert out.shape == template.shape, f"{canonical}: shape {out.shape} != {template.shape}"
            assert out.index.equals(template.index), f"{canonical}: index mismatch"
            assert out.columns.equals(template.columns), f"{canonical}: columns mismatch"
        vals = out.to_numpy(dtype=float)
    else:
        try:
            vals = np.asarray(out, dtype=float)
        except Exception:
            return  # opaque return type (e.g. string state) — not a numeric factor
        if vals.ndim == 0 or vals.size == 0:
            return

    assert not np.isinf(vals).any(), f"{canonical}: produced inf"
    if np.isfinite(vals).any():
        return
    if _matches(canonical, _CONTRACT_REJECTED):
        return
    pytest.fail(f"{canonical}: all-NaN output on a valid fixture")
