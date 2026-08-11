# -*- coding: utf-8 -*-
"""Model-audit fixes for the GARCH/GJR/HAR volatility module (M-081, M-083,
M-084, M-086, M-088).

Direct module import registers the volatility operators (fast; no full
``load_all`` needed for these kernel/registry-level assertions).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cleaned_operators.ts_model.volatility as V  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# M-081 — ts_gjr_leverage timing explicit (in-sample fit-through-t)
# ---------------------------------------------------------------------------

def test_m081_gjr_leverage_fits_through_current_row():
    """The GJR leverage kernel fits its parameters THROUGH r_t: perturbing the
    CURRENT return must move the fitted gamma (in-sample descriptive)."""
    rng = np.random.default_rng(1)
    rets = rng.standard_normal(150) * 0.02
    base = V._gjr_leverage(rets, 120)
    rets2 = rets.copy()
    rets2[-1] = 0.5  # perturb the current return only
    changed = V._gjr_leverage(rets2, 120)
    assert np.isfinite(base) and np.isfinite(changed)
    assert changed != base, (
        "ts_gjr_leverage did not react to the current return — the kernel is "
        "supposed to be an in-sample fit-through-t estimate (M-081)"
    )


def test_m081_gjr_leverage_timing_telemetry():
    """The module telemetry explicitly documents fit-through-t / descriptive and
    flags the reconciler (timing contract + lane re-evaluation)."""
    entry = V._IN_SAMPLE_FIT_THROUGH_T.get("ts_gjr_leverage")
    assert entry is not None, "M-081 telemetry entry for ts_gjr_leverage missing"
    assert entry["fit_through_t"] is True
    assert entry["fit_cutoff_offset"] == 0
    assert entry["descriptive"] is True
    flag = entry["flag_for_reconciler"]
    assert "ModelTimingContract" in flag and "DIAGNOSTIC_RESEARCH" in flag


def test_m081_gjr_leverage_timing_contract_explicit_descriptive():
    """M-081 reconciler closure: ts_gjr_leverage now carries an explicit,
    reviewed ModelTimingContract with fit_cutoff_offset=0 (descriptive) — it must
    MATCH the kernel, which fits through the current row.  If this ever flips to
    a predictive contract, the kernel/contract are inconsistent again."""
    from cleaned_operators.model_timing import (
        get_model_timing_contract,
        model_timing_contract_is_explicit,
    )

    assert model_timing_contract_is_explicit("ts_gjr_leverage") is True, (
        "reconciler: ts_gjr_leverage still lacks an explicit timing contract "
        "(M-081 requires fit_cutoff_offset=0 descriptive)"
    )
    c = get_model_timing_contract("ts_gjr_leverage")
    assert c.fit_cutoff_offset == 0, "M-081: ts_gjr_leverage must be descriptive (fit_cutoff=0)"
    assert c.descriptive is True


def test_m081_gjr_leverage_lane_diagnostic_research():
    """M-081 reconciler closure: an in-sample fit-through-t model must NOT sit in
    an alpha-certified lane — it must be DIAGNOSTIC_RESEARCH."""
    from cleaned_operators.model_lane import assign_model_lane

    assert assign_model_lane("ts_gjr_leverage") == "DIAGNOSTIC_RESEARCH", (
        "M-081: ts_gjr_leverage lane must be DIAGNOSTIC_RESEARCH (in-sample "
        "descriptive fit), not an alpha-certified lane"
    )


# ---------------------------------------------------------------------------
# M-083 — shared GARCH/GJR MLE fit cache (bit-identical, scoped per call)
# ---------------------------------------------------------------------------

def _ret_panel(n: int = 160, cols: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, cols)) * 0.02, columns=list("AB")[:cols])


def test_m083_fit_garch_cached_bit_identical_and_reused():
    rng = np.random.default_rng(1)
    seg = rng.standard_normal(130)
    fresh = V._fit_garch(seg)
    assert V._fit_garch_cached(seg) == fresh, "cached GARCH fit != fresh fit"
    assert V._fit_garch_cached(seg) == fresh, "cache hit != fresh fit"


def test_m083_fit_gjr_cached_bit_identical_and_reused():
    rng = np.random.default_rng(2)
    seg = rng.standard_normal(130)
    fresh = V._fit_gjr(seg)
    assert V._fit_gjr_cached(seg) == fresh, "cached GJR fit != fresh fit"
    assert V._fit_gjr_cached(seg) == fresh, "cache hit != fresh fit"


def test_m083_cache_reuses_mle_within_call():
    """The shared-fit cache must prevent a second MLE for identical input bytes."""
    calls = {"n": 0}
    orig = V._fit_garch

    def _counting(rets):
        calls["n"] += 1
        return orig(rets)

    V._fit_garch = _counting
    try:
        rng = np.random.default_rng(3)
        seg = rng.standard_normal(130)
        a = V._fit_garch_cached(seg)
        b = V._fit_garch_cached(seg)
        assert calls["n"] == 1, "identical fit segment was re-fit (cache not shared)"
        assert a == b
    finally:
        V._fit_garch = orig


def test_m083_cache_scoped_per_operator_call():
    """The cache must be cleared at the end of each operator call — no fit state
    may leak across rows / across calls."""
    op = OperatorRegistry.get("ts_garch_persistence")
    assert op is not None
    op.calculate(_ret_panel(160, 2, seed=4), window=120)
    assert V._GARCH_FIT_CACHE == {}, "fit cache leaked across operator calls"


def test_m083_duplicate_columns_bit_identical():
    """Identical fit segments inside ONE operator call are fit once and reused —
    duplicate columns must produce byte-identical outputs (no cross-column state
    leakage, pure reuse optimisation)."""
    ret = _ret_panel(160, 1, seed=5)
    dup = pd.DataFrame(
        np.column_stack([ret["A"].to_numpy(), ret["A"].to_numpy()]),
        columns=["X", "Y"],
    )
    op = OperatorRegistry.get("ts_garch_standardized_shock")
    out = op.calculate(dup, window=120)
    np.testing.assert_array_equal(
        np.nan_to_num(out["X"].to_numpy()), np.nan_to_num(out["Y"].to_numpy())
    )
    # and the single-column reference agrees
    ref = op.calculate(ret, window=120)["A"]
    np.testing.assert_array_equal(
        np.nan_to_num(out["X"].to_numpy()), np.nan_to_num(ref.to_numpy())
    )


# ---------------------------------------------------------------------------
# M-084 — explicit GARCH missing-gap policy (fail-closed on gap)
# ---------------------------------------------------------------------------

def test_m084_missing_policy_constant():
    assert V._GARCH_MISSING_POLICY == "fail_closed_on_gap"


def test_m084_nan_inside_window_fails_closed():
    rng = np.random.default_rng(6)
    rets = rng.standard_normal(150) * 0.02
    rn = rets.copy()
    rn[100] = np.nan  # a gap inside the trailing fit window
    for stat in ("persistence", "forecast", "shock"):
        assert np.isnan(V._garch_path(rn, 120, stat, False, 0.0)), (
            f"GARCH {stat} did not fail closed on a NaN in the fit window"
        )
    assert np.isnan(V._garch_vol_surprise(rn, 120))
    assert np.isnan(V._gjr_leverage(rn, 120))
    # a NaN current row also fails the forecast closed (never drop-and-rescale)
    rn2 = rets.copy()
    rn2[-1] = np.nan
    assert np.isnan(V._garch_path(rn2, 120, "forecast", False, 0.0))


# ---------------------------------------------------------------------------
# M-086 — HAR window vs min-train-obs split
# ---------------------------------------------------------------------------

def test_m086_har_min_train_obs_constant():
    assert V._HAR_MIN_TRAIN_OBS == 25


def test_m086_har_insufficient_train_rows_fails_closed():
    """A HAR window whose valid OLS training rows fall below _HAR_MIN_TRAIN_OBS
    must fail closed to NaN — and lowering the policy floor (a versioned knob,
    NOT a user param) rescues it, proving the constant is the gate.

    P1 (sample-coverage governance): the gate is now a DOUBLE constraint —
    ``N_effective >= _HAR_MIN_TRAIN_OBS`` AND
    ``N_effective >= _HAR_MIN_COVERAGE * window``.  Lowering ONLY the absolute
    floor no longer rescues a window=40 / ~18-valid-row sample, because the
    fractional-coverage floor (0.6*40=24) still binds; lowering BOTH versioned
    knobs to exactly the valid-row count rescues.
    """
    rng = np.random.default_rng(7)
    rv = np.abs(rng.standard_normal(40)) + 1.0  # monthly rolling leaves ~18 valid rows
    assert np.isnan(V._har_rv(rv, 40, "forecast"))
    assert np.isnan(V._har_rv(rv, 40, "innovation_z"))
    old_min = V._HAR_MIN_TRAIN_OBS
    old_cov = V._HAR_MIN_COVERAGE
    try:
        V._HAR_MIN_TRAIN_OBS = 18  # exactly the valid-row count
        assert np.isnan(V._har_rv(rv, 40, "forecast")), (
            "P1 double gate: the fractional-coverage floor (0.6*40=24) still "
            "rejects 18 valid rows even after the absolute floor is met"
        )
        V._HAR_MIN_COVERAGE = 18 / 40.0  # relax coverage to exactly the valid rows
        assert np.isfinite(V._har_rv(rv, 40, "forecast"))
    finally:
        V._HAR_MIN_TRAIN_OBS = old_min
        V._HAR_MIN_COVERAGE = old_cov


# ---------------------------------------------------------------------------
# M-088 — HAR legacy duplicates are compat aliases, not canonicals
# ---------------------------------------------------------------------------

def test_m088_legacy_har_names_resolve_to_canonicals():
    assert OperatorRegistry.resolve_canonical("ts_har_rv_forecast") == "ts_har_rv_next_vol_forecast"
    assert OperatorRegistry.resolve_canonical("ts_har_rv_innovation_z") == "ts_har_rv_forecast_error_z"
    # not separate canonicals -> mining never double-searches the same kernel
    assert "ts_har_rv_forecast" not in OperatorRegistry.list_canonical()
    assert "ts_har_rv_innovation_z" not in OperatorRegistry.list_canonical()


def test_m088_har_alias_output_equals_canonical():
    rng = np.random.default_rng(8)
    rv = pd.DataFrame(np.abs(rng.standard_normal(90)) * 1e-4 + 1e-4, columns=["A"])
    a = OperatorRegistry.get("ts_har_rv_forecast").calculate(rv, window=80)["A"]
    b = OperatorRegistry.get("ts_har_rv_next_vol_forecast").calculate(rv, window=80)["A"]
    pd.testing.assert_series_equal(a, b, check_dtype=False)
    c = OperatorRegistry.get("ts_har_rv_innovation_z").calculate(rv, window=80)["A"]
    d = OperatorRegistry.get("ts_har_rv_forecast_error_z").calculate(rv, window=80)["A"]
    pd.testing.assert_series_equal(c, d, check_dtype=False)
