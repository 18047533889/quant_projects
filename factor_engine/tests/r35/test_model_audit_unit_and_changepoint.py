# -*- coding: utf-8 -*-
"""Model-operators full audit — M-130/M-140/M-141/M-200..M-203/M-180/M-182.

Each test proves one audit fix at current HEAD:
- M-130: FirstPassage ``unit(scale) == unit(x)`` relational constraint — a
  raw-price x + return-volatility scale is REJECTED at runtime while
  log(close)+vol / return+vol combos keep their behaviour.
- M-200: GLR/Pettitt change-point scores are tagged diagnostic_structure /
  state_condition_event (structural state, NOT next-return predictors).
- M-201: GLR cap/penalty/noise constants are versioned into a stable policy
  digest that changes when a cap constant changes.
- M-202: GLR sign convention — positive = post-regime higher.
- M-203: GLR trailing-contiguous window is stable after a NaN gap.
- M-140: state-density ReferenceQueryModel — history reference <= t-1, current
  query = t, query excluded from the reference.
- M-141: state-density min_periods/window split (already fixed).
- M-180/M-182: kernel-Granger/HSIC live names are registered, dead legacy names
  are absent, and the OOS blocked train/test timing is documented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _load():
    load_all()


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()
    return True


def _frame(a: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"A": a},
        index=pd.date_range("2024-01-01", periods=len(a), freq="B"),
    )


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


# --------------------------------------------------------------------------
# M-130: FirstPassage unit(scale) == unit(x) relational enforcement
# --------------------------------------------------------------------------

def test_m130_first_passage_rejects_raw_price_with_return_vol_scale():
    _load()
    price = np.linspace(100.0, 159.0, 60)     # raw price LEVEL
    scale = np.full(60, 0.02)                 # return-volatility (small)
    for name in (
        "ts_first_passage_bias",
        "ts_first_passage_hit_probability",
        "ts_first_passage_conditional_time",
    ):
        op = _op(name)
        with pytest.raises(ValueError, match="unit mismatch"):
            op.calculate(_frame(price), _frame(scale), window=40, barrier=1.0, horizon=5, min_anchors=3)


def test_m130_first_passage_accepts_log_close_with_vol_scale():
    _load()
    log_close = np.log(np.linspace(100.0, 160.0, 80))  # log(close): same-unit with vol
    scale = np.full(80, 0.02)
    op = _op("ts_first_passage_bias")
    out = op.calculate(_frame(log_close), _frame(scale), window=40, barrier=1.0, horizon=5, min_anchors=3)
    assert out.shape == (80, 1)


def test_m130_first_passage_accepts_return_with_vol_scale():
    _load()
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0, 0.02, 80)           # return series (min(x) < 0)
    scale = np.full(80, 0.02)
    op = _op("ts_first_passage_bias")
    out = op.calculate(_frame(ret), _frame(scale), window=40, barrier=1.0, horizon=5, min_anchors=3)
    assert out.shape == (80, 1)


def test_m130_first_passage_metadata_declares_unit_relation():
    _load()
    op = _op("ts_first_passage_bias")
    units = getattr(op.metadata, "input_units", None) or {}
    compatible = getattr(op.metadata, "compatible_units", None) or {}
    assert units.get("x") == "price_or_log_price"
    assert units.get("scale") == "volatility_with_same_unit_as_x"
    assert "return_volatility" in compatible.get("scale", ())


# --------------------------------------------------------------------------
# M-200: change-point role = structural state / diagnostic, not next-return
# --------------------------------------------------------------------------

def test_m200_change_point_tags_diagnostic_structure():
    _load()
    for name in ("ts_glr_mean_shift_score", "ts_glr_variance_shift_score", "ts_pettitt_change_score"):
        tags = _op(name).metadata.tags
        assert "diagnostic_structure" in tags, name
        assert "state_condition_event" in tags, name
        # metadata description states the role: structural-state / diagnostic,
        # NOT a next-return predictor
        desc = (_op(name).metadata.description or "") + (_op(name).__doc__ or "")
        assert "diagnostic" in desc and "next-return predictor" in desc, name


# --------------------------------------------------------------------------
# M-201: GLR policy digest is stable + versioned on constant change
# --------------------------------------------------------------------------

def test_m201_glr_policy_digest_stable():
    from factor_engine.cleaned_operators.glr_change import (
        _GLR_MAX_LLR,
        _GLR_MEASURE_NOISE_MULT,
        _GLR_POLICY_DIGEST,
        _GLR_SCAN_PENALTY_COEF,
    )

    recomputed = (
        f"GLR_ESTIMATOR_POLICY_v1(max_llr={_GLR_MAX_LLR:g},"
        f"penalty_coef={_GLR_SCAN_PENALTY_COEF:g},"
        f"noise_mult={_GLR_MEASURE_NOISE_MULT:g})"
    )
    # same constants -> identical digest (idempotent, versioned)
    assert recomputed == _GLR_POLICY_DIGEST
    assert _GLR_POLICY_DIGEST.startswith("GLR_ESTIMATOR_POLICY_v1")


def test_m201_glr_policy_digest_changes_on_cap_change():
    from factor_engine.cleaned_operators.glr_change import (
        _GLR_MAX_LLR,
        _GLR_MEASURE_NOISE_MULT,
        _GLR_POLICY_DIGEST,
        _GLR_SCAN_PENALTY_COEF,
    )

    perturbed = (
        f"GLR_ESTIMATOR_POLICY_v1(max_llr={_GLR_MAX_LLR + 1.0:g},"
        f"penalty_coef={_GLR_SCAN_PENALTY_COEF:g},"
        f"noise_mult={_GLR_MEASURE_NOISE_MULT:g})"
    )
    assert perturbed != _GLR_POLICY_DIGEST


# --------------------------------------------------------------------------
# M-202: GLR sign convention — positive = post-regime higher
# --------------------------------------------------------------------------

def test_m202_glr_mean_shift_sign_post_regime_higher():
    from factor_engine.cleaned_operators.glr_change import _mean_shift_score

    rng = np.random.default_rng(1)
    up = np.concatenate([rng.normal(0.0, 0.05, 30), rng.normal(1.0, 0.05, 30)])
    dn = np.concatenate([rng.normal(0.0, 0.05, 30), rng.normal(-1.0, 0.05, 30)])
    s_up = _mean_shift_score(up, 6)
    s_dn = _mean_shift_score(dn, 6)
    assert np.isfinite(s_up) and np.isfinite(s_dn)
    assert s_up > 0.0, "post-regime HIGHER must score positive (M-202)"
    assert s_dn < 0.0, "post-regime LOWER must score negative (M-202)"


# --------------------------------------------------------------------------
# M-203: GLR trailing-contiguous window stable after a NaN gap
# --------------------------------------------------------------------------

def test_m203_glr_trailing_contiguous_stable_after_gap():
    _load()
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(0.0, 0.05, n)
    x[40] = np.nan  # gap in the middle
    op = _op("ts_glr_mean_shift_score")
    out = op.calculate(_frame(x), window=40, min_segment=6).to_numpy()[:, 0]
    # rows whose trailing window crosses the gap must be NaN — a gap-shrunken
    # short run must NOT emit as the same estimator
    assert np.isnan(out[41]) and np.isnan(out[60]) and np.isnan(out[79])
    # after a full contiguous 40-row window has re-accumulated past the gap the
    # estimator emits finite again (v.size == min(w, r+1))
    assert np.isfinite(out[80]) and np.isfinite(out[100])


# --------------------------------------------------------------------------
# M-140 / M-141: state-density ReferenceQueryModel + min_periods split
# --------------------------------------------------------------------------

def test_m140_state_density_query_excluded_from_reference():
    _load()
    op = _op("ts_state_density")
    rng = np.random.default_rng(3)
    base = rng.normal(100.0, 2.0, 120)
    # current value far from every past state -> density ~ 0 (query NOT in ref)
    far = np.concatenate([base, [200.0]])
    d_far = op.calculate(_frame(far), window=60, bandwidth=1.0, min_periods=5).to_numpy()[-1, 0]
    # current value at the past median -> density high
    near = np.concatenate([base, [float(np.median(base))]])
    d_near = op.calculate(_frame(near), window=60, bandwidth=1.0, min_periods=5).to_numpy()[-1, 0]
    assert np.isfinite(d_far) and np.isfinite(d_near)
    assert d_near > d_far


def test_m140_state_density_docstring_reference_query_contract():
    _load()
    doc = _op("ts_state_density").__doc__ or ""
    assert "≤t-1" in doc and "查询" in doc and "排除在参考之外" in doc


def test_m141_state_density_min_periods_gate_independent_of_window():
    _load()
    op = _op("ts_state_density")
    # a trailing reference of 5 finite past values: min_periods=6 -> NaN,
    # min_periods=5 -> finite (window and min_periods are SPLIT gates, M-141)
    x = np.full(40, np.nan)
    x[-6:] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    out = op.calculate(_frame(x), window=20, bandwidth=1.0, min_periods=6).to_numpy()[:, 0]
    assert np.isnan(out[-1])
    out2 = op.calculate(_frame(x), window=20, bandwidth=1.0, min_periods=5).to_numpy()[:, 0]
    assert np.isfinite(out2[-1])


# --------------------------------------------------------------------------
# M-180 / M-182: kernel-Granger / HSIC live names + OOS blocked timing
# --------------------------------------------------------------------------

def test_m180_kernel_granger_hsic_live_names_and_no_dead_aliases():
    _load()
    live = set(OperatorRegistry.list_canonical())
    for name in ("ts_kernel_granger_score", "ts_hsic", "ts_residualized_hsic"):
        assert name in live, f"{name} must be a LIVE canonical"
    for dead in ("ts_kernel_granger_causality", "ts_kernel_granger_oos", "ts_hsic_dependence"):
        assert dead not in live, f"{dead} must NOT be re-registered (reconciler removed it)"


def test_m182_kernel_granger_oos_blocked_timing_documented():
    _load()
    import factor_engine.cleaned_operators.research_spectral as rs

    doc = rs.__doc__ or ""
    assert "blocked" in doc and "out-of-sample" in doc
    # the kernel enforces the blocked split: training strictly before test,
    # scaler/bandwidth fitted on the training partition only
    import inspect

    src = inspect.getsource(rs._kernel_granger_score)
    assert "train_n = int(0.7 * n_avail)" in src
    assert "test_n = n_avail - train_n" in src
    assert "_standardize" in src
