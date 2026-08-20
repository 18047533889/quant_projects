# -*- coding: utf-8 -*-
"""Operator Semantic Closure round (Master Spec P0 shared layer).

Covers the framework-level P0 closures delivered this round:

* Central ``MissingPolicy`` / ``WindowSemantics`` vocabularies + per-canonical
  declaration side-registry (Part C-10 / D-16).
* SameAxis axis-contract guards (Part BM-259/260) — multi-input operators must
  never positionally align panels with differing index/column order.
* ``strict_int`` / ``strict_float`` / ``strict_bool`` / ``strict_enum`` uniform
  validators (Part A-4) and the clamp->raise fixes (Part A-5).
* ``ts_valid_count`` / ``ts_coverage_ratio`` treat ±Inf as invalid, with exact
  pandas/polars parity (Part C-14).
* ``event_interval`` strict EventBool {0,1,NaN} + right-aligned Fano blocks +
  ddof=1 + min-valid-blocks gate + ``event_fano_excess`` (Parts M-63/64,
  N-66/67/68/69).
* The registry-level Semantic Closure audit (Part BY invariants).

The round-14 statistical-family policy declarations are asserted too, so the
declared contracts cannot silently drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.closure.axis_contract import (
    SameAxisError,
    assert_same_axes,
    declare_axis_contract,
    has_axis_contract,
)
from cleaned_operators.closure.missing_policy import (
    MissingPolicy,
    declare_missing_policy,
    missing_policy_for,
)
from cleaned_operators.closure.strict_scalar import (
    strict_bool,
    strict_enum,
    strict_float,
    strict_int,
)
from cleaned_operators.closure.window_semantics import (
    WindowSemantics,
    declare_window_semantics,
    window_semantics_for,
)


# ---------------------------------------------------------------------------
# Part C-10 / D-16: central vocabularies
# ---------------------------------------------------------------------------

def test_missing_policy_vocabulary():
    assert {p.value for p in MissingPolicy} == {
        "current_required", "break", "pairwise_valid", "window_valid",
        "censor", "carry_state", "carry_state_with_max_gap",
        "event_unknown", "source_unknown",
    }


def test_window_semantics_vocabulary():
    assert {s.value for s in WindowSemantics} == {
        "full_window", "min_support_window", "contiguous_full_window",
        "event_count_window", "period_count_window", "session_window",
        "expanding", "recursive_state",
    }


def test_declaration_side_registry():
    declare_missing_policy("zz_closure_demo", "break")
    declare_window_semantics("zz_closure_demo", "contiguous_full_window")
    assert missing_policy_for("zz_closure_demo") is MissingPolicy.BREAK
    assert window_semantics_for("zz_closure_demo") is WindowSemantics.CONTIGUOUS_FULL_WINDOW
    # re-declaration without replace=True is a silent-override error
    with pytest.raises(ValueError):
        declare_missing_policy("zz_closure_demo", "break")
    # unknown policy string raises
    with pytest.raises(ValueError):
        declare_missing_policy("zz_closure_demo", "nope")
    declare_missing_policy("zz_closure_demo", "censor", replace=True)
    assert missing_policy_for("zz_closure_demo") is MissingPolicy.CENSOR


# ---------------------------------------------------------------------------
# Part BM-259/260: SameAxis contract
# ---------------------------------------------------------------------------

def test_assert_same_axes_ok():
    a = pd.DataFrame([[1, 2], [3, 4]], index=["t1", "t2"], columns=["A", "B"])
    b = pd.DataFrame([[5, 6], [7, 8]], index=["t1", "t2"], columns=["A", "B"])
    assert_same_axes(a, b)  # no raise


def test_assert_same_axes_rejects_order_change():
    a = pd.DataFrame([[1, 2], [3, 4]], index=["t1", "t2"], columns=["A", "B"])
    # same values, different row order
    b = pd.DataFrame([[5, 6], [7, 8]], index=["t1", "t3"], columns=["A", "B"])
    with pytest.raises(SameAxisError):
        assert_same_axes(a, b)
    # same values, different column order
    c = pd.DataFrame([[5, 6], [7, 8]], index=["t1", "t2"], columns=["B", "A"])
    with pytest.raises(SameAxisError):
        assert_same_axes(a, c)
    # duplicated index
    d = pd.DataFrame([[5, 6], [7, 8]], index=["t1", "t1"], columns=["A", "B"])
    with pytest.raises(SameAxisError):
        assert_same_axes(a, d)


def test_declare_axis_contract():
    declare_axis_contract("zz_demo_pair", ("same_index", "same_columns"))
    assert has_axis_contract("zz_demo_pair")
    with pytest.raises(ValueError):
        declare_axis_contract("zz_demo_pair2", "bogus_contract")


# ---------------------------------------------------------------------------
# Part A-4: strict scalar validators
# ---------------------------------------------------------------------------

def test_strict_int_rejects_truncation_and_strings():
    assert strict_int(20, "x") == 20
    assert strict_int(20.0, "x") == 20
    for bad in (20.2, True, np.float64(20.2), np.nan, np.inf, "20", None):
        with pytest.raises((TypeError, ValueError)):
            strict_int(bad, "x")
    with pytest.raises((TypeError, ValueError)):
        strict_int(1, "x", lower=2)


def test_strict_float_rejects_bool_and_nonfinite():
    assert strict_float(0.5, "x") == 0.5
    assert strict_float(3, "x") == 3.0
    for bad in (True, np.nan, np.inf, "0.5", None):
        with pytest.raises((TypeError, ValueError)):
            strict_float(bad, "x")


def test_strict_bool_only_python_bool():
    assert strict_bool(True, "x") is True
    for bad in (1, 0, "true", 0.5, None, np.bool_(False)):
        with pytest.raises((TypeError, ValueError)):
            strict_bool(bad, "x")


def test_strict_enum_exact_membership():
    assert strict_enum("a", "x", ("a", "b")) == "a"
    with pytest.raises(ValueError):
        strict_enum("c", "x", ("a", "b"))


# ---------------------------------------------------------------------------
# Part C-14: ts_valid_count / ts_coverage_ratio treat ±Inf as invalid
# ---------------------------------------------------------------------------

def _frame(data: dict):
    return pd.DataFrame(data)


def test_valid_count_excludes_inf_pandas():
    from cleaned_operators.safe_ops import pd_ts_valid_count

    df = _frame({"A": [1.0, np.nan, np.inf, 4.0, 5.0, 6.0, np.nan, 8.0]})
    out = pd_ts_valid_count(df, 4, 1)["A"].to_numpy()
    # +Inf at r2 is NOT counted -> 1, not 2
    assert np.allclose(out, [1, 1, 1, 2, 2, 3, 3, 3], equal_nan=True), out


def test_valid_count_pandas_polars_parity_with_inf():
    pl = pytest.importorskip("polars")
    from cleaned_operators.safe_ops import (
        _pl_finite_count_expr,
        pd_ts_valid_count,
    )

    df = _frame(
        {
            "A": [1.0, np.nan, np.inf, 4.0, 5.0, 6.0, np.nan, 8.0],
            "B": [2.0, 5.0, np.nan, -np.inf, 6.0, 7.0, 8.0, 9.0],
        }
    )
    for w, mp in [(4, 1), (4, 3), (4, 4), (6, 2)]:
        pd_out = pd_ts_valid_count(df, w, mp).to_numpy()
        pl_df = pl.DataFrame({c: df[c].tolist() for c in df.columns})
        pl_out = pl_df.with_columns(
            [_pl_finite_count_expr(c, w, mp).alias(c) for c in pl_df.columns]
        ).to_pandas().to_numpy()
        assert np.allclose(pd_out, pl_out, equal_nan=True), (w, mp, pd_out, pl_out)


def test_coverage_ratio_pandas_polars_parity_with_inf():
    pl = pytest.importorskip("polars")
    from cleaned_operators.safe_ops import (
        _pl_finite_count_expr,
        pd_ts_coverage_ratio,
    )

    df = _frame({"A": [1.0, np.nan, np.inf, 4.0, 5.0, 6.0, np.nan, 8.0]})
    for w, mp in [(4, 1), (4, 3)]:
        pd_out = pd_ts_coverage_ratio(df, w, mp).to_numpy()
        pl_df = pl.DataFrame({"A": df["A"].tolist()})
        pl_out = pl_df.with_columns(
            [(_pl_finite_count_expr("A", w, mp) / w).alias("A")]
        ).to_pandas().to_numpy()
        assert np.allclose(pd_out, pl_out, equal_nan=True), (w, mp, pd_out, pl_out)


# ---------------------------------------------------------------------------
# Parts M-63/64, N-66/67/68/69: event_interval
# ---------------------------------------------------------------------------

def _event_frame(vals):
    return pd.DataFrame({"A": np.asarray(vals, dtype=float)})


def test_event_interval_strict_eventbool():
    from cleaned_operators.event_interval import _event_mask

    # only 0 / 1 / NaN accepted
    assert _event_mask(np.array([0.0, 1.0, np.nan, 1.0])).tolist() == [
        False, True, False, True,
    ]
    for bad in (0.2, -1.0, 2.0):
        with pytest.raises(ValueError):
            _event_mask(np.array([0.0, bad, 1.0]))


def test_event_fano_right_aligned_blocks_and_ddof1():
    from cleaned_operators.event_interval import EventFanoFactor

    ev = np.zeros(100, dtype=float)
    ev[95] = ev[96] = ev[97] = ev[98] = ev[99] = 1.0
    out = EventFanoFactor()._calculate_series(_event_frame(ev), window=100, block=20)
    val = out["A"].iloc[99]
    # right-aligned blocks [0..19],[20..39],[40..59],[60..79],[80..99]:
    # counts=[0,0,0,0,5], sample-var(ddof=1)=5, F=5/1=5
    assert np.isclose(val, 5.0, atol=1e-9), val


def test_event_fano_min_valid_blocks_gate():
    from cleaned_operators.event_interval import EventFanoFactor

    # window 45 / block 20 -> only 2 full blocks < min_valid_blocks=5 -> NaN
    ev = np.zeros(45, dtype=float)
    ev[40] = ev[41] = ev[42] = 1.0
    out = EventFanoFactor()._calculate_series(_event_frame(ev), window=45, block=20)
    assert np.isnan(out["A"].iloc[44])


def test_event_fano_excess_is_zero_centered():
    from cleaned_operators.event_interval import EventFanoExcess

    ev = np.zeros(100, dtype=float)
    ev[95] = ev[96] = ev[97] = ev[98] = ev[99] = 1.0
    out = EventFanoExcess()._calculate_series(_event_frame(ev), window=100, block=20)
    assert np.isclose(out["A"].iloc[99], 4.0, atol=1e-9)  # F-1


def test_event_interval_minimum_support():
    from cleaned_operators.event_interval import EventIntervalMemory

    # 5 events -> 4 intervals -> NaN (minimum 6)
    ev5 = np.zeros(40, dtype=float)
    for p in (1, 3, 6, 10, 15):
        ev5[p] = 1.0
    out5 = EventIntervalMemory()._calculate_series(
        _event_frame(ev5), window=40, max_pre_window_age=40
    )
    assert np.isnan(out5["A"].iloc[39])

    # 7 irregular events -> 6 intervals -> finite (monotone -> corr=1)
    ev7 = np.zeros(40, dtype=float)
    for p in (1, 3, 6, 10, 15, 21, 28):
        ev7[p] = 1.0
    out7 = EventIntervalMemory()._calculate_series(
        _event_frame(ev7), window=40, max_pre_window_age=40
    )
    assert np.isclose(out7["A"].iloc[39], 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Part A-5: silent clamp -> strict raise
# ---------------------------------------------------------------------------

def test_rqa_param_clamps_now_raise():
    from cleaned_operators.rqa_ext import _check_params as rqa_check
    from cleaned_operators.recurrence_analysis import _check_params as rec_check

    assert rqa_check(240, 2, 1, 0.1, 2) == (240, 2, 1, 0.1, 2, 1)
    assert rec_check(240, 2, 1, 0.1, 2) == (240, 2, 1, 0.1, 2)
    for kwargs in (
        dict(window=0, dim=2, delay=1, eps_fraction=0.1, min_line=2),
        dict(window=240, dim=0, delay=1, eps_fraction=0.1, min_line=2),
        dict(window=240, dim=1.9, delay=1, eps_fraction=0.1, min_line=2),
        dict(window=240, dim=2, delay=0, eps_fraction=0.1, min_line=2),
    ):
        with pytest.raises((TypeError, ValueError)):
            rqa_check(**kwargs)
        with pytest.raises((TypeError, ValueError)):
            rec_check(**kwargs)


def test_evt_allan_scale_param_raises_instead_of_clamp():
    from cleaned_operators.evt_allan import _allan_factor_single

    v = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    assert _allan_factor_single(v, 2) == 0.0
    with pytest.raises((TypeError, ValueError)):
        _allan_factor_single(v, 0)


# ---------------------------------------------------------------------------
# Round-14 statistical-family declarations
# ---------------------------------------------------------------------------

def test_round14_family_policy_declarations():
    from cleaned_operators.closure.declared_policies import declare_all

    declare_all()
    expectations = {
        "ts_hankel_effective_rank": ("current_required", "contiguous_full_window"),
        "ts_multifractal_spectrum_width": ("current_required", "contiguous_full_window"),
        "ts_vol_pvariation_roughness": ("window_valid", "min_support_window"),
        "ts_evt_threshold_stability": ("window_valid", "full_window"),
        "ts_kernel_granger_score": ("pairwise_valid", "min_support_window"),
        "ts_local_lyapunov_exponent": ("pairwise_valid", "min_support_window"),
        "event_interval_mark_coupling": ("censor", "contiguous_full_window"),
        "event_count": ("break", "event_count_window"),
        "ts_glr_mean_shift_score": ("window_valid", "full_window"),
        "ts_first_passage_hit_probability": ("break", "contiguous_full_window"),
    }
    for canon, (mp, ws) in expectations.items():
        assert missing_policy_for(canon).value == mp, canon
        assert window_semantics_for(canon).value == ws, canon


def test_round14_multi_input_axis_contracts():
    from cleaned_operators.closure.declared_policies import declare_all

    declare_all()
    for canon in (
        "ts_kernel_granger_score",
        "ts_residualized_hsic",
        "ts_transfer_entropy",
        "ts_cross_spectral_coherence",
        "event_interval_mark_coupling",
    ):
        assert has_axis_contract(canon), canon


# ---------------------------------------------------------------------------
# Part BY: the closure audit itself runs and reports invariants
# ---------------------------------------------------------------------------

def test_semantic_closure_audit_runs_on_subset():
    from cleaned_operators.closure.closure_audit import run_semantic_closure_audit

    fake_cat = {
        "demo_good": {
            "canonical": "demo_good", "status": "implemented",
            "param_names": ["x", "window"], "param_specs": {}, "role": None,
            "input_grain": None, "output_grain": None, "same_session_usable": None,
        },
        "demo_pair": {
            "canonical": "demo_pair", "status": "implemented",
            "param_names": ["x", "y", "window"], "param_specs": {},
            "role": "global_state", "input_grain": None, "output_grain": None,
            "same_session_usable": None,
        },
    }
    rep = run_semantic_closure_audit(catalog=fake_cat, include_behavioral=False)
    # a multi-input op without a declared axis contract is flagged
    assert any(
        "demo_pair" in v
        for v in rep.invariants["MULTI_INPUT_WITHOUT_AXIS_CONTRACT"].violations
    )
    # GLOBAL_STATE_AS_ALPHA_TERMINAL is judged by the MINING layer (the
    # authority on terminal-ness), which cannot resolve fake canonicals — so a
    # metadata role=global_state on an unresolvable op is NOT a violation.
    assert rep.invariants["GLOBAL_STATE_AS_ALPHA_TERMINAL"].count == 0
    assert not rep.release_safe()
