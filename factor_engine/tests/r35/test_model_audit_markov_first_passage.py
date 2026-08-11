# -*- coding: utf-8 -*-
"""Model-audit tests for Markov window-maturity (P0/P1) and First-Passage
strict-params / units / bias (P0/P1).

Audit items (this file owns `cleaned_operators/markov_dynamics.py`,
`cleaned_operators/first_passage.py` and their Polars twins in
`cleaned_operators/polars_dynamics.py`):

1. Markov window-maturity contract (P0/P1): ``window`` is a MAX LOOKBACK, not a
   strict full window; maturity is decided by the explicit gates
   ``min_history`` (finite past observations) / ``min_count``
   (a.k.a. ``min_transition_count``, observed lagged transitions) /
   ``min_state_support`` (observed visits to a state).  A ``window=60`` series
   with only 5 points must NOT emit a confident-looking estimate (all NaN),
   while a sufficiently mature window does.

2. First-Passage strict params (P0): ``window`` / ``horizon`` / ``min_anchors``
   / ``scale_horizon`` are validated at binding time via ParamSpec; a fractional
   ``horizon=4.8`` must raise, never silently ``int()``-truncate to ``4``.
   ``barrier`` remains a float (never truncated).

3. First-Passage units (P0): ``unit(scale) == unit(x)`` is enforced at runtime
   (raw-price + return-vol scale is rejected), and ``scale_horizon`` is a
   first-class parameter entering the signature / semantic identity — a 1-day vs
   20-day volatility are DIFFERENT semantic identities.

4. First-Passage bias documentation (P1): bias = direction x speed x
   hit-probability INCLUDING fully-observed non-hit anchors (contribute 0) — it
   is NOT a conditional-on-hit statistic.  The docstrings must say so.

5. Markov PIT contract (P0): the kernel builds every estimate (P / pi / D1 / D2)
   strictly from ``[t-W, t-1]``; ``x_t`` only selects the current state and never
   enters the reference set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.first_passage  # noqa: F401  (registers pandas backends)
import cleaned_operators.markov_dynamics  # noqa: F401  (registers pandas backends)

from cleaned_operators.markov_dynamics import _state_dynamics_series
from cleaned_operators.first_passage import _first_passage_series
from cleaned_operators.registry import OperatorRegistry


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


def _frame(a: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"a": np.asarray(a, dtype=float)})


# ---------------------------------------------------------------------------
# 1. Markov window-maturity contract
# ---------------------------------------------------------------------------

def test_markov_window_maturity_short_series_emits_no_fake_values():
    """5 points with window=60 must NOT produce a confident estimate.

    Before the fix the kernel estimated a transition matrix from ~2-4 lagged
    transitions, so persistence/entropy/surprisal could be finite.  After the
    fix the window is immature (fewer than min_history=10 finite observations)
    and every estimate fails closed to NaN.
    """
    x = _frame(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    for name in ("ts_markov_persistence", "ts_markov_state_entropy",
                 "ts_markov_transition_surprisal"):
        out = _op(name).calculate(x, window=60, bins=3, lag=1, min_count=1).to_numpy(dtype=float)
        assert np.isnan(out).all(), f"{name} emitted a fake value from an immature window"


def test_markov_window_maturity_kernel_gate():
    """The kernel itself NaNs P/pi/D1/D2 for immature windows but keeps the raw
    support fields (total_trans / counts / N_obs) honest."""
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = _state_dynamics_series(s, window=60, bins=3, lag=1, min_count=1, min_history=10)
    assert not any(np.isfinite(res["P"][t]).any() for t in range(5)), "immature window emitted P"
    # Raw support fields are still recorded on the physical axis.
    assert res["total_trans"][4] == pytest.approx(3.0)
    # A lower min_history makes the same window mature enough to estimate.
    res2 = _state_dynamics_series(s, window=60, bins=3, lag=1, min_count=1, min_history=3)
    assert np.isfinite(res2["P"][4]).any(), "mature window should estimate P"


def test_markov_window_maturity_sufficient_history_produces_output():
    """window is a MAX LOOKBACK: a short but sufficiently-observed window still
    emits a real estimate; a long enough history produces finite output."""
    rng = np.random.default_rng(7)
    x = _frame(rng.normal(0.0, 1.0, size=200))
    out = _op("ts_markov_persistence").calculate(x, 60, 3, 1, 3).to_numpy(dtype=float)
    assert np.isfinite(out).any(), "sufficient history should produce a persistence estimate"


def test_markov_window_maturity_relational_gates_declared():
    """min_history / min_state_support are explicit contract params with
    relational feasibility (<= window) and min_count aliases min_transition_count."""
    op = _op("ts_markov_persistence")
    assert "min_history" in op.metadata.param_names
    assert "min_state_support" in op.metadata.param_names
    exprs = {s.expression for s in op.metadata.relational_specs}
    assert "min_history <= window" in exprs
    assert "min_state_support <= window" in exprs
    assert "min_count <= window - lag" in exprs
    assert "min_transition_count" in (op.metadata.param_aliases or {})


def test_markov_min_transition_count_alias_binds_to_min_count():
    x = _frame(np.random.default_rng(1).normal(size=80))
    op = _op("ts_markov_persistence")
    a = op.calculate(x, 60, 3, 1, min_transition_count=3).to_numpy(dtype=float)
    b = op.calculate(x, 60, 3, 1, min_count=3).to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


# ---------------------------------------------------------------------------
# 2. First-Passage strict params
# ---------------------------------------------------------------------------

def _fp_inputs(n: int = 200):
    rng = np.random.default_rng(0)
    lp = np.log(np.cumprod(1 + rng.normal(0.0, 0.01, size=n)))
    sc = np.abs(rng.normal(0.01, 0.002, size=n)) + 0.005
    return _frame(lp), _frame(sc)


def test_first_passage_strict_params_reject_fractional_and_zero():
    op = _op("ts_first_passage_bias")
    x, scale = _fp_inputs()
    for kw in ({"horizon": 4.8}, {"window": 0.5}, {"min_anchors": 0}, {"scale_horizon": 0}):
        with pytest.raises((TypeError, ValueError)):
            op.calculate(x, scale, 120, 1.0, 10, 3, **kw)


def test_first_passage_barrier_is_float_never_truncated():
    op = _op("ts_first_passage_bias")
    x, scale = _fp_inputs()
    # A fractional barrier is legal (float semantics) and is NOT truncated.
    out = op.calculate(x, scale, 120, 1.5, 10, 3).to_numpy(dtype=float)
    assert np.isfinite(out).any()
    # Non-positive barriers are rejected loudly (never collapsed to EPS).
    with pytest.raises(ValueError, match="barrier"):
        op.calculate(x, scale, 120, 0.0, 10, 3)
    with pytest.raises(ValueError, match="barrier"):
        op.calculate(x, scale, 120, -1.0, 10, 3)


# ---------------------------------------------------------------------------
# 3. First-Passage units / scale_horizon semantic identity
# ---------------------------------------------------------------------------

def test_first_passage_scale_horizon_in_signature_and_identity():
    op = _op("ts_first_passage_bias")
    assert "scale_horizon" in op.metadata.param_names
    sig = next((t for t in op.metadata.tags if t.startswith("signature:")), "")
    assert "scale_horizon" in sig
    # 1-day vs 20-day scale_horizon are different bound semantic identities.
    from cleaned_operators.base import bind_operator_call
    x, scale = _fp_inputs()
    b1 = bind_operator_call(op, (x, scale),
                            dict(window=120, barrier=1.0, horizon=10, min_anchors=3, scale_horizon=1))
    b20 = bind_operator_call(op, (x, scale),
                             dict(window=120, barrier=1.0, horizon=10, min_anchors=3, scale_horizon=20))
    assert b1.bound.normalized_values["scale_horizon"] == 1
    assert b20.bound.normalized_values["scale_horizon"] == 20


def test_first_passage_unit_mismatch_raw_price_rejected():
    """unit(scale) == unit(x): a raw price LEVEL with a return-vol scale raises."""
    op = _op("ts_first_passage_bias")
    price = np.linspace(100.0, 159.0, 60)
    scale = np.full(60, 0.02)
    with pytest.raises(ValueError, match="unit mismatch"):
        op.calculate(_frame(price), _frame(scale), window=40, barrier=1.0, horizon=5, min_anchors=3)


# ---------------------------------------------------------------------------
# 4. First-Passage bias includes non-hit anchors (doc + math)
# ---------------------------------------------------------------------------

def test_first_passage_bias_all_nonhit_is_zero_not_nan():
    """Fully-observed anchors that never touch a barrier contribute 0; with no
    hits at all the bias is exactly 0.0 (a conditional-on-hit statistic would be
    NaN here)."""
    x = np.zeros(30)
    scale = np.full(30, 100.0)  # barrier*scale huge -> never touched
    out = _first_passage_series(x, scale, window=20, barrier=1.0, horizon=5, min_anchors=2)
    assert np.isfinite(out[-1]) and out[-1] == 0.0


def test_first_passage_bias_equals_direction_times_speed_times_hit_prob():
    """Reference re-implementation of the bias formula, INCLUDING non-hit
    anchors as 0 contributions, must match the kernel on a mixed-hit series."""
    # x jumps up once at position 1, then stays at a level that never re-touches
    # the barrier for later anchors -> 1 up-hit anchor + 7 non-hit anchors.
    x = np.array([1.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    scale = np.ones(11)
    barrier, horizon, window, min_anchors = 1.0, 3, 10, 3
    t = 10

    def _ref(x, scale, window, barrier, horizon, min_anchors, t):
        signs: list[float] = []
        for s in range(max(0, t - window), t - horizon + 1):
            xs, sc = x[s], scale[s]
            if not np.isfinite(xs) or not np.isfinite(sc) or sc <= 0.0:
                continue
            up, down = xs + barrier * sc, xs - barrier * sc
            tau, d, fully = 0, 0.0, True
            for h in range(1, horizon + 1):
                val = x[s + h]
                if not np.isfinite(val):
                    fully = False
                    break
                if val >= up:
                    tau, d = h, 1.0
                    break
                if val <= down:
                    tau, d = h, -1.0
                    break
            if not fully:
                continue
            if tau > 0:
                signs.append(d * (horizon + 1 - tau) / horizon)
            else:
                signs.append(0.0)  # fully-observed non-hit anchor -> 0
        if len(signs) < min_anchors:
            return np.nan
        return float(np.mean(signs))

    got = _first_passage_series(x, scale, window, barrier, horizon, min_anchors)[t]
    expected = _ref(x, scale, window, barrier, horizon, min_anchors, t)
    assert np.isfinite(got), "bias must be finite even when non-hit anchors are present"
    assert got == pytest.approx(expected)


def test_first_passage_bias_docstrings_no_longer_say_conditional_on_hit():
    """The hit_probability docstring previously claimed bias averages 'only over
    hit anchors' — the bias actually includes fully-observed non-hit anchors
    (0 contribution).  Docstrings must describe the full formula."""
    doc = _op("ts_first_passage_bias").__doc__ or ""
    assert ("未命中" in doc or "未触及" in doc) and "0" in doc
    hp_doc = _op("ts_first_passage_hit_probability").__doc__ or ""
    assert "未命中" in hp_doc or "未触及" in hp_doc


# ---------------------------------------------------------------------------
# 5. Markov PIT: estimates built strictly from [t-W, t-1]
# ---------------------------------------------------------------------------

def test_markov_kernel_pit_excludes_current_value_from_estimates():
    """P / counts / edges at row t0 are built from [t0-W, t0-1] only; perturbing
    x[t0] must not change the row-t0 estimate (it only changes the selected
    state)."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=80)
    t0 = 50
    alt = base.copy()
    alt[t0] = base[t0] + 1000.0
    r_base = _state_dynamics_series(base, window=30, bins=3, lag=1, min_count=2)
    r_alt = _state_dynamics_series(alt, window=30, bins=3, lag=1, min_count=2)
    assert np.allclose(r_base["P"][t0], r_alt["P"][t0], equal_nan=True)
    assert np.allclose(r_base["counts"][t0], r_alt["counts"][t0], equal_nan=True)
    assert np.allclose(r_base["edges"][t0], r_alt["edges"][t0], equal_nan=True)


def test_markov_module_docstring_declares_timing_semantics():
    import cleaned_operators.markov_dynamics as md
    doc = md.__doc__ or ""
    assert "strictly-past window" in doc and "max lookback" in doc
    assert "PRIOR_REFERENCE_CURRENT_QUERY" in doc
