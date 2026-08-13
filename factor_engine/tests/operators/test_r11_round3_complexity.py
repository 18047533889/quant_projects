# -*- coding: utf-8 -*-
"""R11 round-3 complexity audit regression tests (review items #44-#48).

Owned modules: ``ts_model/complexity.py``, ``sequence_complexity.py``,
``complexity_ext.py``, ``threshold_cycle.py``.

* #44 — permutation / ordinal entropy tie handling is mathematically
  consistent: a window whose ordinal pattern contains ANY tie is DROPPED from
  the permutation-count state space (average tie ranks are a continuum, so
  normalizing by log(order!) would be inconsistent).  Applies to the
  permutation-entropy kernel in ``ts_model.complexity`` and to the three
  permutation-entropy operators in ``sequence_complexity``.  The
  forbidden-ordinal family in ``complexity_ext`` already dropped tied
  embeddings in round 2 and is untouched.
* #45 — ``ts_sample_entropy`` A=0 handling: SampEn = -ln(A/B) is NOT log(B)
  when A=0; it is infinity/undefined.  Production fails closed to NaN; the
  separate ``ts_pseudocount_sample_entropy`` keeps a finite +1 pseudocount
  variant.
* #46 — MSE coarse-graining is RIGHT-ALIGNED: when the window is not divisible
  by the scale the OLDEST remainder is dropped, never the newest bars.
* #47 — ``ts_multiscale_entropy_slope`` regresses entropy on LOG scale
  (Entropy(s) ~ a + b·log s), consistent with the log-log DFA convention.
* #48 — ``ts_two_state_regime_probability`` is a genuine Markov switching
  filter: the prediction step pi_{t|t-1} = P^T · pi_{t-1|t-1} uses the
  transition matrix before the likelihood update (it is NOT a cumulative Bayes
  evidence product).

The shared tree's finalize-layer-governance is transiently broken by the
concurrent session editing the static operator surface, so this module imports
the owned operator modules directly (self-contained) instead of relying on a
full ``load_all``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.base import ParamRole
from cleaned_operators.registry import OperatorRegistry

# Owned modules in load_all order: ts_model.complexity registers first and
# sequence_complexity is a DECLARED_OVERRIDE_SOURCES layer that re-registers the
# research-surface names (ts_permutation_entropy / ts_sample_entropy).
import cleaned_operators.ts_model.complexity as _ts_cplx  # noqa: E402,F401
import cleaned_operators.sequence_complexity as _seq_cplx  # noqa: E402,F401
import cleaned_operators.complexity_ext as _cplx_ext  # noqa: E402,F401
import cleaned_operators.threshold_cycle as _thr  # noqa: E402,F401


def _frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame({c: np.asarray(values, dtype=float) for c in ("A", "B")}, index=idx)


def _assert_same(a: float, b: float) -> None:
    if np.isnan(a) and np.isnan(b):
        return
    assert a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------------------
# #44 — permutation entropy drops tied embeddings
# ---------------------------------------------------------------------------
def _ref_perm_entropy(values, order, window):
    """Reference permutation entropy that DROPS tied embeddings (item #44)."""
    seg = np.asarray(values, dtype=float)[-int(window):]
    counts: dict[tuple[int, ...], int] = {}
    for i in range(len(seg) - order + 1):
        block = seg[i : i + order]
        if np.unique(block).size < block.size:
            continue  # ordinal tie -> drop the embedding
        oi = np.argsort(block, kind="stable")
        pattern = tuple(int(x) for x in np.argsort(oi, kind="stable"))
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(counts.values())
    if total <= 1:
        return np.nan
    h = -sum(c / total * math.log(c / total) for c in counts.values())
    return h / math.log(math.factorial(order))


def test_permutation_entropy_matches_tie_drop_reference():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=200)
    tied = vals.copy()
    tied[::3] = tied[1::3]  # force a tie in ~1/3 of embeddings
    for order in (3, 4):
        for window in (80, 120):
            _assert_same(_ts_cplx._permutation_entropy(vals, order, window),
                         _ref_perm_entropy(vals, order, window))
            _assert_same(_ts_cplx._permutation_entropy(tied, order, window),
                         _ref_perm_entropy(tied, order, window))


def test_permutation_entropy_constant_is_nan():
    # A fully tied window has zero no-tie embeddings -> NaN (never a fabricated
    # "single permutation state" from an average-tie-rank continuum).
    assert np.isnan(_ts_cplx._permutation_entropy(np.ones(60), 3, 60))
    op = OperatorRegistry.get("ts_permutation_entropy", "pandas_numpy")
    assert int(op.calculate(_frame(np.zeros(160)), window=60, order=3).notna().to_numpy().sum()) == 0


def test_sequence_permutation_pattern_tie_returns_none():
    assert _seq_cplx._permutation_pattern(np.array([0.3, 0.3, 0.2])) is None
    assert _seq_cplx._permutation_pattern(np.array([0.3, 0.1, 0.2])) == (2, 0, 1)
    # constant window -> all embeddings dropped -> tie_fraction = 1.0
    codes, tf = _seq_cplx._permutation_codes(np.ones(60), 3, 1)
    assert len(codes) == 0 and tf == 1.0


def test_sequence_permutation_entropy_heavily_tied_fails_closed():
    rng = np.random.default_rng(2)
    # Discrete values over {0,1,2}: only ~22% of order-3 embeddings are tie-free
    # -> tie_fraction > 0.5, so the fail-closed guard (P1-14) emits NaN.
    discrete = rng.integers(0, 3, 160).astype(float)
    op = OperatorRegistry.get("ts_permutation_entropy", "pandas_numpy")
    assert int(op.calculate(_frame(discrete), window=60, order=3).notna().to_numpy().sum()) == 0
    # Continuous random data: essentially no ties -> finite output.
    rnd = rng.normal(size=160)
    assert int(op.calculate(_frame(rnd), window=60, order=3).notna().to_numpy().sum() > 0


def test_weighted_and_transition_entropy_drop_ties():
    op_w = OperatorRegistry.get("ts_weighted_permutation_entropy", "pandas_numpy")
    op_t = OperatorRegistry.get("ts_permutation_transition_entropy", "pandas_numpy")
    const = _frame(np.zeros(160))
    assert int(op_w.calculate(const, window=60, order=3).notna().to_numpy().sum()) == 0
    assert int(op_t.calculate(const, window=60, order=3).notna().to_numpy().sum()) == 0
    rng = np.random.default_rng(9)
    rnd = _frame(rng.normal(size=160))
    assert int(op_w.calculate(rnd, window=60, order=3).notna().to_numpy().sum() > 0
    assert int(op_t.calculate(rnd, window=60, order=3).notna().to_numpy().sum() > 0


def test_ordinal_knob_params_declared_estimator_resolution():
    # Cross-cutting rule: order / delay are estimator-resolution knobs, not
    # economic search dimensions — declared searchable=False.
    for name in (
        "ts_permutation_entropy",
        "ts_weighted_permutation_entropy",
        "ts_permutation_transition_entropy",
    ):
        op = OperatorRegistry.get(name, "pandas_numpy")
        for knob in ("order", "delay"):
            spec = op.metadata.param_specs.get(knob)
            assert spec is not None, f"{name}: {knob} ParamSpec missing"
            assert spec.param_role == ParamRole.ESTIMATOR_RESOLUTION, name
            assert spec.searchable is False, name


# ---------------------------------------------------------------------------
# #45 — sample entropy A=0 handling
# ---------------------------------------------------------------------------
def test_sample_entropy_a_zero_is_nan():
    # Length-2 block (1,2) repeats with different continuations -> B > 0, A = 0.
    x = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 6.0])
    assert np.isnan(_ts_cplx._sample_entropy(x, 2, 0.05, len(x)))
    # A > 0 normal case is finite.
    y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    assert np.isfinite(_ts_cplx._sample_entropy(y, 2, 0.05, len(y)))
    # B == 0 also NaN.
    z = np.arange(12, dtype=float)
    assert np.isnan(_ts_cplx._sample_entropy(z, 2, 0.01, len(z)))


def test_pseudocount_sample_entropy_finite_at_a_zero():
    x = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 6.0])
    pc = _ts_cplx._pseudocount_sample_entropy(x, 2, 0.05, len(x))
    assert np.isfinite(pc)
    # At A=0 the +1 pseudocount gives -ln(1/(B+1)) = ln(B+1) > 0.
    assert pc > 0.0


def test_pseudocount_sample_entropy_registered():
    op = OperatorRegistry.get("ts_pseudocount_sample_entropy", "pandas_numpy")
    assert op is not None


# ---------------------------------------------------------------------------
# #46 / #47 — MSE right-aligned coarse-graining + log-scale slope
# ---------------------------------------------------------------------------
def _ref_multiscale_slope(values, max_scale, window, left=False, log_scale=True):
    """Reference multiscale-entropy slope.

    ``left=True`` reproduces the old (wrong) LEFT-aligned coarse-graining;
    ``log_scale=False`` reproduces a RAW-scale regression.  Default is the
    reviewed behavior: right-aligned coarse-graining + log-scale OLS.
    """
    finite = _ts_cplx.trailing_contiguous_finite(
        np.asarray(values, dtype=float)[-int(window):]
    )
    if len(finite) < 16:
        return np.nan
    scales: list[float] = []
    ents: list[float] = []
    for s in range(1, max(2, int(max_scale)) + 1):
        if len(finite) < s * 3:
            continue
        if left:
            coarse = finite[: len(finite) // s * s].reshape(-1, s).mean(axis=1)
        else:
            coarse = finite[len(finite) % s :].reshape(-1, s).mean(axis=1)
        e = _ts_cplx._permutation_entropy(coarse, 3, len(coarse))
        if np.isfinite(e):
            scales.append(float(s))
            ents.append(e)
    if len(scales) < 2:
        return np.nan
    xs = np.asarray(scales, dtype=float)
    if log_scale:
        xs = np.log(xs)
    sx = xs - float(np.mean(xs))
    sy = np.asarray(ents, dtype=float) - float(np.mean(ents))
    denom = float(np.dot(sx, sx))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(sx, sy) / denom)


def test_multiscale_slope_matches_right_aligned_log_reference():
    rng = np.random.default_rng(5)
    for n, max_scale in ((118, 4), (157, 5), (200, 5)):
        vals = rng.normal(size=n)
        got = _ts_cplx._multiscale_entropy_slope(vals, max_scale, n)
        ref = _ref_multiscale_slope(vals, max_scale, n)
        _assert_same(got, ref)


def test_multiscale_coarse_grain_is_right_aligned_not_left():
    rng = np.random.default_rng(5)
    vals = rng.normal(size=118)  # 118 % 4 == 2, 118 % 3 == 1 -> alignments differ
    got = _ts_cplx._multiscale_entropy_slope(vals, 4, 118)
    ref_left = _ref_multiscale_slope(vals, 4, 118, left=True)
    assert np.isfinite(got) and np.isfinite(ref_left)
    # The right-aligned kernel must NOT equal the old left-aligned coarse grain.
    assert not np.isclose(got, ref_left, atol=1e-9)


def test_multiscale_slope_is_log_scale_not_raw():
    rng = np.random.default_rng(6)
    vals = rng.normal(size=200)
    got = _ts_cplx._multiscale_entropy_slope(vals, 5, 200)
    ref_raw = _ref_multiscale_slope(vals, 5, 200, log_scale=False)
    assert np.isfinite(got) and np.isfinite(ref_raw)
    # Entropy(s) ~ a + b·log s: the slope must regress on log scale.
    assert not np.isclose(got, ref_raw, atol=1e-9)


# ---------------------------------------------------------------------------
# #48 — two-state regime probability is a Markov switching filter
# ---------------------------------------------------------------------------
def _ref_markov_filter(vals, window, p_switch):
    """Reference two-state Gaussian filter with the Markov prediction step."""
    r = np.asarray(vals, dtype=float)[-int(window):]
    var_all = float(np.var(r))
    sigma_hi = np.sqrt(max(var_all * 2.0, 1e-12))
    sigma_lo = np.sqrt(max(var_all * 0.5, 1e-12))
    P = np.array([[1.0 - p_switch, p_switch], [p_switch, 1.0 - p_switch]])
    p_hi = 0.5
    for x in r:
        like_hi = np.exp(-0.5 * (x / sigma_hi) ** 2) / sigma_hi
        like_lo = np.exp(-0.5 * (x / sigma_lo) ** 2) / sigma_lo
        p_lo = 1.0 - p_hi
        pred_lo = P[0, 0] * p_lo + P[1, 0] * p_hi
        pred_hi = P[0, 1] * p_lo + P[1, 1] * p_hi
        post_lo = like_lo * pred_lo
        post_hi = like_hi * pred_hi
        z = post_lo + post_hi
        p_hi = min(max(post_hi / max(z, 1e-300), 1e-6), 1.0 - 1e-6)
    return float(p_hi)


def test_regime_filter_matches_markov_reference():
    rng = np.random.default_rng(10)
    vals = rng.normal(size=150)
    for p in (0.01, 0.1, 0.3):
        got = _ts_cplx._regime_filter(vals, 150, "prob", p)
        ref = _ref_markov_filter(vals, 150, p)
        assert got == pytest.approx(ref, abs=1e-12)


def test_regime_filter_is_not_cumulative_bayes():
    # A cumulative-Bayes recursion (no transition matrix) has no switching knob;
    # a Markov filter must respond to it.  On a low-vol -> high-vol switch, a
    # sticky transition keeps the latched posterior high while a fast transition
    # pulls it back toward the stationary 0.5 — so the two must differ.
    rng = np.random.default_rng(7)
    series = np.concatenate([rng.normal(0, 0.5, 80), rng.normal(0, 3.0, 80)])
    p_sticky = _ts_cplx._regime_filter(series, 160, "prob", 1e-5)
    p_fast = _ts_cplx._regime_filter(series, 160, "prob", 0.5)
    assert np.isfinite(p_sticky) and np.isfinite(p_fast)
    assert abs(p_sticky - p_fast) > 1e-3


def test_two_state_regime_probability_in_unit_range():
    rng = np.random.default_rng(8)
    out = OperatorRegistry.get("ts_two_state_regime_probability", "pandas_numpy").calculate(
        _frame(rng.normal(size=160)), window=60
    )
    vals = out.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_regime_transition_prob_knob_declared_policy():
    op = OperatorRegistry.get("ts_two_state_regime_probability", "pandas_numpy")
    spec = op.metadata.param_specs.get("transition_prob")
    assert spec is not None
    assert spec.param_role == ParamRole.POLICY
    assert spec.searchable is False
    # The parameter is declared on all three regime-family canonicals.
    for name in ("ts_regime_duration", "ts_change_point_probability"):
        op2 = OperatorRegistry.get(name, "pandas_numpy")
        assert op2 is not None and "transition_prob" in op2.metadata.param_names


def test_regime_ops_accept_transition_prob_kwarg():
    rng = np.random.default_rng(11)
    x = _frame(rng.normal(size=160))
    for name in ("ts_two_state_regime_probability", "ts_regime_duration", "ts_change_point_probability"):
        op = OperatorRegistry.get(name, "pandas_numpy")
        out = op.calculate(x, window=60, transition_prob=0.1)
        assert int(out.notna().to_numpy().sum() > 0, name
        # Default (no explicit transition_prob) still works.
        out2 = op.calculate(x, window=60)
        assert int(out2.notna().to_numpy().sum() > 0, name
