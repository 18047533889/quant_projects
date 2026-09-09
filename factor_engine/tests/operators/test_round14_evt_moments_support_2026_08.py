# -*- coding: utf-8 -*-
"""R14 feasibility/support round: EVT declared-vs-runtime feasibility, moments
effective-sample gates, Lyapunov unit + physical-time Theiler.

* ``ts_evt_threshold_stability`` — the declared relational contract must match
  the runtime guard (``window >= 2*k_max + 2``), so window=30 / k_max=20 is
  REJECTED by the declared contract AND all-NaN at runtime (never a bogus
  value).
* ``ts_l_skewness`` / ``ts_l_kurtosis`` / ``ts_hartigan_dip`` — statistical-
  usability gates ("不要把数学上可算当成统计上可用"): effective finite n
  (``min_periods``) and coverage of the nominal trailing window
  (``min_coverage_fraction``); a sparse window emits NaN instead of an
  incomparable number.
* ``ts_local_lyapunov_exponent`` — declared unit ``rate``; a NaN at the current
  row must not emit a stale value; Theiler exclusion uses ORIGINAL physical
  time when NaN rows are present.

NOTE on loading: this workspace's full ``cleaned_operators.load_all()`` is
currently blocked by an unrelated concurrent-session registration error in
``advanced_information.py`` / ``polars_geometry_math.py``.  These tests import
the three modules under test directly, which registers exactly the canonicals
they exercise, so the suite is decoupled from that pre-existing breakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.evt_allan  # noqa: F401  (registers the EVT family)
import factor_engine.cleaned_operators.moments_ext  # noqa: F401  (registers the L-moment/dip family)
import factor_engine.cleaned_operators.local_lyapunov  # noqa: F401  (registers the Lyapunov op)
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _frame(values: np.ndarray) -> pd.DataFrame:
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    return pd.DataFrame(
        v,
        index=pd.date_range("2024-01-01", periods=v.shape[0], freq="D"),
        columns=[f"S{i}" for i in range(v.shape[1])],
    )


def _op(canonical: str):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, f"{canonical} not registered"
    return op


# ---------------------------------------------------------------------------
# ITEM 1 — EVT threshold stability: declared feasibility == runtime guard
# ---------------------------------------------------------------------------
def test_evt_declared_relation_rejects_infeasible_region():
    from factor_engine.cleaned_operators.evt_allan import _EVT_RELATIONAL_SPECS

    # window=30 / k_max=20 was previously declared legal ("k_max <= window - 2")
    # but the runtime guard needs v.size >= 2*20+2 = 42, so every call was
    # guaranteed all-NaN.  The declared contract now expresses the real region.
    feasibility = next(
        r for r in _EVT_RELATIONAL_SPECS if "k_max" in r.param_names and "window" in r.param_names
    )
    assert not feasibility.check({"window": 30, "k_max": 20})
    assert feasibility.check({"window": 60, "k_max": 12})


def test_evt_contract_rejects_window30_kmax20():
    op = _op("ts_evt_threshold_stability")
    df = _frame(np.random.default_rng(0).normal(size=(50, 1)))
    with pytest.raises(ValueError):
        op.calculate(df, window=30, k_min=5, k_max=20)


def test_evt_runtime_emits_nan_not_bogus_for_infeasible():
    from factor_engine.cleaned_operators.evt_allan import _ts_evt_threshold_stability

    df = _frame(np.random.default_rng(1).normal(size=(50, 1)))
    # Bypass the declared contract on purpose: the runtime guard itself must
    # refuse the combination (all-NaN), never emit a bogus stability score.
    out = _ts_evt_threshold_stability(df, window=30, k_min=5, k_max=20)
    assert np.isnan(out.to_numpy(dtype=float)).all()


# ---------------------------------------------------------------------------
# ITEM 2 — moments effective-sample support
# ---------------------------------------------------------------------------
def test_l_moment_low_effective_n_nan():
    op = _op("ts_l_skewness")
    base = np.full((120, 1), np.nan)
    base[114:120, 0] = np.random.default_rng(2).normal(size=6)  # only ~6 finite
    out = op.calculate(_frame(base), window=120)
    assert np.isnan(out.to_numpy(dtype=float)).all()


def test_l_moment_high_effective_n_finite():
    op = _op("ts_l_skewness")
    base = np.full((120, 1), np.nan)
    base[0:40, 0] = np.random.default_rng(3).normal(size=40)  # ~40 finite
    arr = op.calculate(_frame(base), window=120).to_numpy(dtype=float)
    # At r=39 the trailing window is exactly the 40 finite rows: effective
    # n = 40 >= min_periods(20) and coverage = 40/40 = 1.0 >= 0.5 -> finite.
    assert np.isfinite(arr[39, 0])


def test_l_kurtosis_high_effective_n_finite():
    op = _op("ts_l_kurtosis")
    base = np.full((120, 1), np.nan)
    base[0:40, 0] = np.random.default_rng(4).normal(size=40)
    arr = op.calculate(_frame(base), window=120).to_numpy(dtype=float)
    assert np.isfinite(arr[39, 0])


def test_hartigan_dip_low_coverage_nan():
    op = _op("ts_hartigan_dip")
    base = np.full((120, 1), np.nan)
    base[115:120, 0] = np.random.default_rng(5).normal(size=5)  # ~5 finite
    # min_periods=30 exercises the stricter test-time threshold; effective n=5
    # is far below it, so the whole column is NaN.
    out = op.calculate(_frame(base), window=120, min_periods=30)
    assert np.isnan(out.to_numpy(dtype=float)).all()


def test_hartigan_dip_high_coverage_finite():
    op = _op("ts_hartigan_dip")
    base = np.full((120, 1), np.nan)
    base[0:100, 0] = np.random.default_rng(6).normal(size=100)  # high coverage
    arr = op.calculate(_frame(base), window=120, min_periods=30).to_numpy(dtype=float)
    # effective n = 100 >= 30 and coverage 100/100 = 1.0 (r=99) or
    # 100/120 = 0.833 (r=119) >= 0.8 -> finite.
    assert np.isfinite(arr[99, 0])
    assert np.isfinite(arr[119, 0])


def test_moments_effective_sample_params_declared():
    # The gates must be visible catalog parameters, not hidden kernel knobs.
    import inspect

    for canon, default_cov in (
        ("ts_l_skewness", 0.5),
        ("ts_l_kurtosis", 0.5),
        ("ts_hartigan_dip", 0.8),
    ):
        meta = _op(canon).metadata
        names = list(meta.param_names)
        assert "min_periods" in names, canon
        assert "min_coverage_fraction" in names, canon
        # default thresholds as specified in the R14 audit.
        params = inspect.signature(_op(canon)._calculate_series).parameters
        assert params["min_coverage_fraction"].default == pytest.approx(default_cov), canon
        assert params["min_periods"].default == 20, canon


# ---------------------------------------------------------------------------
# ITEM 3 — Lyapunov unit + physical-time Theiler
# ---------------------------------------------------------------------------
def test_lyapunov_declared_unit_is_rate():
    op = _op("ts_local_lyapunov_exponent")
    unit_tags = [t for t in op.metadata.tags if t.startswith("unit:")]
    assert unit_tags == ["unit:rate"]


def test_lyapunov_current_row_nan_no_stale_value():
    op = _op("ts_local_lyapunov_exponent")
    rng = np.random.default_rng(7)
    x = rng.normal(size=120)
    # sanity: with a fully-finite window the operator emits a value at the last
    # row (so the NaN below is caused by the current-row guard, not by a general
    # failure to produce output).
    full = op.calculate(_frame(x), window=120, tau=1, embedding_dim=3, horizon=5, min_anchors=3)
    assert np.isfinite(full.to_numpy(dtype=float)[-1, 0])
    # With the CURRENT row NaN the operator must not fall back to the older
    # window (a stale "yesterday" factor) — it emits NaN.
    x_nan = x.copy()
    x_nan[-1] = np.nan
    nan_out = op.calculate(_frame(x_nan), window=120, tau=1, embedding_dim=3, horizon=5, min_anchors=3)
    assert np.isnan(nan_out.to_numpy(dtype=float)[-1, 0])


def test_lyapunov_theiler_uses_physical_time():
    op = _op("ts_local_lyapunov_exponent")
    # A gap changes true anchor spacing without compressing the physical clock.
    # Both policies remain usable, but they need not select the same neighbours.
    x = np.random.default_rng(14).normal(size=48).cumsum()
    x[[18, 19]] = np.nan
    common = dict(window=32, tau=1, embedding_dim=2, horizon=2, min_anchors=1)
    compressed = op.calculate(_frame(x[:, None]), **common, physical_time=False)
    physical = op.calculate(_frame(x[:, None]), **common, physical_time=True)
    comp = compressed.to_numpy(dtype=float)[:, 0]
    phys = physical.to_numpy(dtype=float)[:, 0]
    assert np.isfinite(comp).any()
    assert np.isfinite(phys).any()
    assert not np.allclose(comp, phys, equal_nan=True)


# ---------------------------------------------------------------------------
# smoke import / eval of the three modules
# ---------------------------------------------------------------------------
def test_smoke_import_and_kernel_eval():
    from factor_engine.cleaned_operators.evt_allan import _ts_evt_threshold_stability
    from factor_engine.cleaned_operators.moments_ext import _dip_series, _l_ratio_series
    from factor_engine.cleaned_operators.local_lyapunov import _lyapunov_series

    rng = np.random.default_rng(9)
    x = rng.normal(size=(60, 2))
    assert _ts_evt_threshold_stability(_frame(x), window=30, k_min=3, k_max=8) is not None
    assert _l_ratio_series(x, 40, "skew") is not None
    assert _dip_series(x, 40) is not None
    assert _lyapunov_series(x[:, 0], 40, 1, 3, 3, 2) is not None
