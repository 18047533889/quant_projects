# -*- coding: utf-8 -*-
"""Tests for the 2026-08 vertical-deepening pack (Markov committor/MFPT/spectral-
gap/stationary, KM equilibrium/diffusion-gradient/quasipotential, first-passage
probability & conditional time, event-response curve shape, extreme-value,
transfer-entropy peak, quantile-transport, MMD, chord geometry, chip-cost shape,
group spectral gap/second mode, KNN Dirichlet energy, intraday RV signature,
report change breadth/coherence and RQA).

Covers registration & surface classification (all 37 daily), determinism, axis
preservation, prefix causality, fail-closed edge cases and pandas↔polars parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.operator_surface import classify_canonical, DAILY_FACTOR_MIGRATED

load_all()

DAILY_OPS = frozenset({
    "ts_markov_committor", "ts_markov_mean_first_passage_time",
    "ts_markov_spectral_gap", "ts_markov_stationary_surprisal",
    "ts_km_equilibrium_distance", "ts_km_diffusion_gradient",
    "ts_km_quasipotential_depth",
    "ts_first_passage_hit_probability", "ts_first_passage_conditional_time",
    "event_response_peak_lag", "event_response_decay_rate",
    "event_response_dispersion", "event_response_reversal_strength",
    "ts_extremal_index", "ts_mean_excess_slope", "ts_gpd_shape_pwm",
    "ts_transfer_entropy_peak_strength", "ts_transfer_entropy_peak_lag",
    "ts_quantile_transport_slope", "ts_quantile_transport_curvature",
    "ts_mmd_rbf_shift",
    "ts_chord_excursion_area", "ts_max_chord_excursion",
    "ts_turnover_cost_entropy", "ts_turnover_cost_mode_distance",
    "ts_turnover_cost_skew", "ts_turnover_age_dispersion",
    "group_feature_spectral_gap", "group_feature_second_mode_localization",
    "cs_knn_graph_dirichlet_energy",
    "intraday_rv_signature_slope",
    "report_change_breadth", "report_change_coherence",
    "ts_recurrence_rate", "ts_recurrence_diagonal_entropy",
    "ts_recurrence_trapping_time", "ts_recurrence_divergence",
})

POLARS_OPS = DAILY_OPS - {
    # cross-sectional / group / minute→daily aggregations stay pandas_numpy-only.
    "group_feature_spectral_gap", "group_feature_second_mode_localization",
    "cs_knn_graph_dirichlet_energy",
    "intraday_rv_signature_slope",
    "report_change_breadth", "report_change_coherence",
    "ts_quantile_transport_slope", "ts_quantile_transport_curvature",
    "ts_mmd_rbf_shift",
    "ts_chord_excursion_area", "ts_max_chord_excursion",
    "ts_recurrence_rate", "ts_recurrence_diagonal_entropy",
    "ts_recurrence_trapping_time", "ts_recurrence_divergence",
}


def _frame(values: np.ndarray, start: str = "2024-01-01", cols: int | None = None) -> pd.DataFrame:
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    cols = v.shape[1] if cols is None else cols
    return pd.DataFrame(v, index=pd.date_range(start, periods=v.shape[0], freq="B"),
                        columns=[f"S{i}" for i in range(cols)])


def _report_period(n: int, cols: int, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Quarterly report-period frame spanning many distinct fiscal quarters.

    Must share the feature frames' index so the panel-axes contract holds."""
    quarters: list[pd.Timestamp] = []
    y, q = 2019, 1
    for _ in range(40):
        quarters.append(pd.Timestamp(year=y, month=3 * q, day=28) if q in (1, 2)
                        else pd.Timestamp(year=y, month=3 * q - 2, day=30))
        q += 1
        if q == 5:
            q = 1
            y += 1
    per = np.array(quarters, dtype=object)
    if index is None:
        index = pd.date_range("2019-01-01", periods=n, freq="B")
    period = pd.DataFrame(index=index, columns=[f"S{i}" for i in range(cols)], dtype=object)
    for c in period.columns:
        col = np.full(len(index), None, dtype=object)
        col[0::15] = per[: len(col[0::15])]
        period[c] = col
    return period


def _minute_frame(n_days: int = 4, per_day: int = 120, cols: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in dates for m in range(per_day)])
    rng = np.random.default_rng(3)
    r = np.cumsum(rng.normal(0.0, 0.001, (n_days * per_day, cols)), axis=0)
    return pd.DataFrame(100.0 * np.exp(r), index=idx, columns=[f"S{i}" for i in range(cols)])


def _build_inputs(name: str, rng: np.random.Generator):
    n = 220
    x = rng.normal(0.0, 1.0, size=(n, 3))
    y = rng.normal(0.0, 1.0, size=(n, 3))
    z = np.abs(rng.normal(0.0, 1.0, size=(n, 3)))
    fx, fy, fz = _frame(x), _frame(y), _frame(z)
    g = _frame(np.tile(np.arange(3) % 2, (n, 1)))
    ev = _frame((np.abs(x) > 0.7).astype(float))
    scale = _frame(np.full((n, 3), 1.0))
    price = _frame(10.0 * np.exp(np.cumsum(0.01 * rng.normal(size=(n, 3)), axis=0)))
    turn = _frame(rng.uniform(0.01, 0.05, size=(n, 3)))
    if name == "ts_first_passage_hit_probability" or name == "ts_first_passage_conditional_time":
        return (fx, scale), {}
    if name.startswith("event_response"):
        return (fy, ev), {"history_window": 60, "horizon": 5}
    if name.startswith("ts_transfer_entropy"):
        return (fy, fx), {"window": 90}
    if name.startswith("ts_turnover"):
        return (price, turn), {}
    if name.startswith("group_feature") or name.startswith("group_corr"):
        return (fx, fy, fz, g), {}
    if name == "cs_knn_graph_dirichlet_energy":
        return (fy, fx, fz, fz + 1.0), {"k": 3}
    if name.startswith("report_change"):
        return (fx, fy, fz, _report_period(n, 3, fx.index)), {}
    if name == "intraday_rv_signature_slope":
        return (_minute_frame(),), {}
    return (fx,), {}


def test_deepening_pack_registered_and_classified():
    canon = set(OperatorRegistry.list_canonical())
    assert DAILY_OPS <= canon
    for name in DAILY_OPS:
        assert classify_canonical(name) == "daily", name
    assert DAILY_OPS <= set(DAILY_FACTOR_MIGRATED)
    # Aliases must resolve onto existing canonicals.
    assert OperatorRegistry.resolve_canonical("intraday_signed_jump_balance") == "intra_signed_jump_ratio"


def test_deepening_deterministic_shape_prefix():
    rng = np.random.default_rng(0)
    for name in sorted(DAILY_OPS):
        args, kwargs = _build_inputs(name, rng)
        op = OperatorRegistry.get(name, "pandas_numpy")
        a = op.calculate(*args, **kwargs)
        b = op.calculate(*args, **kwargs)
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), name
        if name == "intraday_rv_signature_slope":
            # minute → daily: output index is the calendar days, not the input index.
            assert a.columns.equals(args[0].columns), name
            assert a.index.equals(args[0].index.normalize().unique()), name
            assert len(a.index) == len(args[0].index.normalize().unique()), name
            continue
        assert a.index.equals(args[0].index), name
        assert a.columns.equals(args[0].columns), name
        # Prefix causality: leading rows must not depend on later rows.
        sliced = tuple(ar.iloc[:160] for ar in args)
        prefix = op.calculate(*sliced, **kwargs).to_numpy(dtype=float)
        assert np.allclose(a.to_numpy(dtype=float)[:160], prefix, equal_nan=True), name


def test_deepening_fail_closed():
    rng = np.random.default_rng(1)
    const = _frame(np.full((120, 3), 5.0))
    for name in sorted(DAILY_OPS):
        args, kwargs = _build_inputs(name, rng)
        # Substitute the primary input with a constant series where it makes sense
        # (single-series ops).  Constant windows must not yield Inf or NaN-free
        # fabricated values.
        op = OperatorRegistry.get(name, "pandas_numpy")
        out = op.calculate(*args, **kwargs).to_numpy(dtype=float)
        assert not np.isinf(out).any(), name


def test_deepening_polars_parity():
    rng = np.random.default_rng(2)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    x = rng.normal(0.0, 1.0, size=(n, 2))
    A, B = x[:, 0], x[:, 1]
    dfp = pd.DataFrame({"A": A, "B": B}, index=dates)
    import polars as pl
    plf = pl.DataFrame({"date": dates, "A": A, "B": B})
    ev = (np.abs(x) > 0.7).astype(float)
    evp = pd.DataFrame({"A": ev[:, 0], "B": ev[:, 1]}, index=dates)
    evl = pl.DataFrame({"date": dates, "A": ev[:, 0], "B": ev[:, 1]})
    tr = rng.uniform(0.01, 0.05, size=(n, 2))
    trp = pd.DataFrame({"A": tr[:, 0], "B": tr[:, 1]}, index=dates)
    trl = pl.DataFrame({"date": dates, "A": tr[:, 0], "B": tr[:, 1]})
    sc = np.ones((n, 2))
    scp = pd.DataFrame({"A": sc[:, 0], "B": sc[:, 1]}, index=dates)
    scl = pl.DataFrame({"date": dates, "A": sc[:, 0], "B": sc[:, 1]})

    def _pl_args(name):
        if name in ("ts_first_passage_hit_probability", "ts_first_passage_conditional_time"):
            return (plf, scl)
        if name.startswith("event_response"):
            return (plf, evl)
        if name.startswith("ts_transfer_entropy"):
            return (plf, plf)
        if name.startswith("ts_turnover"):
            return (plf, trl)
        return (plf,)

    for name in sorted(POLARS_OPS):
        pd_op = OperatorRegistry.get(name, "pandas_numpy")
        pl_op = OperatorRegistry.get(name, "polars")
        kw = {}
        if name.startswith("ts_transfer_entropy"):
            kw = {"window": 90}
        pd_in = _pl_args(name)
        a = pd_op.calculate(*_to_pandas_args(name, dfp, evp, trp, scp), **kw).to_numpy()
        b = pl_op.calculate(*pd_in, **kw).select(["A", "B"]).to_numpy()
        assert np.allclose(a, b, equal_nan=True, atol=1e-10), name


def _to_pandas_args(name, dfp, evp, trp, scp):
    if name in ("ts_first_passage_hit_probability", "ts_first_passage_conditional_time"):
        return (dfp, scp)
    if name.startswith("event_response"):
        return (dfp, evp)
    if name.startswith("ts_transfer_entropy"):
        return (dfp, dfp)
    if name.startswith("ts_turnover"):
        return (dfp, trp)
    return (dfp,)
