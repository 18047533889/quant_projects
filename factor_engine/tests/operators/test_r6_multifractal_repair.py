"""Exact contracts and independent model oracles for multifractal operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES = ("ts_generalized_hurst_spread_q1_q4", "ts_multifractal_spectrum_width", "ts_multifractal_curvature")
QS = np.asarray([.5, 1., 2., 3., 4.])
LAGS = np.asarray([1, 2, 4, 8])


def _panel(values): return pd.DataFrame({"A": np.asarray(values, dtype=float)})


def _backend(panel, backend):
    if backend == "pandas_numpy": return panel
    import polars as pl
    return pl.DataFrame({"A": panel["A"].to_list()})


def _values(frame): return frame.to_numpy(dtype=float) if isinstance(frame, pd.DataFrame) else frame.to_numpy()


def _ops(name):
    load_all(); backends = OperatorRegistry.backends_for(name)
    assert set(backends) == {"pandas_numpy", "polars"}
    return [(b, OperatorRegistry.get(name, b, mode="any")) for b in backends]


def _hurst(values, q):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    values = values / np.max(np.abs(values[finite]))
    if not np.isfinite(values[-1]): return np.nan
    start = len(values) - 1
    while start > 0 and np.isfinite(values[start - 1]): start -= 1
    run = values[start:]
    lx, ly = [], []
    for lag in LAGS:
        d = np.abs(run[lag:] - run[:-lag]); d = d[np.isfinite(d)]
        if len(d) < max(8, int(np.ceil(8 * abs(q)))): continue
        moment = np.mean(d ** q)
        if moment > 0 and np.isfinite(moment): lx.append(np.log(lag)); ly.append(np.log(moment))
    slope, intercept = np.polyfit(lx, ly, 1)
    fit = slope * np.asarray(lx) + intercept
    r2 = 1 - np.sum((np.asarray(ly) - fit) ** 2) / np.sum((np.asarray(ly) - np.mean(ly)) ** 2)
    return slope / q if r2 >= .9 else np.nan


def _oracle(values):
    hs = np.asarray([_hurst(values, q) for q in QS])
    spread = hs[1] - hs[4]
    tau_coef = np.polyfit(QS, QS * hs - 1, 2)
    alpha = np.polyval(np.polyder(tau_coef), QS)
    width = np.max(alpha) - np.min(alpha)
    curvature = np.polyfit(QS, hs, 2)[0]
    return spread, width, curvature


def test_exact_contracts_defaults_topology_units_and_final_backend():
    for name in NAMES:
        for backend, op in _ops(name):
            meta = op.metadata
            assert tuple(meta.panel_params) == ("x",) and meta.panel_arity == 1
            assert tuple(meta.scalar_params) == ("window",)
            spec = meta.param_specs["window"]
            assert spec.default == 120 and spec.min == 40
            assert spec.history_semantics == "max_rows" and spec.param_role is ParamRole.HORIZON
            assert meta.input_units == {"x": "level"} and meta.output_unit == "dimensionless"
            if backend == "polars": assert op._physical_spec.execution_kind.value == "polars_pandas_delegate"


def test_independent_model_oracles_call_forms_prefix_and_parity():
    values = np.cumsum(np.random.default_rng(0).normal(size=180))
    expected = dict(zip(NAMES, _oracle(values[-120:])))
    panel = _panel(values)
    for name in NAMES:
        reference = None
        for backend, op in _ops(name):
            bp = _backend(panel, backend)
            a = _values(op.calculate(bp, 120)); b = _values(op.calculate(x=bp, window=120)); c = _values(op.calculate(bp, window=120))
            prefix = _values(op.calculate(_backend(panel.iloc[:151], backend), 120))
            np.testing.assert_allclose(a, b, equal_nan=True); np.testing.assert_allclose(a, c, equal_nan=True)
            np.testing.assert_allclose(prefix, a[:151], equal_nan=True)
            np.testing.assert_allclose(a[-1, 0], expected[name], rtol=1e-10, atol=1e-10)
            if reference is None: reference = a
            else: np.testing.assert_allclose(a, reference, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_scale_invariance_at_tiny_and_huge_amplitudes():
    values = np.cumsum(np.random.default_rng(0).normal(size=160))
    for name in NAMES:
        for backend, op in _ops(name):
            got = []
            for factor in (1., 1e-250, 1e250):
                got.append(_values(op.calculate(_backend(_panel(values * factor), backend), 120))[-1, 0])
            assert np.isfinite(got).all()
            np.testing.assert_allclose(got, got[0], rtol=1e-9, atol=1e-9)


def test_gap_uses_only_trailing_contiguous_cohort_without_reconnecting():
    values = np.cumsum(np.random.default_rng(0).normal(size=180)); values[-80] = np.nan
    suffix = values[-79:]
    expected = dict(zip(NAMES, _oracle(suffix)))
    compressed = values[-120:][np.isfinite(values[-120:])]
    assert any(not np.isclose(a, b) for a, b in zip(expected.values(), _oracle(compressed)))
    for name in NAMES:
        for backend, op in _ops(name):
            got = _values(op.calculate(_backend(_panel(values), backend), 120))[-1, 0]
            np.testing.assert_allclose(got, expected[name], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("bad", [39, 40.5, np.nan, np.inf])
def test_invalid_window_and_required_panel_rejected(bad):
    panel = _panel(np.arange(60.0))
    for name in NAMES:
        for backend, op in _ops(name):
            with pytest.raises(Exception): op.calculate(_backend(panel, backend), window=bad)
            with pytest.raises(Exception): op.calculate(window=bad)


def test_current_nan_and_short_post_gap_suffix_fail_closed():
    base = np.cumsum(np.random.default_rng(4).normal(size=120))
    for pos in (-1, -20):
        values = base.copy(); values[pos] = np.nan
        for name in NAMES:
            for backend, op in _ops(name):
                got = _values(op.calculate(_backend(_panel(values), backend), 120))[-1, 0]
                assert np.isnan(got)
