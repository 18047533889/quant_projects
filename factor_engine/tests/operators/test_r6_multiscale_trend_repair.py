"""Exact contracts and numerical oracles for multiscale trend operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES = ("ts_multiscale_trend_consensus", "ts_multiscale_trend_dispersion", "ts_multiscale_trend_curvature")


def _panel(values):
    return pd.DataFrame({"A": np.asarray(values, dtype=float)})


def _backend(panel, backend):
    if backend == "pandas_numpy":
        return panel
    import polars as pl
    return pl.DataFrame({"A": panel["A"].to_list()})


def _values(frame):
    return frame.to_numpy(dtype=float) if isinstance(frame, pd.DataFrame) else frame.to_numpy()


def _ops(name):
    load_all()
    backends = OperatorRegistry.backends_for(name)
    assert set(backends) == {"pandas_numpy", "polars"}
    return [(backend, OperatorRegistry.get(name, backend, mode="any")) for backend in backends]


def _trend(values, scale):
    y = np.asarray(values[-scale:], dtype=float)
    y = y / np.max(np.abs(y))
    x = np.arange(scale, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    slope = xc @ yc / (xc @ xc)
    residual = y - (y.mean() + slope * xc)
    return slope / np.sqrt(np.mean(residual ** 2))


def _oracle(values, scales):
    trends = np.asarray([_trend(values, scale) for scale in scales])
    consensus = np.mean(np.sign(trends))
    dispersion = np.median(np.abs(trends - np.median(trends)))
    logs = np.log(np.asarray(scales, dtype=float))
    curvature = np.linalg.lstsq(np.column_stack([np.ones(len(logs)), logs, logs ** 2]), trends, rcond=None)[0][2]
    return consensus, dispersion, curvature


def test_exact_contracts_and_honest_final_backends():
    minimum = {NAMES[0]: 1, NAMES[1]: 2, NAMES[2]: 3}
    for name in NAMES:
        for backend, op in _ops(name):
            meta = op.metadata
            assert tuple(meta.panel_params) == ("x",) and meta.panel_arity == 1
            assert tuple(meta.scalar_params) == ("window", "scales")
            assert meta.param_specs["window"].default == 60
            assert meta.param_specs["window"].param_role is ParamRole.POLICY
            spec = meta.param_specs["scales"]
            assert spec.default == (5, 10, 20, 40)
            assert spec.param_role is ParamRole.ESTIMATOR_RESOLUTION
            assert all(alt.min_items == minimum[name] for alt in spec.alternatives)
            assert meta.input_units == {"x": "price_level_or_log_price_level"}
            assert meta.output_unit == "dimensionless"
            if backend == "polars":
                assert op._physical_spec.execution_kind.value == "polars_pandas_delegate"


def test_independent_oracles_all_call_forms_prefix_and_parity():
    x = np.arange(96, dtype=float)
    values = 3.0 + 0.04 * x + np.sin(x / 2.7) + 0.25 * np.cos(x / 7.0)
    panel = _panel(values); scales = (5, 9, 17, 31)
    expected = dict(zip(NAMES, _oracle(values, scales)))
    for name in NAMES:
        reference = None
        for backend, op in _ops(name):
            bp = _backend(panel, backend)
            positional = _values(op.calculate(bp, 40, scales))
            keyword = _values(op.calculate(x=bp, window=40, scales=scales))
            mixed = _values(op.calculate(bp, window=40, scales=scales))
            prefix = _values(op.calculate(_backend(panel.iloc[:73], backend), 40, scales))
            np.testing.assert_allclose(positional, keyword, equal_nan=True)
            np.testing.assert_allclose(positional, mixed, equal_nan=True)
            np.testing.assert_allclose(prefix, positional[:73], equal_nan=True)
            np.testing.assert_allclose(positional[-1, 0], expected[name], rtol=1e-11, atol=1e-11)
            if reference is None:
                reference = positional
            else:
                np.testing.assert_allclose(positional, reference, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("name,scales", [
    (NAMES[0], ()), (NAMES[0], (1, 3)), (NAMES[0], (3, 3)),
    (NAMES[1], (5,)), (NAMES[2], (5, 9)), (NAMES[2], (5, 9.5, 17)),
    (NAMES[2], (5, np.inf, 17)),
])
def test_invalid_scale_collections_rejected(name, scales):
    panel = _panel(np.arange(50.0))
    for backend, op in _ops(name):
        with pytest.raises(Exception):
            op.calculate(_backend(panel, backend), window=40, scales=scales)


def test_window_relation_required_panel_and_fraction_rejected():
    panel = _panel(np.arange(50.0) + np.sin(np.arange(50.0)))
    for name in NAMES:
        for backend, op in _ops(name):
            bp = _backend(panel, backend)
            with pytest.raises(Exception): op.calculate(bp, window=20.5, scales=(5, 10, 20))
            with pytest.raises(Exception): op.calculate(bp, window=10, scales=(5, 11, 12))
            with pytest.raises(Exception): op.calculate(window=40, scales=(5, 10, 20))


def test_nan_gap_invalidates_full_declared_scale_set():
    values = 2 + np.sin(np.arange(50) / 3.0)
    values[-17] = np.nan
    for name in NAMES:
        for backend, op in _ops(name):
            out = _values(op.calculate(_backend(_panel(values), backend), 40, (5, 9, 20)))
            assert np.isnan(out[-1, 0])


def test_scale_invariance_at_tiny_and_huge_magnitudes():
    x = np.arange(70, dtype=float)
    values = 2.0 + 0.03 * x + np.sin(x / 3.1)
    for name in NAMES:
        for backend, op in _ops(name):
            outputs = []
            for factor in (1.0, 1e-250, 1e250):
                panel = _backend(_panel(values * factor), backend)
                outputs.append(_values(op.calculate(panel, 40, (5, 10, 20, 40)))[-1, 0])
            assert np.isfinite(outputs).all()
            np.testing.assert_allclose(outputs, outputs[0], rtol=1e-10, atol=1e-10)


def test_exact_linear_and_constant_windows_fail_closed():
    for values in (np.ones(50), np.arange(50.0)):
        for name in NAMES:
            for backend, op in _ops(name):
                result = _values(op.calculate(_backend(_panel(values), backend), 40, (5, 10, 20, 40)))
                assert np.isnan(result[-1, 0])
