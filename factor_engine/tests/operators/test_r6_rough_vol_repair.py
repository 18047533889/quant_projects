"""Exact contracts and stable independent oracles for rough-volatility operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES = ("ts_vol_pvariation_roughness", "ts_vol_scaling_break")


def _panel(values):
    return pd.DataFrame({"A": np.asarray(values, dtype=float)})


def _backend(panel, backend):
    if backend == "pandas_numpy": return panel
    import polars as pl
    return pl.DataFrame({"A": panel["A"].to_list()})


def _values(frame):
    return frame.to_numpy(dtype=float) if isinstance(frame, pd.DataFrame) else frame.to_numpy()


def _ops(name):
    load_all(); backends = OperatorRegistry.backends_for(name)
    assert set(backends) == {"pandas_numpy", "polars"}
    return [(b, OperatorRegistry.get(name, b, mode="any")) for b in backends]


def _scaled_log_moment(values, lag, p):
    inc = np.asarray(values[lag:]) - np.asarray(values[:-lag])
    inc = np.abs(inc[np.isfinite(inc)])
    logs = np.log(inc[inc > 0])
    anchor = np.max(logs)
    return anchor + (np.log(np.exp(p * (logs - anchor)).sum()) - np.log(len(inc))) / p


def _rough_oracle(values, p, scales, min_pairs=5, min_fraction=.5):
    used = []
    for lag in scales:
        possible = len(values) - lag
        pairs = np.isfinite(np.asarray(values[lag:]) - np.asarray(values[:-lag])).sum()
        if pairs >= min_pairs and pairs / possible >= min_fraction:
            try: used.append((np.log(lag), _scaled_log_moment(values, lag, p)))
            except ValueError: pass
    x, y = np.asarray(used).T
    return np.polyfit(x, y, 1)[0]


def _break_oracle(values, p):
    m = {lag: _scaled_log_moment(values, lag, p) for lag in (1, 2, 8, 16)}
    return ((m[2] - m[1]) - (m[16] - m[8])) / np.log(2.0)


def test_exact_contracts_defaults_topology_units_and_backend_honesty():
    expected = {
        NAMES[0]: {"window": 120, "p": 2.0, "scales": (1, 2, 4), "min_pairs": 5, "min_pair_fraction": .5},
        NAMES[1]: {"window": 120, "p": 2.0, "min_pairs": 5, "min_pair_fraction": .5},
    }
    for name, defaults in expected.items():
        for backend, op in _ops(name):
            meta = op.metadata
            assert tuple(meta.panel_params) == ("x",) and meta.panel_arity == 1
            assert tuple(meta.scalar_params) == tuple(defaults)
            assert {k: meta.param_specs[k].default for k in defaults} == defaults
            assert meta.param_specs["window"].history_semantics == "max_rows"
            assert meta.param_specs["window"].param_role is ParamRole.HORIZON
            assert meta.param_specs["min_pairs"].param_role is ParamRole.SUPPORT_POLICY
            assert meta.param_specs["min_pair_fraction"].param_role is ParamRole.SUPPORT_POLICY
            assert meta.input_units == {"x": "level"} and meta.output_unit == "dimensionless"
            if backend == "polars":
                assert op._physical_spec.execution_kind.value == "polars_pandas_delegate"


def test_independent_oracles_call_forms_prefix_and_backend_parity():
    t = np.arange(90.0); values = np.sin(t / 3.7) + .03 * t + .2 * np.cos(t / 11)
    panel = _panel(values)
    cases = {
        NAMES[0]: ((50, 2.5, (1, 2, 4, 8), 5, .5), {"window": 50, "p": 2.5, "scales": (1, 2, 4, 8), "min_pairs": 5, "min_pair_fraction": .5}, _rough_oracle(values[-50:], 2.5, (1, 2, 4, 8))),
        NAMES[1]: ((50, 2.5, 5, .5), {"window": 50, "p": 2.5, "min_pairs": 5, "min_pair_fraction": .5}, _break_oracle(values[-50:], 2.5)),
    }
    for name, (args, kwargs, expected) in cases.items():
        reference = None
        for backend, op in _ops(name):
            bp = _backend(panel, backend)
            a = _values(op.calculate(bp, *args)); b = _values(op.calculate(x=bp, **kwargs)); c = _values(op.calculate(bp, **kwargs))
            prefix = _values(op.calculate(_backend(panel.iloc[:71], backend), *args))
            np.testing.assert_allclose(a, b, equal_nan=True); np.testing.assert_allclose(a, c, equal_nan=True)
            np.testing.assert_allclose(prefix, a[:71], equal_nan=True)
            np.testing.assert_allclose(a[-1, 0], expected, rtol=1e-11, atol=1e-11)
            if reference is None: reference = a
            else: np.testing.assert_allclose(a, reference, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_high_p_log_domain_avoids_power_overflow_and_underflow():
    t = np.arange(80.0); values = np.sin(t / 4.1) + .017 * t
    for p in (1_000.0, 100_000.0):
        expected_r = _rough_oracle(values[-60:], p, (1, 2, 4, 8))
        expected_b = _break_oracle(values[-60:], p)
        for name, expected, args in ((NAMES[0], expected_r, (60, p, (1, 2, 4, 8), 5, .5)), (NAMES[1], expected_b, (60, p, 5, .5))):
            for backend, op in _ops(name):
                got = _values(op.calculate(_backend(_panel(values), backend), *args))[-1, 0]
                assert np.isfinite(got)
                np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)


def test_physical_gaps_are_not_compressed_and_partial_valid_scales_are_retained():
    t = np.arange(50.0); values = np.sin(t / 3.2) + .02 * t
    values[[35, 41, 46]] = np.nan
    expected = _rough_oracle(values[-40:], 2.0, (1, 2, 4, 12), min_pairs=12, min_fraction=.45)
    compressed = _rough_oracle(values[-40:][np.isfinite(values[-40:])], 2.0, (1, 2, 4, 12), min_pairs=12, min_fraction=.45)
    assert np.isfinite(expected) and not np.isclose(expected, compressed)
    for backend, op in _ops(NAMES[0]):
        got = _values(op.calculate(_backend(_panel(values), backend), 40, 2.0, (1, 2, 4, 12), 12, .45))[-1, 0]
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("name,kwargs", [
    (NAMES[0], {"window": 10.5}), (NAMES[0], {"p": 0.0}),
    (NAMES[0], {"scales": (1, 2.5)}), (NAMES[0], {"scales": (2, 2)}),
    (NAMES[0], {"min_pairs": 1.5}), (NAMES[1], {"p": np.inf}),
    (NAMES[1], {"min_pair_fraction": 0.0}), (NAMES[1], {"min_pair_fraction": 1.1}),
])
def test_strict_invalid_parameters_and_required_panel(name, kwargs):
    panel = _panel(np.arange(30.0))
    for backend, op in _ops(name):
        with pytest.raises(Exception): op.calculate(_backend(panel, backend), **kwargs)
        with pytest.raises(Exception): op.calculate(**kwargs)
