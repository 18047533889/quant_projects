"""Final contracts and independent numerical checks for Allan event operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES = ("event_allan_factor", "event_allan_scaling_slope", "event_allan_log_mean")


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


def _manual_af(values, scale):
    values = np.asarray(values, dtype=float)
    count = len(values) // scale
    counts = values[-count * scale:].reshape(count, scale).sum(axis=1)
    return np.mean(np.diff(counts) ** 2) / (2 * np.mean(counts))


def _manual_scaling(values, max_scale):
    pairs = []
    scale = 1
    while scale <= max_scale:
        if len(values) >= 3 * scale:
            af = _manual_af(values, scale)
            if np.isfinite(af) and af > 1e-12:
                pairs.append((scale, np.log2(af)))
        scale *= 2
    x = np.log2([p[0] for p in pairs])
    y = np.asarray([p[1] for p in pairs])
    return np.polyfit(x, y, 1)[0], np.mean(y)


def test_exact_contracts_and_backend_honesty():
    expected = {
        "event_allan_factor": {"window": 120, "scale": 8},
        "event_allan_scaling_slope": {"window": 120, "max_scale": 16},
        "event_allan_log_mean": {"window": 120, "max_scale": 16},
    }
    for name, defaults in expected.items():
        for backend, op in _ops(name):
            meta = op.metadata
            assert tuple(meta.panel_params) == ("event",) and meta.panel_arity == 1
            assert tuple(meta.scalar_params) == tuple(defaults)
            assert {key: meta.param_specs[key].default for key in defaults} == defaults
            assert meta.param_specs["window"].history_semantics == "max_rows"
            assert meta.param_specs["window"].param_role is ParamRole.HORIZON
            assert meta.output_unit == "dimensionless"
            assert "domain:event" in meta.tags and "unit:dimensionless" in meta.tags
            if backend == "polars":
                assert op._physical_spec.execution_kind.value == "polars_pandas_delegate"


def test_call_forms_prefix_and_backend_parity():
    values = np.tile([0, 0, 1, 0, 1, 1, 0, 1], 10)
    panel = _panel(values)
    cases = {
        "event_allan_factor": ((24, 4), {"window": 24, "scale": 4}),
        "event_allan_scaling_slope": ((32, 8), {"window": 32, "max_scale": 8}),
        "event_allan_log_mean": ((32, 8), {"window": 32, "max_scale": 8}),
    }
    for name, (args, kwargs) in cases.items():
        reference = None
        for backend, op in _ops(name):
            bp = _backend(panel, backend)
            positional = _values(op.calculate(bp, *args))
            keyword = _values(op.calculate(event=bp, **kwargs))
            mixed = _values(op.calculate(bp, **kwargs))
            prefix = _values(op.calculate(_backend(panel.iloc[:61], backend), *args))
            np.testing.assert_allclose(positional, keyword, equal_nan=True)
            np.testing.assert_allclose(positional, mixed, equal_nan=True)
            np.testing.assert_allclose(prefix, positional[:61], equal_nan=True)
            if reference is None:
                reference = positional
            else:
                np.testing.assert_allclose(positional, reference, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_independent_allan_and_scaling_oracles():
    values = np.asarray([0, 1, 1, 0, 0, 1, 0, 0] * 5, dtype=float)
    expected_af = _manual_af(values[-24:], 4)
    expected_slope, expected_mean = _manual_scaling(values[-32:], 8)
    for backend, op in _ops("event_allan_factor"):
        got = _values(op.calculate(_backend(_panel(values), backend), 24, 4))[-1, 0]
        np.testing.assert_allclose(got, expected_af, rtol=1e-12, atol=1e-12)
    for name, expected in (("event_allan_scaling_slope", expected_slope), ("event_allan_log_mean", expected_mean)):
        for backend, op in _ops(name):
            got = _values(op.calculate(_backend(_panel(values), backend), 32, 8))[-1, 0]
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name,kwargs", [
    ("event_allan_factor", {"window": 8, "scale": 3}),
    ("event_allan_factor", {"window": 8.5, "scale": 2}),
    ("event_allan_factor", {"window": 12, "scale": 0}),
    ("event_allan_scaling_slope", {"window": 11, "max_scale": 4}),
    ("event_allan_log_mean", {"window": 12, "max_scale": 3}),
    ("event_allan_log_mean", {"window": np.inf, "max_scale": 4}),
])
def test_invalid_scalars_and_required_panel(name, kwargs):
    panel = _panel(np.tile([0, 1], 10))
    for backend, op in _ops(name):
        with pytest.raises(Exception):
            op.calculate(_backend(panel, backend), **kwargs)
        with pytest.raises(Exception):
            op.calculate(**kwargs)


def test_nonbinary_and_nonfinite_input_fail_closed():
    base = np.tile([0, 1, 0, 0, 1, 1], 6).astype(float)
    for bad in (0.5, np.inf):
        values = base.copy(); values[-1] = bad
        for name in NAMES:
            args = (24, 4) if name == "event_allan_factor" else (24, 4)
            for backend, op in _ops(name):
                assert np.isnan(_values(op.calculate(_backend(_panel(values), backend), *args))[-1, 0])


def test_nan_gap_breaks_history_instead_of_bridging():
    values = np.tile([0, 1], 16).astype(float)
    values[-9] = np.nan
    for backend, op in _ops("event_allan_factor"):
        out = _values(op.calculate(_backend(_panel(values), backend), 24, 4))
        assert np.isnan(out[-1, 0])
