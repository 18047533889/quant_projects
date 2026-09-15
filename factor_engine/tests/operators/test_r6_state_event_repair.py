"""Final-backend contract and numerical coverage for state_event operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES = ("event_decay_asof", "ts_event_spacing_cv", "ts_event_spacing_mean", "ts_time_since_change", "ts_transition_count")


def _panel(v):
    return pd.DataFrame({"A": v}, index=pd.date_range("2024-01-02", periods=len(v)), dtype=float)


def _backend(x, backend):
    if backend == "pandas_numpy": return x
    import polars as pl
    return pl.DataFrame({"A": x["A"].to_list()})


def _values(x):
    return x["A"].to_numpy() if not isinstance(x, pd.DataFrame) else x["A"].to_numpy(dtype=float)


def test_final_registry_topology_defaults_all_backends():
    load_all()
    expected = {
        "ts_transition_count": ("condition", {"window": 20, "missing_policy": "break"}),
        "ts_time_since_change": ("condition", {"max_lookback": None, "missing_policy": "break", "initial_semantics": "since_transition"}),
        "ts_event_spacing_mean": ("condition", {"window": 60, "min_events": 2}),
        "ts_event_spacing_cv": ("condition", {"window": 60, "min_events": 3}),
        "event_decay_asof": ("event", {"half_life": 20.0, "missing_policy": "carry", "event_kind": "marked"}),
    }
    for name, (panel, defaults) in expected.items():
        assert set(OperatorRegistry.backends_for(name)) == {"pandas_numpy", "polars"}
        for backend in OperatorRegistry.backends_for(name):
            meta = OperatorRegistry.get(name, backend, mode="any").metadata
            assert tuple(meta.panel_params) == (panel,)
            assert meta.panel_arity == 1
            assert tuple(meta.scalar_params) == tuple(defaults)
            assert {k: v.default for k, v in meta.param_specs.items()} == defaults


def test_numerical_oracles_call_forms_and_prefix_all_backends():
    load_all()
    condition = _panel([0, 1, 1, 0, np.nan, 0, 1, 0, 1, 0])
    event = _panel([np.nan, 2, 0, -1, np.nan, 0, 3, 0, 0, 1])
    cases = {
        "ts_transition_count": (condition, (5, "carry"), {"window": 5, "missing_policy": "carry"}, [0,1,1,2,np.nan,1,2,2,3,4]),
        "ts_time_since_change": (condition, (5, "carry", "state_age"), {"max_lookback": 5, "missing_policy": "carry", "initial_semantics": "state_age"}, [0,0,1,0,1,2,0,0,0,0]),
        "ts_event_spacing_mean": (condition, (10, 2), {"window": 10, "min_events": 2}, [np.nan,np.nan,1,1,1,1,np.nan,np.nan,np.nan,np.nan]),
        "ts_event_spacing_cv": (condition, (10, 3), {"window": 10, "min_events": 3}, [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]),
        "event_decay_asof": (event, (2.0, "carry", "marked"), {"half_life": 2.0, "missing_policy": "carry", "event_kind": "marked"}, None),
    }
    for name, (panel, positional, keyword, expected) in cases.items():
        for backend in OperatorRegistry.backends_for(name):
            op = OperatorRegistry.get(name, backend, mode="any"); bp = _backend(panel, backend)
            a = op.calculate(bp, *positional); b = op.calculate(**{op.metadata.panel_params[0]: bp}, **keyword)
            np.testing.assert_allclose(_values(a), _values(b), equal_nan=True)
            prefix = op.calculate(_backend(panel.iloc[:7], backend), *positional)
            np.testing.assert_allclose(_values(prefix), _values(a)[:7], equal_nan=True)
            if expected is not None:
                np.testing.assert_allclose(_values(a), expected, equal_nan=True)
    weight = 0.5 ** 0.5
    expected_decay = [np.nan,2,2*weight,2*weight**2-1,(2*weight**2-1)*weight,(2*weight**2-1)*weight**2,
                      (2*weight**2-1)*weight**3+3,((2*weight**2-1)*weight**3+3)*weight,
                      ((2*weight**2-1)*weight**3+3)*weight**2,((2*weight**2-1)*weight**3+3)*weight**3+1]
    for backend in OperatorRegistry.backends_for("event_decay_asof"):
        out = OperatorRegistry.get("event_decay_asof", backend, mode="any").calculate(_backend(event, backend), 2.0, "carry", "marked")
        np.testing.assert_allclose(_values(out), expected_decay, equal_nan=True)


@pytest.mark.parametrize("name,param,bad", [
    ("ts_transition_count", "window", 0), ("ts_transition_count", "window", 2.5),
    ("ts_event_spacing_mean", "min_events", 1), ("ts_event_spacing_cv", "min_events", np.nan),
    ("ts_time_since_change", "max_lookback", 1.5), ("event_decay_asof", "half_life", np.inf),
    ("event_decay_asof", "event_kind", "unknown"), ("ts_transition_count", "missing_policy", "unknown"),
])
def test_invalid_scalars_rejected_every_backend(name, param, bad):
    load_all(); panel = _panel([0,1,0,1,0])
    for backend in OperatorRegistry.backends_for(name):
        with pytest.raises(Exception):
            OperatorRegistry.get(name, backend, mode="any").calculate(_backend(panel, backend), **{param: bad})


def test_required_panel_and_bool_domain_are_not_bypassed():
    load_all()
    for name in NAMES:
        for backend in OperatorRegistry.backends_for(name):
            with pytest.raises(Exception):
                OperatorRegistry.get(name, backend, mode="any").calculate()
    bad = _panel([0, 2, 1])
    for name in ("ts_transition_count", "ts_time_since_change", "ts_event_spacing_mean", "ts_event_spacing_cv"):
        for backend in OperatorRegistry.backends_for(name):
            with pytest.raises(Exception): OperatorRegistry.get(name, backend, mode="any").calculate(_backend(bad, backend))
    for backend in OperatorRegistry.backends_for("event_decay_asof"):
        with pytest.raises(Exception): OperatorRegistry.get("event_decay_asof", backend, mode="any").calculate(_backend(bad, backend), event_kind="bool")
