from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.contracts import Accelerator, ExecutionKind
from factor_engine.backend.polars_backend_kind import (
    PolarsImplementationKind,
    canonical_polars_kind,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

NAMES = ("cross_event", "event_refractory")


def op(name, backend="pandas_numpy"):
    value = OperatorRegistry.get(name, backend)
    assert value is not None
    return value


def frame(values):
    return pd.DataFrame({"A": values, "B": values})


def calculate(name, args, backend="pandas_numpy", **kwargs):
    values = [pl.from_pandas(x) for x in args] if backend == "polars" else args
    result = op(name, backend).calculate(*values, **kwargs)
    return result.to_pandas() if isinstance(result, pl.DataFrame) else result


def test_complete_defaults_topology_units_and_physical_route():
    for name in NAMES:
        pandas_op, polars_op = op(name), op(name, "polars")
        assert pandas_op.metadata == polars_op.metadata
        meta = pandas_op.metadata
        if name == "cross_event":
            assert meta.panel_params == ("x", "y") and meta.panel_arity == 2
            assert meta.scalar_params == ("direction",)
            assert meta.param_specs["direction"].default == "up"
        else:
            assert meta.panel_params == ("condition",) and meta.panel_arity == 1
            assert meta.scalar_params == ("cooldown",)
            assert meta.param_specs["cooldown"].default == 5
            assert meta.param_specs["cooldown"].min == 0
        assert meta.output_unit == "state"
        spec = polars_op._physical_spec
        assert spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE
        assert spec.accelerator is Accelerator.NONE
        assert spec.stateful is (name == "event_refractory")
        assert spec.materializes_full_panel and spec.requires_sorted
        assert not spec.supports_lazy and not spec.supports_streaming
        assert len(spec.implementation_source_hash) == 64
        assert name in spec.kernel_identity
        assert canonical_polars_kind(name) is PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_cross_event_direction_missingness_and_prefix_causality(backend):
    x = frame([0., 1., 2., np.nan, 0., -1., 1.])
    y = frame([1., 1., 1., 1., 1., 0., 0.])
    up = calculate("cross_event", [x, y], backend, direction="up")
    down = calculate("cross_event", [x, y], backend, direction="down")
    np.testing.assert_allclose(up["A"], [np.nan, 0., 1., np.nan, np.nan, 0., 1.], equal_nan=True)
    np.testing.assert_allclose(down["A"], [np.nan, 0., 0., np.nan, np.nan, 0., 0.], equal_nan=True)
    x2 = pd.concat([x, frame([1e9])], ignore_index=True)
    y2 = pd.concat([y, frame([-1e9])], ignore_index=True)
    pd.testing.assert_frame_equal(up, calculate("cross_event", [x2, y2], backend).iloc[:-1].reset_index(drop=True))
    with pytest.raises((TypeError, ValueError)):
        calculate("cross_event", [x, y], backend, direction="sideways")


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_event_refractory_oracle_missing_reset_and_strict_cooldown(backend):
    condition = frame([1., 1., 0., 1., np.nan, 1., 1., 0., 1.])
    result = calculate("event_refractory", [condition], backend, cooldown=2)
    expected = [1., 0., 0., 1., np.nan, 1., 0., 0., 1.]
    np.testing.assert_allclose(result["A"], expected, equal_nan=True)
    zero = calculate("event_refractory", [condition], backend, cooldown=0)
    np.testing.assert_allclose(zero["A"], condition["A"], equal_nan=True)
    for bad in (-1, 2.5, True):
        with pytest.raises((TypeError, ValueError)):
            calculate("event_refractory", [condition], backend, cooldown=bad)
    invalid = frame([0., 0.5, 1.])
    with pytest.raises((TypeError, ValueError)):
        calculate("event_refractory", [invalid], backend)
