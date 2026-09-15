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

NAMES = ("ts_vector_state_mahalanobis", "ts_vector_state_local_density")


def op(name, backend="pandas_numpy"):
    value = OperatorRegistry.get(name, backend)
    assert value is not None
    return value


def features(values):
    x = pd.DataFrame({"A": np.asarray(values, dtype=float)})
    constants = [pd.DataFrame({"A": np.full(len(x), float(i))}) for i in (2, 3, 4)]
    return [x, *constants]


def calculate(name, frames, backend="pandas_numpy", **kwargs):
    args = [pl.from_pandas(x) for x in frames] if backend == "polars" else frames
    result = op(name, backend).calculate(*args, **kwargs)
    return result.to_pandas() if isinstance(result, pl.DataFrame) else result


def test_complete_defaults_topology_units_and_physical_route():
    for name in NAMES:
        pandas_op, polars_op = op(name), op(name, "polars")
        assert pandas_op.metadata == polars_op.metadata
        meta = pandas_op.metadata
        assert meta.panel_params == ("f1", "f2", "f3", "f4")
        assert meta.panel_arity == 4
        scalar = "shrinkage" if name.endswith("mahalanobis") else "k"
        assert meta.scalar_params == ("window", scalar)
        assert meta.output_unit == "dimensionless"
        assert meta.param_specs["window"].default == 60
        assert meta.param_specs["window"].history_semantics == "max_rows"
        assert meta.param_specs[scalar].default == (0.5 if scalar == "shrinkage" else 5)
        spec = polars_op._physical_spec
        assert spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE
        assert spec.accelerator is Accelerator.NONE
        assert spec.materializes_full_panel
        assert not spec.supports_lazy and not spec.supports_streaming
        assert spec.supports_nulls and spec.supports_nan and spec.supports_inf
        assert len(spec.implementation_source_hash) == 64
        assert canonical_polars_kind(name) is PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_one_dimensional_numerical_oracles_use_exactly_window_prior_rows(backend):
    values = np.array([0., 1., 2., 3., 4., 7., 8.])
    frames = features(values)
    mahal = calculate(NAMES[0], frames, backend, window=5, shrinkage=0.5)["A"]
    expected_mahal = abs(values[5] - values[:5].mean()) / values[:5].std(ddof=1)
    assert mahal.iloc[5] == pytest.approx(expected_mahal)

    density = calculate(NAMES[1], frames, backend, window=5, k=2)["A"]
    hist = values[:5]
    sd = hist.std(ddof=0)
    distances = np.abs((hist - hist.mean()) / sd - (values[5] - hist.mean()) / sd)
    expected_density = -np.log(max(np.partition(distances, 1)[1], 1e-3))
    assert density.iloc[5] == pytest.approx(expected_density)
    assert mahal.iloc[:5].isna().all()
    assert density.iloc[:5].isna().all()


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_incomplete_history_rows_drop_as_rows_and_incomplete_query_fails_closed(backend):
    frames = features(np.arange(12.0))
    frames[2].iloc[3, 0] = np.nan
    for name in NAMES:
        kwargs = {"window": 6, "shrinkage": 0.5} if name.endswith("mahalanobis") else {"window": 6, "k": 1}
        result = calculate(name, frames, backend, **kwargs)
        # One incomplete historical vector is removed atomically. With five
        # remaining points in a one-dimensional effective space, the fit works.
        assert np.isfinite(result.iloc[6, 0])
        broken = [x.copy() for x in frames]
        broken[1].iloc[8, 0] = np.inf
        assert np.isnan(calculate(name, broken, backend, **kwargs).iloc[8, 0])


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_prefix_causality_panel_identity_and_strict_parameter_domains(backend):
    frames = features(np.arange(15.0) + np.sin(np.arange(15.0)))
    for name in NAMES:
        kwargs = {"window": 6, "shrinkage": 0.25} if name.endswith("mahalanobis") else {"window": 6, "k": 2}
        base = calculate(name, frames, backend, **kwargs)
        longer = [pd.concat([x, pd.DataFrame({"A": [1e100]})], ignore_index=True) for x in frames]
        extended = calculate(name, longer, backend, **kwargs)
        pd.testing.assert_frame_equal(base, extended.iloc[:-1].reset_index(drop=True))
        with pytest.raises((TypeError, ValueError)):
            calculate(name, frames, backend, **{**kwargs, "window": 6.5})

    with pytest.raises((TypeError, ValueError)):
        calculate(NAMES[0], frames, backend, window=6, shrinkage=True)
    with pytest.raises((TypeError, ValueError)):
        calculate(NAMES[0], frames, backend, window=6, shrinkage=1.1)
    with pytest.raises((TypeError, ValueError)):
        calculate(NAMES[1], frames, backend, window=6, k=7)

    mismatched = [x.copy() for x in frames]
    mismatched[1] = mismatched[1].iloc[:-1]
    with pytest.raises((TypeError, ValueError)):
        calculate(NAMES[1], mismatched, backend, window=6, k=2)
