from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.contracts import ExecutionKind

ensure_cleaned_loaded()

NAMES = ("event_historical_response_mean", "event_historical_response_sign_balance")


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend=backend)
    assert op is not None
    return op


def _oracle(response, event, *, history_window, horizon, min_events, sign_balance):
    out = np.full(len(response), np.nan)
    for t in range(len(response)):
        lo = max(0, t - history_window)
        last = t - horizon
        if last < lo:
            continue
        values = []
        for s in range(lo, last + 1):
            if not np.isfinite(event[s]) or event[s] == 0:
                continue
            path = response[s + 1:s + horizon + 1]
            if len(path) != horizon or not np.all(np.isfinite(path)):
                continue
            values.append(float(np.mean(path)))
        if len(values) >= min_events:
            out[t] = np.mean(np.sign(values)) if sign_balance else np.mean(values)
    return out


def _call(name, response, event, backend="pandas_numpy", **kwargs):
    response = pd.DataFrame({"B": response, "A": np.asarray(response) * 2.0})
    event = pd.DataFrame({"B": event, "A": event})
    if backend == "polars":
        response, event = pl.from_pandas(response), pl.from_pandas(event)
    result = _op(name, backend).calculate(response, event, **kwargs)
    return result.to_pandas() if isinstance(result, pl.DataFrame) else result


def test_fresh_load_contract_topology_defaults_and_metadata_parity():
    expected = ["response", "event", "history_window", "horizon", "min_events", "require_full_horizon", "refractory"]
    for name in NAMES:
        pandas_op, polars_op = _op(name), _op(name, "polars")
        assert pandas_op.metadata == polars_op.metadata
        meta = pandas_op.metadata
        if name.endswith("mean"):
            expected_with_mode = expected[:4] + ["mode"] + expected[4:]
            assert meta.param_names == expected_with_mode
            assert meta.param_specs["mode"].default == "mean"
        else:
            assert meta.param_names == expected
        assert meta.panel_params == ("response", "event") and meta.panel_arity == 2
        assert meta.scalar_params == tuple(meta.param_names[2:])
        assert meta.param_specs["history_window"].default == 120
        assert meta.param_specs["history_window"].history_semantics == "max_rows"
        assert meta.param_specs["horizon"].default == 5
        assert meta.param_specs["min_events"].default == 5
        assert meta.param_specs["require_full_horizon"].default is True
        assert meta.param_specs["refractory"].default == 0
        spec = polars_op.physical_spec()
        assert spec.execution_kind is ExecutionKind.POLARS_NUMPY_KERNEL
        assert spec.materializes_full_panel
        assert not spec.supports_lazy and not spec.supports_streaming
        assert len(spec.implementation_source_hash) == 64
        assert "_horizon_response" in spec.kernel_identity


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_independent_matured_outcome_oracle_and_prefix_causality(backend):
    response = np.array([0., 1., 3., 0., -2., -4., 0., 2., 4., 0., -3., -1., 5.])
    event = np.array([1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0.])
    kwargs = dict(history_window=8, horizon=2, min_events=2)
    for name in NAMES:
        actual = _call(name, response, event, backend, **kwargs).iloc[:, 0]
        expected = _oracle(response, event, sign_balance=name.endswith("sign_balance"), **kwargs)
        np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=0, atol=1e-15)
        longer = _call(name, np.r_[response, 1e200], np.r_[event, 1.], backend, **kwargs)
        np.testing.assert_allclose(actual, longer.iloc[:-1, 0], equal_nan=True)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_unknown_event_is_not_an_event_and_incomplete_response_paths_are_excluded(backend):
    response = np.arange(14.0)
    event = np.zeros(14)
    event[[0, 3, 6, 9]] = 1
    kwargs = dict(history_window=8, horizon=2, min_events=2)
    for name in NAMES:
        missing_event = event.copy()
        missing_event[5] = np.nan
        got = _call(name, response, missing_event, backend, **kwargs).iloc[:, 0]
        expected = _oracle(response, missing_event, sign_balance=name.endswith("sign_balance"), **kwargs)
        np.testing.assert_allclose(got, expected, equal_nan=True)
        missing_response = response.copy()
        missing_response[7] = np.nan
        got = _call(name, missing_response, event, backend, **kwargs).iloc[:, 0]
        expected = _oracle(missing_response, event, sign_balance=name.endswith("sign_balance"), **kwargs)
        np.testing.assert_allclose(got, expected, equal_nan=True)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_strict_domains_relationships_and_default_backend_parity(backend):
    response = pd.DataFrame({"A": np.arange(140.0)})
    event = pd.DataFrame({"A": (np.arange(140) % 7 == 0).astype(float)})
    left, right = response, event
    if backend == "polars":
        left, right = pl.from_pandas(left), pl.from_pandas(right)
    for name in NAMES:
        op = _op(name, backend)
        for bad in (
            {"history_window": 10, "horizon": 11},
            {"history_window": 10.5}, {"horizon": 1.5},
            {"min_events": 0}, {"refractory": -1},
            {"require_full_horizon": "yes"},
        ):
            with pytest.raises((TypeError, ValueError)):
                op.calculate(left, right, **bad)
        impossible = op.calculate(left, right, history_window=10, horizon=3, min_events=9)
        impossible = impossible.to_pandas() if isinstance(impossible, pl.DataFrame) else impossible
        assert impossible.isna().all().all()
    if backend == "polars":
        for name in NAMES:
            expected = _op(name).calculate(response, event)
            actual = _op(name, "polars").calculate(left, right).to_pandas()
            pd.testing.assert_frame_equal(expected, actual)


def test_factor_engine_run_and_run_many_research_path():
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.columns import col, field
    from factor_engine.api.factor import Factor
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["timestamp", "instrument"])
    response = np.tile(np.sin(np.arange(40) / 3.0), 2).reshape(2, 40).T.reshape(-1)
    event = np.zeros((40, 2), dtype=float)
    event[::4, :] = 1.0
    source = InMemorySeriesSource(data={
        "close": pd.Series(100.0 + np.cumsum(response), index=index),
        "event": pd.Series(event.reshape(-1), index=index),
    })
    kwargs = dict(history_window=20, horizon=3, min_events=3)
    returns = make_cleaned_call_factory("ts_log_return")(field("close"), 1)
    mean_expr = make_cleaned_call_factory(NAMES[0])(returns, col("event"), **kwargs)
    sign_expr = make_cleaned_call_factory(NAMES[1])(returns, col("event"), **kwargs)
    engine = FactorEngine(
        backend=build_backend("polars_long"), data_source=source, run_mode="research"
    )
    single = engine.run(Factor(name="mean", expr=mean_expr))["result"]
    many = engine.run_many([
        Factor(name="mean", expr=mean_expr), Factor(name="sign", expr=sign_expr),
    ])
    pd.testing.assert_series_equal(single, many["results"]["mean"], check_names=False)
    assert np.isfinite(single.to_numpy()).any()
    assert np.isfinite(many["results"]["sign"].to_numpy()).any()
