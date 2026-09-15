from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.session_recovery import _recovery_day
from factor_engine.runtime.session_panel import default_ashare_calendar

ensure_cleaned_loaded()

NAME = "session_event_recovery_score"


def _op(backend="pandas_numpy"):
    op = OperatorRegistry.get(NAME, backend=backend)
    assert op is not None
    return op


def _session_frames():
    day = pd.Timestamp("2026-01-05")
    idx = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min").append(
        pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min")
    )
    x = np.full(len(idx), 100.0)
    event = np.zeros(len(idx))
    for s, tau in ((10, 1), (40, 2), (150, 3)):
        x[s] = 110.0
        x[s + 1:s + tau] = 106.0
        x[s + tau] = 102.0
        event[s] = 1.0
    return pd.DataFrame({"A": x}, index=idx), pd.DataFrame({"A": event}, index=idx)


def _to_polars(frame):
    return pl.from_pandas(frame.rename_axis("__fe_time__").reset_index())


def test_fresh_load_topology_defaults_metadata_and_delegate_classification():
    pandas_op, polars_op = _op(), _op("polars")
    assert pandas_op.metadata == polars_op.metadata
    meta = pandas_op.metadata
    assert meta.panel_params == ("x", "event") and meta.panel_arity == 2
    assert meta.scalar_params == tuple(meta.param_names[2:])
    assert meta.output_unit == "dimensionless"
    assert set(meta.param_specs) == set(meta.param_names[2:])
    assert {name: meta.param_specs[name].default for name in meta.param_names[2:]} == {
        "horizon": 10, "residual_fraction": 0.25, "refractory": 1,
        "session_tz": None, "min_events": 3, "calendar": None,
    }
    spec = polars_op.physical_spec()
    assert spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE
    assert spec.materializes_full_panel and not spec.supports_lazy and not spec.supports_streaming


def test_independent_recovery_times_refractory_and_right_censoring():
    x = np.array([100., 110., 102., 100., 120., 115., 104., 100., 130., 120., 106., 100.])
    event = np.zeros_like(x)
    event[[1, 4, 8]] = 1
    # taus 1, 2, 2 under f=.25, H=3 => median 2 / 4.
    assert _recovery_day(x, event, 3, 0.25, refractory=0, min_events=3) == pytest.approx(0.5)
    late = event.copy()
    late[-1] = 1  # right-censored and excluded, not counted as a failure
    assert _recovery_day(x, late, 3, 0.25, refractory=0, min_events=3) == pytest.approx(0.5)
    clustered = event.copy()
    clustered[2] = 1
    assert _recovery_day(x, clustered, 3, 0.25, refractory=2, min_events=3) == pytest.approx(0.5)


def test_missing_horizon_and_nonbinary_events_fail_closed_and_params_are_strict():
    x = np.array([100., 110., 102., 100., 120., 115., 104., 100., 130., 120., 106., 100.])
    event = np.zeros_like(x)
    event[[1, 4, 8]] = 1
    missing = x.copy()
    missing[5] = np.nan
    assert np.isnan(_recovery_day(missing, event, 3, 0.25, min_events=3))
    invalid = event.copy()
    invalid[2] = 2
    assert np.isnan(_recovery_day(x, invalid, 3, 0.25, min_events=3))
    for args in ((2.5, 0.25, 1, 3), (3, 0.25, 1.5, 3), (3, 0.25, 1, 2.5)):
        with pytest.raises((TypeError, ValueError)):
            _recovery_day(x, event, *args)


def test_full_session_default_and_explicit_backend_parity():
    x, event = _session_frames()
    calendar = default_ashare_calendar(bar_freq="1min")
    expected = _op().calculate(x, event, calendar=calendar)
    actual = _op("polars").calculate(_to_polars(x), _to_polars(event), calendar=calendar).to_pandas()
    actual = actual.set_index("__fe_time__") if "__fe_time__" in actual.columns else actual
    actual.index.name = expected.index.name
    pd.testing.assert_frame_equal(expected, actual)
    assert expected.iloc[0, 0] == pytest.approx(2.0 / 11.0)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_binding_rejects_invalid_support_and_fractional_counts(backend):
    x, event = _session_frames()
    if backend == "polars":
        x, event = _to_polars(x), _to_polars(event)
    calendar = default_ashare_calendar(bar_freq="1min")
    for bad in ({"horizon": 2.5}, {"refractory": -1}, {"min_events": 2},
                {"residual_fraction": 0.0}, {"residual_fraction": 1.1}):
        with pytest.raises((TypeError, ValueError)):
            _op(backend).calculate(x, event, calendar=calendar, **bad)


def test_session_calendar_snapshot_schema_is_typed_object_and_identity_stable():
    from factor_engine.runtime.operator_snapshot import _parameter_contract

    op = _op()
    schema1, identity1, verified1 = _parameter_contract(op, op.metadata.panel_params)
    schema2, identity2, verified2 = _parameter_contract(op, op.metadata.panel_params)
    calendar = schema1["properties"]["calendar"]
    assert verified1 and verified2
    assert calendar["type"] == ["object", "null"]
    assert calendar["default"] is None
    assert calendar["x-factor-engine-role"] == "session_policy"
    assert "x-factor-engine-verification" not in calendar
    assert identity1 == identity2
