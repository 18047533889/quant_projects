from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _registry():
    load_all()


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend=backend)
    assert op is not None
    return op


def _polars_panel(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({"date": list(frame.index.to_pydatetime()), **{
        col: frame[col].to_numpy() for col in frame.columns
    }})


def test_contracts_defaults_and_backend_parity():
    impulse = _op("intra_impulse_event_detector")
    assert impulse.metadata.panel_params == ("price", "volume")
    assert impulse.metadata.panel_arity == 2
    assert impulse.metadata.scalar_params == (
        "event", "threshold", "z", "min_bars", "merge_gap", "output",
    )
    assert impulse.metadata.available_at == "session_close"
    assert impulse.metadata.same_session_usable is False
    defaults = {k: v.default for k, v in impulse.metadata.param_specs.items()}
    assert defaults == {"event": "up", "threshold": "robust_z", "z": 3.0,
                        "min_bars": 1, "merge_gap": 2, "output": "count"}

    for backend in ("pandas_numpy", "polars"):
        cycle = _op("ts_dominant_cycle_period", backend)
        md = cycle.metadata
        assert md.panel_params == ("x",) and md.panel_arity == 1
        assert md.scalar_params == ("window", "min_peak_share")
        assert md.param_specs["window"].default == 60
        assert md.param_specs["window"].searchable is False
        assert md.param_specs["window"].param_role is ParamRole.ESTIMATOR_RESOLUTION
        assert md.param_specs["min_peak_share"].default == pytest.approx(.10)
        assert md.param_specs["min_peak_share"].param_role is ParamRole.STATE_THRESHOLD
        assert md.input_units == {"x": "return_or_continuous_price"}
        assert md.output_unit == "bars"


def test_impulse_quantile_oracle_outputs_volume_noop_and_no_event_semantics():
    day = pd.Timestamp("2026-01-05")
    times = pd.DatetimeIndex([
        day + pd.Timedelta(minutes=m) for m in (570, 571, 572, 573, 780, 781, 782, 783)
    ])
    returns = np.array([0., .01, .02, .03, 0., .04, .05, .06])
    price = pd.DataFrame({"A": 100 * np.exp(np.cumsum(returns))}, index=times)
    volume = pd.DataFrame({"A": np.arange(8.) + 10}, index=times)
    op = _op("intra_impulse_event_detector")
    expected = {"count": 1., "strength": .15, "max_strength": .15,
                "first_time": 5 / 8, "last_time": 7 / 8, "duration": 3.}
    for output, value in expected.items():
        got = op.calculate(price, volume, threshold="quantile", z=.5,
                           merge_gap=2, output=output).iloc[0, 0]
        assert got == pytest.approx(value, abs=1e-12)
        without_volume = op.calculate(price, threshold="quantile", z=.5,
                                      merge_gap=2, output=output).iloc[0, 0]
        assert without_volume == pytest.approx(got, abs=1e-12)

    flat = pd.DataFrame({"A": np.ones(8) * 100}, index=times)
    assert op.calculate(flat, volume, output="count").iloc[0, 0] == 0.0
    assert np.isnan(op.calculate(flat, volume, output="strength").iloc[0, 0])


def test_impulse_lunch_gap_never_merges_and_strict_domains():
    day = pd.Timestamp("2026-01-05")
    times = pd.DatetimeIndex([day + pd.Timedelta(minutes=m) for m in (570, 571, 572, 780, 781)])
    # Events at indices 2 and 4 are only two rows apart, but live in separate
    # contiguous trading segments and therefore form two episodes.
    price = pd.DataFrame({"A": 100 * np.exp(np.cumsum([0., 0., .2, 0., .2]))}, index=times)
    volume = pd.DataFrame({"A": np.ones(5)}, index=times)
    op = _op("intra_impulse_event_detector")
    got = op.calculate(price, volume, threshold="quantile", z=.2,
                       merge_gap=10, output="count")
    assert got.iloc[0, 0] == 2.0
    for bad in ({"z": True}, {"min_bars": True}, {"min_bars": 1.5},
                {"merge_gap": -1}, {"threshold": "quantile", "z": 1.01}):
        with pytest.raises((TypeError, ValueError)):
            op.calculate(price, volume, **bad)


def test_impulse_polars_daily_axis_and_parity():
    days = pd.date_range("2026-01-05", periods=2, freq="D")
    times = pd.DatetimeIndex([d + pd.Timedelta(minutes=m) for d in days for m in (570, 571, 572, 573)])
    r = np.tile([0., .01, .08, .01], 2)
    price = pd.DataFrame({"A": 100 * np.exp(np.cumsum(r))}, index=times)
    volume = pd.DataFrame({"A": np.arange(8.) + 1}, index=times)
    kwargs = dict(threshold="quantile", z=.5, output="count")
    expected = _op("intra_impulse_event_detector").calculate(price, volume, **kwargs)
    actual = _op("intra_impulse_event_detector", "polars").calculate(
        _polars_panel(price), _polars_panel(volume), **kwargs,
    )
    assert actual.columns == ["date", "A"]
    pd.testing.assert_index_equal(pd.DatetimeIndex(actual["date"].to_list()), expected.index,
                                  exact=False)
    np.testing.assert_allclose(actual["A"].to_numpy(), expected["A"].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_dominant_cycle_independent_sinusoid_oracle_gate_defaults_and_prefix(backend):
    n, window, period = 112, 64, 8
    values = np.sin(2 * np.pi * np.arange(n) / period)
    pdf = pd.DataFrame({"A": values, "B": 2 * values})
    panel = pl.from_pandas(pdf) if backend == "polars" else pdf
    op = _op("ts_dominant_cycle_period", backend)
    result = op.calculate(panel, window=window, min_peak_share=.1)
    actual = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    assert actual.iloc[:window - 1].isna().all().all()
    np.testing.assert_allclose(actual.iloc[window - 1:].to_numpy(), period, atol=0, rtol=0)
    prefix_panel = panel[:90] if backend == "polars" else panel.iloc[:90]
    prefix = op.calculate(prefix_panel, window=window, min_peak_share=.1)
    prefix = prefix.to_pandas() if isinstance(prefix, pl.DataFrame) else prefix
    pd.testing.assert_frame_equal(actual.iloc[:90].reset_index(drop=True),
                                  prefix.reset_index(drop=True))
    rejected = op.calculate(panel, window=window, min_peak_share=1.0)
    rejected = rejected.to_pandas() if isinstance(rejected, pl.DataFrame) else rejected
    assert rejected.isna().all().all()

    default = op.calculate(panel)
    explicit = op.calculate(panel, window=60, min_peak_share=.10)
    default = default.to_pandas() if isinstance(default, pl.DataFrame) else default
    explicit = explicit.to_pandas() if isinstance(explicit, pl.DataFrame) else explicit
    pd.testing.assert_frame_equal(default, explicit)
    for bad in ({"window": True}, {"window": 16.5}, {"window": 15},
                {"min_peak_share": True}, {"min_peak_share": -.1},
                {"min_peak_share": 1.1}):
        with pytest.raises((TypeError, ValueError)):
            op.calculate(panel, **bad)


def test_dominant_cycle_pandas_polars_parity():
    values = np.sin(2 * np.pi * np.arange(100) / 8)
    pdf = pd.DataFrame({"A": values})
    expected = _op("ts_dominant_cycle_period").calculate(pdf, window=64)
    actual = _op("ts_dominant_cycle_period", "polars").calculate(
        pl.from_pandas(pdf), window=64,
    ).to_pandas()
    pd.testing.assert_frame_equal(expected, actual)
