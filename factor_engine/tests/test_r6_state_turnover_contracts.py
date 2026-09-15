from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.ashare.state_machine
import factor_engine.cleaned_operators.turnover_survival
from factor_engine.cleaned_operators.registry import OperatorRegistry


_STATE = (
    "ashare_limit_up_streak", "ashare_limit_down_streak",
    "ashare_days_since_limit_up", "ashare_days_since_limit_down",
    "ashare_limit_touch_count", "ashare_failed_limit_count",
    "ashare_one_price_limit_streak", "ashare_limit_event_density",
    "ashare_limit_asymmetry", "ashare_suspension_episode_length",
    "ashare_limit_open_up_streak", "ashare_limit_open_down_streak",
    "ashare_limit_up_volume_ratio", "ashare_limit_down_volume_ratio",
)
_TURNOVER = (
    "ts_turnover_reference_price", "ts_turnover_cost_dispersion",
    "ts_turnover_profit_share", "ts_turnover_holding_age",
    "ts_turnover_near_cost_mass", "ts_turnover_cost_quantile_distance",
    "ts_turnover_cost_entropy", "ts_turnover_cost_mode_distance",
    "ts_turnover_cost_skew", "ts_turnover_age_dispersion",
    "ts_turnover_old_mass", "ts_turnover_cost_entropy_vol_scaled",
)


def _inputs(rows: int = 96, cols: int = 4) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=rows)
    columns = [f"S{i}" for i in range(cols)]
    t = np.arange(rows, dtype=float)[:, None]
    offsets = np.arange(cols, dtype=float)[None, :]

    def frame(values) -> pd.DataFrame:
        values = np.broadcast_to(values, (rows, cols)).astype(float).copy()
        return pd.DataFrame(values, index=index, columns=columns)

    close = frame(10.0 + 0.03 * t + 0.12 * offsets + 0.08 * np.sin(t / 5 + offsets))
    high_limit = close * 1.10
    low_limit = close * 0.90
    up_event = frame((t.astype(int) + offsets.astype(int)) % 7 == 0)
    down_event = frame((t.astype(int) + 2 * offsets.astype(int)) % 11 == 0)
    volume = frame(1000.0 + 5.0 * t + 20.0 * offsets)
    return {
        "close": close,
        "high": high_limit * 0.997,
        "low": low_limit * 1.003,
        "high_limit": high_limit,
        "low_limit": low_limit,
        "open": close.copy(),
        "valid_trade": frame(1.0),
        "event": up_event,
        "known_status": frame(1.0),
        "up_event": up_event,
        "down_event": down_event,
        "is_suspend": frame((t.astype(int) + offsets.astype(int)) % 17 == 0),
        "limit_up_event": up_event,
        "limit_down_event": down_event,
        "volume": volume,
        "price": close,
        "turnover": frame(0.12 + 0.01 * np.sin(t / 5 + offsets)),
    }


def _scalar_value(name: str):
    return {
        "window": 20,
        "max_lookback": 30,
        "side": "up",
        "tick_tolerance": 0.005,
        "band": 0.05,
        "q_high": 0.75,
        "q_low": 0.25,
    }[name]


@pytest.mark.parametrize("name", _STATE + _TURNOVER)
def test_state_turnover_real_default_keyword_positional_contract(name):
    panels = _inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert metadata.panel_params
    assert metadata.panel_arity == len(metadata.panel_params)
    assert metadata.total_positional_arity == len(metadata.param_names)
    assert set(metadata.scalar_params) == set(metadata.param_specs)

    panel_args = [panels[param] for param in metadata.panel_params]
    panel_kwargs = {param: panels[param] for param in metadata.panel_params}
    defaults_by_name = {
        param: metadata.param_specs[param].default for param in metadata.scalar_params
    }
    default_ordered = [
        panels[param] if param in panel_kwargs else defaults_by_name[param]
        for param in metadata.param_names
    ]
    default_positional = op.calculate(*default_ordered)
    default_keyword = op.calculate(**panel_kwargs)
    default_mixed = (
        op.calculate(panels["event"])
        if name == "ashare_limit_event_density"
        else op.calculate(*panel_args)
    )
    pd.testing.assert_frame_equal(default_positional, default_keyword)
    pd.testing.assert_frame_equal(default_positional, default_mixed)

    scalar_values = [_scalar_value(param) for param in metadata.scalar_params]
    scalar_by_name = dict(zip(metadata.scalar_params, scalar_values))
    ordered = [
        panels[param] if param in panel_kwargs else scalar_by_name[param]
        for param in metadata.param_names
    ]
    result = op.calculate(*ordered)
    keyword = op.calculate(
        **panel_kwargs,
        **dict(zip(metadata.scalar_params, scalar_values)),
    )
    pd.testing.assert_frame_equal(result, keyword)
    assert result.shape == panels["close"].shape
    assert np.isfinite(result.to_numpy()).any(), name

    prefix = op.calculate(
        *[
            panels[param].iloc[:72] if param in panel_kwargs else scalar_by_name[param]
            for param in metadata.param_names
        ],
    )
    pd.testing.assert_frame_equal(
        result.iloc[:72], prefix, check_exact=False, rtol=1e-12, atol=1e-12,
    )


@pytest.mark.parametrize("name", _STATE + _TURNOVER)
def test_state_turnover_scalar_specs_reject_invalid_values(name):
    panels = _inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    if not metadata.scalar_params:
        return
    panel_args = [panels[param] for param in metadata.panel_params]
    scalar_values = {
        param: _scalar_value(param) for param in metadata.scalar_params
    }
    target = metadata.scalar_params[0]
    spec = metadata.param_specs[target]
    if spec.choices:
        bad = "__invalid__"
    elif spec.min is not None:
        bad = spec.min - 1
    else:
        bad = np.nan
    scalar_values[target] = bad
    with pytest.raises(ValueError):
        op.calculate(**{
            **{param: panels[param] for param in metadata.panel_params},
            **scalar_values,
        })


@pytest.mark.parametrize("name", _STATE + _TURNOVER)
def test_state_turnover_polars_twin_metadata_matches(name):
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    polars_op = OperatorRegistry.get(name, backend="polars", mode="research")
    assert polars_op.metadata.panel_params == pandas_op.metadata.panel_params
    assert polars_op.metadata.scalar_params == pandas_op.metadata.scalar_params
    assert polars_op.metadata.param_specs == pandas_op.metadata.param_specs
