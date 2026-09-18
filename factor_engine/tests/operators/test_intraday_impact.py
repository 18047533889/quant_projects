"""Meaningful tests for the minute-to-daily impact-decay operator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.operator_errors import OperatorParameterError

ensure_cleaned_loaded()


def _op():
    operator = OperatorRegistry.get("intraday_impact_decay_rate", "pandas_numpy")
    assert operator is not None
    return operator


def _decay_day(day: str, kappa: float, horizon: int = 3) -> pd.Series:
    """Ten minute returns with one isolated shock and a known log-price decay."""
    idx = pd.date_range(f"{day} 09:30", periods=10, freq="min")
    log_price = np.array([0.0, 0.001, 0.0, 0.001, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0])
    shock = 0.10
    event = 6
    anchor = log_price[event - 1]
    for h in range(horizon + 1):
        log_price[event + h] = anchor + shock * np.exp(-kappa * h)
    previous = np.r_[log_price[0], log_price[:-1]]
    returns = np.exp(log_price - previous) - 1.0
    returns[0] = 0.0
    return pd.Series(returns, index=idx)


def _panels(kappas: tuple[float, ...] = (0.2, 0.7)):
    days = ["2024-01-02", "2024-01-03"]
    columns = {}
    for symbol, offset in (("A", 0.0), ("B", 0.15)):
        columns[symbol] = pd.concat(
            [_decay_day(day, kappa + offset) for day, kappa in zip(days, kappas)]
        )
    ret = pd.DataFrame(columns)
    amount = pd.DataFrame(1_000_000.0, index=ret.index, columns=ret.columns)
    return ret, amount


def test_decay_rate_matches_independent_exponential_oracle():
    ret, amount = _panels()
    result = _op().calculate(ret, amount, horizon=3, shock_quantile=0.8)

    assert result.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_allclose(result["A"], [0.2, 0.7], atol=1e-12)
    np.testing.assert_allclose(result["B"], [0.35, 0.85], atol=1e-12)


def test_amount_is_required_and_invalid_amount_fails_closed():
    ret, amount = _panels()
    with pytest.raises(OperatorParameterError, match="missing"):
        _op().calculate(ret, horizon=3, shock_quantile=0.8)

    amount.loc[pd.Timestamp("2024-01-02 09:35"), "A"] = np.nan
    result = _op().calculate(ret, amount, horizon=3, shock_quantile=0.8)
    assert np.isnan(result.loc[pd.Timestamp("2024-01-02"), "A"])
    assert np.isfinite(result.loc[pd.Timestamp("2024-01-02"), "B"])


def test_completed_day_is_prefix_stable_when_later_day_is_appended():
    ret, amount = _panels()
    first_day = ret.index.normalize() == pd.Timestamp("2024-01-02")
    prefix = _op().calculate(ret.loc[first_day], amount.loc[first_day], horizon=3, shock_quantile=0.8)
    full = _op().calculate(ret, amount, horizon=3, shock_quantile=0.8)
    pd.testing.assert_frame_equal(prefix, full.loc[[pd.Timestamp("2024-01-02")]])


def test_decay_rate_metadata_declares_two_panels_and_eod_availability():
    meta = _op().metadata
    assert meta.param_names[:2] == ["ret", "amount"]
    assert meta.input_grain == "minute"
    assert meta.output_grain == "daily"
    assert meta.available_at == "session_close"
    assert meta.same_session_usable is False
