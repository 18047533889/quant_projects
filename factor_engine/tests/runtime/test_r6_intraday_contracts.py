"""R6 direct-call contracts for intraday sufficient-statistics operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.intraday import sufficient_stats_ops as module
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _panel(values) -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2024-01-02 09:30",
            "2024-01-02 09:31",
            "2024-01-02 09:32",
            "2024-01-03 09:30",
            "2024-01-03 09:31",
            "2024-01-03 09:32",
        ]
    )
    return pd.DataFrame({"000001.SZ": values}, index=index)


def test_sufficient_statistics_family_declares_panel_and_scalar_authority() -> None:
    for canonical in module._SCALAR_FAMILY_CANONICALS:
        metadata = OperatorRegistry.get(canonical, mode="any").metadata
        assert metadata.panel_params == ("close",)
        assert metadata.scalar_params == ("window",)
        spec = metadata.param_specs["window"]
        assert spec.dtype is int
        assert spec.min == 1
        assert spec.default is None
        assert spec.param_role is ParamRole.HORIZON

    for canonical in module._BUILTIN_ONE:
        metadata = OperatorRegistry.get(canonical, mode="any").metadata
        assert metadata.panel_params == ("close",)
        assert metadata.panel_arity == 1
        assert metadata.scalar_params == ()

    for canonical in module._BUILTIN_TWO:
        metadata = OperatorRegistry.get(canonical, mode="any").metadata
        assert metadata.panel_params == ("close", "volume")
        assert metadata.panel_arity == 2
        assert metadata.scalar_params == ()


def test_intraday_window_call_is_session_bounded_and_rejects_invalid_scalars() -> None:
    close = _panel([3.0, 1.0, 2.0, 10.0, 20.0, 30.0])
    operator = OperatorRegistry.get("intra_ts_mean", mode="any")

    full = operator.calculate(close)
    last_two = operator.calculate(close, window=2)
    positional = operator.calculate(close, 2)
    keyword = operator.calculate(close=close, window=2)

    expected_index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    pd.testing.assert_index_equal(full.index, expected_index)
    np.testing.assert_allclose(full.iloc[:, 0], [2.0, 20.0])
    np.testing.assert_allclose(last_two.iloc[:, 0], [1.5, 25.0])
    pd.testing.assert_frame_equal(last_two, positional)
    pd.testing.assert_frame_equal(last_two, keyword)
    with pytest.raises((TypeError, ValueError), match="window"):
        operator.calculate(close, window=0)
    with pytest.raises((TypeError, ValueError), match="window"):
        operator.calculate(close, window=True)
    with pytest.raises((TypeError, ValueError), match="close|required|missing"):
        operator.calculate(window=2)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|window"):
        operator.calculate(close, 2, window=2)
    with pytest.raises((TypeError, ValueError), match="undeclared|unknown|unexpected"):
        operator.calculate(close, mystery=2)


def test_two_panel_direct_call_checks_grid_and_uses_joint_positive_support() -> None:
    close = _panel([10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
    volume = _panel([1.0, 2.0, 3.0, 2.0, 0.0, 2.0])
    operator = OperatorRegistry.get("intra_ts_vwap", mode="any")

    result = operator.calculate(close, volume)

    np.testing.assert_allclose(result.iloc[:, 0], [68.0 / 6.0, 21.0])
    shifted = volume.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="grid|index|session"):
        operator.calculate(close, shifted)


def test_realized_beta_family_declares_two_panels_and_breaks_overnight_returns() -> None:
    from factor_engine.cleaned_operators.intraday import realized_beta

    canonicals = tuple(realized_beta._CANONICALS)
    assert len(canonicals) == 17
    for canonical in canonicals:
        metadata = OperatorRegistry.get(canonical, mode="any").metadata
        assert metadata.panel_params == ("close", "free_market_cap")
        assert metadata.panel_arity == 2
        assert metadata.scalar_params == ()

    index = pd.to_datetime([
        "2024-01-02 09:30", "2024-01-02 09:31", "2024-01-02 09:32",
        "2024-01-02 09:33", "2024-01-03 09:30", "2024-01-03 09:31",
        "2024-01-03 09:32", "2024-01-03 09:33",
    ])
    close = pd.DataFrame({
        "A": [10, 11, 10, 12, 12, 13, 12, 14],
        "B": [20, 19, 21, 18, 18, 17, 19, 16],
    }, index=index, dtype=float)
    caps = pd.DataFrame({"A": [100.0, 100.0], "B": [200.0, 200.0]},
                        index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    operator = OperatorRegistry.get("intra_realized_beta", mode="any")

    positional = operator.calculate(close, caps)
    keyword = operator.calculate(close=close, free_market_cap=caps)
    mixed = operator.calculate(close, free_market_cap=caps)
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    assert positional.index.tolist() == list(caps.index)

    changed_prior_close = close.copy()
    changed_prior_close.iloc[3] *= 100.0
    changed = operator.calculate(changed_prior_close, caps)
    pd.testing.assert_series_equal(positional.loc[caps.index[1]], changed.loc[caps.index[1]])
    with pytest.raises((TypeError, ValueError), match="free_market_cap|required|missing"):
        operator.calculate(close)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|free_market_cap"):
        operator.calculate(close, caps, free_market_cap=caps)
    with pytest.raises((TypeError, ValueError), match="undeclared|unknown|unexpected"):
        operator.calculate(close, caps, mystery=True)


def test_same_clock_lag_contract_and_missing_clock_boundary() -> None:
    from factor_engine.cleaned_operators import same_clock_lag as module

    operator = module.SameClockLagPandas()
    metadata = operator.metadata
    assert metadata.panel_params == ("x",)
    assert metadata.scalar_params == ("lag", "clock_unit")
    assert metadata.param_specs["lag"].param_role is ParamRole.HORIZON
    assert metadata.param_specs["clock_unit"].param_role is ParamRole.POLICY

    index = pd.to_datetime([
        "2024-01-02 09:30", "2024-01-02 09:31",
        "2024-01-03 09:30", "2024-01-03 09:31",
        "2024-01-04 09:30",
    ])
    panel = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    positional = operator.calculate(panel, 1, "minute_of_day")
    keyword = operator.calculate(x=panel, lag=1, clock_unit="minute_of_day")
    mixed = operator.calculate(panel, lag=1, clock_unit="minute_of_day")
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    np.testing.assert_allclose(
        positional.iloc[:, 0].to_numpy(), [np.nan, np.nan, 1.0, 2.0, 3.0],
        equal_nan=True,
    )
    with pytest.raises((TypeError, ValueError), match="lag"):
        operator.calculate(panel, lag=-1)
    with pytest.raises((TypeError, ValueError), match="lag"):
        operator.calculate(panel, lag=True)
    with pytest.raises((TypeError, ValueError), match="clock_unit"):
        operator.calculate(panel, clock_unit="hour")
    with pytest.raises((TypeError, ValueError), match="x|required|missing"):
        operator.calculate(lag=1)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|lag"):
        operator.calculate(panel, 1, lag=1)


def test_overnight_contracts_support_named_open_and_reject_dead_windows() -> None:
    from factor_engine.cleaned_operators.intraday import overnight

    assert len(overnight._CANONICALS) == 7
    for canonical in overnight._CANONICALS:
        metadata = OperatorRegistry.get(canonical, mode="any").metadata
        assert metadata.panel_params == ("close", "open", "pre_close")
        assert set(metadata.scalar_params) == set(metadata.param_specs)

    index = pd.date_range("2024-01-01", periods=8, freq="D")
    pre_close = pd.DataFrame({"A": np.arange(100.0, 108.0)}, index=index)
    open_px = pre_close * 1.01
    close = open_px * np.array([1.0, .99, 1.01, .98, 1.02, .99, 1.01, 1.0])[:, None]
    operator = OperatorRegistry.get("ts_overnight_intraday_spread", mode="any")
    positional = operator.calculate(close, open_px, pre_close, 5)
    keyword = operator.calculate(
        close=close, open=open_px, pre_close=pre_close, window=5
    )
    mixed = operator.calculate(close, open=open_px, pre_close=pre_close, window=5)
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    with pytest.raises((TypeError, ValueError), match="window"):
        OperatorRegistry.get("ts_overnight_intraday_cov", mode="any").calculate(
            close, open_px, pre_close, window=4
        )
    with pytest.raises((TypeError, ValueError), match="window"):
        operator.calculate(close, open_px, pre_close, window=True)
    with pytest.raises((TypeError, ValueError), match="threshold"):
        OperatorRegistry.get("ts_gap_reversion_ratio", mode="any").calculate(
            close, open_px, pre_close, threshold=-0.01
        )
    with pytest.raises((TypeError, ValueError), match="open|required|missing"):
        operator.calculate(close=close, pre_close=pre_close)


def test_intraday_volatility_shape_contract_and_gap_boundary() -> None:
    from factor_engine.cleaned_operators import intraday_vol_ext

    for canonical in intraday_vol_ext._NEW_CANONICALS:
        metadata = OperatorRegistry.get(canonical, mode="any").metadata
        assert metadata.panel_params == ("returns",)
        assert metadata.scalar_params == ("window",)
        spec = metadata.param_specs["window"]
        assert spec.dtype is int
        assert spec.min == 2
        assert spec.default == 240
        assert spec.param_role is ParamRole.HORIZON

    index = pd.date_range("2024-01-02 09:30", periods=14, freq="min")
    returns = pd.DataFrame(
        {"A": [.01, -.02, .03, .01, -.01, .04, -.02, .01, .03, -.01, .02, .01, -.03, .02]},
        index=index,
    )
    operator = OperatorRegistry.get("intraday_volatility_time_centroid", mode="any")
    positional = operator.calculate(returns, 10)
    keyword = operator.calculate(returns=returns, window=10)
    mixed = operator.calculate(returns, window=10)
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    pd.testing.assert_index_equal(positional.index, returns.index)
    pd.testing.assert_index_equal(positional.columns, returns.columns)
    assert positional.iloc[-1].notna().all()

    curvature = OperatorRegistry.get("intraday_rv_signature_curvature", mode="any")
    assert curvature.calculate(returns, 10).iloc[-1].notna().all()
    with_gap = returns.copy()
    with_gap.iloc[-5, 0] = np.nan
    assert curvature.calculate(with_gap, 10).iloc[-1].isna().all()

    with pytest.raises((TypeError, ValueError), match="window"):
        operator.calculate(returns, 0)
    with pytest.raises((TypeError, ValueError), match="window"):
        operator.calculate(returns, -1)
    with pytest.raises((TypeError, ValueError), match="window"):
        operator.calculate(returns, True)
    with pytest.raises((TypeError, ValueError), match="returns|required|missing"):
        operator.calculate(window=10)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|window"):
        operator.calculate(returns, 10, window=10)
    with pytest.raises((TypeError, ValueError), match="undeclared|unknown|unexpected"):
        operator.calculate(returns, window=10, mystery=True)


def test_intraday_session_contract_and_completed_session_boundary() -> None:
    from factor_engine.cleaned_operators import intraday_session
    from factor_engine.runtime.session_calendar import SessionCalendar

    novelty = OperatorRegistry.get("intraday_session_shape_novelty", mode="any")
    pca = OperatorRegistry.get("intraday_profile_pca_residual", mode="any")
    assert novelty.metadata.panel_params == ("x", "session_id")
    assert novelty.metadata.scalar_params == (
        "history_days", "min_history_sessions", "calendar",
    )
    assert pca.metadata.panel_params == ("x", "session_id")
    assert pca.metadata.scalar_params == (
        "history_days", "n_components", "min_history_sessions", "calendar",
    )
    assert novelty.metadata.param_specs["history_days"].default == 20
    assert pca.metadata.param_specs["n_components"].param_role is ParamRole.ESTIMATOR_RESOLUTION

    calendar = SessionCalendar(
        market="CN",
        segments=(("09:30", "09:34"),),
        timestamp_convention="bar_start",
        bar_freq="1min",
    )
    index = pd.DatetimeIndex([
        pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30 + minute)
        for day in ("2024-01-02", "2024-01-03", "2024-01-04")
        for minute in range(4)
    ])
    x = pd.DataFrame(
        {"A": [1., 2., 3., 4., 1., 2., 4., 8., 2., 1., 4., 3.]}, index=index,
    )
    session_id = pd.DataFrame({"A": np.repeat([1, 2, 3], 4)}, index=index)
    positional = novelty.calculate(x, session_id, 2, 1, calendar)
    keyword = novelty.calculate(
        x=x, session_id=session_id, history_days=2,
        min_history_sessions=1, calendar=calendar,
    )
    mixed = novelty.calculate(
        x, session_id=session_id, history_days=2,
        min_history_sessions=1, calendar=calendar,
    )
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    assert positional.iloc[-1].notna().all()
    assert positional.iloc[:-1].notna().sum().sum() == 1

    truncated = novelty.calculate(x.iloc[:-1], session_id.iloc[:-1], 2, 1, calendar)
    assert truncated.iloc[-1].isna().all()
    bad_sid = session_id.astype(float)
    bad_sid.iloc[0, 0] = 1.5
    with pytest.raises((TypeError, ValueError), match="session_id.*integral"):
        novelty.calculate(x, bad_sid, 2, 1, calendar)
    with pytest.raises((TypeError, ValueError), match="history_days"):
        novelty.calculate(x, session_id, history_days=True, calendar=calendar)
    with pytest.raises((TypeError, ValueError), match="session_id|required|missing"):
        novelty.calculate(x=x, calendar=calendar)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|history_days"):
        novelty.calculate(x, session_id, 2, history_days=2, calendar=calendar)
    with pytest.raises((TypeError, ValueError), match="undeclared|unknown|unexpected"):
        novelty.calculate(x, session_id, calendar=calendar, mystery=True)


def test_volume_clock_contract_optional_open_and_gap_policy() -> None:
    from factor_engine.cleaned_operators import volume_clock  # noqa: F401

    canonicals = (
        "intraday_volume_clock_path_efficiency",
        "intraday_volume_clock_roughness",
    )
    for canonical in canonicals:
        metadata = OperatorRegistry.get(canonical, mode="any").metadata
        assert metadata.panel_params == ("price", "activity", "open")
        assert metadata.scalar_params == ("buckets",)
        assert metadata.param_specs["buckets"].choices == (8, 16, 32)
        assert metadata.param_specs["buckets"].default == 16
        assert metadata.param_specs["buckets"].param_role is ParamRole.ESTIMATOR_RESOLUTION

    index = pd.date_range("2024-01-02 09:30", periods=18, freq="min")
    price = pd.DataFrame({"A": np.linspace(10.0, 12.0, 18)}, index=index)
    activity = pd.DataFrame({"A": np.arange(1.0, 19.0)}, index=index)
    open_px = pd.DataFrame({"A": np.full(18, 9.8)}, index=index)
    operator = OperatorRegistry.get("intraday_volume_clock_path_efficiency", mode="any")
    positional = operator.calculate(price, activity, 16, open_px)
    keyword = operator.calculate(
        price=price, activity=activity, buckets=16, open=open_px,
    )
    mixed = operator.calculate(price, activity, buckets=16, open=open_px)
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    assert positional.iloc[-1].notna().all()
    assert operator.calculate(price, activity, buckets=16).iloc[-1].notna().all()

    with_gap = price.copy()
    with_gap.iloc[8, 0] = np.nan
    assert operator.calculate(with_gap, activity, 16).iloc[-1].isna().all()
    with pytest.raises((TypeError, ValueError), match="buckets"):
        operator.calculate(price, activity, buckets=7)
    with pytest.raises((TypeError, ValueError), match="buckets"):
        operator.calculate(price, activity, buckets=True)
    with pytest.raises((TypeError, ValueError), match="activity|required|missing"):
        operator.calculate(price=price, buckets=16)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|buckets"):
        operator.calculate(price, activity, 16, buckets=16)
    with pytest.raises((TypeError, ValueError), match="undeclared|unknown|unexpected"):
        operator.calculate(price, activity, buckets=16, mystery=True)


def test_activity_clock_factory_contracts_and_prior_boundary() -> None:
    from factor_engine.cleaned_operators import activity_clock

    expected_panels = {
        "ts_activity_clock_lagged_value": ("x", "activity"),
        "ts_activity_clock_lagged_value_prior": ("x", "activity"),
        "ts_activity_clock_age": ("activity",),
        "ts_max_drawdown_activity_cost": ("x", "activity"),
    }
    for canonical in activity_clock._DAILY_CANONICALS:
        operator = OperatorRegistry.get(canonical, mode="pandas_numpy")
        assert operator is not None
        metadata = operator.metadata
        assert metadata.panel_params == expected_panels[canonical]
        assert metadata.scalar_params == tuple(
            name for name in activity_clock._PARAMS[canonical]
            if name not in expected_panels[canonical]
        )
        assert set(metadata.param_specs) == set(metadata.scalar_params)

    index = pd.date_range("2024-01-01", periods=30, freq="D")
    x = pd.DataFrame({"A": np.arange(100.0, 130.0)}, index=index)
    activity = pd.DataFrame({"A": np.ones(30)}, index=index)
    operator = OperatorRegistry.get("ts_activity_clock_lagged_value_prior", mode="pandas_numpy")
    positional = operator.calculate(x, activity, 2.0, 5, 10)
    keyword = operator.calculate(
        x=x, activity=activity, budget=2.0, scale_window=5, max_lookback=10,
    )
    mixed = operator.calculate(
        x, activity=activity, budget=2.0, scale_window=5, max_lookback=10,
    )
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    assert positional.iloc[-1, 0] == x.iloc[-3, 0]

    current = OperatorRegistry.get("ts_activity_clock_lagged_value", mode="pandas_numpy")
    current_value = current.calculate(x, activity, 2.0, 5, 10, True)
    assert current_value.iloc[-1, 0] == x.iloc[-2, 0]
    age = OperatorRegistry.get("ts_activity_clock_age", mode="pandas_numpy")
    assert age.calculate(activity, 2.0, 5, 10, True).iloc[-1, 0] == 1.0

    shifted = activity.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="aligned|index"):
        operator.calculate(x, shifted, 2.0, 5, 10)
    with pytest.raises((TypeError, ValueError), match="scale_window"):
        operator.calculate(x, activity, scale_window=4)
    with pytest.raises((TypeError, ValueError), match="max_lookback"):
        operator.calculate(x, activity, max_lookback=True)
    with pytest.raises((TypeError, ValueError), match="activity|required|missing"):
        operator.calculate(x=x)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|budget"):
        operator.calculate(x, activity, 2.0, budget=2.0)
    with pytest.raises((TypeError, ValueError), match="undeclared|unknown|unexpected"):
        operator.calculate(x, activity, mystery=True)


def test_intraday_activity_duration_contract_and_official_grid() -> None:
    from factor_engine.cleaned_operators import intraday_activity_duration  # noqa: F401
    from factor_engine.runtime.session_calendar import SessionCalendar

    operator = OperatorRegistry.get(
        "intraday_activity_duration_curvature", mode="pandas_numpy",
    )
    metadata = operator.metadata
    assert metadata.panel_params == ("activity",)
    assert metadata.scalar_params == ("buckets", "calendar")
    assert metadata.param_specs["buckets"].min == 3
    assert metadata.param_specs["buckets"].default == 10
    assert metadata.param_specs["buckets"].param_role is ParamRole.ESTIMATOR_RESOLUTION

    calendar = SessionCalendar(
        market="CN", segments=(("09:30", "09:42"),),
        timestamp_convention="bar_start", bar_freq="1min",
    )
    index = pd.date_range("2024-01-02 09:30", periods=12, freq="min")
    activity = pd.DataFrame({"A": np.arange(1.0, 13.0)}, index=index)
    positional = operator.calculate(activity, 4, calendar)
    keyword = operator.calculate(activity=activity, buckets=4, calendar=calendar)
    mixed = operator.calculate(activity, buckets=4, calendar=calendar)
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    assert positional.iloc[0, 0] == pytest.approx(-2.0 / 3.0)

    truncated = operator.calculate(activity.iloc[:-1], 4, calendar)
    assert truncated.iloc[0].isna().all()
    with_gap = activity.drop(index[5])
    assert operator.calculate(with_gap, 4, calendar).iloc[0].isna().all()
    with pytest.raises((TypeError, ValueError), match="buckets"):
        operator.calculate(activity, buckets=2, calendar=calendar)
    with pytest.raises((TypeError, ValueError), match="buckets"):
        operator.calculate(activity, buckets=True, calendar=calendar)
    with pytest.raises((TypeError, ValueError), match="activity|required|missing"):
        operator.calculate(buckets=4, calendar=calendar)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|buckets"):
        operator.calculate(activity, 4, buckets=4, calendar=calendar)
    with pytest.raises((TypeError, ValueError), match="undeclared|unknown|unexpected"):
        operator.calculate(activity, calendar=calendar, mystery=True)
