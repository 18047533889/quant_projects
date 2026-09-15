"""R6 runtime contracts for both intraday time-structure operator packs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.backend.operator_semantic_version import semantic_version
from factor_engine.cleaned_operators.intraday import time_structure, time_structure_v2
from factor_engine.cleaned_operators.intraday import polars_intraday_full, polars_next_stage
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import load_all


@pytest.fixture(scope="module", autouse=True)
def _finalized_registry():
    load_all()


EXPECTED_PANELS = {
    "intra_interval_return": ("close",),
    "intra_interval_volume_share": ("volume",),
    "intra_interval_amount_share": ("amount",),
    "intra_interval_realized_variance": ("close",),
    "intra_interval_vwap_deviation": ("close", "amount", "volume"),
    "intra_interval_illiquidity": ("close", "amount"),
    "intra_same_slot_momentum": ("close",),
    "intra_same_slot_reversal": ("close",),
    "intra_return_profile_cosine": ("x",),
    "intra_volume_profile_cosine": ("x",),
    "intra_amount_profile_cosine": ("x",),
    "intra_volume_profile_jsd": ("x",),
    "intra_amount_profile_jsd": ("x",),
    "intra_profile_earth_mover_distance": ("x",),
    "intra_signed_return_profile_cosine": ("close",),
    "intra_abs_return_profile_cosine": ("close",),
    "intra_bar_range_persistence": ("high", "low"),
    "intra_bar_range_deviation": ("high", "low"),
    "intra_tail_volume_share": ("close", "volume"),
    "intra_volume_price_alignment": ("close", "volume"),
    "intra_ute_high": ("close",),
    "intra_ute_low": ("close",),
    "intra_slot_volume_surprise": ("volume",),
    "intra_slot_amount_surprise": ("amount",),
    "intra_slot_volatility_surprise": ("high", "low"),
    "intra_market_lead_lag_ex_self": ("close",),
    "intra_industry_lead_lag_ex_self": ("close", "industry"),
    "intra_session_return_asymmetry": ("close",),
    "intra_close_participation": ("volume",),
    "intra_high_low_affinity": ("high", "low"),
}


def _local_panel(values):
    index = pd.to_datetime([
        "2024-01-02 09:30", "2024-01-02 09:31", "2024-01-02 09:32",
        "2024-01-03 09:30", "2024-01-03 09:31", "2024-01-03 09:32",
    ])
    return pd.DataFrame({"A": values}, index=index, dtype=float)


def _rich_panels():
    days = pd.bdate_range("2024-01-02", periods=30)
    minute_offsets = tuple(range(570, 591)) + tuple(range(780, 801))
    index = pd.DatetimeIndex([
        day + pd.Timedelta(minutes=minute) for day in days for minute in minute_offsets
    ])
    nslot = len(minute_offsets)
    close_data, volume_data = {}, {}
    for j, symbol in enumerate(("A", "B", "C")):
        values, volumes = [], []
        for d in range(len(days)):
            k = np.arange(nslot, dtype=float)
            returns = 0.0015 * np.sin((k + d + 2 * j) / 3.0) + 0.0004 * np.cos((2 * k + j) / 5.0)
            values.extend((50.0 + 10 * j + d * 0.1) * np.exp(np.cumsum(returns)))
            volumes.extend(1000.0 + 13 * k + 7 * d + 40 * np.sin((k + j + d) / 4.0))
        close_data[symbol] = values
        volume_data[symbol] = volumes
    close = pd.DataFrame(close_data, index=index)
    volume = pd.DataFrame(volume_data, index=index)
    high = close * (1.002 + 0.0002 * np.sin(np.arange(len(index)) / 7.0)[:, None])
    low = close * (0.998 - 0.0002 * np.cos(np.arange(len(index)) / 9.0)[:, None])
    amount = close * volume
    industry = pd.DataFrame(1.0, index=days, columns=close.columns)
    return {"close": close, "x": volume, "volume": volume, "amount": amount,
            "high": high, "low": low, "industry": industry}


def _to_polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({"date": list(frame.index.to_pydatetime()), **{
        column: frame[column].to_numpy() for column in frame.columns
    }})


def test_all_time_structure_canonicals_declare_exact_topology_and_timezone_policy() -> None:
    canonicals = tuple(time_structure._CANONICALS + time_structure_v2._CANONICALS)
    assert len(canonicals) == 30
    assert set(canonicals) == set(EXPECTED_PANELS)
    assert all(name.startswith("intra_") for name in canonicals)
    for canonical in canonicals:
        metadata = OperatorRegistry.get(canonical, mode="any").metadata
        assert metadata.panel_params == EXPECTED_PANELS[canonical]
        assert metadata.panel_arity == len(EXPECTED_PANELS[canonical])
        assert tuple(metadata.param_names) == metadata.panel_params + metadata.scalar_params
        assert set(metadata.scalar_params) == set(metadata.param_specs)
        tz = metadata.param_specs["session_tz"]
        assert tz.dtype is str
        assert tz.default is None
        assert tz.searchable is False
        assert tz.param_role is ParamRole.POLICY


def test_interval_return_is_daily_numeric_and_honours_explicit_session_timezone() -> None:
    local = _local_panel([10, 11, 12, 20, 18, 22])
    utc = local.copy()
    utc.index = utc.index.tz_localize("Asia/Shanghai").tz_convert("UTC")
    operator = OperatorRegistry.get("intra_interval_return", mode="any")

    expected = pd.Series([0.2, 0.1], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    local_result = operator.calculate(local, 570, 572)
    utc_result = operator.calculate(close=utc, start_minute=570, end_minute=572, session_tz="Asia/Shanghai")
    np.testing.assert_allclose(local_result["A"], expected)
    pd.testing.assert_index_equal(local_result.index, expected.index)
    pd.testing.assert_frame_equal(local_result, utc_result)
    # Regression: 09:30 must be minute 570, not UInt8-overflowed and filtered out.
    assert local_result.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(0.2)
    for canonical in (
        "intra_interval_return", "intra_interval_volume_share", "intra_interval_amount_share",
    ):
        assert semantic_version(canonical) >= 2


def test_three_panel_calls_and_bad_call_shapes_fail_closed() -> None:
    close = _local_panel([10, 11, 12, 20, 21, 22])
    amount = _local_panel([100, 220, 360, 200, 420, 660])
    volume = _local_panel([10, 20, 30, 10, 20, 30])
    operator = OperatorRegistry.get("intra_interval_vwap_deviation", mode="any")

    positional = operator.calculate(close, amount, volume, 570, 572, "Asia/Shanghai")
    keyword = operator.calculate(
        close=close, amount=amount, volume=volume,
        start_minute=570, end_minute=572, session_tz="Asia/Shanghai",
    )
    mixed = operator.calculate(close, amount=amount, volume=volume, end_minute=572)
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    assert positional.notna().all().all()

    with pytest.raises((TypeError, ValueError), match="start_minute"):
        operator.calculate(close, amount, volume, start_minute=True)
    with pytest.raises((TypeError, ValueError), match="amount|required|missing"):
        operator.calculate(close=close, volume=volume)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|end_minute"):
        operator.calculate(close, amount, volume, 570, 572, end_minute=572)
    with pytest.raises((TypeError, ValueError), match="undeclared|unknown|unexpected"):
        operator.calculate(close, amount, volume, mystery=1)


def test_v2_scalar_boundaries_are_enforced_by_registry_call() -> None:
    close = _local_panel([10, 11, 12, 20, 21, 22])
    volume = _local_panel([1, 2, 5, 2, 4, 8])
    assert OperatorRegistry.get("intra_close_participation", mode="any").calculate(volume, 1).notna().all().all()
    with pytest.raises((TypeError, ValueError), match="tail_minutes"):
        OperatorRegistry.get("intra_close_participation", mode="any").calculate(volume, 0)
    with pytest.raises((TypeError, ValueError), match="tail_quantile"):
        OperatorRegistry.get("intra_tail_volume_share", mode="any").calculate(close, volume, 1.0)
    with pytest.raises((TypeError, ValueError), match="lag"):
        OperatorRegistry.get("intra_market_lead_lag_ex_self", mode="any").calculate(close, 0)


def test_every_canonical_executes_from_final_registry_as_a_finite_daily_operator() -> None:
    panels = _rich_panels()
    expected_days = panels["close"].index.normalize().unique()
    for canonical, panel_names in EXPECTED_PANELS.items():
        operator = OperatorRegistry.get(canonical, backend="pandas_numpy")
        result = operator.calculate(**{name: panels[name] for name in panel_names})
        assert isinstance(result, pd.DataFrame), canonical
        assert result.index.is_unique and result.columns.is_unique, canonical
        assert result.index.isin(expected_days).all(), canonical
        assert 1 <= len(result.index) <= len(expected_days), canonical
        assert np.isfinite(result.to_numpy(dtype=float)).any(), canonical


POLARS_TWINS = (
    "intra_interval_return", "intra_interval_volume_share", "intra_interval_amount_share",
    "intra_interval_realized_variance", "intra_interval_vwap_deviation", "intra_interval_illiquidity",
    "intra_same_slot_momentum", "intra_same_slot_reversal",
    "intra_return_profile_cosine", "intra_volume_profile_cosine", "intra_amount_profile_cosine",
    "intra_volume_profile_jsd", "intra_amount_profile_jsd", "intra_profile_earth_mover_distance",
)


def test_each_polars_twin_matches_local_positional_and_utc_keyword_timezone_calls() -> None:
    pandas_panels = _rich_panels()
    local = {name: _to_polars(frame) for name, frame in pandas_panels.items() if name != "industry"}
    utc = {}
    for name, frame in pandas_panels.items():
        if name == "industry":
            continue
        converted = frame.copy()
        converted.index = converted.index.tz_localize("Asia/Shanghai").tz_convert("UTC")
        utc[name] = _to_polars(converted)
    for canonical in POLARS_TWINS:
        operator = OperatorRegistry.get(canonical, backend="polars")
        panel_names = EXPECTED_PANELS[canonical]
        scalar_values = [operator.metadata.param_specs[name].default
                         for name in operator.metadata.scalar_params if name != "session_tz"]
        positional = operator.calculate(
            *(local[name] for name in panel_names), *scalar_values, "Asia/Shanghai"
        ).sort("date")
        keyword_args = {name: utc[name] for name in panel_names}
        keyword_args.update(zip(
            (name for name in operator.metadata.scalar_params if name != "session_tz"), scalar_values
        ))
        keyword_args["session_tz"] = "Asia/Shanghai"
        keyword = operator.calculate(**keyword_args).sort("date")
        ordered = ["date", *sorted(column for column in positional.columns if column != "date")]
        positional = positional.select(ordered)
        keyword = keyword.select(ordered)
        assert positional.columns == keyword.columns, canonical
        assert positional["date"].to_list() == keyword["date"].to_list(), canonical
        values = [column for column in positional.columns if column != "date"]
        assert np.isfinite(positional.select(values).to_numpy()).any(), canonical
        np.testing.assert_allclose(
            positional.select(values).to_numpy(), keyword.select(values).to_numpy(),
            rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=canonical,
        )
