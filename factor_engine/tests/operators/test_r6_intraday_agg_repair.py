"""Independent finalized-registry checks for the intraday_agg surface."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.microstructure.intraday_agg import __all__ as NAMES

load_all()


def _inputs(days: int = 2):
    stamps = []
    for offset in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=offset)
        stamps.extend(day + pd.Timedelta(minutes=m) for m in (*range(571, 691), *range(781, 901)))
    idx = pd.DatetimeIndex(stamps)
    n = len(idx)
    returns = 0.0015 * np.sin(np.arange(n) / 7.0) + 0.0003
    close = pd.DataFrame({"A": 100.0 * np.exp(np.cumsum(returns))}, index=idx)
    volume = pd.DataFrame({"A": 10.0 + 2e4 * np.abs(returns)}, index=idx)
    amount = close * volume
    high, low, opening = close * 1.001, close * 0.999, close * 0.9995
    dates = pd.DatetimeIndex(sorted(set(idx.normalize())))
    high_limit = pd.DataFrame({"A": [float(high.loc[high.index.normalize() == d].iloc[80, 0]) for d in dates]}, index=dates)
    low_limit = pd.DataFrame({"A": [float(low.loc[low.index.normalize() == d].iloc[80, 0]) for d in dates]}, index=dates)
    return close, high, low, opening, volume, amount, high_limit, low_limit


def _calls(data):
    close, high, low, opening, volume, amount, high_limit, low_limit = data
    return {
        "intra_segment_return": ((close,), {"segment": "morning"}),
        "intra_segment_volume_share": ((volume,), {"segment": "morning"}),
        "intra_segment_amount_share": ((amount,), {"segment": "morning"}),
        "intra_segment_vwap_deviation": ((close, amount, volume), {"segment": "morning"}),
        "intra_segment_realized_vol": ((close,), {"segment": "morning"}),
        "intra_realized_variance": ((close,), {}),
        "intra_realized_semivariance": ((close,), {"side": "up"}),
        "intra_bipower_variation": ((close,), {}),
        "intra_jump_ratio": ((close,), {}),
        "intra_path_efficiency": ((close,), {}),
        "intra_high_time": ((high,), {}),
        "intra_low_time": ((low,), {}),
        "intra_vwap_above_ratio": ((close, amount, volume), {}),
        "intra_vwap_cross_count": ((close, amount, volume), {}),
        "intra_concentration": ((volume,), {}),
        "intra_entropy": ((volume,), {"normalize": True}),
        "intra_signed_imbalance_proxy": ((close, volume), {}),
        "intra_return_activity_corr": ((close, volume), {"absolute_return": True}),
        "intra_amihud": ((close, amount), {"scale": 1e8}),
        "intra_kyle_lambda_proxy": ((close, amount), {}),
        "intra_extreme_bar_return": ((close,), {"side": "max"}),
        "intra_lunch_gap_return": ((close, opening), {"endpoint_policy": "exact"}),
        "intra_limit_first_hit_time": ((close,), {"high": high, "high_limit": high_limit, "side": "up"}),
        "intra_limit_duration": ((close,), {"high_limit": high_limit, "side": "up"}),
        "intra_limit_reopen_count": ((close,), {"high_limit": high_limit, "side": "up", "transition": "open"}),
    }


def test_all_unique_canonicals_execute_after_finalize_and_are_prefix_causal():
    assert len(NAMES) == len(set(NAMES)) == 25
    data = _inputs()
    calls = _calls(data)
    assert set(calls) == set(NAMES)
    for name, (args, kwargs) in calls.items():
        op = OperatorRegistry.get(name)
        assert type(op).__module__.endswith("microstructure.intraday_agg")
        out = op.calculate(*args, **kwargs)
        assert out.shape == (2, 1), name
        assert np.isfinite(out.to_numpy()).any(), name
        one_day_args = tuple(x.loc[:"2024-01-03 23:59:59"] if isinstance(x, pd.DataFrame) else x for x in args)
        one_day_kwargs = {k: (v.loc[:"2024-01-03 23:59:59"] if isinstance(v, pd.DataFrame) else v) for k, v in kwargs.items()}
        prefix = op.calculate(*one_day_args, **one_day_kwargs)
        pd.testing.assert_frame_equal(prefix, out.iloc[:1], check_dtype=False, obj=name)


def test_canonical_keywords_and_positional_calls_are_equivalent():
    close, _, _, opening, volume, _, _, _ = _inputs(days=1)
    lunch = OperatorRegistry.get("intra_lunch_gap_return")
    pd.testing.assert_frame_equal(
        lunch.calculate(close, opening, "11:30", "13:01", None, "exact"),
        lunch.calculate(close=close, open=opening, morning_cutoff="11:30", afternoon_start="13:01", session_tz=None, endpoint_policy="exact"),
    )
    share = OperatorRegistry.get("intra_segment_volume_share")
    pd.testing.assert_frame_equal(
        share.calculate(volume, "afternoon", None, "ashare"),
        share.calculate(volume=volume, segment="afternoon", session_tz=None, market="ashare"),
    )


def test_final_five_have_explicit_topology_defaults_and_call_equivalence():
    close, high, low, _, volume, amount, high_limit, low_limit = _inputs(days=1)
    expected = {
        "intra_segment_volume_share": (("volume",), ("segment", "session_tz", "market")),
        "intra_segment_amount_share": (("amount",), ("segment", "session_tz", "market")),
        "intra_limit_first_hit_time": (("close", "high", "low", "high_limit", "low_limit"), ("side",)),
        "intra_limit_duration": (("close", "high_limit", "low_limit"), ("side",)),
        "intra_limit_reopen_count": (("close", "high_limit", "low_limit"), ("side", "transition")),
    }
    for name, (panels, scalars) in expected.items():
        metadata = OperatorRegistry.get(name).metadata
        assert tuple(metadata.panel_params) == panels
        assert tuple(metadata.scalar_params) == scalars

    for name, panel in (("intra_segment_volume_share", volume), ("intra_segment_amount_share", amount)):
        op = OperatorRegistry.get(name)
        default = op.calculate(panel)
        explicit = op.calculate(panel, "morning", None, "ashare")
        mixed = op.calculate(panel, segment="morning", market="ashare")
        pd.testing.assert_frame_equal(default, explicit)
        pd.testing.assert_frame_equal(default, mixed)
        assert np.isfinite(default.to_numpy()).all()

    first = OperatorRegistry.get("intra_limit_first_hit_time")
    duration = OperatorRegistry.get("intra_limit_duration")
    reopen = OperatorRegistry.get("intra_limit_reopen_count")
    pd.testing.assert_frame_equal(
        first.calculate(close, high, low, high_limit, low_limit, "up"),
        first.calculate(close=close, high=high, low=low, high_limit=high_limit, low_limit=low_limit, side="up"),
    )
    pd.testing.assert_frame_equal(
        duration.calculate(close, high_limit, low_limit, "up"),
        duration.calculate(close=close, high_limit=high_limit, low_limit=low_limit, side="up"),
    )
    pd.testing.assert_frame_equal(
        reopen.calculate(close, high_limit, low_limit, "up", "open"),
        reopen.calculate(close=close, high_limit=high_limit, low_limit=low_limit, side="up", transition="open"),
    )
    assert np.isfinite(first.calculate(close, high=high, high_limit=high_limit).to_numpy()).all()
    assert np.isfinite(duration.calculate(close, high_limit=high_limit).to_numpy()).all()
    assert np.isfinite(reopen.calculate(close, high_limit=high_limit).to_numpy()).all()


def test_final_five_reject_invalid_or_duplicate_scalar_binding():
    _, _, _, _, volume, _, _, _ = _inputs(days=1)
    share = OperatorRegistry.get("intra_segment_volume_share")
    with pytest.raises((TypeError, ValueError)):
        share.calculate(volume, "morning", None, "ashare", market="ashare")
    with pytest.raises((TypeError, ValueError)):
        share.calculate(volume, market=123)


@pytest.mark.parametrize("name,args,kwargs", [
    ("intra_segment_return", (), {"segment": "overnight"}),
    ("intra_realized_semivariance", (), {"side": "bad"}),
    ("intra_extreme_bar_return", (), {"side": "up"}),
    ("intra_amihud", (None,), {"scale": float("nan")}),
    ("intra_limit_reopen_count", (), {"side": "up", "transition": "bad"}),
])
def test_invalid_scalars_rejected(name, args, kwargs):
    close, _, _, _, _, amount, high_limit, _ = _inputs(days=1)
    if name == "intra_amihud":
        args = (close, amount)
    elif name == "intra_limit_reopen_count":
        args = (close,)
        kwargs["high_limit"] = high_limit
    else:
        args = (close,)
    with pytest.raises((TypeError, ValueError)):
        OperatorRegistry.get(name).calculate(*args, **kwargs)
