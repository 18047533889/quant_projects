"""Final-registry contracts for higher moments, state, and slice/profile packs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import MISSING, ParamRole
from factor_engine.cleaned_operators.intraday import higher_moments, slice_profile, state_ops
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _finalized_registry():
    load_all()


def _panels():
    days = pd.bdate_range("2024-01-02", periods=30)
    offsets = tuple(range(570, 591)) + tuple(range(780, 801))
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=m) for d in days for m in offsets])
    nslot = len(offsets)
    price, volume, state = {}, {}, {}
    for j, col in enumerate(("A", "B")):
        pv, vv, sv = [], [], []
        for d in range(len(days)):
            k = np.arange(nslot)
            r = .002 * np.sin((k + d + j) / 3) + .001 * np.cos(k / 2)
            r[8] += .08; r[9] -= .06; r[30] += .05
            pv.extend((20 + j + .05*d) * np.exp(np.cumsum(r)))
            vv.extend(100 + 5*k + 20*np.cos((k+d+j)/4))
            sv.extend(((k + d + j) % 4 < 2).astype(float))
        price[col], volume[col], state[col] = pv, vv, sv
    price = pd.DataFrame(price, index=idx)
    volume = pd.DataFrame(volume, index=idx)
    state = pd.DataFrame(state, index=idx)
    event = state.copy()
    high = price * 1.003
    low = price * .997
    return {"close": price, "price": price, "x": price.pct_change().fillna(0),
            "y": volume, "mask_field": volume, "volume": volume,
            "pre_close": price.shift(1).bfill(), "state": state, "state_a": state,
            "state_b": 1-state, "event_mask": event, "high": high, "low": low}


def _scalar_value(name, spec):
    if spec.default is not MISSING:
        if name == "threshold_scale": return 1.0
        if name == "min_coverage": return 0.1
        return spec.default
    if name.startswith("target"): return 1.0
    raise AssertionError(f"no test value for required scalar {name}")


def test_all_42_unique_canonicals_have_explicit_topology_and_finite_real_calls():
    modules = (higher_moments, state_ops, slice_profile)
    names = tuple(dict.fromkeys(n for module in modules for n in module._CANONICALS))
    assert len(names) == 42
    panels = _panels()
    for name in names:
        op = OperatorRegistry.get(name, backend="pandas_numpy")
        md = op.metadata
        assert md.panel_params and md.panel_arity == len(md.panel_params), name
        assert set(md.param_names) == set(md.panel_params) | set(md.scalar_params), name
        assert set(md.scalar_params) == set(md.param_specs), name
        kwargs = {p: panels[p] for p in md.panel_params}
        kwargs.update({p: _scalar_value(p, md.param_specs[p]) for p in md.scalar_params})
        result = op.calculate(**kwargs)
        assert isinstance(result, pd.DataFrame), name
        assert result.index.is_unique and result.columns.is_unique, name
        assert np.isfinite(result.to_numpy(dtype=float)).any(), name
        cutoff = panels[md.panel_params[0]].index.normalize().unique()[20]
        prefix_kwargs = {
            key: (value.loc[value.index < cutoff] if key in md.panel_params else value)
            for key, value in kwargs.items()
        }
        prefix = op.calculate(**prefix_kwargs)
        pd.testing.assert_frame_equal(result.loc[prefix.index], prefix, obj=f"{name} causal prefix")


def test_state_and_slice_named_mixed_calls_and_timezone_are_real():
    p = _panels()
    utc = p["state"].copy()
    utc.index = utc.index.tz_localize("Asia/Shanghai").tz_convert("UTC")
    op = OperatorRegistry.get("intra_state_count", backend="pandas_numpy")
    positional = op.calculate(p["state"], 1.0, 1, "Asia/Shanghai")
    keyword = op.calculate(state=utc, target=1.0, window_days=1, session_tz="Asia/Shanghai")
    pd.testing.assert_frame_equal(positional, keyword)
    with pytest.raises((TypeError, ValueError), match="window_days"):
        op.calculate(p["state"], target=1.0, window_days=0)
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|target"):
        op.calculate(p["state"], 1.0, target=1.0)
    with pytest.raises((TypeError, ValueError), match="unknown|unexpected|undeclared"):
        op.calculate(p["state"], mystery=1)

    pair = OperatorRegistry.get("intra_slice_mask_pair_reduce", backend="pandas_numpy")
    a = pair.calculate(p["x"], p["y"], p["mask_field"], min_pairs=4)
    b = pair.calculate(x=p["x"], y=p["y"], mask_field=p["mask_field"], min_pairs=4)
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a.to_numpy()).any()


def test_threshold_role_and_jump_reference():
    p = _panels()
    op = OperatorRegistry.get("intra_jump_count", backend="pandas_numpy")
    assert op.metadata.param_specs["threshold_scale"].param_role is ParamRole.STATE_THRESHOLD
    result = op.calculate(p["close"], 1.0)
    assert (result > 0).any().any()
    with pytest.raises((TypeError, ValueError), match="threshold_scale"):
        op.calculate(p["close"], 0.0)


def test_replaced_polars_placeholders_execute_as_exact_positional_keyword_bridges():
    affected = (
        "intraday_rv_signature_slope", "intra_multiresolution_resample_reduce",
        "intra_neighbor_event_class", "intra_slice_mask_pair_reduce",
        "intra_slice_mask_reduce", "intra_state_dwell_stats",
        "intra_state_interval_moment", "intra_state_pair_same_slot_corr",
    )
    pandas_panels = _panels()
    panels = {name: pl.DataFrame({"date": list(frame.index.to_pydatetime()), **{
        col: frame[col].to_numpy() for col in frame.columns
    }}) for name, frame in pandas_panels.items()}
    for name in affected:
        op = OperatorRegistry.get(name, backend="polars")
        assert op.__class__.__module__ == "factor_engine.cleaned_operators.rolling_pack", name
        md = op.metadata
        scalar_map = {param: _scalar_value(param, md.param_specs[param]) for param in md.scalar_params}
        positional = op.calculate(*(
            panels[param] if param in md.panel_params else scalar_map[param]
            for param in md.param_names
        ))
        kwargs = {param: panels[param] for param in md.panel_params}
        kwargs.update(scalar_map)
        keyword = op.calculate(**kwargs)
        pandas_kwargs = {param: pandas_panels[param] for param in md.panel_params}
        pandas_kwargs.update(scalar_map)
        pandas_result = OperatorRegistry.get(name, backend="pandas_numpy").calculate(**pandas_kwargs)
        assert isinstance(positional, pl.DataFrame), name
        ordered = ["date", *sorted(col for col in positional.columns if col != "date")]
        positional, keyword = positional.select(ordered), keyword.select(ordered)
        assert positional["date"].to_list() == keyword["date"].to_list(), name
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(pd.to_datetime(positional["date"].to_list())),
            pd.DatetimeIndex(pandas_result.index), obj=f"{name} output clock", exact=False,
        )
        np.testing.assert_allclose(
            positional.drop("date").to_numpy(), keyword.drop("date").to_numpy(),
            equal_nan=True, err_msg=name,
        )
        assert np.isfinite(positional.drop("date").to_numpy()).any(), name
    broken = panels["close"].drop("date")
    with pytest.raises((TypeError, ValueError), match="time|date|axis|index"):
        OperatorRegistry.get("intraday_rv_signature_slope", backend="polars").calculate(broken)
