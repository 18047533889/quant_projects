from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.base import MISSING
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

TARGETS = (
    "cs_multi_resid", "cs_neutralize", "cs_wls_resid", "ts_max_drawdown",
    "ts_nth_value", "ts_partial_corr", "ts_regression_in_sample_resid",
    "ts_regression_intercept", "ts_regression_r2", "ts_regression_resid",
    "ts_regression_resid_mean", "ts_regression_tstat", "ts_time_slope",
    "ts_trend_tstat",
)


def _frames(n=50, p=10):
    rng = np.random.default_rng(614)
    idx = pd.date_range("2025-01-01", periods=n)
    cols = [f"S{i}" for i in range(p)]
    x = pd.DataFrame(rng.normal(size=(n, p)), index=idx, columns=cols)
    z = pd.DataFrame(rng.normal(size=(n, p)), index=idx, columns=cols)
    y = 1.2 + 1.7*x - .4*z + pd.DataFrame(rng.normal(scale=.05, size=(n,p)), index=idx, columns=cols)
    w = pd.DataFrame(rng.uniform(.5, 2, size=(n,p)), index=idx, columns=cols)
    group = pd.DataFrame(np.tile(np.array(["a"]*5+["b"]*5), (n,1)), index=idx, columns=cols)
    level = (1 + x.abs()/100).cumprod()
    return y, x, z, w, group, level


def _call(name, frames, stop=None):
    y,x,z,w,g,level = (f.iloc[:stop] if stop else f for f in frames)
    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    if name == "cs_multi_resid": return op.calculate(y, x, z, min_obs=5)
    if name == "cs_neutralize": return op.calculate(y, (x, z), group=g, weight=w, min_obs=5)
    if name == "cs_wls_resid": return op.calculate(y=y, x=x, weight=w, min_obs=5)
    if name == "ts_max_drawdown": return op.calculate(level, 12, min_periods=5)
    if name == "ts_nth_value": return op.calculate(x, 12, n=2, order="largest", min_periods=5)
    if name == "ts_partial_corr": return op.calculate(x=x, y=y, z=z, window=12, min_periods=6)
    if name == "ts_time_slope": return op.calculate(x, window=12, min_periods=5)
    if name == "ts_trend_tstat": return op.calculate(x=x, window=12, min_periods=5)
    return op.calculate(y, x, 12, min_periods=6, add_intercept=True)


def test_exact_contract_inventory_and_topology():
    assert len(TARGETS) == len(set(TARGETS)) == 14
    for name in TARGETS:
        op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
        assert op is not None
        m = op.metadata
        assert m.panel_params and m.scalar_params
        assert not set(m.panel_params) & set(m.scalar_params)
        for scalar in m.scalar_params:
            spec = m.param_specs[scalar]
            assert spec.param_role is not None
            if scalar in {"min_periods", "add_intercept", "n", "order", "min_obs"}:
                assert spec.default is not MISSING


@pytest.mark.parametrize("name", TARGETS)
def test_each_canonical_finite_shape_prefix_and_invalid_scalar(name):
    frames = _frames()
    out = _call(name, frames)
    assert out.index.equals(frames[0].index) and out.columns.equals(frames[0].columns)
    assert np.isfinite(out.to_numpy()).any(), name
    short = _call(name, frames, 37)
    np.testing.assert_allclose(out.iloc[:37], short, equal_nan=True, rtol=1e-10, atol=1e-10)
    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    if "window" in op.metadata.scalar_params:
        args = [frames[-1]] if name == "ts_max_drawdown" else [frames[1]]
        with pytest.raises((TypeError, ValueError)):
            op.calculate(*args, window=-1)


def test_independent_regression_order_and_drawdown_oracles_and_call_styles():
    y,x,z,w,g,level = _frames()
    intercept = OperatorRegistry.get("ts_regression_intercept", "pandas_numpy", mode="any").calculate(y, x, 20, min_periods=10)
    assert intercept.iloc[-1].mean() == pytest.approx(1.2, abs=.08)
    nth = OperatorRegistry.get("ts_nth_value", "pandas_numpy", mode="any").calculate(x, window=7, n=2, order="largest", min_periods=7)
    assert nth.iloc[-1,0] == pytest.approx(np.sort(x.iloc[-7:,0])[-2])
    dd = OperatorRegistry.get("ts_max_drawdown", "pandas_numpy", mode="any").calculate(level, window=10, min_periods=10)
    v=level.iloc[-10:,0].to_numpy(); expected=np.min(v/np.maximum.accumulate(v)-1)
    assert dd.iloc[-1,0] == pytest.approx(expected)
    op=OperatorRegistry.get("ts_time_slope", "pandas_numpy", mode="any")
    a=op.calculate(x,12,min_periods=5); b=op.calculate(x,window=12,min_periods=5); c=op.calculate(x=x,window=12,min_periods=5)
    np.testing.assert_allclose(a,b,equal_nan=True); np.testing.assert_allclose(b,c,equal_nan=True)


def test_registered_polars_twin_contracts_match_when_present():
    mismatches=[]
    for name in TARGETS:
        pdop=OperatorRegistry.get(name,"pandas_numpy",mode="any")
        plop=OperatorRegistry.get(name,"polars",mode="any")
        if plop is None: continue
        for field in ("param_names","param_specs","panel_params","scalar_params"):
            if getattr(plop.metadata,field) != getattr(pdop.metadata,field):
                mismatches.append((name,field,getattr(plop.metadata,field),getattr(pdop.metadata,field)))
    assert not mismatches, mismatches
