"""Final backend oracles for audited product, MAD and grouped percentiles."""
from decimal import Decimal
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()

def frame(values):
    a = np.asarray(values)
    if a.ndim == 1:
        a = a[:, None]
    return pd.DataFrame(a, index=pd.date_range("2025-01-01", periods=len(a), tz="Asia/Hong_Kong"),
        columns=list("ABCDEF")[:a.shape[1]])

def run(name, backend, x, **kwargs):
    op = OperatorRegistry.get(name, backend, mode="research")
    assert _parameter_contract(op, op.metadata.panel_params)[2]
    if backend == "polars":
        convert = lambda v: pl.from_pandas(v.rename_axis("date").reset_index()) if isinstance(v, pd.DataFrame) else v
        x = convert(x)
        kwargs = {k:convert(v) for k,v in kwargs.items()}
    out = op.calculate(x=x, **kwargs)
    return out.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else out

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_product_extreme_intermediates_nan_inf_and_boolean_policy(backend):
    values = [1e308, 1e308, 1e-308, -1e-308]
    exact = Decimal(1)
    for v in values:
        exact *= Decimal.from_float(v)
    out = run("ts_product", backend, frame(values), window=4)
    assert out.iloc[-1,0] == pytest.approx(float(exact), rel=1e-14)
    assert np.isnan(run("ts_product", backend, frame([1e308, 1e308]), window=2).iloc[-1,0])
    assert run("ts_product", backend, frame([1e308, 0.]), window=2).iloc[-1,0] == 0
    assert np.isnan(run("ts_product", backend, frame([0., np.inf]), window=2).iloc[-1,0])
    assert run("ts_product", backend, frame([2., np.nan, -3.]), window=3, min_periods=2).iloc[-1,0] == -6
    assert np.isnan(run("ts_product", backend, frame([2., np.nan, -3.]), window=3, min_periods=2, skipna=False).iloc[-1,0])
    for bad in (1, "yes"):
        with pytest.raises((TypeError, ValueError)):
            run("ts_product", backend, frame([1.,2.]), skipna=bad)

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_mad_wide_median_and_strict_controls(backend):
    x=frame([-1e308, -1e308, 1e308, 1e308])
    got=run("ts_mad", backend, x, window=4)
    assert got.iloc[-1,0] == pytest.approx(1e308)
    assert run("ts_mad", backend, x, window=4, scale=0).iloc[-1,0] == 0
    for bad in (-1, True, np.inf):
        with pytest.raises((ValueError, TypeError)):
            run("ts_mad", backend, x, scale=bad)
    for name in ("ts_product", "ts_mad"):
        for kwargs in (dict(window=True), dict(window=3.5), dict(window=2,min_periods=3)):
            with pytest.raises((ValueError, TypeError)):
                run(name, backend, x, **kwargs)
        defaults=run(name, backend, x, window=20, min_periods=None)
        np.testing.assert_allclose(run(name, backend, x), defaults, equal_nan=True)

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_group_true_labels_finite_support_and_missing_policy(backend):
    x=frame([[3.,1.,np.inf,9.,7.,5.]])
    groups=frame([["g","g","g","h","h","h"]])
    out=run("group_percentile",backend,x,group=groups,p=.5)
    np.testing.assert_allclose(out,[[1.,0.,np.nan,1.,0.,0.]],equal_nan=True)
    bottom=run("group_percentile",backend,x,group=groups,p=.5,side="bottom")
    np.testing.assert_allclose(bottom,[[0.,1.,np.nan,0.,0.,1.]],equal_nan=True)
    with pytest.raises((ValueError,TypeError)):
        run("group_percentile",backend,x)
    assert run("group_percentile",backend,x,missing_group_policy="null").isna().all().all()
    with pytest.raises((ValueError,TypeError)):
        run("group_percentile",backend,x,group=groups.rename(columns={"A":"Z"}))
    for bad in (0., True, np.inf):
        with pytest.raises((ValueError,TypeError)):
            run("group_percentile",backend,x,group=groups,p=bad)
