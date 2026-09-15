"""Final registry checks for the fifteen foundational rolling operators."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES = ("WMA", "ts_decay_linear", "ts_delay", "ts_delta", "ts_ema", "ts_log_return",
         "ts_max", "ts_median", "ts_min", "ts_pct", "ts_skew", "ts_std", "ts_sum",
         "ts_var", "ts_zscore")

@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()

def reference(name, x, window):
    if name in ("WMA", "ts_decay_linear"):
        return x.rolling(window, min_periods=1).apply(
            lambda y: np.dot(y, np.arange(window-len(y)+1, window+1)) /
                      np.arange(window-len(y)+1, window+1).sum(), raw=True)
    if name == "ts_delay":
        return x.shift(window)
    if name == "ts_delta":
        return x - x.shift(window)
    if name == "ts_pct":
        return x / x.shift(window) - 1
    if name == "ts_log_return":
        return np.log(x / x.shift(window))
    if name == "ts_ema":
        return x.ewm(span=window, adjust=False, ignore_na=True, min_periods=1).mean()
    if name == "ts_zscore":
        roll = x.rolling(window, min_periods=1)
        return (x-roll.mean())/roll.std()
    method = {"ts_max":"max", "ts_min":"min", "ts_median":"median", "ts_skew":"skew",
              "ts_std":"std", "ts_sum":"sum", "ts_var":"var"}[name]
    return getattr(x.rolling(window, min_periods=1), method)()

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("name", NAMES)
def test_final_rolling_real_numerics_keywords_prefix_and_contract(name, backend):
    op = OperatorRegistry.get(name, backend, mode="research")
    if op is None:
        pytest.skip(f"{name}: no registered {backend} implementation")
    x = pd.DataFrame({"A": [2.,4.,3.,7.,5.,8.,6.,9.,10.,7.,12.,11.],
                      "B": [3.,2.,5.,4.,8.,6.,9.,7.,11.,10.,14.,12.]})
    window = 5
    source = pl.from_pandas(x) if backend == "polars" else x
    result = op.calculate(source, window)
    if backend == "polars":
        result = result.to_pandas()
    pd.testing.assert_frame_equal(result, reference(name,x,window), atol=1e-9, rtol=1e-9)
    assert np.isfinite(result.to_numpy()).any(), name
    param = "span" if name=="ts_ema" else "n" if name in ("ts_delay","ts_delta") else "d" if name in ("ts_pct","ts_log_return") else "window"
    keyword = op.calculate(x=source, **{param:window})
    if backend=="polars":
        keyword=keyword.to_pandas()
    pd.testing.assert_frame_equal(keyword,result)
    short = op.calculate(source[:9],window)
    if backend=="polars":
        short=short.to_pandas()
    pd.testing.assert_frame_equal(short,result.iloc[:9])
    _, _, verified = _parameter_contract(op, ("x",))
    assert verified, (name,backend)
    for invalid in (True,-1):
        with pytest.raises((ValueError,TypeError)):
            op.calculate(source,invalid)

def test_ema_retains_supported_fractional_alpha():
    x = pd.DataFrame({"A":[2.,5.,3.,8.]})
    op=OperatorRegistry.get("ts_ema","pandas_numpy",mode="research")
    pd.testing.assert_frame_equal(op.calculate(x,.2),x.ewm(alpha=.2,adjust=False,ignore_na=True,min_periods=1).mean())

@pytest.mark.parametrize("params", [{"ddof":-1},{"min_periods":0},{"window":2.5}])
def test_variance_invalid_integer_policy(params):
    op=OperatorRegistry.get("ts_var","pandas_numpy",mode="research")
    with pytest.raises((ValueError,TypeError)):
        op.calculate(pd.DataFrame({"A":[1.,2.,3.]}),**params)

@pytest.mark.parametrize("params", [{"ddof":2},{"nan_policy":"guess"},{"includes_current_bar":1},{"zero_std_policy":"guess"}])
def test_zscore_invalid_policy(params):
    op=OperatorRegistry.get("ts_zscore","pandas_numpy",mode="research")
    with pytest.raises((ValueError,TypeError)):
        op.calculate(pd.DataFrame({"A":[1.,2.,3.]}),**params)


@pytest.mark.parametrize("name", ["ts_std","ts_sum","ts_max","ts_min","ts_median","ts_delta","ts_delay"])
def test_native_legacy_d_alias_is_actually_consumed(name):
    op=OperatorRegistry.get(name,"polars",mode="research")
    x=pl.DataFrame({"A":[2.,4.,3.,7.,5.,8.,6.,9.,10.,7.]})
    expected=op.calculate(x,3).to_pandas()
    pd.testing.assert_frame_equal(op.calculate(x,d=3).to_pandas(),expected)

@pytest.mark.parametrize("name", ["ts_std","ts_sum","ts_max","ts_min","ts_median"])
def test_native_rolling_finite_support_matches_reference(name):
    x=pd.DataFrame({"A":[2.,np.nan,3.,np.inf,5.,8.,-np.inf,9.,10.,7.]})
    pandas_op=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    polars_op=OperatorRegistry.get(name,"polars",mode="research")
    expected=pandas_op.calculate(x,3)
    actual=polars_op.calculate(pl.from_pandas(x),3).to_pandas()
    pd.testing.assert_frame_equal(actual,expected,atol=1e-9,rtol=1e-9)


@pytest.mark.parametrize("ddof",[0,1])
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_std_explicit_ddof_consistent_across_backends(backend,ddof):
    x=pd.DataFrame({"A":[2.,4.,np.nan,3.,8.,np.inf,7.,10.]})
    op=OperatorRegistry.get("ts_std",backend,mode="research")
    frame=pl.from_pandas(x) if backend=="polars" else x
    expected=x.replace([np.inf,-np.inf],np.nan).rolling(3,min_periods=1).std(ddof=ddof)
    actual=op.calculate(x=frame,window=3,ddof=ddof)
    positional=op.calculate(frame,3,ddof)
    legacy=op.calculate(frame,d=3,ddof=ddof)
    for output in (actual,positional,legacy):
        if backend=="polars":
            output=output.to_pandas()
        pd.testing.assert_frame_equal(output,expected,atol=1e-10,rtol=1e-10)
    for invalid in (-1,2,True,0.5):
        with pytest.raises((TypeError,ValueError)):
            op.calculate(frame,3,invalid)
