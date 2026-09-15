"""Finalized-registry and numerical-oracle coverage for flow-impact repairs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


NAMES = (
    "intraday_bvc_imbalance", "intraday_impact_beta",
    "intraday_impact_asymmetry", "intraday_return_wasserstein_shift",
    "micro_bvc_vpin",
)


def _panel(values, *, days=1):
    per_day = len(values) // days
    indices = []
    for d in range(days):
        indices.extend(pd.date_range(pd.Timestamp("2024-01-02") + pd.Timedelta(days=d), periods=per_day, freq="min"))
    return pd.DataFrame({"A": values}, index=pd.DatetimeIndex(indices), dtype=float)


def _backend(frame, backend):
    if backend == "pandas_numpy":
        return frame
    import polars as pl
    out = frame.copy(); out.index.name = "date"
    return pl.from_pandas(out.reset_index())


def _values(frame):
    if isinstance(frame, pd.DataFrame):
        return frame["A"].to_numpy(dtype=float)
    return frame["A"].to_numpy()


def _op(name, backend):
    return OperatorRegistry.get(name, backend, mode="any")


def test_fresh_registry_exact_topology_defaults_and_roles_every_backend():
    load_all()
    expected = {
        "intraday_bvc_imbalance": (("close", "volume", "locked"), {"scale_window": 20}),
        "intraday_impact_beta": (("returns", "flow"), {"min_periods": 20}),
        "intraday_impact_asymmetry": (("returns", "flow"), {"min_periods": 20}),
        "intraday_return_wasserstein_shift": (("returns",), {"lookback_days": 10}),
        "micro_bvc_vpin": (("close", "volume"), {"scale_window": 40, "bucket_count": 20}),
    }
    for name, (panels, defaults) in expected.items():
        assert set(OperatorRegistry.backends_for(name)) == {"pandas_numpy", "polars"}
        for backend in OperatorRegistry.backends_for(name):
            meta = _op(name, backend).metadata
            assert tuple(meta.panel_params) == panels
            assert meta.panel_arity == len(panels)
            assert tuple(meta.scalar_params) == tuple(defaults)
            assert {k: v.default for k, v in meta.param_specs.items()} == defaults
            assert all(v.dtype is int and v.searchable is False for v in meta.param_specs.values())


def test_all_backends_call_forms_prefix_and_locked_optional_semantics():
    load_all()
    close = _panel(np.exp(np.cumsum(np.r_[0.0, np.linspace(-0.03, 0.04, 79)])), days=2)
    volume = _panel(np.linspace(10.0, 89.0, 80), days=2)
    flow = _panel(np.tile(np.linspace(-2.0, 2.0, 40), 2), days=2)
    returns = _panel(0.3 + 1.7 * flow["A"].to_numpy(), days=2)
    locked = _panel(np.zeros(80), days=2); locked.iloc[-1, 0] = 1.0
    calls = {
        "intraday_bvc_imbalance": (close, volume, (2, locked), {"scale_window": 2, "locked": locked}),
        "intraday_impact_beta": (returns, flow, (5,), {"min_periods": 5}),
        "intraday_impact_asymmetry": (returns, flow, (6,), {"min_periods": 6}),
        "intraday_return_wasserstein_shift": (returns, None, (2,), {"lookback_days": 2}),
        "micro_bvc_vpin": (close, volume, (2, 4), {"scale_window": 2, "bucket_count": 4}),
    }
    for name, (a, b, scalars, kwargs) in calls.items():
        for backend in OperatorRegistry.backends_for(name):
            ba = _backend(a, backend)
            bb = _backend(b, backend) if b is not None else None
            pos_scalars = tuple(_backend(x, backend) if isinstance(x, pd.DataFrame) else x for x in scalars)
            kw = {k: (_backend(v, backend) if isinstance(v, pd.DataFrame) else v) for k, v in kwargs.items()}
            pos = _op(name, backend).calculate(*((ba,) if bb is None else (ba, bb)), *pos_scalars)
            keyed = _op(name, backend).calculate(**{a_name: ba for a_name in (_op(name, backend).metadata.panel_params[:1])}, **({} if bb is None else {_op(name, backend).metadata.panel_params[1]: bb}), **kw)
            np.testing.assert_allclose(_values(pos), _values(keyed), equal_nan=True)
            cut = 40
            p_args = ((_backend(a.iloc[:cut], backend),) if b is None else (_backend(a.iloc[:cut], backend), _backend(b.iloc[:cut], backend)))
            p_scalars = tuple(_backend(x.iloc[:cut], backend) if isinstance(x, pd.DataFrame) else x for x in scalars)
            prefix = _op(name, backend).calculate(*p_args, *p_scalars)
            np.testing.assert_allclose(_values(prefix), _values(pos)[:1], equal_nan=True)

        if name == "intraday_bvc_imbalance":
            plain = _op(name, "pandas_numpy").calculate(close, volume, scale_window=2)
            explicit_none = _op(name, "pandas_numpy").calculate(close, volume, scale_window=2, locked=None)
            np.testing.assert_allclose(_values(plain), _values(explicit_none), equal_nan=True)
            assert not np.allclose(_values(plain), _values(_op(name, "pandas_numpy").calculate(close, volume, 2, locked)), equal_nan=True)


@pytest.mark.parametrize("name,param,bad", [
    ("intraday_bvc_imbalance", "scale_window", 1),
    ("intraday_bvc_imbalance", "scale_window", 2.5),
    ("intraday_impact_beta", "min_periods", 4),
    ("intraday_impact_asymmetry", "min_periods", np.nan),
    ("intraday_return_wasserstein_shift", "lookback_days", np.inf),
    ("micro_bvc_vpin", "bucket_count", 1.5),
])
def test_invalid_scalars_fail_closed_on_every_backend(name, param, bad):
    load_all()
    x = _panel(np.arange(40.0) + 100.0)
    y = _panel(np.arange(40.0) + 1.0)
    for backend in OperatorRegistry.backends_for(name):
        args = [_backend(x, backend)]
        if name != "intraday_return_wasserstein_shift":
            args.append(_backend(y, backend))
        with pytest.raises(Exception):
            _op(name, backend).calculate(*args, **{param: bad})


def test_alignment_and_negative_volume_fail_closed():
    load_all()
    x = _panel(np.arange(40.0) + 100.0)
    y = _panel(np.arange(40.0) + 1.0)
    shifted = y.copy(); shifted.index = shifted.index + pd.Timedelta(minutes=1)
    for name in ("intraday_impact_beta", "intraday_impact_asymmetry"):
        with pytest.raises(ValueError, match="misaligned|same index and columns"):
            _op(name, "pandas_numpy").calculate(x, shifted, min_periods=5)
    for name in ("intraday_bvc_imbalance", "micro_bvc_vpin"):
        negative = y.copy(); negative.iloc[3, 0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            _op(name, "pandas_numpy").calculate(x, negative)


def test_independent_beta_asymmetry_bvc_and_wasserstein_oracles():
    load_all()
    q = np.r_[np.linspace(-4, -1, 15), np.linspace(1, 4, 15)]
    beta = _op("intraday_impact_beta", "pandas_numpy").calculate(_panel(0.7 + 2.25*q), _panel(q), min_periods=5)
    np.testing.assert_allclose(_values(beta), [2.25], atol=1e-12)
    r = np.where(q > 0, 0.2 + 3.0*q, -0.4 + 1.0*q)
    asym = _op("intraday_impact_asymmetry", "pandas_numpy").calculate(_panel(r), _panel(q), min_periods=6)
    np.testing.assert_allclose(_values(asym), [(3.0-1.0)/(3.0+1.0+1e-12)], atol=1e-12)

    prices = np.exp(np.cumsum([0.0, 0.1, -0.2, 0.3, -0.1]))
    vols = np.array([3., 5., 7., 11., 13.])
    rets = np.r_[np.nan, np.diff(np.log(prices))]
    scales = np.array([np.nan, np.nan] + [np.std(rets[max(0, i-1):i+1][np.isfinite(rets[max(0, i-1):i+1])]) for i in range(2, 5)])
    classified = np.isfinite(rets) & np.isfinite(scales)
    expected_bvc = np.sum(vols[classified] * (2*norm.cdf(rets[classified]/(scales[classified]+1e-12))-1)) / np.sum(vols[classified])
    actual_bvc = _op("intraday_bvc_imbalance", "pandas_numpy").calculate(_panel(prices), _panel(vols), scale_window=2)
    np.testing.assert_allclose(_values(actual_bvc), [expected_bvc], atol=1e-12)

    hist = np.linspace(-2, 2, 30); today = np.linspace(-1, 3, 30)
    data = _panel(np.r_[hist, today], days=2)
    actual_w = _op("intraday_return_wasserstein_shift", "pandas_numpy").calculate(data, lookback_days=2)
    grid = np.linspace(0, 1, 512)
    expected_w = np.mean(np.abs(np.quantile(today, grid)-np.quantile(hist, grid))) / np.median(np.abs(hist-np.median(hist)))
    np.testing.assert_allclose(_values(actual_w), [np.nan, expected_w], equal_nan=True, atol=1e-12)


def test_public_vpin_independent_bucket_split_oracle():
    load_all()
    prices = np.exp(np.cumsum([0.0, 0.1, -0.2, 0.3, -0.1, 0.2]))
    vols = np.array([2., 3., 5., 7., 11., 13.])
    rets = np.r_[np.nan, np.diff(np.log(prices))]
    scales = np.full(6, np.nan)
    for i in range(2, 6):
        scales[i] = np.std(rets[i-1:i+1])
    mask = np.isfinite(scales)
    cvol = np.where(mask, vols, 0.0)
    flow = np.where(mask, vols*(2*norm.cdf(np.where(mask, rets/(scales+1e-12), 0.0))-1), 0.0)
    total = cvol.sum(); target = total/3; bucket_flows=[]; current=0.; capacity=target
    for vm, fm in zip(cvol, flow):
        left = vm
        while left > 0:
            take = min(left, capacity); current += fm*take/vm; left -= take; capacity -= take
            if capacity <= 1e-14:
                bucket_flows.append(current); current=0.; capacity=target
    if capacity < target:
        bucket_flows.append(current)
    expected = np.sum(np.abs(bucket_flows))/total
    actual = _op("micro_bvc_vpin", "pandas_numpy").calculate(_panel(prices), _panel(vols), scale_window=2, bucket_count=3)
    np.testing.assert_allclose(_values(actual), [expected], atol=1e-12)
