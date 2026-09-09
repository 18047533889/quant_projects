from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
import subprocess
import sys


def _load(canonical: bool = False):
    from factor_engine.cleaned_operators import extreme_tail as module
    return module


PARAMS = dict(window=20, side="upper", tail_fraction=0.25, min_tail_count=3)


def test_all_negative_upper_tail_is_outside_positive_hill_domain():
    et = _load()
    x = pd.DataFrame({"A": -np.arange(20.0, 0.0, -1.0)})
    out = et.TsHillTailIndex().calculate(x, **PARAMS)
    assert np.isnan(out.iloc[-1, 0])


def test_positive_tail_matches_direct_log_ratio_reference_and_scale():
    et = _load()
    values = np.arange(1.0, 21.0)
    u = float(np.quantile(values, 0.75))
    exc = values[values > u]
    expected = float(np.mean(np.log(exc / u)))
    base = et.TsHillTailIndex().calculate(pd.DataFrame({"A": values}), **PARAMS)
    scaled = et.TsHillTailIndex().calculate(
        pd.DataFrame({"A": values * 1e100}), **PARAMS
    )
    assert base.iloc[-1, 0] == pytest.approx(expected)
    assert scaled.iloc[-1, 0] == pytest.approx(expected)


def test_lower_tail_is_exact_upper_tail_of_mirrored_series():
    et = _load()
    values = np.array(
        [-20, -15, -11, -8, -5, -3, -2, -1, 1, 2, 3, 5, 8, 11, 15, 20, 25, 30, 35, 40],
        dtype=float,
    )
    lower = et.TsHillTailIndex().calculate(
        pd.DataFrame({"A": values}), **{**PARAMS, "side": "lower"}
    )
    mirrored = et.TsHillTailIndex().calculate(
        pd.DataFrame({"A": -values}), **PARAMS
    )
    np.testing.assert_allclose(lower, mirrored, equal_nan=True)


def test_zero_threshold_and_insufficient_tail_remain_nan():
    et = _load()
    zero_threshold = np.array([-3.0] + [0.0] * 15 + [1.0] * 4)
    out = et.TsHillTailIndex().calculate(
        pd.DataFrame({"A": zero_threshold}), **PARAMS
    )
    assert np.isnan(out.iloc[-1, 0])
    insufficient = et.TsHillTailIndex().calculate(
        pd.DataFrame({"A": np.arange(1.0, 21.0)}),
        window=20,
        side="upper",
        tail_fraction=0.1,
        min_tail_count=3,
    )
    assert np.isnan(insufficient.iloc[-1, 0])


def test_nextafter_exceedances_are_not_rounded_back_to_ratio_one():
    et = _load()
    above = np.nextafter(1.0, np.inf)
    values = np.array([1.0] * 16 + [above] * 4)
    out = et.TsHillTailIndex().calculate(pd.DataFrame({"A": values}), **PARAMS)
    expected = np.log1p(above - 1.0)
    assert out.iloc[-1, 0] == pytest.approx(expected, rel=0, abs=1e-30)


def test_extreme_finite_dynamic_range_avoids_ratio_overflow_and_scales():
    et = _load()
    values = np.array([1e-300] * 16 + [1e-200, 1.0, 1e200, 1e300])
    u = float(np.quantile(values, 0.75))
    exc = values[values > u]
    expected = float(np.mean(np.log(exc) - np.log(u)))
    base = et.TsHillTailIndex().calculate(pd.DataFrame({"A": values}), **PARAMS)
    scaled = et.TsHillTailIndex().calculate(
        pd.DataFrame({"A": values * 1e-5}), **PARAMS
    )
    assert base.iloc[-1, 0] == pytest.approx(expected)
    assert scaled.iloc[-1, 0] == pytest.approx(expected)


def test_pytest_surface_polars_winner_uses_same_hill_kernel():
    pl = pytest.importorskip("polars")
    from factor_engine.cleaned_operators import load_all
    load_all()
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_hill_tail_index", "polars")
    assert type(op).__module__ == "factor_engine.cleaned_operators.polars_native.ts_advanced_batch5"
    out = op.calculate(pl.DataFrame({"A": -np.arange(20.0, 0.0, -1.0)}), **PARAMS)
    assert np.isnan(out["A"][-1])


def test_active_polars_winner_matches_pandas_across_domain_edges():
    pl = pytest.importorskip("polars")
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    native = OperatorRegistry.get("ts_hill_tail_index", "polars")
    pandas_op = _load().TsHillTailIndex()
    cases = [
        np.arange(1.0, 21.0),
        np.array([1.0] * 16 + [np.nextafter(1.0, np.inf)] * 4),
        np.array([1e-300] * 16 + [1e-200, 1.0, 1e200, 1e300]),
        np.array([-20, -15, -11, -8, -5, -3, -2, -1, 1, 2, 3, 5, 8, 11, 15, 20, 25, 30, 35, 40], dtype=float),
    ]
    for side in ("upper", "lower"):
        for values in cases:
            expected = pandas_op.calculate(pd.DataFrame({"A": values}), **{**PARAMS, "side": side})
            actual = native.calculate(pl.DataFrame({"A": values}), **{**PARAMS, "side": side})
            np.testing.assert_allclose(actual["A"].to_numpy(), expected["A"].to_numpy(), equal_nan=True)


def test_raw_registry_research_mode_keeps_first_registered_batch5():
    pl = pytest.importorskip("polars")
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    op = OperatorRegistry.get("ts_hill_tail_index", "polars", mode="research")
    assert type(op).__module__ == "factor_engine.cleaned_operators.polars_native.ts_advanced_batch5"
    values = np.array([1e-300] * 16 + [1e-200, 1.0, 1e200, 1e300])
    expected = _load().TsHillTailIndex().calculate(pd.DataFrame({"A": values}), **PARAMS)
    actual = op.calculate(pl.DataFrame({"A": values}), **PARAMS)
    np.testing.assert_allclose(actual["A"].to_numpy(), expected["A"].to_numpy(), equal_nan=True)


def test_batch5_is_explicit_python_delegate_and_multistock_long_fails_closed():
    pl = pytest.importorskip("polars")
    from factor_engine.backend.polars_backend_kind import (
        PolarsImplementationKind,
        polars_backend_kind,
    )
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    op = OperatorRegistry.get("ts_hill_tail_index", "polars", mode="research")
    assert polars_backend_kind(op, production_mode=True) == PolarsImplementationKind.UNSUPPORTED
    long_frame = pl.DataFrame(
        {
            "date": [1, 1, 2, 2],
            "stock_code": ["A", "B", "A", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="multi-stock long input"):
        op.calculate(long_frame, **PARAMS)


def test_plain_load_all_fresh_process_binds_polars_dynamics_and_is_domain_correct():
    code = r'''
import numpy as np
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
load_all()
op = OperatorRegistry.get("ts_hill_tail_index", "polars")
assert type(op).__module__ == "factor_engine.cleaned_operators.polars_dynamics"
params = dict(window=20, side="upper", tail_fraction=0.25, min_tail_count=3)
negative = op.calculate(pl.DataFrame({"A": -np.arange(20.0, 0.0, -1.0)}), **params)
assert np.isnan(negative["A"][-1])
values = np.arange(1.0, 21.0)
u = float(np.quantile(values, 0.75))
expected = float(np.mean(np.log(values[values > u] / u)))
actual = op.calculate(pl.DataFrame({"A": values}), **params)["A"][-1]
assert np.isclose(actual, expected)
'''
    subprocess.run([sys.executable, "-c", code], check=True)
