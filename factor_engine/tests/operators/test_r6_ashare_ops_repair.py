"""Independent finalized-registry checks for the repaired A-share operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.market.price_basis import PriceBasis, PriceBasisMismatchError


NAMES = (
    "float_share_ratio",
    "free_float_share_ratio",
    "true_turnover_rate",
    "benchmark_relative_price",
    "limit_up_close",
    "limit_down_close",
)


def _frame(values):
    return pd.DataFrame({"A": values}, index=pd.date_range("2024-01-02", periods=len(values)), dtype=float)


def _cases():
    numerator = _frame([2.0, 3.0, np.nan, 4.0])
    denominator = _frame([4.0, 0.0, 5.0, 8.0])
    upper = _frame([10.0] * 4)
    lower = _frame([8.0] * 4)
    return {
        "float_share_ratio": ((numerator, denominator), {"float_shares": numerator, "total_shares": denominator}),
        "free_float_share_ratio": ((numerator, denominator), {"free_float_shares": numerator, "total_shares": denominator}),
        "true_turnover_rate": ((numerator, denominator), {"volume": numerator, "free_float_shares": denominator}),
        "benchmark_relative_price": ((numerator, denominator), {"price": numerator, "benchmark_price": denominator}),
        "limit_up_close": ((_frame([10.0, 9.994, 9.0, np.nan]), upper), {"close": _frame([10.0, 9.994, 9.0, np.nan]), "upper_limit": upper}),
        "limit_down_close": ((_frame([8.0, 8.006, 9.0, np.nan]), lower), {"close": _frame([8.0, 8.006, 9.0, np.nan]), "lower_limit": lower}),
    }


def test_finalized_pandas_owners_have_exact_topology():
    load_all()
    expected = {
        "float_share_ratio": (("float_shares", "total_shares"), ()),
        "free_float_share_ratio": (("free_float_shares", "total_shares"), ()),
        "true_turnover_rate": (("volume", "free_float_shares"), ()),
        "benchmark_relative_price": (("price", "benchmark_price"), ()),
        "limit_up_close": (("close", "upper_limit"), ("tick_tolerance",)),
        "limit_down_close": (("close", "lower_limit"), ("tick_tolerance",)),
    }
    for name, (panels, scalars) in expected.items():
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert type(op).__module__.endswith("ashare.ops")
        assert tuple(op.metadata.panel_params) == panels
        assert tuple(op.metadata.scalar_params) == scalars
        assert op.metadata.panel_arity == len(panels)


def test_default_keyword_positional_and_prefix_equivalence():
    load_all()
    for name, (args, kwargs) in _cases().items():
        op = OperatorRegistry.get(name)
        positional = op.calculate(*args)
        keyword = op.calculate(**kwargs)
        pd.testing.assert_frame_equal(positional, keyword, obj=name)
        if name.startswith("limit_"):
            explicit = op.calculate(*args, 0.005)
            pd.testing.assert_frame_equal(positional, explicit, obj=f"{name}: default")
        prefix = op.calculate(*(panel.iloc[:2] for panel in args))
        pd.testing.assert_frame_equal(prefix, positional.iloc[:2], obj=f"{name}: prefix")


def test_independent_numeric_oracle_and_missingness():
    load_all()
    cases = _cases()
    ratio_expected = np.asarray([0.5, np.nan, np.nan, 0.5])
    for name in NAMES[:4]:
        actual = OperatorRegistry.get(name).calculate(*cases[name][0])["A"].to_numpy()
        np.testing.assert_allclose(actual, ratio_expected, equal_nan=True, err_msg=name)
    up = OperatorRegistry.get("limit_up_close").calculate(*cases["limit_up_close"][0])
    down = OperatorRegistry.get("limit_down_close").calculate(*cases["limit_down_close"][0])
    np.testing.assert_allclose(up["A"], [1.0, 0.0, 0.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(down["A"], [1.0, 0.0, 0.0, np.nan], equal_nan=True)


def test_limit_contract_rejects_bad_tolerance_and_adjusted_prices():
    load_all()
    close, upper = _cases()["limit_up_close"][0]
    op = OperatorRegistry.get("limit_up_close")
    for bad in (-0.001, np.nan, np.inf):
        with pytest.raises((TypeError, ValueError)):
            op.calculate(close, upper, bad)
    with pytest.raises(PriceBasisMismatchError):
        op._calculate_series(
            close, upper,
            price_basis={"close": PriceBasis.ADJUSTED, "upper_limit": PriceBasis.RAW_OFFICIAL_LIMIT},
        )
