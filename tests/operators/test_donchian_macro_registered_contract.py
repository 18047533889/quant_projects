import inspect

import numpy as np
import pandas as pd

from factor_engine.api.operator_registry import build_dsl_allowlist
from factor_engine.api.technical_macros_v2 import augment_technical_macros
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def test_compat_macro_matches_registered_legacy_donchian_parameter_order():
    load_all()
    macro = build_dsl_allowlist(surface="compat_research")["donchian_position"]
    registered = OperatorRegistry.get("donchian_position", "pandas_numpy")

    assert tuple(inspect.signature(macro).parameters) == (
        "close", "high", "low", "window"
    )
    assert tuple(registered.metadata.param_names) == (
        "close", "high", "low", "window"
    )


def test_daily_catalog_shape_is_not_reinterpreted_as_high_low_close():
    index = pd.date_range("2026-01-01", periods=7)
    close = pd.DataFrame({"A": [10, 11, 12, 11, 14, 13, 15]}, index=index)
    high = pd.DataFrame({"A": [20, 22, 24, 23, 28, 27, 30]}, index=index)
    low = pd.DataFrame({"A": [2, 3, 4, 3, 6, 5, 7]}, index=index)

    def binary(fn):
        return lambda left, right: fn(left, right)

    noop = lambda *args, **kwargs: None
    base = {name: noop for name in (
        "add subtract multiply divide maximum minimum ts_max ts_min ts_mean "
        "ts_std ts_sum ts_delay ts_ema true_range"
    ).split()}
    base.update(
        add=binary(lambda a, b: a + b),
        subtract=binary(lambda a, b: a - b),
        multiply=binary(lambda a, b: a * b),
        divide=binary(lambda a, b: a / b),
        ts_max=lambda x, window: x.rolling(window, min_periods=window).max(),
        ts_min=lambda x, window: x.rolling(window, min_periods=window).min(),
        ts_delay=lambda x, lag: x.shift(lag),
    )
    macro = augment_technical_macros(base)["donchian_position"]
    actual = macro(close, high, low, 3)
    wrong_old_macro_order = macro(high, low, close, 3)

    registered = OperatorRegistry.get("donchian_position", "pandas_numpy")
    expected = registered.calculate(close, high, low, window=3)
    pd.testing.assert_frame_equal(actual, expected)
    assert not np.allclose(
        actual.to_numpy(), wrong_old_macro_order.to_numpy(), equal_nan=True
    )
