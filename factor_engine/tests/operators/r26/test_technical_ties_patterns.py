# -*- coding: utf-8 -*-
"""R26-076..087: Aroon latest-extreme ties + candlestick tri-state.

* Aroon ties reference the LATEST extreme (plateau);
* candlestick NaN input -> NaN (never confirmed False / 0);
* spinning-top / outside-bar neutral direction -> NaN (not 0 / not +1);
* invalid OHLC geometry -> NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.price_volume.technical_extensions as te


def _pd(vals):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="D")
    return pd.DataFrame(vals, index=idx, columns=["A"])


def test_aroon_uses_latest_extreme():
    # [10,12,12,11] window=4: the latest high is the SECOND 12 (index 2).
    x = _pd([10.0, 12.0, 12.0, 11.0])
    out = te._aroon_component(x, 4, high=True)
    # periods-since = 0 (index 3 is not the high; the high is index 2, one bar ago)
    # Aroon = 100 * (4 - 1) / 4 = 75 (latest-extreme) vs 50 (first-extreme).
    assert out.iloc[-1, 0] == pytest.approx(75.0)


def test_candlestick_nan_input_is_nan():
    O, H, L, C = (_pd([10.0, 10.0, 10.0]), _pd([11.0, 11.0, 11.0]),
                  _pd([9.0, 9.0, 9.0]), _pd([10.0, 10.0, 10.0]))
    C2 = C.copy()
    C2.iloc[1, 0] = np.nan
    out = te._cdl_doji(O, H, L, C2)
    assert np.isnan(out.iloc[1, 0]), "R26-083: NaN input -> NaN, never confirmed False/0"
    assert out.iloc[0, 0] == 1.0


def test_candlestick_invalid_geometry_is_nan():
    # low > high -> geometry invalid
    O = _pd([10.0]); H = _pd([10.0]); L = _pd([11.0]); C = _pd([10.0])
    out = te._cdl_doji(O, H, L, C)
    assert np.isnan(out.iloc[0, 0])


def test_spinning_top_zero_body_neutral():
    # body == 0 everywhere -> spinning top fires with NEUTRAL direction -> NaN
    O, H, L, C = (_pd([10.0, 10.0]), _pd([11.0, 11.0]),
                  _pd([9.0, 9.0]), _pd([10.0, 10.0]))
    out = te._cdl_spinning_top(O, H, L, C)
    assert np.all(np.isnan(out.to_numpy())), "R26-085/086: neutral event must not be +1"


def test_outside_bar_neutral_direction_not_zero():
    # outside bar with open == close -> event present, direction neutral -> NaN
    O = _pd([10.0, 10.0]); H = _pd([11.0, 12.0]); L = _pd([9.0, 8.0]); C = _pd([10.0, 10.0])
    out = te._cdl_outside_bar(O, H, L, C)
    # row1: prev (10,11,9) inside current (10,12,8) -> outside bar, neutral dir
    assert np.isnan(out.iloc[1, 0])
