# -*- coding: utf-8 -*-
"""R11 #144/#145: conditional/event operators enforce ConditionBool and censor
unknown event states.

- ``ts_count_if`` / ``ts_sum_if`` / ``ts_mean_if`` / ``ts_std_if`` /
  ``ts_last_if`` / ``ts_days_since`` / ``ts_true_streak`` reject any finite
  condition value outside {0, 1} (NaN = missing) instead of silently coercing a
  NumericSeries into a truthy boolean.
- ``ts_days_since`` / ``ts_true_streak`` treat an unknown (NaN) event state as
  censored: output NaN AND reset the running state certainty.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.common.daily_panel import (
    ts_count_if,
    ts_days_since,
    ts_last_if,
    ts_mean_if,
    ts_std_if,
    ts_sum_if,
    ts_true_streak,
)


def _frame(values) -> pd.DataFrame:
    return pd.DataFrame({"A": list(values)})


def _x5() -> pd.DataFrame:
    return _frame([1.0, 2.0, 3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# ISSUE 1 — every finite condition value must be in {0, 1} (NaN = missing).
# ---------------------------------------------------------------------------

_CASES: list[pytest.mark.structures.ParameterSet] = [
    pytest.param(
        lambda: ts_count_if(_frame([1.0, 1.0, 0.0, 1.0, 1.0]), 3, 2),
        [np.nan, 2.0, 2.0, 2.0, 2.0],
        lambda b: ts_count_if(_frame([1.0, b, 0.0, 1.0, 1.0]), 3, 2),
        id="ts_count_if",
    ),
    pytest.param(
        lambda: ts_sum_if(_x5(), _frame([1.0, 0.0, 1.0, 1.0, np.nan]), 2, 1),
        [1.0, 1.0, 3.0, 7.0, 4.0],
        lambda b: ts_sum_if(_x5(), _frame([1.0, b, 1.0, 1.0, np.nan]), 2, 1),
        id="ts_sum_if",
    ),
    pytest.param(
        lambda: ts_mean_if(_x5(), _frame([1.0, 1.0, 0.0, 1.0, 1.0]), 3, 2),
        [np.nan, 1.5, 1.5, 3.0, 4.5],
        lambda b: ts_mean_if(_x5(), _frame([1.0, b, 0.0, 1.0, 1.0]), 3, 2),
        id="ts_mean_if",
    ),
    pytest.param(
        lambda: ts_std_if(_x5(), _frame([1.0, 1.0, 1.0, 1.0, 1.0]), 3, 2, 1),
        [np.nan, 0.7071067811865476, 1.0, 1.0, 1.0],
        lambda b: ts_std_if(_x5(), _frame([1.0, b, 1.0, 1.0, 1.0]), 3, 2, 1),
        id="ts_std_if",
    ),
    pytest.param(
        lambda: ts_last_if(_x5(), _frame([1.0, 0.0, 1.0, np.nan, 1.0]), 3),
        [1.0, 1.0, 3.0, 3.0, 5.0],
        lambda b: ts_last_if(_x5(), _frame([1.0, b, 1.0, np.nan, 1.0]), 3),
        id="ts_last_if",
    ),
    pytest.param(
        lambda: ts_days_since(_frame([1.0, 0.0, 0.0, 1.0, 0.0])),
        [0.0, 1.0, 2.0, 0.0, 1.0],
        lambda b: ts_days_since(_frame([1.0, b, 0.0, 1.0, 0.0])),
        id="ts_days_since",
    ),
    pytest.param(
        lambda: ts_true_streak(_frame([1.0, 1.0, 0.0, 1.0, 1.0])),
        [1.0, 2.0, 0.0, 1.0, 2.0],
        lambda b: ts_true_streak(_frame([1.0, b, 0.0, 1.0, 1.0])),
        id="ts_true_streak",
    ),
]


@pytest.mark.parametrize("valid_call, expected, bad_builder", _CASES)
def test_condition_bool_rejected_or_accepted(
    valid_call: Callable[[], pd.DataFrame],
    expected: list[float],
    bad_builder: Callable[[float], pd.DataFrame],
) -> None:
    # A nonzero non-boolean numeric (e.g. -0.03, 5.0) must raise, never be
    # silently coerced into a truthy boolean.
    for bad_value in (-0.03, 5.0):
        with pytest.raises(ValueError, match="ConditionBool"):
            bad_builder(bad_value)
    # The same call with a {0, 1, NaN} condition succeeds and is computed.
    result = valid_call()
    np.testing.assert_allclose(
        result["A"].to_numpy(dtype=float),
        np.asarray(expected, dtype=float),
        equal_nan=True,
    )


def test_boolean_condition_is_accepted() -> None:
    cond = _frame([True, False, True])
    result = ts_true_streak(cond)
    np.testing.assert_allclose(result["A"].to_numpy(dtype=float), [1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# ISSUE 2 — NaN event state is unknown: censored output AND lost certainty.
# ---------------------------------------------------------------------------


def test_days_since_censors_unknown_event_state() -> None:
    cond = _frame([1.0, np.nan, 0.0, 1.0, 0.0])
    result = ts_days_since(cond)
    # Row 1 (NaN): output NaN.  The NaN also destroys certainty, so row 2 (a
    # confirmed 0) can no longer claim a distance and stays NaN.  Row 3 (1)
    # rebuilds the state -> 0; row 4 (0) -> 1.
    np.testing.assert_allclose(
        result["A"].to_numpy(dtype=float),
        [0.0, np.nan, np.nan, 0.0, 1.0],
        equal_nan=True,
    )


def test_days_since_control_without_nan_is_uncensored() -> None:
    cond = _frame([1.0, 0.0, 0.0, 1.0, 0.0])
    result = ts_days_since(cond)
    np.testing.assert_allclose(
        result["A"].to_numpy(dtype=float),
        [0.0, 1.0, 2.0, 0.0, 1.0],
        equal_nan=True,
    )


def test_days_since_max_lookback_cap_still_applies() -> None:
    cond = _frame([1.0, 0.0, 0.0, 0.0])
    result = ts_days_since(cond, max_lookback=2)
    np.testing.assert_allclose(
        result["A"].to_numpy(dtype=float),
        [0.0, 1.0, np.nan, np.nan],
        equal_nan=True,
    )


def test_true_streak_censors_and_resets_on_unknown_state() -> None:
    cond = _frame([1.0, 1.0, np.nan, 1.0, 0.0, 1.0])
    result = ts_true_streak(cond)
    # Row 2 (NaN): output NaN AND the streak is reset, so row 3 (a confirmed 1)
    # restarts at 1.  Row 4 (a confirmed 0) emits 0.
    np.testing.assert_allclose(
        result["A"].to_numpy(dtype=float),
        [1.0, 2.0, np.nan, 1.0, 0.0, 1.0],
        equal_nan=True,
    )
