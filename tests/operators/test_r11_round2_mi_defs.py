# -*- coding: utf-8 -*-
"""R11 round-2 MI / distance operator contract regression tests (review #14-18).

Covers the P0 definition fixes in ``cleaned_operators.nonlinear_dependence``:

* #14 — ``ts_lagged_mutual_information.window`` now counts ALIGNED PAIRS, so the
  raw history requirement is ``window + lag`` bars and the parameter means the
  same thing across lags (window=60,lag=1 and window=60,lag=10 both report 60
  aligned pairs instead of 59 / 50).
* #15 — ``lag`` is a strict-integer ParamSpec (``min=1``) with a relational
  ``lag < window`` gate; ``lag=1.9`` (previously silently truncated to 1) and
  ``lag == window`` raise.
* #16 — ``min_periods`` is a strict lower bound (``min=10``) on
  ts_distance_corr / ts_distance_cov / ts_mutual_information /
  ts_lagged_mutual_information; the old ``max(10, int(min_periods))`` clamp
  (which silently turned 2..10 into 10 — a fake parameter interval) is removed.
* #17 — ``ts_mutual_information.normalized`` is a strict bool (only True/False);
  the old ``bool(normalized)`` truthiness coercion (1/2/"False"/-1) is gone.
* #18 — ``ts_distance_cov`` declares its output unit as the sqrt-product form
  ``sqrt(unit(x)*unit(y))`` so cross-scale mixes (price x turnover vs return x
  amount) are visible in the metadata instead of silently scaling the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.nonlinear_dependence  # noqa: F401  (import-triggered registration)
from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _op(canonical: str):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, canonical
    return op


def _frame(values: np.ndarray, start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame(values, index=idx, columns=["A", "B"])


# ---------------------------------------------------------------------------
# #14 — window counts ALIGNED PAIRS; raw history = window + lag
# ---------------------------------------------------------------------------
def test_lagged_mi_window_counts_aligned_pairs_across_lags():
    rng = np.random.default_rng(0)
    n = 150
    x = _frame(rng.normal(size=(n, 2)))
    y = _frame(rng.normal(size=(n, 2)))
    op = _op("ts_lagged_mutual_information")
    window = 60
    # ``min_periods=window`` is satisfiable only when exactly ``window`` aligned
    # pairs are available.  The redefined contract makes lag=1 and lag=10 both
    # report ``window`` aligned pairs (raw history = window + lag), so BOTH stay
    # finite.  The old code reported window - lag pairs (59 / 50), so this probe
    # would have been all-NaN.
    for lag in (1, 10):
        out = op.calculate(x, y, window=window, lag=lag, min_periods=window)
        assert np.isfinite(out.to_numpy(dtype=float)).any(), (
            f"lag={lag}: window={window} must supply {window} aligned pairs "
            "(raw history = window + lag)"
        )


def test_lagged_mi_raw_history_is_window_plus_lag():
    rng = np.random.default_rng(0)
    n = 150
    x = _frame(rng.normal(size=(n, 2)))
    y = _frame(rng.normal(size=(n, 2)))
    op = _op("ts_lagged_mutual_information")
    window = 60
    # First non-NaN output (0-indexed) sits at row window + lag - 1: the kernel
    # consumes window + lag raw bars before the lag-shift yields window pairs.
    for lag, expected_first in ((1, 60), (10, 69)):
        out = op.calculate(x, y, window=window, lag=lag)
        arr = out.to_numpy(dtype=float)
        first = int(np.where(np.isfinite(arr))[0][0])
        assert first == expected_first, (
            f"lag={lag}: first finite at row {first}, expected {expected_first} "
            "(raw history = window + lag)"
        )


# ---------------------------------------------------------------------------
# #15 — lag is a strict integer (min=1) and must be < window
# ---------------------------------------------------------------------------
def test_lagged_mi_lag_strict_integer_and_relational():
    rng = np.random.default_rng(1)
    n = 80
    x = _frame(rng.normal(size=(n, 2)))
    y = _frame(rng.normal(size=(n, 2)))
    op = _op("ts_lagged_mutual_information")
    # Non-integer finite values raise instead of int()-truncating (1.9 -> 1).
    for bad in (1.9, 2.5, np.float64(3.7)):
        with pytest.raises((OperatorParameterError, ValueError), match="lag"):
            op.calculate(x, y, window=60, lag=bad)
    # lag >= 1: a 0/-1 lag is a synchronous / future reference.
    for bad in (0, -1):
        with pytest.raises((OperatorParameterError, ValueError), match="lag"):
            op.calculate(x, y, window=60, lag=bad)
    # Relational gate: lag == window (or larger) leaves zero aligned pairs.
    with pytest.raises(ValueError, match="lag"):
        op.calculate(x, y, window=60, lag=60)
    with pytest.raises(ValueError, match="lag"):
        op.calculate(x, y, window=60, lag=61)
    # A valid integer lag still runs and reports the same window-pair count.
    out = op.calculate(x, y, window=60, lag=5)
    assert out.shape == x.shape
    assert np.isfinite(out.to_numpy(dtype=float)).any()


# ---------------------------------------------------------------------------
# #16 — min_periods is a strict lower bound (min=10), no silent clamp
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "canonical",
    [
        "ts_distance_corr",
        "ts_distance_cov",
        "ts_mutual_information",
        "ts_lagged_mutual_information",
    ],
)
def test_min_periods_below_10_raises(canonical):
    rng = np.random.default_rng(2)
    n = 80
    x = _frame(rng.normal(size=(n, 2)))
    y = _frame(rng.normal(size=(n, 2)))
    op = _op(canonical)
    # Values 2..10 used to be silently coerced to 10; now they fail at binding.
    with pytest.raises((OperatorParameterError, ValueError), match="min_periods"):
        op.calculate(x, y, window=40, min_periods=5)
    with pytest.raises((OperatorParameterError, ValueError), match="min_periods"):
        op.calculate(x, y, window=40, min_periods=2)
    # The boundary value 10 is legal and the declared default.
    out = op.calculate(x, y, window=40, min_periods=10)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# #17 — normalized is a strict bool (only True/False)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad", [1, 0, 2, -1, 1.0, "False", "true", "0", np.bool_(True)]
)
def test_mi_normalized_strict_bool(bad):
    rng = np.random.default_rng(3)
    n = 80
    x = _frame(rng.normal(size=(n, 2)))
    y = _frame(rng.normal(size=(n, 2)))
    op = _op("ts_mutual_information")
    with pytest.raises((OperatorParameterError, ValueError), match="boolean"):
        op.calculate(x, y, window=40, normalized=bad)
    # Both strict booleans are accepted.
    out_true = op.calculate(x, y, window=40, normalized=True)
    out_false = op.calculate(x, y, window=40, normalized=False)
    assert out_true.shape == x.shape
    assert out_false.shape == x.shape
    assert np.isfinite(out_true.to_numpy(dtype=float)).any()
    assert np.isfinite(out_false.to_numpy(dtype=float)).any()


def test_mi_normalized_accepts_bool_literals():
    rng = np.random.default_rng(4)
    n = 80
    x = _frame(rng.normal(size=(n, 2)))
    y = _frame(rng.normal(size=(n, 2)))
    op = _op("ts_mutual_information")
    assert op.metadata.param_specs["normalized"].dtype is bool
    # Strict bool type: bool is a subclass of int, so a plain ``int`` input must
    # NOT be accepted through the bool path.
    out = op.calculate(x, y, window=40, normalized=True)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# #18 — ts_distance_cov output unit is the sqrt-product form
# ---------------------------------------------------------------------------
def test_distance_cov_unit_is_sqrt_product():
    op = _op("ts_distance_cov")
    assert op.metadata.output_unit == "sqrt(unit(x)*unit(y))"
    assert any(t == "unit:sqrt(unit(x)*unit(y))" for t in op.metadata.tags)
    # ts_distance_corr is a dimensionless ratio and stays unchanged.
    corr = _op("ts_distance_corr")
    assert corr.metadata.output_unit is None
    assert any(t == "unit:ratio" for t in corr.metadata.tags)
