# -*- coding: utf-8 -*-
"""R9-P0-004/005/006 — history-layer contract: no param truncation, DAG-path
composition, per-operator lag-vs-window HistoryTransforms.

The history authority (``runtime.execution_contract``) must:

  * P0-004 — NEVER bind/truncate a parameter itself (``window=5.9`` must not
    silently become 5); an un-resolvable bound value is UNKNOWN history.
  * P0-005 — compose nested lookbacks along DAG PATHS, not max-over-tree:
    ``H(node) = own_history_transform(max(H(children)))``.
  * P0-006 — treat a pure lag as ``value`` and a rolling window as ``value-1``,
    declared per operator (never guessed uniformly from the parameter NAME).

Test cases (from the review):
  (a) ts_mean(ts_delay(x, 20), 60)         -> 20 + (60-1) = 79, NOT 59.
  (b) ts_corr(ts_delay(x, 5), ts_delay(y, 20), 60)
                                          -> max(5, 20) + (60-1) = 79 (additive).
  (c) ts_zscore(ts_mean(x, 20), 60)       -> 19 + (60-1) = 78 (nested additive).
  (d) full-run vs auto-warmup: evaluating each expression on a window of exactly
      ``rows + 1`` bars reproduces the full-run output (and a too-small window
      does NOT, proving the check is not vacuous).
  (e) ``_bound_param`` never truncates a fractional window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.ir.nodes import IRNode
from factor_engine.runtime.execution_contract import (
    _UNKNOWN,
    _bound_param,
    factor_history_requirement,
    history_requirement,
)


# ---------------------------------------------------------------------------
# IR builders
# ---------------------------------------------------------------------------
def _col(name: str) -> IRNode:
    return IRNode(op="column", attrs={"name": name})


def _lit(value: object) -> IRNode:
    return IRNode(op="literal", attrs={"value": value})


def _call(canonical: str, *args: object, **attrs: object) -> IRNode:
    return IRNode(op=canonical, inputs=tuple(args), attrs=dict(attrs))


def _delay(x: object, n: int) -> IRNode:
    return _call("ts_delay", x, _lit(n))


# ---------------------------------------------------------------------------
# Direct-kernel executors for the three expressions (no full engine needed).
# ---------------------------------------------------------------------------
def _get_operator(canonical: str):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry.get(canonical)


def _expr_mean_delay(df: pd.DataFrame, delay_n: int = 20, mean_w: int = 60) -> pd.DataFrame:
    delayed = _get_operator("ts_delay")._calculate_series(df[["x"]], n=delay_n)
    return _get_operator("ts_mean")._calculate_series(delayed, window=mean_w)


def _expr_corr_delay(
    df: pd.DataFrame, n1: int = 5, n2: int = 20, w: int = 60
) -> pd.DataFrame:
    # ts_corr computes per-column rolling correlation between two panels that
    # share the same universe columns — align both series to a common column.
    dx = _get_operator("ts_delay")._calculate_series(
        df[["x"]].rename(columns={"x": "v"}), n=n1
    )
    dy = _get_operator("ts_delay")._calculate_series(
        df[["y"]].rename(columns={"y": "v"}), n=n2
    )
    return _get_operator("ts_corr")._calculate_series(dx, dy, window=w)


def _expr_zscore_mean(df: pd.DataFrame, mean_w: int = 20, z_w: int = 60) -> pd.DataFrame:
    meaned = _get_operator("ts_mean")._calculate_series(df[["x"]], window=mean_w)
    return _get_operator("ts_zscore")._calculate_series(meaned, window=z_w)


def _panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    x = np.sin(np.arange(n) / 7.0) + 0.01 * rng.randn(n)
    y = np.cos(np.arange(n) / 5.0) + 0.01 * rng.randn(n)
    return pd.DataFrame({"x": x, "y": y})


# ---------------------------------------------------------------------------
# (a)/(b)/(c) DAG-path composition — NOT max-over-tree
# ---------------------------------------------------------------------------
def test_mean_of_delay_is_additive_not_max():
    tree = _call("ts_mean", _delay(_col("x"), 20), _lit(60))
    req = factor_history_requirement(tree)
    assert req.kind == "finite"
    assert req.rows == 79  # 20 + (60 - 1)
    assert req.rows != 59  # the old max-over-tree answer


def test_corr_of_delays_is_additive_across_branches():
    tree = _call(
        "ts_corr",
        _delay(_col("x"), 5),
        _delay(_col("y"), 20),
        _lit(60),
    )
    req = factor_history_requirement(tree)
    assert req.kind == "finite"
    assert req.rows == 79  # max(5, 20) + (60 - 1)
    assert req.rows >= 20 + 59


def test_zscore_of_mean_is_nested_additive():
    tree = _call("ts_zscore", _call("ts_mean", _col("x"), _lit(20)), _lit(60))
    req = factor_history_requirement(tree)
    assert req.kind == "finite"
    assert req.rows == 78  # (20 - 1) + (60 - 1) nested additive
    assert req.rows >= 20 + 59 - 1
    assert req.rows != 59  # the old max-over-tree answer


def test_parallel_branches_take_max_not_sum():
    tree = _call("add", _delay(_col("x"), 5), _delay(_col("y"), 20))
    assert factor_history_requirement(tree).rows == 20


# ---------------------------------------------------------------------------
# (d) full-run vs auto-warmup produce identical results
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tree, expr, expected_rows",
    [
        (
            _call("ts_mean", _delay(_col("x"), 20), _lit(60)),
            _expr_mean_delay,
            79,
        ),
        (
            _call("ts_corr", _delay(_col("x"), 5), _delay(_col("y"), 20), _lit(60)),
            _expr_corr_delay,
            79,
        ),
        (
            _call("ts_zscore", _call("ts_mean", _col("x"), _lit(20)), _lit(60)),
            _expr_zscore_mean,
            78,
        ),
    ],
)
def test_full_run_matches_auto_warmup(tree, expr, expected_rows):
    data = _panel(n=400)
    full = expr(data)
    rows = factor_history_requirement(tree).rows
    assert rows == expected_rows

    # Auto-warmup: exactly ``rows`` prior bars + the current bar must reproduce
    # the full-run output at every sampled bar (incl. the last one).
    for t in (rows, rows + 25, len(data) - 1):
        windowed = expr(data.iloc[t - rows: t + 1])
        assert np.allclose(
            windowed.iloc[-1].values, full.iloc[t].values, equal_nan=True
        )

    # Negative control: a window that is 5 bars too small must NOT reproduce
    # the full-run output — proving the warmup bound is load-bearing.
    too_small = max(2, rows - 5)
    short = expr(data.iloc[-(too_small + 1):])
    assert not np.allclose(short.iloc[-1].values, full.iloc[-1].values, equal_nan=True)


# ---------------------------------------------------------------------------
# (e) _bound_param never truncates; unknown history is conservative
# ---------------------------------------------------------------------------
def test_bound_param_does_not_truncate_fractional_window():
    # A fractional window is UNKNOWN history — never silently floored to 5.
    assert _bound_param({"window": 5.9}, "window", {}) is _UNKNOWN
    assert _bound_param({"n": 5.9}, "n", {}) is _UNKNOWN
    # An integral float is a genuine row count.
    assert _bound_param({"window": 20.0}, "window", {}) == 20
    # A real int is unchanged.
    assert _bound_param({"window": 5}, "window", {}) == 5
    # Unbound param -> None.
    assert _bound_param({"window": 20}, "lag", {}) is None
    # bool is never a row count.
    assert _bound_param({"window": True}, "window", {}) is _UNKNOWN


def test_fractional_window_requires_full_history():
    req = history_requirement("ts_mean", {"window": 5.9})
    assert req.is_full_history  # conservative, never a silently-truncated 5


# ---------------------------------------------------------------------------
# single-op authority: lag is `value`, rolling window stays `value - 1`
# ---------------------------------------------------------------------------
def test_lag_operators_need_value_prior_bars_not_value_minus_one():
    assert history_requirement("ts_delay", {"n": 20}).rows == 20
    assert history_requirement("ts_delta", {"n": 20}).rows == 20
    assert history_requirement("ts_pct", {"d": 20}).rows == 20
    assert history_requirement("ts_log_return", {"d": 20}).rows == 20
    assert history_requirement("MOM", {"window": 10}).rows == 10
    assert history_requirement("ROC", {"window": 10}).rows == 10


def test_zero_lag_is_finite_not_full_history():
    # ts_delay(x, 0) is a valid no-op shift: needs 0 prior bars, finite floor 2.
    req = history_requirement("ts_delay", {"n": 0})
    assert not req.is_full_history
    assert req.rows == 2


def test_rolling_window_leaf_keeps_window_minus_one():
    # Backward-compatible leaf behavior: a single rolling window is W - 1.
    assert history_requirement("ts_mean", {"window": 60}).rows == 59
    assert history_requirement("ts_corr", {"window": 60}).rows == 59
    assert history_requirement("ts_zscore", {"window": 60}).rows == 59
    assert history_requirement("ts_ema", {"span": 20}).rows == 19


def test_compound_macd_still_composes():
    req = history_requirement("MACD_line", {"fast": 12, "slow": 26, "signal": 9})
    assert req.rows == (26 - 1) + (9 - 1)  # max(fast, slow)-1 + (signal-1)
