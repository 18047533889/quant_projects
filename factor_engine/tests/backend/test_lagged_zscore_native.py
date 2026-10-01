"""Parity and execution-shape checks for native long-panel lagged z-score."""
import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.long_smoothing import lagged_zscore


def _reference(frame, *, window, min_periods, ddof):
    out = np.full(len(frame), np.nan, dtype=np.float64)
    history = {}
    required = max(1, min_periods)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for row, (asset, current) in enumerate(zip(frame.asset_id, frame.value)):
            if pd.isna(asset):
                continue
            prior = history.setdefault(asset, [])[-window:]
            finite = np.asarray([v for v in prior if np.isfinite(v)], dtype=np.float64)
            if finite.size >= required and finite.size > ddof:
                mean = finite.mean()
                std = finite.std(ddof=0) * np.sqrt(finite.size / (finite.size - ddof))
                out[row] = (float(current) - mean) / std
            history[asset].append(float(current))
    return out


def _assert_ieee_equal(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=5e-14, atol=0.0)


@pytest.mark.parametrize("window,min_periods", [(1, 1), (3, 1), (3, 3)])
@pytest.mark.parametrize("ddof", [-1, 0, 1, 2])
def test_native_lagged_zscore_matches_reference_across_window_contracts(
    window, min_periods, ddof
):
    frame = pd.DataFrame({
        "asset_id": ["A", "B"] * 5,
        "date": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "value": [2.0, 10.0, 2.0, 12.0, 2.0, 14.0, 3.0, 14.0, 4.0, 16.0],
    }, index=[5, 5, 3, 3, 5, 5, 3, 3, 5, 5])
    actual = lagged_zscore(frame, window=window, min_periods=min_periods, ddof=ddof)
    expected = _reference(frame, window=window, min_periods=min_periods, ddof=ddof)
    _assert_ieee_equal(actual.to_numpy(), expected)
    assert actual.index.equals(frame.index)


def test_native_lagged_zscore_preserves_nonfinite_current_and_sanitizes_history():
    frame = pd.DataFrame({
        "asset_id": ["A", "B", "A", "B", "A", "B", "A", "B", "A"],
        "date": [0, 0, 1, 1, 2, 2, 3, 3, 4],
        "value": [2.0, 10.0, 2.0, 12.0, np.nan, 14.0, np.inf, -np.inf, 5.0],
    }, index=[4, 4, 2, 2, 4, 4, 2, 2, 4])
    actual = lagged_zscore(frame, window=3, min_periods=1, ddof=0)
    expected = _reference(frame, window=3, min_periods=1, ddof=0)
    _assert_ieee_equal(actual.to_numpy(), expected)


def test_native_lagged_zscore_preserves_zero_std_ieee_results():
    frame = pd.DataFrame({
        "asset_id": ["A"] * 4,
        "date": [0, 1, 2, 3],
        "value": [2.0, 2.0, 3.0, 2.0],
    })
    actual = lagged_zscore(frame, window=1, min_periods=1, ddof=0).to_numpy()
    expected = _reference(frame, window=1, min_periods=1, ddof=0)
    _assert_ieee_equal(actual, expected)
    assert np.isnan(actual[1])
    assert np.isposinf(actual[2])


def test_lagged_zscore_is_causal_under_future_perturbation():
    frame = pd.DataFrame({
        "asset_id": ["A"] * 6,
        "date": list(range(6)),
        "value": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0],
    })
    changed = frame.copy()
    changed.loc[4:, "value"] = [1e100, -1e100]
    before = lagged_zscore(frame, window=3, min_periods=1, ddof=1).to_numpy()
    after = lagged_zscore(changed, window=3, min_periods=1, ddof=1).to_numpy()
    _assert_ieee_equal(before[:4], after[:4])


def test_zscore_arithmetic_is_a_native_polars_expression_not_a_python_kernel():
    from factor_engine.backend import native_long_rolling_moments as moments
    from factor_engine.backend import long_smoothing

    collector_tree = ast.parse(inspect.getsource(moments.collect_lagged_moments))
    zscore_aliases = [
        node for node in ast.walk(collector_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "alias"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "_zscore"
    ]
    assert len(zscore_aliases) == 1
    expression = zscore_aliases[0].func.value
    assert isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Div)
    assert isinstance(expression.left, ast.BinOp) and isinstance(expression.left.op, ast.Sub)
    called_attributes = {
        node.func.attr for node in ast.walk(collector_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called_attributes.intersection({"map_elements", "map_batches", "map_groups", "apply"})

    caller_tree = ast.parse(inspect.getsource(long_smoothing.lagged_zscore))
    python_numeric_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    assert not any(isinstance(node, ast.BinOp) and isinstance(node.op, python_numeric_ops)
                   for node in ast.walk(caller_tree))


def test_native_zscore_collects_single_plan_without_joins(monkeypatch):
    import polars as pl

    plans = []
    original = pl.LazyFrame.collect

    def counted(plan, *args, **kwargs):
        plans.append(plan.explain().upper())
        return original(plan, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", counted)
    frame = pd.DataFrame({
        "asset_id": ["A", "B", "A", "B", "A", "B"],
        "date": [0, 0, 1, 1, 2, 2],
        "value": [1.0, 10.0, 3.0, 12.0, 5.0, 14.0],
    }, index=[1, 1, 1, 1, 1, 1])
    result = lagged_zscore(frame, window=2, min_periods=1, ddof=0)

    assert len(result) == len(frame)
    assert len(plans) == 1
    assert "JOIN" not in plans[0]
    assert "_ZSCORE" in plans[0]
    assert "_CURRENT" in plans[0]
