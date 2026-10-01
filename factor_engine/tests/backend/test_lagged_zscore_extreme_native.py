"""High-precision regressions for the bounded stable z-score fallback."""
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.long_smoothing import lagged_zscore


def _decimal_z(current, history, ddof=0):
    with localcontext() as ctx:
        # Preserve exact binary64 decimals, including subnormals and 1e308.
        ctx.prec = 1200
        values = [Decimal.from_float(float(value)) for value in history]
        mean = sum(values) / Decimal(len(values))
        variance = sum((value - mean) ** 2 for value in values) / Decimal(len(values) - ddof)
        return float((Decimal.from_float(float(current)) - mean) / variance.sqrt())


def _frame(values):
    return pd.DataFrame({"asset_id": ["A"] * len(values),
                         "date": np.arange(len(values)), "value": values})


@pytest.mark.parametrize("scale", [1e308, np.nextafter(0.0, 1.0)])
def test_extreme_opposite_sign_window_matches_decimal(scale):
    values = [scale, -scale, scale, -scale]
    actual = lagged_zscore(_frame(values), window=2,
                           min_periods=2, ddof=0).to_numpy()
    expected = [np.nan, np.nan, _decimal_z(values[2], values[:2]),
                _decimal_z(values[3], values[1:3])]
    np.testing.assert_allclose(actual[2:], expected[2:], rtol=1e-14, atol=0.0)


def test_high_offset_nextafter_window_matches_decimal():
    base = 1e308
    adjacent = np.nextafter(base, np.inf)
    values = [base, adjacent, np.nextafter(adjacent, np.inf), base]
    actual = lagged_zscore(_frame(values), window=2,
                           min_periods=2, ddof=0).to_numpy()
    expected = [np.nan, np.nan, _decimal_z(values[2], values[:2]),
                _decimal_z(values[3], values[1:3])]
    np.testing.assert_allclose(actual[2:], expected[2:], rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("values", [
    [1e308, -1e308],
    [np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0)],
    [1e308, np.nextafter(1e308, np.inf)],
])
def test_hazard_detector_selects_stable_route(values):
    from factor_engine.backend.long_smoothing import _native_std_domain_safe

    assert not _native_std_domain_safe(
        np.asarray(values), np.ones(len(values), dtype=bool),
        np.zeros(len(values), dtype=np.int64), window=2,
    )


def test_stable_fallback_preserves_nonfinite_current_ieee_results():
    from factor_engine.backend.native_long_rolling_moments import collect_lagged_zscores_stable

    history = np.asarray([1e308, -1e308, 1e308, -1e308])
    current = np.asarray([1e308, -1e308, np.inf, -np.inf])
    result = collect_lagged_zscores_stable(
        np.arange(4), np.zeros(4, dtype=np.int64), history,
        np.ones(4), current, window=2, min_periods=2, ddof=0,
        max_working_pairs=4,
    ).get_column("_zscore").to_numpy()
    assert np.isposinf(result[2])
    assert np.isneginf(result[3])


def test_interval_fallback_is_causal_and_crosses_tile_boundary():
    values = [(-1.0 if i % 2 else 1.0) * 1e308 for i in range(40)]
    frame = _frame(values)
    from factor_engine.backend.native_long_rolling_moments import collect_lagged_zscores_stable

    direct = collect_lagged_zscores_stable(
        np.arange(len(values)), np.zeros(len(values), dtype=np.int64),
        np.asarray(values), np.ones(len(values)), np.asarray(values),
        window=2, min_periods=2, ddof=0, max_working_pairs=4,
    ).get_column("_zscore").to_numpy()
    np.testing.assert_allclose(
        direct[2:], np.tile([1.0, -1.0], 19), rtol=1e-14, atol=0.0
    )
    before = lagged_zscore(frame, window=2, min_periods=2, ddof=0).to_numpy()
    changed = frame.copy()
    changed.loc[20:, "value"] = [1e-300] * 20
    after = lagged_zscore(changed, window=2, min_periods=2, ddof=0).to_numpy()
    np.testing.assert_array_equal(before[:20], after[:20])


def test_interval_fallback_handles_valid_large_window():
    values = np.asarray([(-1.0 if i % 2 else 1.0) * 1e308
                        for i in range(133)], dtype=np.float64)
    actual = lagged_zscore(_frame(values), window=129,
                           min_periods=129, ddof=0).to_numpy()
    expected = [np.nan] * 129 + [
        _decimal_z(values[i], values[i - 129:i]) for i in range(129, 133)
    ]
    np.testing.assert_allclose(actual[129:], expected[129:], rtol=1e-13, atol=0.0)


def test_extreme_fallback_has_no_python_numeric_kernel():
    import ast
    import inspect
    from factor_engine.backend import native_long_rolling_moments as moments

    tree = ast.parse(inspect.getsource(moments.collect_lagged_zscores_stable))
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called.intersection({"map_elements", "map_batches", "map_groups", "apply"})
    numpy_reductions = {"sum", "mean", "std", "var", "min", "max", "sqrt", "abs"}
    assert not any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr in numpy_reductions
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def _decimal_series(values, *, window, min_periods, ddof):
    output = np.full(len(values), np.nan, dtype=np.float64)
    for row, current in enumerate(values):
        history = [float(x) for x in values[max(0, row - window):row]
                   if np.isfinite(x)]
        if len(history) < max(1, min_periods) or len(history) <= ddof:
            continue
        if not np.isfinite(current):
            output[row] = float(current)
            continue
        with localcontext() as ctx:
            # Avoid rounding exact large constants into artificial variance.
            ctx.prec = 1200
            observations = [Decimal.from_float(x) for x in history]
            mean = sum(observations) / Decimal(len(observations))
            variance = sum((x - mean) ** 2 for x in observations)
            variance /= Decimal(len(observations) - ddof)
            difference = Decimal.from_float(float(current)) - mean
            if variance == 0:
                output[row] = np.nan if difference == 0 else (
                    np.inf if difference > 0 else -np.inf
                )
            else:
                output[row] = float(difference / variance.sqrt())
    return output


@pytest.mark.parametrize("ddof", [-1, 0, 1, 2])
def test_extreme_window_matches_decimal_with_missing_history_and_ddof(ddof):
    values = np.asarray([1e308, -1e308, np.nan, 1e308,
                         1e308, -1e308, 0.0], dtype=np.float64)
    actual = lagged_zscore(_frame(values), window=3,
                           min_periods=1, ddof=ddof).to_numpy()
    expected = _decimal_series(values, window=3, min_periods=1, ddof=ddof)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=2e-13, atol=0.0)


def test_interleaved_groups_and_duplicate_rows_match_decimal_oracle():
    frame = pd.DataFrame({
        "asset_id": ["A", "B"] * 4,
        "date": [0, 0, 1, 1, 2, 2, 3, 3],
        "value": [1e308, -1e308, -1e308, 1e308,
                  np.nan, 1e308, 1e308, -1e308],
    }, index=[7, 7, 7, 7, 2, 2, 2, 2])
    actual = lagged_zscore(frame, window=2, min_periods=1, ddof=0)
    expected = np.full(len(frame), np.nan)
    for asset in ("A", "B"):
        rows = np.flatnonzero(frame.asset_id.to_numpy() == asset)
        expected[rows] = _decimal_series(
            frame.value.to_numpy()[rows], window=2, min_periods=1, ddof=0
        )
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual.to_numpy()[finite], expected[finite], rtol=2e-13, atol=0.0)
    assert actual.index.equals(frame.index)


def test_extreme_constant_window_keeps_ieee_zero_variance_results():
    base = 1e308
    values = np.asarray([base, base, base, np.nextafter(base, np.inf)])
    actual = lagged_zscore(_frame(values), window=2,
                           min_periods=2, ddof=0).to_numpy()
    expected = _decimal_series(values, window=2, min_periods=2, ddof=0)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=2e-13, atol=0.0)


def test_extreme_empty_panel_returns_empty_series():
    frame = pd.DataFrame({"asset_id": pd.Series(dtype=object),
                          "date": pd.Series(dtype=np.int64),
                          "value": pd.Series(dtype=np.float64)})
    actual = lagged_zscore(frame, window=2, min_periods=1, ddof=0)
    assert actual.empty
    assert actual.index.equals(frame.index)


@pytest.mark.parametrize("layout", ["asset_sorted", "date_interleaved"])
def test_stable_group_index_preserves_decimal_oracle_for_sparse_assets(monkeypatch, layout):
    from factor_engine.backend import long_smoothing

    # Force the numerical-hazard route while exercising both global layouts;
    # each asset remains chronologically ordered and B/C have sparse dates.
    monkeypatch.setattr(long_smoothing, "_native_std_domain_safe", lambda *args, **kwargs: False)
    base = 1e12
    by_asset = {
        "A": [(0, base), (1, base + 1), (2, base + 2), (3, np.nan),
              (4, base + 1), (5, base + 8), (6, np.inf)],
        "B": [(0, base + 4), (2, base + 5), (3, base + 9), (5, base + 7)],
        "C": [(1, base - 4), (2, base - 3), (6, base - 2)],
    }
    if layout == "asset_sorted":
        records = [(asset, date, value) for asset, points in by_asset.items()
                   for date, value in points]
    else:
        records = sorted(
            ((asset, date, value) for asset, points in by_asset.items()
             for date, value in points), key=lambda row: (row[1], row[0])
        )
    frame = pd.DataFrame(records, columns=["asset_id", "date", "value"],
                         index=np.arange(len(records)) % 5)
    actual = lagged_zscore(frame, window=3, min_periods=1, ddof=1)
    expected = np.full(len(frame), np.nan, dtype=np.float64)
    for asset in by_asset:
        rows = np.flatnonzero(frame["asset_id"].to_numpy() == asset)
        expected[rows] = _decimal_series(
            frame["value"].to_numpy()[rows], window=3, min_periods=1, ddof=1
        )
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual.to_numpy()[finite], expected[finite], rtol=2e-13, atol=0.0)
    assert actual.index.equals(frame.index)


def test_stable_group_rows_uses_int64_views_and_preserves_group_order():
    from factor_engine.backend.native_long_rolling_moments import _stable_group_rows

    codes = np.asarray([2, 0, 2, 1, 0, 2, 1, 0], dtype=np.int64)
    groups = list(_stable_group_rows(codes))
    assert [code for code, _ in groups] == [0, 1, 2]
    for code, rows in groups:
        assert rows.dtype == np.dtype(np.int64)
        np.testing.assert_array_equal(rows, np.flatnonzero(codes == code))

@pytest.mark.parametrize("codes", [
    np.asarray([[0, 1]], dtype=np.int64),
    np.asarray([-1, 0], dtype=np.int64),
    np.asarray([0, 2], dtype=np.int64),
    np.asarray([0.0, 1.0]),
    np.asarray([True, False]),
])
def test_stable_group_rows_rejects_invalid_codes(codes):
    from factor_engine.backend.native_long_rolling_moments import _stable_group_rows
    with pytest.raises(ValueError, match="group_codes"):
        list(_stable_group_rows(codes))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
def test_stable_group_rows_partitions_all_rows_once(dtype):
    from factor_engine.backend.native_long_rolling_moments import _stable_group_rows
    assert list(_stable_group_rows(np.asarray([], dtype=dtype))) == []
    # Sparse codes are valid; empty groups must not lose row identity.
    codes = np.asarray([2, 0, 2, 0, 2], dtype=dtype)
    groups = list(_stable_group_rows(codes))
    assert [code for code, _ in groups] == [0, 2]
    rows = np.concatenate([row_ids for _, row_ids in groups])
    np.testing.assert_array_equal(np.sort(rows), np.arange(len(codes)))
    for code, row_ids in groups:
        np.testing.assert_array_equal(row_ids, np.flatnonzero(codes == code))


def test_stable_fallback_has_no_python_row_list_staging():
    import ast
    import inspect
    from factor_engine.backend import native_long_rolling_moments as moments

    tree = ast.parse(inspect.getsource(moments.collect_lagged_zscores_stable))
    assert not any(isinstance(node, ast.Name) and node.id == "rows_by_group"
                   for node in ast.walk(tree))
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append" and node.args
        and isinstance(node.args[0], ast.Name) and node.args[0].id == "row"
        for node in ast.walk(tree)
    )
