from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from factor_engine.cleaned_operators.downside_risk import (
    TsBestLagCorrExcess,
    TsPriceDelay,
    _price_delay_model,
    _stable_surrogate_seed,
)
from factor_engine.cleaned_operators.common.polars_robust_stats import (
    ts_best_lag_corr_excess as pl_best_lag_corr_excess,
)


def _frame(values, index, columns=("A",)):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return pd.DataFrame(arr, index=index, columns=list(columns))


def test_price_delay_scale_translation_invariance():
    rng = np.random.default_rng(12014)
    n = 100
    bench = rng.normal(size=n)
    stock = 0.3 + 0.8 * bench + 0.7 * np.roll(bench, 2) + 0.1 * rng.normal(size=n)
    stock[:2] = 0.0
    expected = _price_delay_model(stock, bench, n - 1, 40, 3, 6)
    assert np.isfinite(expected)
    assert _price_delay_model(stock, bench * 1e-11, n - 1, 40, 3, 6) == pytest.approx(expected, abs=2e-12)
    assert _price_delay_model(stock, bench + 1e8, n - 1, 40, 3, 6) == pytest.approx(expected, abs=2e-8)


def test_price_delay_common_mask_and_rank_fail_closed():
    rng = np.random.default_rng(12015)
    bench = rng.normal(size=90)
    stock = bench + np.roll(bench, 1) + 0.1 * rng.normal(size=90)
    bench[[55, 61]] = np.nan
    stock[58] = np.nan
    assert np.isfinite(_price_delay_model(stock, bench, 89, 40, 3, 6))
    constant = np.ones(90)
    assert np.isnan(_price_delay_model(stock, constant, 89, 40, 3, 6))
    alternating = np.tile([1.0, -1.0], 45)
    assert np.isnan(_price_delay_model(stock, alternating, 89, 40, 3, 6))


def test_price_delay_restricted_r2_near_zero_is_valid():
    rng = np.random.default_rng(12016)
    bench = rng.normal(size=200)
    stock = np.roll(bench, 3) + 0.02 * rng.normal(size=200)
    value = _price_delay_model(stock, bench, 199, 80, 4, 6)
    assert np.isfinite(value)
    assert 0.8 < value <= 1.0


def test_price_delay_static_joint_domain_declared_and_enforced():
    op = TsPriceDelay()
    relation = op.metadata.relational_specs[0]
    assert relation.expression == "window >= 2 * (max_lag + 2)"
    assert relation.check({"window": 14, "max_lag": 5})
    assert not relation.check({"window": 13, "max_lag": 5})
    idx = pd.date_range("2025-01-01", periods=30)
    frame = _frame(np.arange(30.0), idx)
    with pytest.raises(ValueError, match="STATIC_DOMAIN_INFEASIBLE"):
        op.calculate(frame, frame, window=13, max_lag=5, min_periods=3)


def test_lag_corr_excess_full_slice_replay_with_sufficient_history():
    rng = np.random.default_rng(14001)
    idx = pd.date_range("2021-01-01", periods=140)
    x = rng.normal(size=140)
    y = np.roll(x, 2) + 0.2 * rng.normal(size=140)
    full = TsBestLagCorrExcess().calculate(_frame(y, idx), _frame(x, idx), window=20, max_lag=3)
    sliced = TsBestLagCorrExcess().calculate(_frame(y[50:], idx[50:]), _frame(x[50:], idx[50:]), window=20, max_lag=3)
    np.testing.assert_array_equal(full.loc[idx[73]:, "A"].to_numpy(), sliced.loc[idx[73]:, "A"].to_numpy())


def test_lag_corr_excess_security_reorder_and_isolation():
    rng = np.random.default_rng(14002)
    idx = pd.date_range("2022-01-01", periods=90)
    xa, xb = rng.normal(size=(2, 90))
    ya = np.roll(xa, 1) + 0.1 * rng.normal(size=90)
    yb = rng.normal(size=90)
    x = _frame(np.column_stack([xa, xb]), idx, ("A", "B"))
    y = _frame(np.column_stack([ya, yb]), idx, ("A", "B"))
    op = TsBestLagCorrExcess()
    base = op.calculate(y, x, window=24, max_lag=3)
    reordered = op.calculate(y[["B", "A"]], x[["B", "A"]], window=24, max_lag=3)
    np.testing.assert_array_equal(base["A"].to_numpy(), reordered["A"].to_numpy())
    x_changed = x.copy()
    y_changed = y.copy()
    x_changed["B"] = rng.normal(size=90) * 100
    y_changed["B"] = rng.normal(size=90) * 100
    isolated = op.calculate(y_changed, x_changed, window=24, max_lag=3)
    np.testing.assert_array_equal(base["A"].to_numpy(), isolated["A"].to_numpy())


def test_surrogate_seed_is_typed_stable_identity():
    t = pd.Timestamp("2025-03-04", tz="Asia/Hong_Kong")
    seed = _stable_surrogate_seed(42, t, "000001.SZ", "ts_best_lag_corr_excess@v2")
    assert seed == _stable_surrogate_seed(42, t, "000001.SZ", "ts_best_lag_corr_excess@v2")
    assert seed != _stable_surrogate_seed(42, t, "000002.SZ", "ts_best_lag_corr_excess@v2")
    assert seed != _stable_surrogate_seed(42, t + pd.Timedelta(days=1), "000001.SZ", "ts_best_lag_corr_excess@v2")
    assert seed != _stable_surrogate_seed(42, t, "000001.SZ", "ts_best_lag_corr_excess@v3")


def _pl_frame(values, dates, columns=("A",)):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    payload = {"date": list(dates)}
    payload.update({column: arr[:, i] for i, column in enumerate(columns)})
    return pl.DataFrame(payload)


def test_polars_cpu_bridge_parity_and_slice_replay():
    rng = np.random.default_rng(14003)
    idx = pd.date_range("2023-01-01", periods=110)
    x = rng.normal(size=110)
    y = np.roll(x, 2) + 0.15 * rng.normal(size=110)
    pandas_out = TsBestLagCorrExcess().calculate(
        _frame(y, idx), _frame(x, idx), window=20, max_lag=3
    )["A"].to_numpy()
    full = pl_best_lag_corr_excess(
        _pl_frame(y, idx), _pl_frame(x, idx), window=20, max_lag=3
    )["A"].to_numpy()
    np.testing.assert_array_equal(full, pandas_out)
    sliced = pl_best_lag_corr_excess(
        _pl_frame(y[40:], idx[40:]), _pl_frame(x[40:], idx[40:]),
        window=20, max_lag=3,
    )["A"].to_numpy()
    np.testing.assert_array_equal(full[63:], sliced[23:])


def test_polars_cpu_bridge_security_reorder_and_isolation():
    rng = np.random.default_rng(14004)
    idx = pd.date_range("2023-06-01", periods=70)
    xa, xb = rng.normal(size=(2, 70))
    ya = np.roll(xa, 2) + 0.1 * rng.normal(size=70)
    yb = rng.normal(size=70)
    x = _pl_frame(np.column_stack([xa, xb]), idx, ("A", "B"))
    y = _pl_frame(np.column_stack([ya, yb]), idx, ("A", "B"))
    base = pl_best_lag_corr_excess(y, x, 20, 3)
    reordered = pl_best_lag_corr_excess(
        y.select("date", "B", "A"), x.select("date", "B", "A"), 20, 3
    )
    np.testing.assert_array_equal(base["A"].to_numpy(), reordered["A"].to_numpy())
    changed_x = x.with_columns(pl.Series("B", rng.normal(size=70) * 100))
    changed_y = y.with_columns(pl.Series("B", rng.normal(size=70) * 100))
    isolated = pl_best_lag_corr_excess(changed_y, changed_x, 20, 3)
    np.testing.assert_array_equal(base["A"].to_numpy(), isolated["A"].to_numpy())


def test_polars_bridge_requires_and_preserves_absolute_coordinates():
    idx = list(pd.date_range("2024-01-01", periods=50))
    values = np.arange(50.0)
    with pytest.raises(ValueError, match="explicit date coordinate"):
        pl_best_lag_corr_excess(
            pl.DataFrame({"A": values}), pl.DataFrame({"A": values}), 20, 3
        )
    baseline = pl_best_lag_corr_excess(
        _pl_frame(values, idx), _pl_frame(values, idx), 20, 3
    )["A"].to_numpy()
    idx[25] = None
    missing = pl_best_lag_corr_excess(
        _pl_frame(values, idx), _pl_frame(values, idx), 20, 3
    )["A"].to_numpy()
    assert np.isnan(missing[25])
    np.testing.assert_array_equal(missing[26:], baseline[26:])
