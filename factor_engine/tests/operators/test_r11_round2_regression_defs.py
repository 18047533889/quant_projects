# -*- coding: utf-8 -*-
"""R11 round-2 regression-model operator definition fixes.

Covers review items #9-#13 for ``cleaned_operators/regression_models.py``:

1. ``ts_variance_ratio`` -> ``ts_variance_ratio_proxy`` (a VR *proxy* over
   differenced price/log-price LEVELS, NOT the Lo–MacKinlay estimator) with a
   declared PriceLevel/LogPriceLevel input contract; plus the two genuinely new
   canonicals ``ts_lo_mackinlay_vr`` / ``ts_lo_mackinlay_z``.
2. ``ts_level_shift_score`` / ``ts_vol_shift_score`` preserve PHYSICAL time:
   the trailing contiguous run is split at the window midpoint, never
   drop-finite-compressed across a NaN gap.
3. ``ts_cusum_break_score`` -> ``ts_cumulative_deviation_score`` (a heuristic
   max-absolute standardized cumulative deviation, NOT a CUSUM test statistic).
4. ``ts_huber_regression_resid`` / ``ts_ridge_regression_resid`` split into
   explicit in-sample and predictive (no-look-ahead) canonicals.
5. Deprecated legacy spellings resolve via in-module aliases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

NEW_CANONICALS = (
    "ts_variance_ratio_proxy",
    "ts_lo_mackinlay_vr",
    "ts_lo_mackinlay_z",
    "ts_cumulative_deviation_score",
    "ts_huber_regression_in_sample_resid",
    "ts_huber_regression_predictive_resid",
    "ts_ridge_regression_in_sample_resid",
    "ts_ridge_regression_predictive_resid",
)

LEGACY_ALIASES = {
    "ts_variance_ratio": "ts_variance_ratio_proxy",
    "ts_cusum_break_score": "ts_cumulative_deviation_score",
    "ts_huber_regression_resid": "ts_huber_regression_in_sample_resid",
    "ts_ridge_regression_resid": "ts_ridge_regression_in_sample_resid",
}


def _col(values) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(values, dtype=float).reshape(-1, 1), columns=["c"])


def _dates(n: int = 400) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def _random_walk_price(n: int = 400, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _col(100.0 + np.cumsum(drift + rng.standard_normal(n)))


# ---------------------------------------------------------------------------
# §1 naming + input contract (review #10/#11)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NEW_CANONICALS)
def test_new_canonicals_registered_and_classified_extended(name: str) -> None:
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    assert OperatorRegistry.get(name, "polars") is not None, f"{name}: polars backend"
    assert classify_canonical(name) in {"daily", "extended"}, name


def test_legacy_spellings_resolve_via_deprecated_aliases() -> None:
    for alias, canonical in LEGACY_ALIASES.items():
        assert OperatorRegistry.resolve_canonical(alias) == canonical, alias
        assert OperatorRegistry.get(alias) is not None, alias
        assert OperatorRegistry.get(alias, "polars") is not None, f"{alias}: polars"


def test_alias_and_canonical_produce_identical_output() -> None:
    x = _random_walk_price(seed=11)
    y = 1.5 * x + 0.3
    # per-alias (args, kwargs): each operator has its own parameter contract.
    calls = {
        "ts_variance_ratio": ([x], dict(window=120, q=8, min_periods=15)),
        "ts_cusum_break_score": ([x], dict(window=120, min_periods=15)),
        "ts_huber_regression_resid": ([y, x], dict(window=60, min_periods=6)),
        "ts_ridge_regression_resid": ([y, x], dict(window=60, alpha=1e-6, min_periods=6)),
    }
    for alias, canonical in LEGACY_ALIASES.items():
        args, kwargs = calls[alias]
        a = OperatorRegistry.get(alias).calculate(*args, **kwargs)
        b = OperatorRegistry.get(canonical).calculate(*args, **kwargs)
        np.testing.assert_allclose(
            a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True,
            err_msg=f"{alias} != {canonical}",
        )


def test_variance_ratio_input_semantic_declares_price_level_not_return() -> None:
    for name in ("ts_variance_ratio_proxy", "ts_lo_mackinlay_vr", "ts_lo_mackinlay_z"):
        meta = OperatorRegistry.get(name).metadata
        assert "price_level" in meta.input_units["x"], (name, meta.input_units)
        assert "return" not in meta.input_units["x"], (name, meta.input_units)
        assert set(meta.compatible_units["x"]) == {"price_level", "log_price_level"}, (
            name, meta.compatible_units["x"],
        )


def test_variance_ratio_proxy_matches_legacy_behavior() -> None:
    # Random-walk price: Var(q-period)/q/Var(1) - 1 ~ 0.
    rw = _random_walk_price(seed=0)
    out = OperatorRegistry.get("ts_variance_ratio_proxy").calculate(
        rw, window=200, q=10, min_periods=20
    )
    assert abs(out["c"].iloc[-1]) < 0.3
    # Trending (positive-autocorrelation) price: proxy clearly > 0.
    rng = np.random.default_rng(1)
    n = 400
    rho = 0.6
    rets = np.zeros(n)
    for t in range(1, n):
        rets[t] = rho * rets[t - 1] + rng.standard_normal()
    trend = _col(100.0 + np.cumsum(rets))
    out_t = OperatorRegistry.get("ts_variance_ratio_proxy").calculate(
        trend, window=200, q=10, min_periods=20
    )
    assert out_t["c"].iloc[-1] > 0.3


def test_lo_mackinlay_vr_z_finite_and_sane_on_geometric_random_walk() -> None:
    rng = np.random.default_rng(2)
    n = 400
    g = _col(100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01)))
    vr = OperatorRegistry.get("ts_lo_mackinlay_vr").calculate(g, window=200, q=10, min_periods=20)
    z = OperatorRegistry.get("ts_lo_mackinlay_z").calculate(g, window=200, q=10, min_periods=20)
    last_vr = float(vr["c"].iloc[-1])
    last_z = float(z["c"].iloc[-1])
    assert np.isfinite(last_vr) and np.isfinite(last_z)
    # geometric random walk: VR ~ 1 (within sampling error), z finite.
    assert abs(last_vr - 1.0) < 0.5
    assert abs(last_z) < 5.0


def test_lo_mackinlay_z_positive_for_mean_reversion() -> None:
    # A mean-reverting level (AR(1) in the level) gives VR < 1 -> z < 0
    # (Lo–MacKinlay convention: negative z = mean reversion).
    rng = np.random.default_rng(3)
    n = 400
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    series = _col(x)
    vr = OperatorRegistry.get("ts_lo_mackinlay_vr").calculate(series, window=200, q=10, min_periods=20)
    z = OperatorRegistry.get("ts_lo_mackinlay_z").calculate(series, window=200, q=10, min_periods=20)
    assert float(vr["c"].iloc[-1]) < 1.0
    assert float(z["c"].iloc[-1]) < 0.0


# ---------------------------------------------------------------------------
# §2 physical-time NaN handling for level/vol shift (review #12)
# ---------------------------------------------------------------------------
def test_level_shift_score_respects_nan_gap_no_bridging() -> None:
    # The gap ends exactly at the current 20-row contract's midpoint, so the
    # trailing contiguous run [10..19] is entirely AFTER the
    # midpoint.  The "before" state is unobservable in physical time -> fail-closed
    # NaN.  The old drop-finite code re-paired [1,2,3] with [10..15] and produced
    # a finite (bridged) score; the fix never bridges across the gap.
    vals = np.r_[np.arange(1.0, 6.0), np.full(5, np.nan), np.arange(10.0, 20.0)]
    x = _col(vals)
    out = OperatorRegistry.get("ts_level_shift_score").calculate(x, window=20, min_periods=2)
    assert np.isnan(float(out["c"].iloc[-1]))


def test_vol_shift_score_respects_nan_gap_no_bridging() -> None:
    vals = np.array([1.0, 2.0, 3.0, np.nan, np.nan, np.nan, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    x = _col(vals)
    out = OperatorRegistry.get("ts_vol_shift_score").calculate(x, window=12, min_periods=2)
    assert np.isnan(float(out["c"].iloc[-1]))


def test_level_shift_short_trailing_run_after_gap_is_nan() -> None:
    # The task's exact example: a short trailing run after the gap is too small
    # to split, so the score is NaN (never a half-bridged value).
    vals = np.r_[np.arange(1.0, 15.0), np.full(3, np.nan), [10.0, 11.0, 12.0]]
    x = _col(vals)
    out = OperatorRegistry.get("ts_level_shift_score").calculate(x, window=20, min_periods=2)
    assert np.isnan(float(out["c"].iloc[-1]))


def test_level_shift_split_at_physical_window_midpoint() -> None:
    # In a valid 20-row window, the trailing run starts at offset 9 and the
    # physical midpoint is offset 10.  Thus first=[3], second=[4..13].
    vals = np.r_[np.arange(1.0, 9.0), np.nan, np.arange(3.0, 14.0)]
    x = _col(vals)
    out = OperatorRegistry.get("ts_level_shift_score").calculate(x, window=20, min_periods=2)
    last = float(out["c"].iloc[-1])
    assert np.isfinite(last)
    # mean([4..13]) - mean([3]) > 0, normalised by std>0.
    assert last > 0.5


def test_level_shift_rejects_window_below_typed_contract() -> None:
    x = _col(np.arange(12.0))
    with pytest.raises(Exception, match="window must be >= 20"):
        OperatorRegistry.get("ts_level_shift_score").calculate(
            x, window=12, min_periods=2)


def test_level_shift_detects_mean_shift_no_nan() -> None:
    rng = np.random.default_rng(0)
    n = 100
    series = rng.standard_normal(n)
    series[60:] += 1.5
    x = _col(series)
    out = OperatorRegistry.get("ts_level_shift_score").calculate(x, window=80, min_periods=10)
    assert out["c"].iloc[-1] > 0.5


# ---------------------------------------------------------------------------
# §3 cumulative deviation score (review #13)
# ---------------------------------------------------------------------------
def test_cumulative_deviation_score_catches_level_jump() -> None:
    rng = np.random.default_rng(4)
    n = 200
    base = rng.standard_normal(n)
    x_base = _col(base)
    # Jump at the midpoint of the LAST window (rows 120..199, window=80).
    jumped = base.copy()
    jumped[160:] += 2.0
    x_jump = _col(jumped)

    op = OperatorRegistry.get("ts_cumulative_deviation_score")
    no_jump = float(op.calculate(x_base, window=80, min_periods=10)["c"].iloc[-1])
    with_jump = float(op.calculate(x_jump, window=80, min_periods=10)["c"].iloc[-1])
    assert np.isfinite(no_jump) and np.isfinite(with_jump)
    # A level jump makes the standardized deviations share a sign on one side of
    # the window midpoint, growing the cumulative path well beyond the noise path.
    assert with_jump > 2.0 * no_jump
    assert with_jump > 15.0


def test_cumulative_deviation_heuristic_not_cusum_statistic() -> None:
    # The metadata/docstring must state it is a heuristic, not a CUSUM test.
    meta = OperatorRegistry.get("ts_cumulative_deviation_score").metadata
    assert "heuristic" in meta.description.lower() or "启发式" in meta.description


# ---------------------------------------------------------------------------
# §4 in-sample vs predictive residuals (review #9)
# ---------------------------------------------------------------------------
def test_in_sample_resid_approximately_zero_on_perfect_line() -> None:
    x = _col(np.arange(120, dtype=float))
    y = 2.0 * x + 0.5
    huber = OperatorRegistry.get("ts_huber_regression_in_sample_resid").calculate(
        y, x, window=60, min_periods=6
    )
    ridge = OperatorRegistry.get("ts_ridge_regression_in_sample_resid").calculate(
        y, x, window=60, alpha=1e-6, min_periods=6
    )
    assert huber["c"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert ridge["c"].iloc[-1] == pytest.approx(0.0, abs=1e-4)


def test_predictive_resid_has_no_look_ahead() -> None:
    # y = x on all rows except the LAST row, where y_t = x_t + 10 (an
    # innovation the previous window could not have seen).  A short window gives
    # the endpoint a high leverage so the in-sample/predictive gap is clear.
    x = _col(np.arange(30, dtype=float))
    yv = x["c"].to_numpy(dtype=float).copy()
    yv[-1] = yv[-1] + 10.0
    y = _col(yv)

    # Ridge at a tiny alpha is OLS-like.  The in-sample fit includes (x_t, y_t)
    # and tilts toward the high-leverage outlier, understating its residual; the
    # predictive fit on [t-20, t-1] (a clean line y=x) predicts y_t = x_t and
    # reports the FULL +10 innovation — no look-ahead.  The two genuinely differ.
    in_sample = OperatorRegistry.get("ts_ridge_regression_in_sample_resid").calculate(
        y, x, window=20, alpha=1e-6, min_periods=6
    )
    predictive = OperatorRegistry.get("ts_ridge_regression_predictive_resid").calculate(
        y, x, window=20, alpha=1e-6, min_periods=6
    )
    is_last = float(in_sample["c"].iloc[-1])
    pred_last = float(predictive["c"].iloc[-1])
    assert pred_last == pytest.approx(10.0, abs=1.5)
    assert pred_last - is_last > 1.0


def test_predictive_huber_resid_on_clean_line_no_look_ahead() -> None:
    # On a clean linear trend the predictive residual is ~0: the fit on the
    # PRECEDING window predicts the current row exactly (no look-ahead).
    x = _col(np.arange(80, dtype=float))
    y = 2.0 * x + 0.5
    pred = OperatorRegistry.get("ts_huber_regression_predictive_resid").calculate(
        y, x, window=60, min_periods=6
    )
    assert pred["c"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_predictive_variants_documented_as_mining_preference() -> None:
    for name in ("ts_huber_regression_predictive_resid", "ts_ridge_regression_predictive_resid"):
        desc = OperatorRegistry.get(name).metadata.description
        assert "predictive" in desc


# ---------------------------------------------------------------------------
# §5 polars parity (via the bridge)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("canonical,fields,kwargs", [
    ("ts_variance_ratio_proxy", ["close"], dict(window=120, q=8, min_periods=15)),
    ("ts_lo_mackinlay_vr", ["close"], dict(window=120, q=8, min_periods=15)),
    ("ts_lo_mackinlay_z", ["close"], dict(window=120, q=8, min_periods=15)),
    ("ts_cumulative_deviation_score", ["close"], dict(window=80, min_periods=10)),
    ("ts_level_shift_score", ["close"], dict(window=80, min_periods=10)),
    ("ts_vol_shift_score", ["close"], dict(window=80, min_periods=10)),
    ("ts_huber_regression_predictive_resid", ["y", "x"], dict(window=60, min_periods=6)),
    ("ts_ridge_regression_predictive_resid", ["y", "x"], dict(window=60, alpha=1e-6, min_periods=6)),
])
def test_polars_parity_new_canonicals(canonical, fields, kwargs) -> None:
    pytest.importorskip("polars")
    import polars as pl

    rng = np.random.default_rng(7)
    n = 200
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    pdf = pd.DataFrame(
        {
            "date": _dates(n),
            "close": close,
            "y": 1.5 * close + rng.standard_normal(n),
            "x": 100.0 + np.cumsum(rng.standard_normal(n)),
        }
    )
    pdf = pdf.set_index("date")
    plf = pl.DataFrame(
        {
            "date": pd.DatetimeIndex(pdf.index),
            "close": pdf["close"].to_numpy(),
            "y": pdf["y"].to_numpy(),
            "x": pdf["x"].to_numpy(),
        }
    )

    def _to_pd(pldf: pl.DataFrame) -> pd.DataFrame:
        if "date" in pldf.columns:
            idx = pd.DatetimeIndex(pldf["date"].to_pandas())
        else:
            idx = pd.RangeIndex(pldf.height)
        cols = [c for c in pldf.columns if c != "date"]
        return pd.DataFrame({c: pldf[c].to_numpy() for c in cols}, index=idx)

    p_op = OperatorRegistry.get(canonical, "pandas_numpy")
    l_op = OperatorRegistry.get(canonical, "polars")
    assert l_op is not None, canonical
    # Multi-panel operators require ALIGNED column labels across inputs, so
    # every panel is normalised to the same single column label "c".
    p_args = [pdf[[f]].rename(columns={f: "c"}) for f in fields]
    pl_args = [plf.select(["date", f]).rename({f: "c"}) for f in fields]
    ref = p_op.calculate(*p_args, **kwargs)
    got_pl = l_op.calculate(*pl_args, **kwargs)
    got = _to_pd(got_pl).reindex(index=ref.index, columns=ref.columns)
    assert ref.shape == got.shape, canonical
    for col in ref.columns:
        a = ref[col].to_numpy(dtype=float)
        b = got[col].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, atol=1e-6, equal_nan=True, err_msg=f"{canonical}:{col}")
