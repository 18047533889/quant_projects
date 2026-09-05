# -*- coding: utf-8 -*-
"""Tests for the 2026-08 intraday feature extensions and ``intra_*`` aliases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.storage.sources.intraday_feature_extension import _calc


_TIMES = ["09:31", "09:36", "09:41", "09:46", "09:51", "09:56", "13:01", "13:06", "13:11", "13:16"]


def _bar(rows: int | None = None, seed: int = 0, cross_lunch: bool = False) -> pd.DataFrame:
    if rows is None:
        rows = len(_TIMES)
    times = _TIMES[:rows]
    idx = pd.to_datetime(["2024-01-02 " + h for h in times])
    rng = np.random.default_rng(seed)
    close = 10.0 + np.cumsum(rng.normal(0, 0.01, rows))
    open_px = np.concatenate([[10.0], close[:-1]])
    high = np.maximum(open_px, close) + 0.01
    low = np.minimum(open_px, close) - 0.01
    volume = rng.integers(100, 500, rows).astype(float)
    amount = volume * close
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        }
    )


def test_intraday_features_and_intra_native_ops_are_exposed() -> None:
    from factor_engine.api.intraday_daily import INTRADAY_DAILY_DSL_FUNCTIONS
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    for name in (
        "intraday_lunch_gap_return",
        "intraday_return_activity_corr",
        "intraday_vwap_above_ratio",
        "intraday_kyle_lambda_proxy",
        "intraday_extreme_bar_return",
        "intraday_segment_return",
        "intraday_segment_volume_share",
        "intraday_segment_amount_share",
        "intraday_segment_vwap_deviation",
        "intraday_segment_realized_vol",
        "intraday_limit_first_hit_time",
        "intraday_limit_duration",
        "intraday_limit_reopen_count",
    ):
        assert name in INTRADAY_DAILY_DSL_FUNCTIONS, name
    for name in (
        "intra_realized_variance",
        "intra_bipower_variation",
        "intra_jump_ratio",
        "intra_path_efficiency",
        "intra_high_time",
        "intra_low_time",
        "intra_vwap_cross_count",
        "intra_amihud",
        "intra_lunch_gap_return",
        "intra_segment_return",
        "intra_limit_reopen_count",
    ):
        assert OperatorRegistry.get(name) is not None, name


def test_lunch_gap_return_is_afternoon_open_over_morning_close_minus_one() -> None:
    bar = _bar(rows=8)  # 6 morning bars then 2 afternoon bars
    out = _calc("lunch_gap_return", bar, {})
    morning_last_close = bar["close"].iloc[5]
    afternoon_first_open = bar["open"].iloc[6]
    assert out == pytest.approx(afternoon_first_open / morning_last_close - 1.0)


def test_vwap_above_ratio_is_mean_close_over_bar_vwap() -> None:
    bar = _bar(rows=10)
    out = _calc("vwap_above_ratio", bar, {})
    bar_vwap = (bar["amount"] / bar["volume"]).to_numpy()
    expected = float(np.mean(bar["close"].to_numpy() >= bar_vwap))
    assert out == pytest.approx(expected)


def test_extreme_bar_return_side() -> None:
    bar = _bar(rows=10)
    r = np.log(bar["close"].to_numpy() / np.roll(bar["close"].to_numpy(), 1))[1:]
    assert _calc("extreme_bar_return", bar, {"side": "max"}) == pytest.approx(float(np.nanmax(r)))
    assert _calc("extreme_bar_return", bar, {"side": "min"}) == pytest.approx(float(np.nanmin(r)))


def test_segment_return_first_bars() -> None:
    bar = _bar(rows=10)
    out = _calc("segment_return", bar, {"segment": "first", "minutes": 10, "bar_minutes": 5})
    first_two = bar.iloc[:2]
    expected = first_two["close"].iloc[-1] / first_two["open"].iloc[0] - 1.0
    assert out == pytest.approx(expected)


def test_segment_volume_share() -> None:
    bar = _bar(rows=10)
    out = _calc("segment_volume_share", bar, {"segment": "first", "minutes": 10, "bar_minutes": 5})
    expected = bar["volume"].iloc[:2].sum() / bar["volume"].sum()
    assert out == pytest.approx(expected)


def test_kyle_lambda_proxy_is_positive() -> None:
    bar = _bar(rows=10)
    out = _calc("kyle_lambda_proxy", bar, {})
    assert np.isfinite(out) and out >= 0.0


def test_return_activity_corr_matches_signed_corr() -> None:
    bar = _bar(rows=10)
    out = _calc("return_activity_corr", bar, {"activity": "volume"})
    close = bar["close"].to_numpy()
    r = np.full(len(close), 0.0)
    r[1:] = np.log(close[1:] / close[:-1])
    corr = np.corrcoef(r, bar["volume"].to_numpy())[0, 1]
    assert out == pytest.approx(float(corr))


def test_limit_features_nan_without_limit_price() -> None:
    bar = _bar(rows=10)
    # prev_close=0 -> ashare_limit_prices returns NaN -> features return NaN
    assert np.isnan(_calc("limit_first_hit_time", bar, {}))
    assert np.isnan(_calc("limit_duration", bar, {}))
    assert np.isnan(_calc("limit_reopen_count", bar, {}))


def _fake_source(anchor: pd.MultiIndex) -> Any:
    """Minimal LQTPLogicalDataSource-like fixture for compute_many parity."""

    class Inner:
        start_date = None
        end_date = None
        instrument_filter = None
        data_snapshot_id = "snap-1"
        params = {}
        dataset = "ashare_stock_minute"

    class Source:
        inner = Inner()

        def _anchor_index(self):
            return anchor

        def _record_dependency(self, *args, **kwargs):
            self._dep = (args, kwargs)

        @staticmethod
        def _align_exact_by_instrument(anchor, series):
            s = series.copy()
            s.index = s.index.set_names(["timestamp", "instrument"])
            idx = anchor.set_names(["timestamp", "instrument"])
            out = s.reindex(idx)
            out.index = anchor
            return out

    return Source()


def _grouped_fixture() -> list[tuple[pd.Timestamp, str, pd.DataFrame]]:
    """Two instruments x two days of 5-minute bars (bar_end clock)."""
    rows: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []
    for day in ("2024-01-02", "2024-01-03"):
        for inst in ("A", "B"):
            times = pd.to_datetime(
                [f"{day} 09:31", f"{day} 09:36", f"{day} 09:41", f"{day} 09:46"]
            )
            rng = np.random.default_rng(hash((day, inst)) % (2**32))
            close = 10.0 + np.cumsum(rng.normal(0, 0.01, 4))
            open_px = np.concatenate([[10.0], close[:-1]])
            high = np.maximum(open_px, close) + 0.01
            low = np.minimum(open_px, close) - 0.01
            volume = rng.integers(100, 500, 4).astype(float)
            amount = volume * close
            bar = pd.DataFrame(
                {
                    "timestamp": times,
                    "open": open_px,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                }
            )
            rows.append((pd.Timestamp(day), inst, bar))
    return rows


def test_compute_many_matches_per_feature_loop() -> None:
    """P0#5: single-scan compute_many is numerically identical to the old
    per-feature loop (which recomputed shared intermediates each time)."""
    from factor_engine.storage.sources import intraday_feature_runtime_v2 as v2

    anchor = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-03"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    source = _fake_source(anchor)
    grouped = _grouped_fixture()
    features = [
        "realized_variance",
        "realized_vol",
        "trend_slope",
        "trend_r2",
        "vwap",
        "close_to_vwap",
        "volume_hhi",
        "return_volume_corr",
        "max_drawdown",
        "jump_ratio",
        "lunch_gap_return",
        "segment_return",
    ]
    params = {"bar_minutes": 5, "min_coverage": 0.8, "min_bars": 2}

    # Reference: original per-feature loop (recompute shared per feature).
    reference: dict[str, pd.Series] = {}
    for f in features:
        previous_close: dict[str, float] = {}
        values: dict[tuple[pd.Timestamp, str], float] = {}
        for date, instrument, bars in grouped:
            calc_params = {**params, "instrument": instrument, "trade_date": date}
            values[(date, instrument)] = _calc(
                f, bars, calc_params, prev_close=previous_close.get(instrument, np.nan)
            )
            previous_close[instrument] = float(bars["close"].iloc[-1])
        idx = pd.MultiIndex.from_tuples(list(values), names=["timestamp", "instrument"])
        reference[f] = (
            pd.Series(list(values.values()), index=idx, name=f"intraday_{f}")
            .sort_index()
            .reindex(anchor)
        )

    # compute_many with a stubbed single grouped-bars pass.
    import factor_engine.storage.sources.intraday_feature_runtime_v2 as _v2

    original = _v2._grouped_bars
    _v2._grouped_bars = lambda *a, **k: (object(), grouped, 4)
    try:
        out = v2.compute_many(source, features, params)
    finally:
        _v2._grouped_bars = original

    assert set(out.keys()) == set(features)
    for f in features:
        got = out[f].reindex(anchor)
        np.testing.assert_allclose(
            got.to_numpy(), reference[f].to_numpy(), equal_nan=True, rtol=1e-12, atol=1e-12
        )


def test_compute_many_single_feature_equals_load_intraday_feature() -> None:
    """P0#5: load_intraday_feature (single) delegates to compute_many and is
    identical to the direct compute_many single-feature result."""
    from factor_engine.storage.sources import intraday_feature_runtime_v2 as v2

    anchor = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-03"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    source = _fake_source(anchor)
    grouped = _grouped_fixture()
    params = {"bar_minutes": 5, "min_coverage": 0.8, "min_bars": 2}

    import factor_engine.storage.sources.intraday_feature_runtime_v2 as _v2

    original = _v2._grouped_bars
    _v2._grouped_bars = lambda *a, **k: (object(), grouped, 4)
    try:
        single = v2.load_intraday_feature(source, {**params, "feature": "vwap"})
        many = v2.compute_many(source, ["vwap"], params)["vwap"]
    finally:
        _v2._grouped_bars = original

    np.testing.assert_allclose(
        single.reindex(anchor).to_numpy(),
        many.reindex(anchor).to_numpy(),
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


class _CountingWideSource:
    """InMemory-ish wide-frame source that counts physical ``load_columns`` calls.

    Resolution happens through ``_has_column`` (no scan); the ONLY physical
    scan is a single batch ``load_columns`` — so the scan count is exactly one
    for a frame needing up to six columns (R61 P0 #61 regression).
    """

    def __init__(self, data: dict[str, pd.Series]) -> None:
        self._data = data
        self.scan_count = 0

    def _has_column(self, name: str) -> bool:
        return name in self._data

    def load_column(self, name: str):
        return self._data[name]

    def load_columns(self, names: list[str]) -> dict[str, pd.Series]:
        self.scan_count += 1
        return {n: self._data[n] for n in names if n in self._data}


def _counting_src(data: dict[str, pd.Series]) -> _CountingWideSource:
    return _CountingWideSource(data)


def test_wide_frame_is_exactly_one_physical_scan() -> None:
    """R61 P0 #61: the 6-column wide frame must be ONE multi-column scan.

    The old implementation called ``_load_field`` → ``load_column`` once per
    field (up to 6 physical scans for one frame, plus N merges).  The fix
    resolves all candidates once (plan-level, no scan) and reads the union in a
    single ``load_columns`` batch.
    """
    from factor_engine.storage.sources.intraday_feature_extension import _wide_frame

    rng = np.random.default_rng(61)
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02 09:31"), "A"),
            (pd.Timestamp("2024-01-02 09:36"), "A"),
            (pd.Timestamp("2024-01-02 09:31"), "B"),
            (pd.Timestamp("2024-01-02 09:36"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    data = {
        "Open": pd.Series(10.0 + rng.normal(0, 0.01, len(idx)), index=idx),
        "High": pd.Series(11.0 + rng.normal(0, 0.01, len(idx)), index=idx),
        "Low": pd.Series(9.0 + rng.normal(0, 0.01, len(idx)), index=idx),
        "Close": pd.Series(10.5 + rng.normal(0, 0.01, len(idx)), index=idx),
        "Volume": pd.Series(rng.integers(100, 500, len(idx)).astype(float), index=idx),
        "Amount": pd.Series(rng.normal(1000, 50, len(idx)), index=idx),
    }
    src = _counting_src(data)
    frame = _wide_frame(src)
    assert set(frame.columns) == {
        "timestamp", "instrument", "open", "high", "low", "close", "volume", "amount"
    }
    assert len(frame) == 4
    # EXACTLY ONE physical multi-column scan for a 6-column frame.
    assert src.scan_count == 1, f"scan_count={src.scan_count}"

    # Each resolved column landed in the same physical scan (no per-column
    # re-reads): values are bit-identical to the source series.
    for key, phys in (
        ("open", "Open"), ("high", "High"), ("low", "Low"),
        ("close", "Close"), ("volume", "Volume"), ("amount", "Amount"),
    ):
        np.testing.assert_array_equal(frame[key].to_numpy(), data[phys].to_numpy())


def test_wide_frame_single_scan_with_optional_column_missing() -> None:
    """R61 P0 #61: an optional column absent from the source must NOT cause a
    second scan — resolution is already a no-scan probe, and the one batch
    simply omits it; the merge-identical fallback fills the column."""
    from factor_engine.storage.sources.intraday_feature_extension import _wide_frame

    rng = np.random.default_rng(2)
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02 09:31"), "A"),
            (pd.Timestamp("2024-01-02 09:36"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    data = {
        "Open": pd.Series(11.0 + rng.normal(0, 0.01, len(idx)), index=idx),
        "High": pd.Series(12.0 + rng.normal(0, 0.01, len(idx)), index=idx),
        "Low": pd.Series(10.0 + rng.normal(0, 0.01, len(idx)), index=idx),
        "Close": pd.Series(11.5 + rng.normal(0, 0.01, len(idx)), index=idx),
        "Volume": pd.Series(rng.integers(50, 200, len(idx)).astype(float), index=idx),
        # "Amount" intentionally absent → merged-identical close*volume fallback.
    }
    src = _counting_src(data)
    frame = _wide_frame(src)
    assert "amount" in frame.columns
    np.testing.assert_allclose(
        frame["amount"].to_numpy(),
        frame["close"].abs().to_numpy() * frame["volume"].to_numpy(),
    )
    assert src.scan_count == 1, f"scan_count={src.scan_count}"
