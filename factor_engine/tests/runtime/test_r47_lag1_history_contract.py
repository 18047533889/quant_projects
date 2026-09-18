"""Incremental-history contracts for adjacent-pair rolling operators."""
from factor_engine.cleaned_operators.polars_native import ts_advanced_batch5  # noqa: F401
from factor_engine.runtime.execution_contract import (
    forward_impact,
    history_requirement,
    own_history_requirement,
)


def test_lag1_autocorr_window_counts_pairs_not_raw_bars():
    params = {"window": 20, "n_bins": 5}
    assert own_history_requirement("ts_lag1_autocorr", params).rows == 20
    assert history_requirement("ts_lag1_autocorr", params).rows == 20
    assert forward_impact("ts_lag1_autocorr", params) == 20


def test_lag1_autocorr_history_tracks_explicit_window():
    for window in (20, 37, 120):
        params = {"window": window, "n_bins": 5}
        assert own_history_requirement("ts_lag1_autocorr", params).rows == window
        assert forward_impact("ts_lag1_autocorr", params) == window


def test_jump_bipower_keeps_raw_window_history_contract():
    for window in (5, 20, 61):
        params = {"window": window}
        assert own_history_requirement("ts_jump_bipower", params).rows == window - 1
        assert forward_impact("ts_jump_bipower", params) == window - 1


def test_incremental_overlap_reproduces_full_values_and_short_overlap_fails():
    import numpy as np
    import polars as pl
    from factor_engine.cleaned_operators.polars_native.ts_advanced_batch5 import (
        TSLag1AutocorrPolarsNative, TSJumpBipowerPolarsNative,
    )
    values = np.random.default_rng(4701).normal(size=100)
    frame = pl.DataFrame({"A": values})
    for name, cls, window in (
        ("ts_lag1_autocorr", TSLag1AutocorrPolarsNative, 20),
        ("ts_jump_bipower", TSJumpBipowerPolarsNative, 5),
    ):
        overlap = own_history_requirement(name, {"window": window}).rows
        full = cls().calculate(frame, window=window)["A"].to_numpy()
        start = 50
        chunk = cls().calculate(frame.slice(start-overlap), window=window)["A"].to_numpy()[overlap:]
        np.testing.assert_allclose(chunk, full[start:], equal_nan=True, atol=1e-13)
        short = cls().calculate(frame.slice(start-overlap+1), window=window)["A"].to_numpy()[overlap-1:]
        assert np.isnan(short[0]) and np.isfinite(full[start])
