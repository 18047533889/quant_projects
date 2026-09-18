"""Independent parity contracts for next-stage jump and path statistics."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.intraday.polars_next_stage import _jump_stats, _max_path


def _log_returns(values):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.log(values[1:] / values[:-1])
    return out


def _jump_oracle(values, scale):
    returns = _log_returns(values)
    finite = returns[np.isfinite(returns)]
    mask = np.zeros(len(returns), dtype=bool)
    if len(finite) >= 2:
        rv = np.sum(finite * finite)
        if np.isfinite(rv) and rv > 1e-12:
            mask = np.isfinite(returns) & (np.abs(returns) > scale * np.sqrt(rv / len(finite)))
    jumps = returns[mask]
    if not len(jumps):
        return {"count": 0.0, "pos": np.nan, "neg": np.nan, "signed_ratio": np.nan, "concentration": np.nan}
    pos = np.sum(jumps[jumps > 0] ** 2)
    neg = np.sum(jumps[jumps < 0] ** 2)
    total = pos + neg
    return {
        "count": float(len(jumps)),
        "pos": float(pos),
        "neg": float(neg),
        "signed_ratio": float((pos - neg) / (total + 1e-12)),
        "concentration": float(np.sum((jumps * jumps) ** 2) / (total * total)) if total > 1e-12 else np.nan,
    }


def _path_oracle(values, side):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return np.nan
    if side == "down":
        running = np.maximum.accumulate(finite)
        return np.min(finite / running - 1.0)
    running = np.minimum.accumulate(finite)
    return np.max(finite / running - 1.0)


def _fixture():
    times = list(pd.date_range("2025-05-05 09:30", periods=7, freq="min"))
    times += list(pd.date_range("2025-05-06 09:30", periods=4, freq="min"))
    a = [100, 101, 99, 110, 108, 107, 109, 50, np.nan, np.inf, np.nan]
    b = [20, 19, np.nan, 18, 21, 17, 22, 30, 31, 29, 33]
    values = {"A": np.asarray(a, float), "B": np.asarray(b, float)}
    return times, values, pl.DataFrame({"timestamp": times, **values})


@pytest.mark.parametrize("stat", ["count", "pos", "neg", "signed_ratio", "concentration"])
def test_jump_stats_match_independent_finite_return_oracle(stat):
    times, values, frame = _fixture()
    actual = _jump_stats(frame, stat, 1.0).sort("date")
    dates = pd.Series(times).dt.date.to_numpy()
    for column in ("A", "B"):
        expected = [_jump_oracle(values[column][dates == day], 1.0)[stat] for day in sorted(set(dates))]
        np.testing.assert_allclose(actual[column].to_numpy(), expected, equal_nan=True)
    assert actual.columns == ["date", "A", "B"]


@pytest.mark.parametrize("side", ["down", "up"])
def test_max_path_matches_finite_price_oracle_and_one_sample_contract(side):
    times, values, frame = _fixture()
    actual = _max_path(frame, side).sort("date")
    dates = pd.Series(times).dt.date.to_numpy()
    for column in ("A", "B"):
        expected = [_path_oracle(values[column][dates == day], side) for day in sorted(set(dates))]
        np.testing.assert_allclose(actual[column].to_numpy(), expected, equal_nan=True)
    assert np.isnan(actual["A"].to_numpy()[1])
    assert actual.columns == ["date", "A", "B"]


def test_jump_stats_reject_invalid_threshold_and_single_price_is_no_jump():
    _, _, frame = _fixture()
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite number > 0"):
            _jump_stats(frame, "count", bad)
    one = pl.DataFrame({"timestamp": [pd.Timestamp("2025-05-07 09:30")], "A": [100.0]})
    count = _jump_stats(one, "count", 3.0)
    pos = _jump_stats(one, "pos", 3.0)
    assert count["A"].to_list() == [0]
    assert np.isnan(pos["A"].to_numpy()[0])


@pytest.mark.parametrize("side", ["up", "down"])
def test_entirely_missing_symbol_and_day_and_empty_input_keep_grid(side):
    frame = pl.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 09:30","2024-01-01 09:31","2024-01-02 09:30"]),
        "B": [np.nan, np.nan, np.nan], "A": [100., 101., np.nan],
    })
    actual = _max_path(frame, side)
    assert actual.columns == ["date", "B", "A"]
    assert actual.height == 2
    assert np.isnan(actual["B"].to_numpy()).all()
    assert np.isnan(actual["A"].to_numpy()[1])
    assert _max_path(frame.head(0), side).shape == (0,3)
    assert _jump_stats(frame.head(0), "count", 1.).shape == (0,3)


@pytest.mark.parametrize("side", ["up", "down"])
def test_zero_running_reference_is_undefined_not_ignored(side):
    frame = pl.DataFrame({"timestamp": pd.date_range("2024-01-01 09:30", periods=3, freq="min"),
                          "A": [0., 1., 2.]})
    actual = _max_path(frame, side)["A"].to_numpy()
    assert np.isnan(actual).all()
