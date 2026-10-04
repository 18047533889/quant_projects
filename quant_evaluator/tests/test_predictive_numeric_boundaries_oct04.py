"""Public regressions for predictive rolling numeric boundaries."""

import numpy as np
import pytest

from quant_evaluator.metrics.predictive import (
    compute_rank_ic_decay,
    compute_rolling_rank_ic_ir,
)


@pytest.mark.parametrize("window", [0, -1, 2.5, True, False, np.bool_(True)])
def test_public_rolling_ir_rejects_nonpositive_or_nonintegral_windows(window):
    # Prevent division-by-zero / slicing crashes and silently truncated windows.
    series = np.linspace(0.1, 0.4, 8)[:, None]
    with pytest.raises(ValueError):
        compute_rolling_rank_ic_ir(series, window=window, min_periods=2)


@pytest.mark.parametrize("min_periods", [0, -1, 1.5, True, False, np.bool_(True)])
def test_public_rolling_ir_rejects_nonpositive_or_nonintegral_min_periods(min_periods):
    # Prevent a malformed warmup threshold from silently admitting/skipping data.
    series = np.linspace(0.1, 0.4, 8)[:, None]
    with pytest.raises(ValueError):
        compute_rolling_rank_ic_ir(series, window=3, min_periods=min_periods)


def test_public_rolling_ir_accepts_numpy_integrals_and_allows_min_periods_above_window():
    series = np.linspace(0.1, 0.4, 8)[:, None]
    result = compute_rolling_rank_ic_ir(
        series, window=np.int64(3), min_periods=np.int64(2)
    )
    assert result.shape == (1,)
    assert np.isfinite(result[0])

    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        not_enough_for_any_window = compute_rolling_rank_ic_ir(
            series, window=2, min_periods=3
        )
    assert np.isnan(not_enough_for_any_window[0])


def test_public_rank_ic_decay_uses_stable_pairwise_centered_correlation():
    # The raw-moment formula n*sum(x*y)-sum(x)*sum(y) loses the tiny variance
    # of bounded ICs around a nonzero offset, especially at lag 5.
    n = 100_000
    t = np.arange(n)
    values = 0.25 + np.where(t % 2, 2e-8, -2e-8)
    values[t % 17 == 0] = np.nan
    series = values[:, None]

    lags = (1, 5, 10, 20)
    oracle = []
    for lag in lags:
        pair = np.isfinite(values[:-lag]) & np.isfinite(values[lag:])
        oracle.append(np.corrcoef(values[:-lag][pair], values[lag:][pair])[0, 1])
    expected = float(np.mean(oracle))

    actual = compute_rank_ic_decay(series, horizons=lags)
    assert abs(expected) < 1e-12
    np.testing.assert_allclose(actual, [expected], rtol=1e-10, atol=1e-12)


def test_public_rank_ic_decay_reports_nan_for_pairwise_constant_nonbinary_columns():
    # Centering the raw levels can round a constant mean away from its value,
    # creating identical nonzero residuals and a spurious correlation of 1.
    t = np.arange(2_000)
    series = np.full((len(t), 3), np.nan)
    masks = (
        t % 17 != 0,
        t % 5 < 4,
        t % 6 < 4,
    )
    for column, (value, mask) in enumerate(zip((0.1, 0.3, 0.996908), masks)):
        series[mask, column] = value

    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        actual = compute_rank_ic_decay(series)
    assert actual.shape == (3,)
    assert np.isnan(actual).all()


def test_public_rank_ic_decay_pairwise_constant_side_is_undefined():
    # Disjoint valid blocks create pairs whose right-hand values are constant;
    # pairwise correlation is undefined even when the left side varies.
    t = np.arange(130)
    values = np.full(130, np.nan)
    values[:30] = 0.1 + np.arange(30) * 1e-8
    values[100:] = 0.3
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        actual = compute_rank_ic_decay(values[:, None], horizons=(100,))
    assert np.isnan(actual[0])
