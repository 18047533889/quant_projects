"""
IC significance tests.

Covers the small-sample IC inflation fixes in quant_evaluator/metrics/ic.py:

- ``ic_significance`` returns (ir, t_stat, p_value) that rejects pure noise
  (p_value > 0.05) and accepts true signal (p_value < 0.01).
- Sparse-tie factors yield NaN Spearman IC (too few distinct levels) and are
  excluded from the daily mean instead of inflating it.
- Winsorized Pearson recovers near the true correlation in the presence of a
  single outlier; the unwinsorized version is pulled away.
- ``compute_daily_ic`` default ``min_assets`` is 20 (signature-compatible
  bump): a N=10 cross-section yields all-NaN daily IC when min_assets is not
  passed explicitly.

Only quant_evaluator/metrics/ic.py + tests/metrics/ are touched.
"""

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import (
    compute_daily_ic,
    compute_mean_ic,
    ic_significance,
    _pearson_correlation,
    _spearman_rank_correlation,
)

NOISE_SEED = 20260828


def _batch(values):
    T, N, F = values.shape
    return FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(F)),
        time_axis=AxisRef("t", "int", T),
        asset_axis=AxisRef("a", "str", N),
        values=values,
    )


def _bundle(labels):
    T = labels.shape[0]
    return LabelBundle(
        target_id="r",
        values=labels,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )


# ---------------------------------------------------------------------------
# ic_significance
# ---------------------------------------------------------------------------


def test_noise_factor_p_value_accepts_null():
    """Pure-noise factor: daily IC computable, but significance accepts H0."""
    rng = np.random.default_rng(NOISE_SEED)
    T, N = 40, 10
    values = rng.normal(size=(T, N, 1))
    labels = rng.normal(size=(T, N))
    ic_series, _ = compute_daily_ic(
        _batch(values), _bundle(labels), method="pearson", min_assets=5
    )
    assert np.count_nonzero(np.isfinite(ic_series)) >= 30

    ir, t_stat, p_value = ic_significance(ic_series)
    assert p_value > 0.05
    assert abs(t_stat) < 2.0
    assert np.isfinite(ir)


def test_true_signal_p_value_rejects_null():
    """Factor = label + small noise: p_value << 0.01, large t_stat."""
    rng = np.random.default_rng(NOISE_SEED + 1)
    T, N = 60, 20
    labels = rng.normal(size=(T, N))
    values = labels[..., None] + 0.05 * rng.normal(size=(T, N, 1))
    ic_series, _ = compute_daily_ic(
        _batch(values), _bundle(labels), method="pearson", min_assets=10
    )
    assert np.nanmean(ic_series) > 0.5

    ir, t_stat, p_value = ic_significance(ic_series)
    assert p_value < 0.01
    assert t_stat > 2.0
    assert ir > 0.0


def test_ic_significance_small_samples_nan():
    """Fewer than 2 periods -> all-NaN significance."""
    ir, t_stat, p_value = ic_significance(np.array([0.1]))
    assert np.isnan(ir) and np.isnan(t_stat) and np.isnan(p_value)
    ir, t_stat, p_value = ic_significance(np.array([np.nan, 0.2]))
    assert np.isnan(ir) and np.isnan(t_stat) and np.isnan(p_value)


def test_ic_significance_constant_series_nan():
    """Zero std -> all-NaN significance."""
    ir, t_stat, p_value = ic_significance(np.full(10, 0.05))
    assert np.isnan(ir) and np.isnan(t_stat) and np.isnan(p_value)


def test_ic_significance_does_not_change_compute_mean_ic():
    """compute_mean_ic keeps its historical (mean_ic, ic_std) return shape."""
    ic_series = np.array([0.1, 0.2, -0.05, 0.15])
    mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=1)
    assert np.isfinite(mean_ic) and np.isfinite(ic_std)
    ir, t_stat, p_value = ic_significance(ic_series)
    assert np.isfinite(ir) and np.isfinite(t_stat) and np.isfinite(p_value)


# ---------------------------------------------------------------------------
# sparse-tie guard (spearman)
# ---------------------------------------------------------------------------


def test_sparse_tie_factor_spearman_is_nan_and_excluded_from_mean():
    """3-level factor with 30 obs: rank levels < floor -> NaN, no fake IC."""
    factor = np.array(
        [0.0, 1.0, 2.0] * 10, dtype=float
    )  # only 3 distinct levels
    labels = np.arange(30, dtype=float)

    corr = _spearman_rank_correlation(factor, labels, min_obs=10)
    assert np.isnan(corr)

    # And through the daily pipeline: all days NaN, mean excludes them.
    T, N, F = 30, 30, 1
    values = np.tile(factor, (T, 1)).reshape(T, N, F)
    label_panel = np.tile(labels, (T, 1))
    ic_series, counts = compute_daily_ic(
        _batch(values), _bundle(label_panel), method="spearman", min_assets=10
    )
    assert np.all(np.isnan(ic_series))
    mean_ic, _ = compute_mean_ic(ic_series, min_periods=1)
    assert np.isnan(mean_ic)


def test_sparse_tie_factor_excluded_from_daily_mean_but_real_levels_kept():
    """Tie-sparse factor is NaN (excluded); a real-levels factor still computes."""
    rng = np.random.default_rng(11)
    T, N, F = 20, 30, 2
    tie = np.tile(np.array([0.0, 1.0, 2.0] * 10), (T, 1)).reshape(T, N, 1)
    real = rng.normal(size=(T, N, 1))
    values = np.concatenate([tie, real], axis=2)
    labels = rng.normal(size=(T, N))
    ic_series, _ = compute_daily_ic(
        _batch(values), _bundle(labels), method="spearman", min_assets=10
    )
    # tie factor: NaN everywhere; real factor: computable
    assert np.all(np.isnan(ic_series[:, 0]))
    assert np.count_nonzero(np.isfinite(ic_series[:, 1])) >= 15


# ---------------------------------------------------------------------------
# winsorize (pearson)
# ---------------------------------------------------------------------------


def test_winsorize_restores_pearson_with_single_outlier():
    """A single outlier pulls plain Pearson away; winsorize recovers the truth."""
    rng = np.random.default_rng(1234)
    N = 30
    x = rng.normal(size=N)
    y = 0.8 * x + 0.3 * rng.normal(size=N)
    base = _pearson_correlation(x, y, min_obs=5)

    xo = x.copy()
    yo = y.copy()
    xo[0] = 100.0
    yo[0] = -100.0
    raw = _pearson_correlation(xo, yo, min_obs=5)
    win = _pearson_correlation(xo, yo, min_obs=5, winsorize=0.1)

    # Outlier distorts the raw estimate relative to the base truth...
    assert abs(raw - base) > 0.5
    # ...and winsorizing pulls it back toward the true correlation.
    assert abs(win - base) < 0.2
    assert abs(win - base) < abs(raw - base)


def test_winsorize_default_none_keeps_historical_behavior():
    """winsorize=None (default) must equal the unwinsorized correlation."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=40)
    y = x + rng.normal(size=40)
    plain = _pearson_correlation(x, y, min_obs=5)
    default = _pearson_correlation(x, y, min_obs=5, winsorize=None)
    assert plain == default
    assert np.isfinite(plain)


def test_winsorize_rejects_invalid_fraction():
    """winsorize outside (0, 0.5) raises ValueError."""
    x = np.linspace(0, 1, 20)
    y = x.copy()
    for bad in (0.0, 0.5, 0.9, -0.1):
        try:
            _pearson_correlation(x, y, min_obs=5, winsorize=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for winsorize={bad}")


# ---------------------------------------------------------------------------
# min_assets default (20)
# ---------------------------------------------------------------------------


def test_min_assets_default_is_20_all_nan_for_n10():
    """With N=10, the default min_assets=20 yields all-NaN daily IC."""
    rng = np.random.default_rng(99)
    T, N, F = 40, 10, 1
    values = rng.normal(size=(T, N, F))
    labels = rng.normal(size=(T, N))
    ic_series, _ = compute_daily_ic(_batch(values), _bundle(labels))
    assert np.all(np.isnan(ic_series))


def test_min_assets_explicit_10_still_computes_for_n10():
    """Passing min_assets=10 explicitly keeps historical small-cross-section behavior."""
    rng = np.random.default_rng(99)
    T, N, F = 40, 10, 1
    values = rng.normal(size=(T, N, F))
    labels = rng.normal(size=(T, N))
    ic_series, _ = compute_daily_ic(
        _batch(values), _bundle(labels), min_assets=10
    )
    assert np.count_nonzero(np.isfinite(ic_series)) >= 30


def test_min_assets_default_20_computes_for_n30():
    """With N=30, the default min_assets=20 still yields finite daily IC."""
    rng = np.random.default_rng(100)
    T, N, F = 40, 30, 1
    values = rng.normal(size=(T, N, F))
    labels = rng.normal(size=(T, N))
    ic_series, _ = compute_daily_ic(_batch(values), _bundle(labels))
    assert np.count_nonzero(np.isfinite(ic_series)) >= 30


# ---------------------------------------------------------------------------
# constant / degenerate pearson guards
# ---------------------------------------------------------------------------


def test_pearson_constant_nan_via_nanmax_nanmin():
    """Constant input -> NaN via the nanmax-nanmin==0 guard (incl. all-NaN edge)."""
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.arange(5.0)
    assert np.isnan(_pearson_correlation(x, y, min_obs=3))
    # all-NaN y -> pairwise-finite n = 0 < min_obs -> NaN (no crash)
    assert np.isnan(_pearson_correlation(x, np.full(5, np.nan), min_obs=3))


def test_spearman_constant_nan():
    """Constant input -> NaN for spearman."""
    x = np.full(20, 3.0)
    y = np.arange(20.0)
    assert np.isnan(_spearman_rank_correlation(x, y, min_obs=5))
