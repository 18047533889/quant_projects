from math import fsum, sqrt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, rankdata

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts.source_all24_ic_oracle import (
    IC_TILE_METRICS, _observation_counts, _summarize, reference_ic_tile,
)


def _fixture(T=24):
    n = 21
    times = pd.date_range("2026-01-01", periods=T, freq="B")
    time_axis = AxisRef("time", "datetime64[ns]", T, times.to_numpy())
    asset_axis = AxisRef("asset", "int64", n, np.arange(n, dtype=np.int64))
    y = np.tile(np.arange(n, dtype=np.float64), (T, 1))
    x = np.empty((T, n, 4), dtype=np.float64)
    x[:, :, 0] = y
    for t in range(T):
        x[t, :, 1] = y[t] if t % 2 == 0 else -y[t]
    x[:, :, 2] = np.repeat(np.arange(7, dtype=np.float64), 3)
    x[:, :, 3] = 1.0
    label_validity = np.ones((T, n), dtype=bool)
    factor_validity = np.ones((T, n, 4), dtype=bool)
    if T > 2:
        y[2, :2] = (np.nan, np.inf)
    if T > 3:
        factor_validity[3, :2, 0] = False
    if T > 4:
        label_validity[4, 0] = False
    if T > 5:
        x[5, 0, 0] = np.inf
    batch = FactorBatch(tuple(f"f{i}" for i in range(4)), time_axis,
                        asset_axis, x, validity=factor_validity)
    labels = LabelBundle("r", y, 1, decision_time=tuple(time_axis.values),
        label_start_time=tuple(times), label_end_time=tuple(times + pd.Timedelta(days=1)),
        asset_axis=asset_axis, validity=label_validity)
    return batch, labels


def _summary_oracle(series):
    T, F = series.shape
    out = {name: np.full(F, np.nan) for name in (
        "rank_ic", "ic_ir", "ic_std", "ic_median", "pearson_ic",
        "pearson_ic_std", "pearson_ic_ir", "rank_ic_positive_ratio",
        "recent_3m_rank_ic", "rolling_rank_ic_ir")}
    for f in range(F):
        finite = [float(v) for v in series[:, f] if np.isfinite(v)]
        n = len(finite)
        if n:
            out["rank_ic"][f] = fsum(finite) / n
            out["pearson_ic"][f] = fsum(finite) / n
        if n >= 20:
            mean = fsum(finite) / n
            variance = fsum((v - mean) ** 2 for v in finite) / (n - 1)
            std = sqrt(variance)
            out["ic_std"][f] = out["pearson_ic_std"][f] = std
            if any(v != finite[0] for v in finite) and std > 0:
                out["ic_ir"][f] = out["pearson_ic_ir"][f] = mean / std
            ordered = sorted(finite)
            out["ic_median"][f] = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
            out["rank_ic_positive_ratio"][f] = sum(v > 0 for v in finite) / n
            tail = [float(v) for v in series[max(0, T - 63):, f] if np.isfinite(v)]
            if tail:
                out["recent_3m_rank_ic"][f] = fsum(tail) / len(tail)
        rolling = []
        for end in range(T):
            window = [float(v) for v in series[max(0, end - 59):end + 1, f]
                      if np.isfinite(v)]
            if len(window) < 20:
                continue
            mean = fsum(window) / len(window)
            std = sqrt(fsum((v - mean) ** 2 for v in window) / (len(window) - 1))
            if std > 1e-12:
                rolling.append(mean / std)
        if rolling:
            out["rolling_rank_ic_ir"][f] = fsum(rolling) / len(rolling)
    return out


def test_reference_returns_exact_twelve_metrics_with_handchecked_rows_and_counts():
    batch, labels = _fixture()
    result = reference_ic_tile(batch, labels)
    assert isinstance(result, BatchEvaluationBundle)
    assert set(result.series_metrics) == {"rank_ic_series", "pearson_ic_series"}
    assert set(result.scalar_metrics) == set(IC_TILE_METRICS) - set(result.series_metrics)
    assert set(result.observation_counts) == set(IC_TILE_METRICS)
    rank = result.series_metrics["rank_ic_series"]
    pearson = result.series_metrics["pearson_ic_series"]
    assert rank.shape == pearson.shape == (24, 4)
    np.testing.assert_array_equal(rank[0], [1.0, 1.0, pearsonr(
        rankdata(batch.values[0, :, 2]), rankdata(labels.values[0])).statistic, np.nan])
    assert rank[1, 1] == -1.0
    assert np.isnan(rank[2]).all()  # 19 pairwise-valid assets, below the 20 gate.
    assert result.observation_counts["rank_ic_series"].tolist() == [22, 23, 23, 0]
    for metric in IC_TILE_METRICS:
        np.testing.assert_array_equal(result.observation_counts[metric], [22, 23, 23, 0])
    for name, expected in _summary_oracle(rank).items():
        if name.startswith("pearson_ic"):
            continue
        np.testing.assert_allclose(result.scalar_metrics[name], expected,
                                   rtol=1e-12, atol=1e-12, equal_nan=True)
    for name, expected in _summary_oracle(pearson).items():
        if name in {"rank_ic", "ic_ir", "ic_std", "ic_median",
                    "rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir"}:
            continue
        np.testing.assert_allclose(result.scalar_metrics[name], expected,
                                   rtol=1e-12, atol=1e-12, equal_nan=True)


def test_mean_metrics_have_one_day_gate_while_std_ir_and_predictive_use_twenty():
    batch, labels = _fixture(T=3)
    result = reference_ic_tile(batch, labels)
    assert np.isfinite(result.scalar_metrics["rank_ic"][:3]).all()
    assert np.isfinite(result.scalar_metrics["pearson_ic"][:3]).all()
    for name in ("ic_std", "ic_ir", "ic_median", "pearson_ic_std",
                 "pearson_ic_ir", "rank_ic_positive_ratio", "recent_3m_rank_ic",
                 "rolling_rank_ic_ir"):
        assert np.isnan(result.scalar_metrics[name]).all()


def test_scipy_average_rank_ties_and_pearson_match_hand_calculation():
    batch, labels = _fixture(T=1)
    result = reference_ic_tile(batch, labels)
    expected_rank = pearsonr(rankdata(batch.values[0, :, 2], method="average"),
                             rankdata(labels.values[0], method="average")).statistic
    expected_pearson = pearsonr(batch.values[0, :, 2], labels.values[0]).statistic
    assert result.series_metrics["rank_ic_series"][0, 2] == pytest.approx(expected_rank)
    assert result.series_metrics["pearson_ic_series"][0, 2] == pytest.approx(expected_pearson)
    assert np.isnan(result.series_metrics["rank_ic_series"][0, 3])
    assert np.isnan(result.series_metrics["pearson_ic_series"][0, 3])


def test_whole_series_ir_keeps_tiny_real_variation_while_rolling_uses_epsilon_gate():
    series = np.full((20, 1), .123, dtype=np.float64)
    series[-1, 0] = np.nextafter(series[-1, 0], np.inf)
    result = _summarize(series, method="rank")
    assert result["ic_std"][0] > 0
    assert np.isfinite(result["ic_ir"][0])
    assert np.isnan(result["rolling_rank_ic_ir"][0])


def test_metric_family_counts_are_derived_from_their_own_finite_series():
    # Controlled series results exercise count ownership independently of the
    # numerical row algorithm, where valid rank/Pearson eligibility normally agrees.
    rank = np.array([[.2, np.nan], [np.nan, .1], [.5, np.nan]])
    pearson = np.array([[.1, np.nan], [np.nan, .2], [np.nan, .3]])
    counts = _observation_counts(rank, pearson)
    for metric in ("rank_ic", "rank_ic_series", "ic_ir", "ic_std", "ic_median",
                   "rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir"):
        np.testing.assert_array_equal(counts[metric], [2, 1])
    for metric in ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"):
        np.testing.assert_array_equal(counts[metric], [1, 2])


def test_pearson_rows_are_scale_safe_for_extreme_and_near_constant_finite_values():
    T, N = 3, 40
    times = pd.date_range("2026-02-01", periods=T, freq="B")
    time_axis = AxisRef("time", "datetime64[ns]", T, times.to_numpy())
    asset_axis = AxisRef("asset", "int64", N, np.arange(N, dtype=np.int64))
    large = 1e308 * (0.5 + np.linspace(0.0, 0.4, N))
    near = 1e12 + np.arange(N, dtype=np.float64) * 0.00025
    factor = np.stack((large, large, near), axis=0)[:, :, None]
    labels_values = np.stack((large, -large, near), axis=0)
    batch = FactorBatch(("f0",), time_axis, asset_axis, factor)
    labels = LabelBundle("r", labels_values, 1, decision_time=tuple(time_axis.values),
        label_start_time=tuple(times), label_end_time=tuple(times + pd.Timedelta(days=1)),
        asset_axis=asset_axis)
    result = reference_ic_tile(batch, labels)
    np.testing.assert_allclose(result.series_metrics["pearson_ic_series"][:, 0],
                               [1.0, -1.0, 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.series_metrics["rank_ic_series"][:, 0],
                               [1.0, -1.0, 1.0], rtol=0, atol=1e-12)
    for metric in IC_TILE_METRICS:
        np.testing.assert_array_equal(result.observation_counts[metric], [3])
