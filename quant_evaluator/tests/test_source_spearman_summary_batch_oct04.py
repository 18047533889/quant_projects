"""Public source-batch parity for scalar summaries of daily rank IC."""

import numpy as np
import pytest
from scipy.stats import rankdata

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle

METRICS = (
    "rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir",
)


class _TinySource:
    def __init__(self, values, validity, times, assets, factor_ids):
        self.values = values
        self.validity = validity
        self.time_axis = times
        self.asset_axis = assets
        self.factor_ids = tuple(factor_ids)
        self.dtype = "float64"
        self.snapshot_id = "tiny-source-spearman-summary-oct04"
        self.max_tile_size = len(factor_ids)
        self.reads = []

    def read_tile(self, start, end):
        self.reads.append((start, end))
        batch = FactorBatch(
            self.factor_ids[start:end], self.time_axis, self.asset_axis,
            self.values[:, :, start:end],
            validity=self.validity[:, :, start:end],
        )
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        pass


def _inputs():
    t_count, n_assets, n_factors = 100, 36, 4
    times = np.arange(t_count, dtype=np.int64)
    time_axis = AxisRef("time", "int64", t_count, times)
    assets = AxisRef("asset", "str", n_assets,
                     np.asarray([f"A{i:03d}" for i in range(n_assets)]))
    asset_order = np.arange(n_assets, dtype=np.float64)
    labels = np.broadcast_to(asset_order, (t_count, n_assets)).copy()
    values = np.empty((t_count, n_assets, n_factors), dtype=np.float64)
    values[:, :, 0] = np.broadcast_to(asset_order // 3, (t_count, n_assets))
    signs = np.where(np.arange(t_count) % 2 == 0, 1.0, -1.0)
    values[:, :, 1] = values[:, :, 0] * signs[:, None]
    values[:, :, 2] = values[:, :, 0]
    values[:, :, 3] = values[:, :, 0]
    validity = np.ones_like(values, dtype=bool)
    validity[np.r_[5:22, 70:86], :, 1] = False
    validity[20:, :, 2] = False
    validity[19:, :, 3] = False
    validity[np.r_[41:64, 90:100], :, 0] = False
    label_bundle = LabelBundle(
        "tiny-label", labels, 1, decision_time=tuple(times),
        observation_time=tuple(times), signal_available_time=tuple(times),
        execution_time=tuple(times), label_start_time=tuple(times),
        label_end_time=tuple(times + 1), asset_axis=assets,
    )
    source = _TinySource(values, validity, time_axis, assets,
                         ("factor0", "factor1", "factor2", "factor3"))
    return source, label_bundle


def _raw_rank_ic(source, label):
    """Independent day/factor Spearman oracle directly from panel values."""
    t_count, _, f_count = source.values.shape
    daily = np.full((t_count, f_count), np.nan)
    for t in range(t_count):
        y = label.values[t]
        for f in range(f_count):
            x = source.values[t, :, f]
            valid = (source.validity[t, :, f] & np.isfinite(x)
                     & np.isfinite(y))
            if valid.sum() < 20:
                continue
            xr = rankdata(x[valid], method="average")
            yr = rankdata(y[valid], method="average")
            if np.ptp(xr) and np.ptp(yr):
                daily[t, f] = np.corrcoef(xr, yr)[0, 1]
    return daily


def _summary_oracle(daily):
    count = np.isfinite(daily).sum(axis=0)
    positive = np.full(daily.shape[1], np.nan)
    recent = np.full(daily.shape[1], np.nan)
    rolling_ir = np.full(daily.shape[1], np.nan)
    for f in range(daily.shape[1]):
        values = daily[:, f]
        finite = values[np.isfinite(values)]
        if count[f] >= 20:
            positive[f] = np.count_nonzero(finite > 0) / count[f]
            tail = values[-63:]
            if np.isfinite(tail).any():
                recent[f] = np.nanmean(tail)
        window_irs = []
        for end in range(1, len(values) + 1):
            window = values[max(0, end - 60):end]
            window = window[np.isfinite(window)]
            if len(window) < 20:
                continue
            std = np.std(window, ddof=1)
            if std > 1e-12:
                window_irs.append(np.mean(window) / std)
        if window_irs:
            rolling_ir[f] = np.mean(window_irs)
    return count, positive, recent, rolling_ir


def test_public_cpu_source_summaries_match_independent_raw_panel_oracle():
    """Source API admits these summaries and retains IC-series counts."""
    source, label = _inputs()
    try:
        result = evaluate_factor_source_batch(
            source, label, metrics=METRICS, backend="cpu", max_tile_size=4,
        )
    finally:
        source.close()

    daily = _raw_rank_ic(source, label)
    counts, positive, recent, rolling_ir = _summary_oracle(daily)
    assert result.factor_ids == source.factor_ids
    assert set(result.scalar_metrics) == set(METRICS)
    assert result.series_metrics == {}
    assert source.reads == [(0, 4)]
    np.testing.assert_allclose(result.scalar_metrics["rank_ic_positive_ratio"],
                               positive, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(result.scalar_metrics["recent_3m_rank_ic"],
                               recent, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(result.scalar_metrics["rolling_rank_ic_ir"],
                               rolling_ir, rtol=1e-10, atol=1e-12, equal_nan=True)
    for metric in METRICS:
        np.testing.assert_array_equal(result.observation_counts[metric], counts)
    assert counts.tolist() == [67, 67, 20, 19]
    assert positive[0] == 1.0
    assert 0.0 < positive[1] < 1.0
    assert np.isnan(result.scalar_metrics["rank_ic_positive_ratio"][3])
    assert np.isnan(result.scalar_metrics["recent_3m_rank_ic"][3])
    assert np.isnan(result.scalar_metrics["rolling_rank_ic_ir"][2])
def _direct_rolling_ir(series, window=60, min_periods=20):
    expected = np.full(series.shape[1], np.nan)
    for factor in range(series.shape[1]):
        values = series[:, factor]
        valid_irs = []
        for end in range(1, len(values) + 1):
            window_values = values[max(0, end - window):end]
            window_values = window_values[np.isfinite(window_values)]
            if window_values.size < min_periods:
                continue
            mean = np.mean(window_values)
            centered = window_values - mean
            std = np.sqrt(np.sum(centered * centered) / (window_values.size - 1))
            if std > 1e-12:
                valid_irs.append(mean / std)
        if valid_irs:
            expected[factor] = np.mean(valid_irs)
    return expected


def _rolling_fixture():
    values = np.empty((100, 6), dtype=np.float64)
    values[:, 0] = 0.25
    values[:20, 1] = 10.0
    values[20:, 1] = 0.25 + np.where(np.arange(80) % 2, 2e-8, -2e-8)
    values[:, 2] = 0.1 + (np.arange(100) % 5) * 0.01
    values[:, 3] = -0.2 + np.arange(100) * 0.003
    values[:, 4] = 0.25 + np.where(np.arange(100) % 2, 2e-12, -2e-12)
    values[:, 5] = 0.25 + np.where(np.arange(100) % 2, 2.5e-13, -2.5e-13)
    values[np.arange(100) % 4 == 0, 0] = np.nan
    values[np.arange(100) % 9 == 0, 1] = np.nan
    values[np.r_[25:45, 70:78], 2] = np.nan
    values[19:, 3] = np.nan
    return values


def test_cpu_rolling_rank_ic_ir_uses_stable_sample_variance_for_constant_and_nearconstant():
    from quant_evaluator.metrics.predictive import compute_rolling_rank_ic_ir

    series = _rolling_fixture()
    expected = _direct_rolling_ir(series)
    actual = compute_rolling_rank_ic_ir(series)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12,
                               equal_nan=True)
    assert np.isnan(actual[0])
    assert np.isfinite(actual[1])
    assert abs(actual[1] - expected[1]) < 1e-3


def test_gpu_rolling_rank_ic_ir_uses_stable_sample_variance_for_constant_and_nearconstant():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from quant_evaluator.kernels.gpu.predictive import rolling_rank_ic_ir

    series = _rolling_fixture()
    expected = _direct_rolling_ir(series)
    actual = rolling_rank_ic_ir(cp.asarray(series))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12,
                               equal_nan=True)
    assert np.isnan(actual[0])
    assert np.isfinite(actual[1])
    assert abs(actual[1] - expected[1]) < 1e-3
def test_public_cuda_source_summaries_share_one_real_spearman_pass(monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from quant_evaluator.kernels.gpu import correlation

    source, label = _inputs()
    original = correlation.batched_spearman_ic
    calls = []

    def counted_spearman(factors, labels, *args, **kwargs):
        calls.append((tuple(factors.shape), tuple(labels.shape)))
        return original(factors, labels, *args, **kwargs)

    monkeypatch.setattr(correlation, "batched_spearman_ic", counted_spearman)
    try:
        result = evaluate_factor_source_batch(
            source, label, metrics=METRICS, backend="cuda_strict",
            max_tile_size=4,
        )
    finally:
        source.close()

    daily = _raw_rank_ic(source, label)
    counts, positive, recent, rolling_ir = _summary_oracle(daily)
    assert result.factor_ids == source.factor_ids
    assert set(result.scalar_metrics) == set(METRICS)
    assert result.series_metrics == {}
    assert result.metadata["backend_used"] == "cuda"
    assert source.reads == [(0, 4)]
    assert len(calls) == 1
    assert calls[0] == ((100, 4, 36), (100, 36))
    np.testing.assert_allclose(result.scalar_metrics["rank_ic_positive_ratio"],
                               positive, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(result.scalar_metrics["recent_3m_rank_ic"],
                               recent, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(result.scalar_metrics["rolling_rank_ic_ir"],
                               rolling_ir, rtol=1e-10, atol=1e-12, equal_nan=True)
    for metric in METRICS:
        np.testing.assert_array_equal(result.observation_counts[metric], counts)


def _direct_window_statistics(values):
    means = np.full(values.shape, np.nan)
    irs = np.full(values.shape, np.nan)
    for factor in range(values.shape[1]):
        for end in range(1, len(values) + 1):
            sample = values[max(0, end - 60):end, factor]
            sample = sample[np.isfinite(sample)]
            if sample.size < 20:
                continue
            means[end - 1, factor] = np.mean(sample)
            std = np.std(sample, ddof=1)
            if std > 1e-12:
                irs[end - 1, factor] = np.mean(sample) / std
    return means, irs


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
@pytest.mark.parametrize("factor_count", (0, 1, 1093))
def test_rolling_mean_and_ir_preserve_factor_chunks_and_empty_axis(backend, factor_count):
    """Catch writes to the whole output instead of the current factor slice."""
    base = _rolling_fixture()
    columns = np.arange(factor_count) % base.shape[1]
    values = base[:, columns]
    expected_mean, expected_ir = _direct_window_statistics(base)
    if backend == "cpu":
        from quant_evaluator.metrics.predictive import _rolling_mean_ir
        actual_mean, actual_ir = _rolling_mean_ir(values, 60, 20)
    else:
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
        from quant_evaluator.kernels.gpu.predictive import _rolling_mean_ir
        actual_mean, actual_ir = _rolling_mean_ir(cp.asarray(values), 60, 20)
        actual_mean, actual_ir = cp.asnumpy(actual_mean), cp.asnumpy(actual_ir)
    assert actual_mean.shape == actual_ir.shape == (100, factor_count)
    np.testing.assert_allclose(actual_mean, expected_mean[:, columns],
                               rtol=1e-12, atol=1e-14, equal_nan=True)
    np.testing.assert_allclose(actual_ir, expected_ir[:, columns],
                               rtol=1e-10, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_rolling_std_threshold_and_twenty_observation_gate(backend):
    """Catch changing the variance threshold or counting missing dates as observations."""
    values = np.empty((80, 3), dtype=np.float64)
    signs = np.where(np.arange(80) % 2, 1.0, -1.0)
    values[:, 0] = 0.25 + signs * 5e-13
    values[:, 1] = 0.25 + signs * 2e-12
    values[:, 2] = 0.25 + signs * 0.02
    values[10, 2] = np.nan
    expected_mean, expected_ir = _direct_window_statistics(values)
    if backend == "cpu":
        from quant_evaluator.metrics.predictive import _rolling_mean_ir
        means, irs = _rolling_mean_ir(values, 60, 20)
    else:
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
        from quant_evaluator.kernels.gpu.predictive import _rolling_mean_ir
        means, irs = _rolling_mean_ir(cp.asarray(values), 60, 20)
        means, irs = cp.asnumpy(means), cp.asnumpy(irs)
    np.testing.assert_allclose(means, expected_mean, rtol=1e-12,
                               atol=1e-14, equal_nan=True)
    np.testing.assert_allclose(irs, expected_ir, rtol=1e-10,
                               atol=1e-12, equal_nan=True)
    assert np.isnan(irs[:, 0]).all()
    assert np.isfinite(irs[19:, 1]).all()
    assert np.isnan(means[19, 2]) and np.isnan(irs[19, 2])
    assert np.isfinite(means[20, 2]) and np.isfinite(irs[20, 2])
