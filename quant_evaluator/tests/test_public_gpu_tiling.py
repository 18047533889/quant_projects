"""Small real-CUDA public API tests; pandas supplies independent rank oracle."""
import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.runtime.gpu_executor import GPUExecutor
from quant_evaluator.runtime.evaluator import evaluate


def _contracts():
    rng = np.random.default_rng(937)
    x = rng.integers(0, 12, size=(4, 40, 5)).astype(float)
    y = rng.normal(size=(4, 40))
    valid = np.ones(x.shape, dtype=bool)
    valid[:, :3, :] = False
    yvalid = np.ones(y.shape, dtype=bool)
    yvalid[:, 3:6] = False
    x[1, 9, 2] = np.nan
    times = tuple(pd.date_range("2024-01-01", periods=4))
    fb = FactorBatch(tuple(f"f{i}" for i in range(5)),
                     AxisRef("time", "datetime", 4), AxisRef("asset", "str", 40),
                     x, validity=valid)
    lb = LabelBundle("ret", y, 1, decision_time=times,
                     label_start_time=times,
                     label_end_time=tuple(t + pd.Timedelta(days=1) for t in times),
                     validity=yvalid)
    return fb, lb


def _oracle(fb, lb):
    expected = np.full((4, 5), np.nan)
    for t in range(4):
        for f in range(5):
            good = fb.validity[t, :, f] & lb.validity[t]
            good &= np.isfinite(fb.values[t, :, f]) & np.isfinite(lb.values[t])
            a = pd.Series(fb.values[t, good, f]).rank(method="average")
            b = pd.Series(lb.values[t, good]).rank(method="average")
            expected[t, f] = a.corr(b)
    return expected


def test_public_api_tiles_before_upload_and_respects_validity(monkeypatch):
    fb, lb = _contracts()
    uploaded = []
    original = DeviceEvaluationSession.stage_factors

    def stage(self, values, factor_ids, layout="T,F,N"):
        uploaded.append(len(factor_ids))
        return original(self, values, factor_ids, layout)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 2)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", stage)
    out = evaluate(fb, lb, backend="cuda_strict", metrics=("rank_ic_series", "coverage", "quantile_returns_full"))
    assert uploaded == [2, 2, 1]
    assert out.factor_ids == fb.factor_ids
    assert out.vector_metrics["quantile_returns_full"].shape == (5, 5)
    np.testing.assert_allclose(out.series_metrics["rank_ic_series"], _oracle(fb, lb), atol=1e-12)
    valid = fb.validity & lb.validity[:, :, None] & np.isfinite(fb.values)
    np.testing.assert_allclose(out.scalar_metrics["coverage"], valid.mean(axis=(0, 1)))
    assert out.metadata["factor_tiles_processed"] == 3


@pytest.mark.parametrize("retry", [True, False])
def test_public_oom_retry_is_bounded_and_preserves_results(monkeypatch, retry):
    fb, lb = _contracts()
    original = GPUExecutor.run
    attempts = []

    def run(self, factor_ids, metrics, label_id="next_ret"):
        attempts.append(len(factor_ids))
        if len(factor_ids) > 2:
            raise cp.cuda.memory.OutOfMemoryError(100, 100, 100)
        return original(self, factor_ids, metrics, label_id)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 4)
    monkeypatch.setattr(GPUExecutor, "run", run)
    policy = GPUExecutionPolicy(oom_retile=retry)
    if not retry:
        with pytest.raises(cp.cuda.memory.OutOfMemoryError):
            evaluate(fb, lb, backend="cuda", gpu_policy=policy, metrics=("rank_ic_series",))
        assert attempts == [4]
    else:
        out = evaluate(fb, lb, backend="cuda", gpu_policy=policy, metrics=("rank_ic_series",))
        assert attempts == [4, 2, 2, 1]
        assert out.metadata["oom_retries"] == 1
        np.testing.assert_allclose(out.series_metrics["rank_ic_series"], _oracle(fb, lb), atol=1e-12)


def test_budget_does_not_force_unfitting_minimum_tile():
    session = DeviceEvaluationSession()
    session._vram_budget = session._estimate_working_set(None, 500, 2000, 2, 8)
    assert session.estimate_tile(None, 500, 2000, 8) == 2
    session._vram_budget = 1
    with pytest.raises(MemoryError, match="single factor"):
        session.estimate_tile(None, 500, 2000, 8)


def test_public_l20_quantile_monotonicity_uses_strict_shape_kernel(monkeypatch):
    """QE-08: formal metric is monotonicity of the mean profile, not daily votes."""
    T,N,F = 40,100,2
    ranks = np.arange(N,dtype=float)
    values = np.broadcast_to(ranks[None,:,None],(T,N,F)).copy()
    validity = np.ones_like(values,dtype=bool)
    validity[:,30:,1] = False  # five buckets cannot meet min_assets=10 for f1
    labels = np.empty((T,N),dtype=float)
    labels[:25] = ranks
    labels[25:] = -ranks  # 15 daily profiles reverse, mean profile still increases
    fb = FactorBatch(('mean_monotone','insufficient'),AxisRef('time','int',T),
        AxisRef('asset','int',N),values,validity=validity)
    lb = LabelBundle('ret',labels,1,decision_time=tuple(range(T)),
        label_start_time=tuple(range(1,T+1)),label_end_time=tuple(range(2,T+2)))
    metrics = ('quantile_monotonicity','daily_quantile_monotonicity_series',
               'daily_quantile_monotonicity_rate')
    cpu = evaluate(fb,lb,metrics=metrics)
    monkeypatch.setattr(DeviceEvaluationSession,'estimate_tile',lambda *a,**k:1)
    gpu = evaluate(fb,lb,backend='cuda_strict',metrics=metrics)
    for mid in metrics:
        np.testing.assert_allclose(gpu.artifacts[mid].values,cpu.artifacts[mid].values,
            rtol=0,atol=1e-12,equal_nan=True)
    assert cpu.get_metric('quantile_monotonicity','mean_monotone').value == 1.0
    assert not cpu.get_metric('quantile_monotonicity','insufficient').valid
    assert cpu.get_metric('daily_quantile_monotonicity_rate','mean_monotone').value == .625
    assert cpu.get_metric('daily_quantile_monotonicity_rate','mean_monotone').observation_count == 40
    daily = cpu.artifacts['daily_quantile_monotonicity_series'].values
    np.testing.assert_equal(daily[:25,0],np.ones(25))
    np.testing.assert_equal(daily[25:,0],np.zeros(15))
    assert np.isnan(daily[:,1]).all()
    assert gpu.metadata['shape_kernel_backend'] == 'cuda_strict'
    assert gpu.metadata['shape_kernel_no_fallback'] is True
    assert gpu.metadata['shape_kernel_dispatches'] == 6
    assert gpu.metadata['factor_tiles_processed'] == 2
    assert gpu.metadata['peak_vram'] > 0
    assert gpu.metadata['vram_budget_bytes'] >= gpu.metadata['peak_vram']


def test_strict_cuda_does_not_advertise_unwired_shape_metrics(monkeypatch):
    fb,lb = _contracts()
    opened = []
    original = DeviceEvaluationSession._open
    def track(self):
        opened.append(True); return original(self)
    monkeypatch.setattr(DeviceEvaluationSession,'_open',track)
    from quant_evaluator.contracts.errors import UnsupportedMetricError
    with pytest.raises(UnsupportedMetricError,match='u_shape_score'):
        evaluate(fb,lb,backend='cuda_strict',metrics=('u_shape_score',))
    assert opened == []


def test_cuda_respects_sealed_split_before_device_open(monkeypatch):
    from quant_evaluator.contracts.sealed_split import SealedSplitRef
    from quant_evaluator.contracts.errors import SealedSplitOverlapError
    fb, lb = _contracts()
    monkeypatch.setattr(DeviceEvaluationSession, "_open", lambda self: pytest.fail("opened GPU before split guard"))
    split = SealedSplitRef("sealed", start_time=lb.decision_time[0], end_time=lb.label_end_time[-1])
    with pytest.raises(SealedSplitOverlapError):
        evaluate(fb, lb, backend="cuda", split_ref=split)


def test_real_budget_multiple_tiles_transfer_labels_once():
    from dataclasses import replace
    from scipy.stats import rankdata
    fb, lb = _contracts()
    rng = np.random.default_rng(52)
    values = rng.normal(size=(4, 40, 129))
    fb = replace(fb, values=values, validity=None, factor_ids=tuple(f"f{i}" for i in range(129)))
    lb = replace(lb, validity=None)
    result = evaluate(fb, lb, backend="cuda", metrics=("rank_ic_series",))
    xr = rankdata(values, axis=1)
    yr = rankdata(lb.values, axis=1)[:, :, None]
    xr -= xr.mean(axis=1, keepdims=True)
    yr -= yr.mean(axis=1, keepdims=True)
    expected = (xr * yr).sum(axis=1) / np.sqrt((xr * xr).sum(axis=1) * (yr * yr).sum(axis=1))
    np.testing.assert_allclose(result.series_metrics["rank_ic_series"], expected, atol=1e-12)
    assert result.metadata["factor_tiles_processed"] == 2
    assert result.metadata["h2d_bytes"] == values.nbytes + lb.values.nbytes
    # Daily values plus one authoritative int64 observation count per factor.
    assert result.metadata["d2h_bytes"] == expected.nbytes + len(fb.factor_ids) * np.dtype(np.int64).itemsize
    assert result.metadata["peak_vram"] > 0


def test_overlapping_sessions_keep_thread_local_allocators():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    original = cp.cuda.get_allocator()

    def worker():
        with DeviceEvaluationSession() as session:
            barrier.wait(timeout=15)
            active = cp.cuda.get_allocator() == session._pool.malloc
            device_values = cp.arange(32)
            allocated = session._pool.used_bytes()
            del device_values
            barrier.wait(timeout=15)
            return active, allocated

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert all(active and allocated > 0 for active, allocated in results)
    assert cp.cuda.get_allocator() == original


def test_memmap_is_snapshotted_once_then_factor_tiles_share_snapshot(monkeypatch, tmp_path):
    from dataclasses import replace
    fb, lb = _contracts()
    values = np.memmap(tmp_path / "factors.dat", mode="w+", dtype="float64", shape=fb.values.shape)
    values[:] = fb.values
    fb = replace(fb, values=values, validity=None)
    # Writable caller-owned files cannot be a durable input snapshot. The
    # contract copies once; tiling must still avoid one full copy per tile.
    assert not np.shares_memory(fb.values, values)
    values[:] = 999
    assert not np.all(fb.values == 999)
    original = DeviceEvaluationSession.stage_factors
    shared = []

    def stage(self, chunk, ids, layout="T,F,N"):
        shared.append(np.shares_memory(chunk, fb.values))
        return original(self, chunk, ids, layout)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 2)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", stage)
    evaluate(fb, lb, backend="cuda", metrics=("coverage",))
    assert shared == [True, True, True]
