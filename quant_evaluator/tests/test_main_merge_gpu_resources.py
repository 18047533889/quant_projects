"""M17: bounded worker output and actual CUDA upload/compute OOM lifecycle."""
from types import SimpleNamespace
import numpy as np
import pytest
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.runtime.gpu_executor import GPUExecutor


def test_host_budget_and_identity_map():
    executor = GPUExecutor(DeviceEvaluationSession(GPUExecutionPolicy(max_host_result_bytes=64)))
    executor._reserve_host_result(48)
    with pytest.raises(MemoryError, match="queued factor batch"):
        executor._reserve_host_result(24)
    assert executor._host_result_bytes == 48
    executor.portfolio_factor_ids = ("b", "a")
    assert [executor._portfolio_column_map[x] for x in ("a", "b")] == [1, 0]
    with pytest.raises(ValueError, match="unique"):
        executor.portfolio_factor_ids = ("a", "a")


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_host_budget(value):
    with pytest.raises(ValueError):
        GPUExecutionPolicy(max_host_result_bytes=value)


@pytest.mark.parametrize("failure_stage", ["upload", "compute"])
def test_probe_oom_retiles_on_actual_cuda(monkeypatch, failure_stage):
    cp = pytest.importorskip("cupy")
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel, PortfolioSpec
    from quant_evaluator.kernels.gpu import portfolio
    from quant_evaluator.metrics.probe_portfolio import compute_cohort_pnl
    T, N, F = 8, 4, 5
    times = AxisRef("time", "int", T, np.arange(T))
    assets = AxisRef("asset", "int", N, np.arange(N))
    values = np.broadcast_to(np.arange(N)[None, :, None], (T, N, F)).astype(float).copy()
    returns = np.tile(np.array([-.02, -.01, .01, .02]), (T, 1))
    batch = FactorBatch(tuple(f"f{i}" for i in range(F)), times, assets, values)
    holding = HoldingReturnPanel(returns, times, assets, "synthetic:m17", "close_to_close")
    spec = PortfolioSpec(holding=2, n_quantiles=2, per_side_cost=.001)
    original_stage = DeviceEvaluationSession.stage_factors
    original_compute = portfolio.compute_cohort_pnl_batch_gpu
    attempts = []
    def stage(self, chunk, ids, layout="T,N,F"):
        attempts.append(len(ids))
        result = original_stage(self, chunk, ids, layout)
        if failure_stage == "upload" and len(ids) > 2:
            raise cp.cuda.memory.OutOfMemoryError(100, 100, 100)
        return result
    def compute(factors, *args, **kwargs):
        if failure_stage == "compute" and factors.shape[1] > 2:
            raise cp.cuda.memory.OutOfMemoryError(100, 100, 100)
        return original_compute(factors, *args, **kwargs)
    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile", lambda *a, **k: 4)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_factors", stage)
    monkeypatch.setattr(portfolio, "compute_cohort_pnl_batch_gpu", compute)
    with DeviceEvaluationSession() as session:
        executor = GPUExecutor(session)
        result = executor.build_probe_pnl_tiled(batch, holding, spec)
        assert not session._staged_factors
        assert executor.probe_tiles_processed == 3
        assert session._oom_retries == 1
    assert attempts == [4, 2, 2, 1]
    expected = compute_cohort_pnl(values[:, :, 0], returns, np.ones((T, N)),
        holding=2, n_quantiles=2, per_side_cost=.001, require_tradable=False)["pnl_net"]
    np.testing.assert_allclose(result, np.repeat(expected[:, None], F, axis=1), atol=1e-12)


def test_probe_budget_rejects_before_upload(monkeypatch):
    batch = SimpleNamespace(values=np.empty((4, 2, 3)))
    executor = GPUExecutor(DeviceEvaluationSession(GPUExecutionPolicy(max_host_result_bytes=8)))
    import quant_evaluator.runtime.gpu_executor as module
    monkeypatch.setattr(module, "_import_cp", lambda: None)
    with pytest.raises(MemoryError, match="host result budget"):
        executor.build_probe_pnl_tiled(batch, None, None)
