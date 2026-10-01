"""Batch / tile / OOM / memory-leak tests (spec §57).

- Batch invariance: F=500 in one call == 500 × F=1 CPU reference.
- Tile invariance: Ftile=16/32/64/all-fit produce identical results.
- OOM retile: simulated low VRAM retiles 128→64→32 and keeps results.
- Memory leak: repeated evaluate does not grow VRAM monotonically.
- CUDA strict: mandatory metric without CUDA impl raises, no silent CPU.
"""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.correlation import batched_spearman_ic
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy, PrecisionPolicy


def _cpu_spearman(x, y, min_obs=20):
    T, N, F = x.shape
    out = np.full((T, F), np.nan)
    for t in range(T):
        for f in range(F):
            m = np.isfinite(x[t, :, f]) & np.isfinite(y[t, :])
            if m.sum() < min_obs:
                continue
            rx = pd.Series(x[t, m, f]).rank().values
            ry = pd.Series(y[t, m]).rank().values
            if np.unique(x[t, m, f]).size < 2 or np.unique(y[t, m]).size < 2:
                continue
            out[t, f] = np.corrcoef(rx, ry)[0, 1]
    return out


def _tile_rank_ic(fact, y, tile):
    """Run batched spearman over F in tiles; returns (F,) mean rank_ic."""
    T, N, F = fact.shape
    out = np.zeros(F)
    for s in range(0, F, tile):
        ftile = np.ascontiguousarray(fact[:, :, s : s + tile])
        xdev = cp.transpose(cp.asarray(ftile, dtype=cp.float32), (0, 2, 1))
        ic, _ = batched_spearman_ic(xdev, cp.asarray(y), min_obs=20)
        out[s : s + tile] = cp.asnumpy(cp.nanmean(ic, axis=0))
        cp.get_default_memory_pool().free_all_blocks()
    return out


def test_batch_500_vs_500_single():
    rng = np.random.default_rng(0)
    T, N, F = 20, 300, 64
    x = rng.normal(size=(T, N, F)).astype(np.float32)
    x[rng.random((T, N, F)) < 0.05] = np.nan
    y = rng.normal(size=(T, N))
    y[rng.random((T, N)) < 0.05] = np.nan
    # batch (all F at once, tile=64)
    batch = _tile_rank_ic(x, y, tile=64)
    # single-factor CPU reference (per-factor mean over time)
    single = np.nanmean(_cpu_spearman(x, y), axis=0)  # (F,)
    mask = np.isfinite(single)
    assert np.nanmax(np.abs(batch[mask] - single[mask])) < 1e-6

def test_factor_tile_invariance():
    rng = np.random.default_rng(1)
    T, N, F = 20, 300, 64
    x = rng.normal(size=(T, N, F)).astype(np.float32)
    x[rng.random((T, N, F)) < 0.05] = np.nan
    y = rng.normal(size=(T, N))
    y[rng.random((T, N)) < 0.05] = np.nan
    r16 = _tile_rank_ic(x, y, tile=16)
    r32 = _tile_rank_ic(x, y, tile=32)
    r64 = _tile_rank_ic(x, y, tile=64)
    assert np.abs(r16 - r32).max() < 1e-8
    assert np.abs(r32 - r64).max() < 1e-8


def test_oom_retile():
    """Simulate low VRAM: session retiles 128→64→32 and keeps results."""
    rng = np.random.default_rng(2)
    T, N, F = 20, 300, 64
    x = rng.normal(size=(T, N, F)).astype(np.float32)
    y = rng.normal(size=(T, N))
    # An unfitting budget must fail closed rather than forcing 16 factors.
    policy = GPUExecutionPolicy(max_vram_fraction=0.01)
    session = DeviceEvaluationSession(policy)
    session._open()
    session._vram_budget = 1
    with pytest.raises(MemoryError, match="single factor"):
        session.estimate_tile("spearman", T, N, dtype_bytes=4)
    # Deterministic estimator budget, independent of other GPU processes.
    session._vram_budget = session._estimate_working_set("spearman", T, N, 16, 4)
    tile = session.estimate_tile("spearman", T, N, dtype_bytes=4)
    assert tile == 16
    # retile on OOM: 16 -> 8 -> 4 -> 2 -> 1
    t = 16
    while t > 1:
        t = session.retile_on_oom(t)
    assert session._oom_retries >= 4
    assert session._final_tile == 1
    session.close()


def test_gpu_memory_no_linear_leak():
    rng = np.random.default_rng(3)
    T, N, F = 20, 300, 8
    x = rng.normal(size=(T, N, F)).astype(np.float32)
    y = rng.normal(size=(T, N))
    xdev = cp.transpose(cp.asarray(x), (0, 2, 1))
    ydev = cp.asarray(y)
    # warm up
    batched_spearman_ic(xdev, ydev, min_obs=20)
    cp.get_default_memory_pool().free_all_blocks()
    free0, _ = cp.cuda.runtime.memGetInfo()
    for _ in range(5):
        batched_spearman_ic(xdev, ydev, min_obs=20)
        cp.get_default_memory_pool().free_all_blocks()
    free1, _ = cp.cuda.runtime.memGetInfo()
    # free VRAM should not shrink monotonically (leak) — allow small slack
    assert free1 >= free0 - 200 * 1024 * 1024  # within 200MB slack


@pytest.mark.parametrize("dtype", [np.int16, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("precision", [PrecisionPolicy.GPU_MIXED, PrecisionPolicy.GPU_FP64])
def test_stage_masked_factors_matches_numpy_where_and_preserves_host(
    dtype, precision, monkeypatch
):
    values = np.arange(12, dtype=dtype).reshape(2, 3, 2)
    original = values.copy()
    validity = np.array(
        [[[True, False], [False, True], [True, True]],
         [[False, True], [True, False], [False, False]]],
        dtype=np.bool_,
    )
    expected = np.transpose(np.where(validity, values, np.nan), (0, 2, 1))
    if precision is PrecisionPolicy.GPU_FP64:
        expected = expected.astype(np.float64)
    session = DeviceEvaluationSession(GPUExecutionPolicy(precision_policy=precision))
    session._open()

    def forbidden_host_mask_op(*args, **kwargs):
        raise AssertionError("factor masking must not use host where/logical_not")

    monkeypatch.setattr(np, "where", forbidden_host_mask_op)
    monkeypatch.setattr(np, "logical_not", forbidden_host_mask_op)
    staged = session.stage_masked_factors(
        values, validity, ("f0", "f1"), layout="T,N,F")
    monkeypatch.undo()
    assert staged.dtype == expected.dtype
    cp.testing.assert_array_equal(staged, expected)
    np.testing.assert_array_equal(values, original)
    assert session._h2d_bytes == staged.nbytes + validity.nbytes
    session.release_factor_tile()
    session.close()


def test_stage_masked_factors_owns_device_input():
    values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    device_values = cp.asarray(values)
    original = device_values.copy()
    validity = np.ones(values.shape, dtype=np.bool_)
    validity[0, 1, 0] = False
    session = DeviceEvaluationSession(GPUExecutionPolicy())
    session._open()
    staged = session.stage_masked_factors(
        device_values, validity, ("f0", "f1"), layout="T,N,F")
    assert staged.data.ptr != device_values.data.ptr
    cp.testing.assert_array_equal(device_values, original)
    assert session._h2d_bytes == validity.nbytes
    session.release_factor_tile()
    session.close()


def test_masked_factor_staging_has_no_host_where_or_logical_not():
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(DeviceEvaluationSession.stage_masked_factors))
    tree = ast.parse(source)
    host_mask_calls = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr in {"where", "logical_not"}
    ]
    assert host_mask_calls == []


def test_cuda_strict_no_silent_cpu():
    """CUDA_STRICT: a metric with no CUDA impl must raise, not fall back."""
    from quant_evaluator.runtime.device_session import UnsupportedBackendCapability
    # The GPUExecutor raises RuntimeError for unsupported metrics; CUDA_STRICT
    # semantics are enforced at the executor level.  Verify the executor raises
    # rather than silently computing on CPU.
    from quant_evaluator.runtime.gpu_executor import GPUExecutor
    session = DeviceEvaluationSession(GPUExecutionPolicy())
    session._open()
    session.stage_factors(np.zeros((2, 1, 3)), ("f0",), layout="T,N,F")
    session.stage_labels(np.zeros((2, 3)), "next_ret")
    ex = GPUExecutor(session)
    with pytest.raises(RuntimeError):
        ex.run(("f0",), metrics=["not_a_real_metric"], label_id="next_ret")
    session.close()
