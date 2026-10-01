"""Pure tests for conservative GPU working-set estimators."""
import pytest

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy, PrecisionPolicy
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.runtime.gpu_working_set import estimate_coverage_working_set_bytes


def test_coverage_estimate_validates_integer_dimensions_and_item_sizes():
    estimate_coverage_working_set_bytes(2, 3, 1, 8, 8)
    for argument in (0, 1, 2):
        for bad in (-1, True, 1.5, "2"):
            args = [2, 3, 1, 8, 8]
            args[argument] = bad
            with pytest.raises(ValueError):
                estimate_coverage_working_set_bytes(*args)
    for argument in (3, 4):
        for bad in (0, -1, True, 1.5):
            args = [2, 3, 1, 8, 8]
            args[argument] = bad
            with pytest.raises(ValueError):
                estimate_coverage_working_set_bytes(*args)
    with pytest.raises(ValueError):
        estimate_coverage_working_set_bytes(2, 3, 1, 8, 8, force_fp64=1)


def test_coverage_estimate_accounts_for_dtype_promotion_and_headroom():
    mixed = estimate_coverage_working_set_bytes(2, 3, 1, 4, 4)
    promoted = estimate_coverage_working_set_bytes(
        2, 3, 1, 4, 4, force_fp64=True)
    assert promoted > mixed
    base = 2 * 3 * 4 + 2 * 3 * 8 + 3 * 2 * 3
    base += 2 * 2 * 1 * 8 + 16 + 2 * 3
    assert mixed == (3 * base + 1) // 2 + 64 * 1024**2


def test_exact_default_coverage_f32_panel_fits_eight_factor_tile_at_two_gib():
    session = DeviceEvaluationSession()
    session._vram_budget = 2 * 1024**3
    assert session.estimate_tile(
        ("coverage",), 2586, 5461, 8, metric_parameters={}) == 8
    estimate8 = estimate_coverage_working_set_bytes(2586, 5461, 8, 8, 8)
    assert estimate8 <= session._vram_budget
    session._vram_budget = estimate8 - 1
    assert session.estimate_tile(("coverage",), 2586, 5461, 8) == 4
    session._vram_budget = estimate_coverage_working_set_bytes(2586, 5461, 1, 8, 8) - 1
    with pytest.raises(MemoryError, match="single factor"):
        session.estimate_tile(("coverage",), 2586, 5461, 8)


def test_coverage_estimator_adds_other_live_pool_bytes_and_is_narrow():
    class Pool:
        def used_bytes(self):
            return 100 * 1024**2

    session = DeviceEvaluationSession()
    session._vram_budget = 2 * 1024**3
    session._pool = Pool()
    assert session.estimate_tile(("coverage",), 2586, 5461, 8) == 4
    with pytest.raises(MemoryError):
        session.estimate_tile(
            ("coverage",), 2586, 5461, 8,
            metric_parameters={"coverage": {"min_assets": 20}})
    with pytest.raises(MemoryError):
        session.estimate_tile(("coverage", "rank_ic"), 2586, 5461, 8)


def test_fp64_policy_promotes_factor_and_label_estimate():
    session = DeviceEvaluationSession(GPUExecutionPolicy(
        precision_policy=PrecisionPolicy.GPU_FP64))
    session._vram_budget = 2 * 1024**3
    promoted = estimate_coverage_working_set_bytes(
        2586, 5461, 8, 4, 8, force_fp64=True)
    native_fp64 = estimate_coverage_working_set_bytes(
        2586, 5461, 8, 8, 8, force_fp64=False)
    assert promoted == native_fp64
    assert session.estimate_tile(("coverage",), 2586, 5461, 4) == 8


def test_gpu_executor_forwards_parameters_only_for_specialized_plans():
    class Session:
        def __init__(self):
            self.calls = []

        def estimate_tile(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return 8

    from quant_evaluator.runtime.gpu_executor import GPUExecutor

    session = Session()
    executor = GPUExecutor(session)
    assert executor._factor_tile_size(("coverage",), 4, 40, 5, 8) == 5
    assert session.calls[-1][1] == {"metric_parameters": {}}
    assert executor._factor_tile_size(("coverage", "rank_ic"), 4, 40, 5, 8) == 5
    assert session.calls[-1][1] == {}
    assert executor._factor_tile_size(
        ("quantile_returns_full",), 4, 40, 5, 8) == 5
    assert session.calls[-1][1] == {}
