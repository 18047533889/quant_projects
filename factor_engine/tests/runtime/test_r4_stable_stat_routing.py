"""Auto must not advertise the raw-moment Numba routes removed for correctness."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from factor_engine.backend.routing import numba_enabled_for_op
from factor_engine.backend.operator_cost import get_operator_cost
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.multibackend.batch_global_optimizer import PhysicalBatchGlobalOptimizer


@pytest.mark.parametrize("op", ["ts_std", "ts_std_dev", "ts_var", "ts_cov", "ts_corr", "ts_correlation"])
def test_stable_window_kernel_does_not_advertise_raw_moment_numba(op, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_USE_NUMBA", "1")
    assert numba_enabled_for_op(op, window=20, panel_rows=1000) is False
    cost = get_operator_cost(op)
    assert cost.complexity == "O(NW)"
    assert cost.supports_incremental is False


@pytest.mark.parametrize("op", ["ts_std", "ts_var", "ts_cov", "ts_corr"])
def test_optimizer_does_not_offer_disabled_numba_candidate(op, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_USE_NUMBA", "1")
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: False)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_sql", lambda *a, **k: False)
    capability = Mock()
    capability.execution_kind = ExecutionKind.PANDAS_REFERENCE
    capability.is_production_eligible.return_value = False
    capability.implementation_id = None
    capability.canonical = op
    capability.backend = "pandas_numpy"
    monkeypatch.setattr("factor_engine.backend.operator_capability.capability_for", lambda *a, **k: capability)
    choices = PhysicalBatchGlobalOptimizer()._eligible_choices(
        "n", PlanNode(op), 1000, SimpleNamespace(run_mode="research", runtime_stats={}))
    assert choices
    assert all(choice.execution_kind != ExecutionKind.NUMBA_CPU_KERNEL for choice in choices)
