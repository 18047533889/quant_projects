import pytest

from factor_optimizer import package_info
from factor_optimizer.capabilities import ExecutionMode
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import SealedTestResult
from factor_optimizer.adapters import create_mock_qe_adapter
from factor_optimizer.errors import CapabilityError
from factor_optimizer.search.runner import SearchConfig, SearchRunner


def test_package_reports_research_only_production_capability():
    info = package_info()
    capability = info["production_capability"]
    assert capability["status"] == "research_only"
    assert capability["supported"] is False
    assert "trusted_split_aware_qe_integration_missing" in capability["blockers"]


def test_production_search_config_fails_closed():
    with pytest.raises(CapabilityError, match="production execution is unsupported"):
        SearchConfig(
            budget=SearchBudget(max_trials=1, max_evaluations=1),
            execution_mode=ExecutionMode.PRODUCTION,
        )


def test_production_search_runner_fails_closed_even_with_callbacks():
    with pytest.raises(CapabilityError, match="production execution is unsupported"):
        SearchRunner(
            SearchConfig(
                budget=SearchBudget(max_trials=1, max_evaluations=1),
                execution_mode=ExecutionMode.PRODUCTION,
            ),
            lambda: None,
            lambda trial, fidelity: {"cost": 0},
        )


def test_mock_qe_is_explicitly_research_only():
    assert create_mock_qe_adapter(execution_mode=ExecutionMode.RESEARCH_ONLY)
    with pytest.raises(ValueError, match="research_only"):
        create_mock_qe_adapter(execution_mode=ExecutionMode.PRODUCTION)


def test_direct_sealed_test_result_is_not_production_valid():
    with pytest.raises(TypeError, match="frozen_at must be a datetime"):
        SealedTestResult(
            trial_id="trial",
            test_metrics={"rank_ic": 0.1},
            frozen_at=None,
            search_session_id="session",
            split_id="split",
            evaluation_ref="store://evidence-1",
        )
