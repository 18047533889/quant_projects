"""Regression tests for Q Backend factory integration (R21-Q-FACTORY-INTEGRATION)."""

from __future__ import annotations

import pytest

from backend.factory import build_backend
from backend.q_backend import QBackend


class TestQKdbFactoryIntegration:
    """Test build_backend("q_kdb") factory branch."""

    def test_build_backend_q_kdb_returns_q_backend(self):
        """build_backend('q_kdb') must return a QBackend instance."""
        backend = build_backend("q_kdb")
        assert isinstance(backend, QBackend)
        assert backend.runtime_backend_label == "q_kdb"

    def test_build_backend_q_returns_q_backend(self):
        """build_backend('q') (alias) must also return a QBackend instance."""
        backend = build_backend("q")
        assert isinstance(backend, QBackend)
        assert backend.runtime_backend_label == "q_kdb"

    def test_build_backend_q_kdb_production_mode_default(self):
        """QBackend from factory must be in production_mode=True by default."""
        backend = build_backend("q_kdb")
        assert backend._production_mode is True
        assert backend._fallback_to_pandas is False


class TestQBackendExecuteProductionUnavailable:
    """Test QBackend.execute() raises BackendUnavailableError when q is unavailable in production."""

    def test_production_mode_q_unavailable_raises_backend_unavailable(self):
        """If production_mode=True and q runtime is not available, must raise BackendUnavailableError."""
        from backend.operator_capability import BackendUnavailableError
        from backend.q_backend.q_process_manager import QProcessManager, QAvailabilityStatus, QProcessInfo
        from planner.logical_plan import PlanNode

        # Create a QBackend in production mode
        backend = QBackend(fallback_to_pandas=False, production_mode=True)

        # Mock process manager to report unavailable
        mock_pm = QProcessManager()
        mock_pm._checked = True
        mock_pm._process_info = QProcessInfo(
            status=QAvailabilityStatus.UNAVAILABLE,
            error_message="PyKX not installed (test mock)",
        )
        backend._process_manager = mock_pm

        # Create a dummy plan node
        dummy_plan = PlanNode(op="add", inputs=(), attrs={}, node_id="test_node")

        from backend.context import ExecutionContext

        # Create minimal context
        ctx = ExecutionContext(data_source=None, run_mode="research")

        with pytest.raises(BackendUnavailableError, match="q runtime unavailable"):
            backend.execute(dummy_plan, ctx)


class TestQBackendExecuteMinimalPath:
    """Test minimal execution path when q is available."""

    def test_research_mode_checks_availability_before_execution(self):
        """Research mode should check q availability before executing."""
        from backend.q_backend.q_process_manager import QProcessManager, QAvailabilityStatus, QProcessInfo
        from planner.logical_plan import PlanNode
        from backend.context import ExecutionContext

        backend = QBackend(fallback_to_pandas=False, production_mode=False)

        # Mock process manager to report unavailable
        mock_pm = QProcessManager()
        mock_pm._checked = True
        mock_pm._process_info = QProcessInfo(
            status=QAvailabilityStatus.UNAVAILABLE,
            error_message="PyKX not installed (test mock)",
        )
        backend._process_manager = mock_pm

        dummy_plan = PlanNode(op="add", inputs=(), attrs={}, node_id="test_node")
        ctx = ExecutionContext(data_source=None, run_mode="research")

        # Should raise QDataUnavailableError (not BackendUnavailableError) in research mode
        from backend.q_backend.q_errors import QDataUnavailableError, QPlanningFallbackAllowed

        with pytest.raises((QDataUnavailableError, QPlanningFallbackAllowed)):
            backend.execute(dummy_plan, ctx)
