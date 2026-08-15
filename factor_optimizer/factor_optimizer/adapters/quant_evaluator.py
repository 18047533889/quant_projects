"""QuantEvaluatorAdapter: protocol for QE integration (optional dependency)."""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


from factor_optimizer.capabilities import ExecutionMode, require_production_capability


@runtime_checkable
class QuantEvaluatorAdapter(Protocol):
    """
    Protocol for QuantEvaluator integration.

    This is a protocol/interface, not an implementation. FO does NOT directly import
    or depend on QE. Concrete adapters are provided when QE is available.

    Required capabilities:
    - Submit evaluation requests for factor candidates
    - Retrieve evaluation results/evidence bundles
    - Access metric catalog for optimization objectives
    """

    def evaluate(
        self,
        factor_batch: Any,
        labels: Any,
        metrics: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit evaluation request to QE.

        Args:
            factor_batch: FactorBatch object (QE-specific format)
            labels: LabelBundle object (QE-specific format)
            metrics: Optional list of metric IDs (None = default preset)
            context: Optional evaluation context (universe, period, etc.)

        Returns:
            Dictionary with evaluation results:
                - evaluation_id (str): Reference to evaluation
                - metrics (dict): Metric name -> value
                - diagnostics (dict): Warnings, coverage, etc.
                - evidence_ref (str): Reference to full evidence bundle
        """
        ...

    def get_evidence(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Retrieve full evidence bundle by evaluation ID.

        Args:
            evaluation_id: Evaluation reference ID

        Returns:
            Full evidence bundle dictionary with:
                - evaluation_id (str)
                - metrics (dict): Full metric results
                - diagnostics (dict): Coverage, warnings, errors
                - timeseries (dict): Optional time-series metrics
        """
        ...

    def list_metrics(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available metrics from QE catalog.

        Args:
            tier: Optional tier filter (e.g., "core", "advanced", "research")

        Returns:
            List of metric specifications with:
                - metric_id (str): Unique metric identifier
                - name (str): Display name
                - description (str): Metric description
                - tier (str): Metric tier/category
                - higher_is_better (bool): Optimization direction
        """
        ...


class OptionalDependencyMissing(Exception):
    """Raised when optional QE dependency is not available."""

    pass


def create_qe_adapter(*, execution_mode: ExecutionMode = ExecutionMode.RESEARCH_ONLY) -> QuantEvaluatorAdapter:
    """Create a real QE adapter; production remains fail-closed until implemented."""
    if isinstance(execution_mode, str):
        execution_mode = ExecutionMode(execution_mode)
    if execution_mode is ExecutionMode.PRODUCTION:
        require_production_capability()
    try:
        # Try to import QE - this will fail if not installed
        # Note: QE doesn't exist yet, so this is a forward-looking stub
        import quant_evaluator as qe

        if not hasattr(qe, "Evaluator"):
            raise ImportError("quant_evaluator.Evaluator is not available")

        # Concrete adapter implementation
        class ConcreteQEAdapter:
            """Concrete QE adapter implementation."""

            def __init__(self):
                """Initialize with QE evaluator."""
                self.evaluator = qe.Evaluator()

            def evaluate(
                self,
                factor_batch: Any,
                labels: Any,
                metrics: Optional[List[str]] = None,
                context: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                """Submit evaluation to QE."""
                # Real implementation would call QE API
                result = self.evaluator.evaluate(
                    factors=factor_batch,
                    labels=labels,
                    metrics=metrics or qe.DEFAULT_METRICS,
                    context=context or {},
                )

                return {
                    "evaluation_id": result.evaluation_id,
                    "metrics": result.metrics,
                    "diagnostics": result.diagnostics,
                    "evidence_ref": result.evidence_ref,
                }

            def get_evidence(self, evaluation_id: str) -> Dict[str, Any]:
                """Retrieve evidence bundle."""
                evidence = self.evaluator.get_evidence(evaluation_id)
                return {
                    "evaluation_id": evaluation_id,
                    "metrics": evidence.metrics,
                    "diagnostics": evidence.diagnostics,
                    "timeseries": evidence.timeseries,
                }

            def list_metrics(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
                """List available metrics."""
                catalog = qe.MetricCatalog()
                return catalog.list(tier=tier)

        return ConcreteQEAdapter()

    except ImportError as e:
        raise OptionalDependencyMissing(
            "quant-evaluator not installed. "
            "This is a future package; for now, use mock adapters in tests."
        ) from e


def create_mock_qe_adapter(*, execution_mode: ExecutionMode = ExecutionMode.RESEARCH_ONLY) -> QuantEvaluatorAdapter:
    """Create the mock adapter only for explicit research use."""
    if isinstance(execution_mode, str):
        execution_mode = ExecutionMode(execution_mode)
    if execution_mode is not ExecutionMode.RESEARCH_ONLY:
        raise ValueError("mock QE adapters are research_only and cannot run in production")
    import uuid
    from datetime import datetime

    class MockQEAdapter:
        """Mock QE adapter for testing."""

        def __init__(self):
            """Initialize mock adapter."""
            self.evaluations: Dict[str, Dict[str, Any]] = {}

        def evaluate(
            self,
            factor_batch: Any,
            labels: Any,
            metrics: Optional[List[str]] = None,
            context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Mock evaluation."""
            eval_id = f"mock_eval_{uuid.uuid4().hex[:8]}"

            # Generate mock metrics
            import random

            random.seed(hash(eval_id))

            mock_metrics = {
                "rank_ic": random.uniform(-0.1, 0.15),
                "ic_mean": random.uniform(-0.08, 0.12),
                "ic_std": random.uniform(0.05, 0.15),
                "turnover": random.uniform(0.1, 0.3),
                "sharpe": random.uniform(-0.5, 2.0),
            }

            # Filter to requested metrics
            if metrics:
                mock_metrics = {k: v for k, v in mock_metrics.items() if k in metrics}

            result = {
                "evaluation_id": eval_id,
                "metrics": mock_metrics,
                "diagnostics": {
                    "coverage": random.uniform(0.85, 0.98),
                    "warnings": [],
                    "evaluated_at": datetime.now().isoformat(),
                },
                "evidence_ref": f"mock_evidence_{eval_id}",
            }

            # Store for get_evidence
            self.evaluations[eval_id] = result

            return result

        def get_evidence(self, evaluation_id: str) -> Dict[str, Any]:
            """Retrieve mock evidence bundle."""
            if evaluation_id in self.evaluations:
                eval_result = self.evaluations[evaluation_id]
                return {
                    "evaluation_id": evaluation_id,
                    "metrics": eval_result["metrics"],
                    "diagnostics": eval_result["diagnostics"],
                    "timeseries": {},
                }

            # Return empty if not found
            return {
                "evaluation_id": evaluation_id,
                "metrics": {},
                "diagnostics": {"error": "Evaluation not found"},
                "timeseries": {},
            }

        def list_metrics(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
            """List mock metrics."""
            all_metrics = [
                {
                    "metric_id": "rank_ic",
                    "name": "Rank IC",
                    "description": "Rank information coefficient",
                    "tier": "core",
                    "higher_is_better": True,
                },
                {
                    "metric_id": "ic_mean",
                    "name": "IC Mean",
                    "description": "Mean information coefficient",
                    "tier": "core",
                    "higher_is_better": True,
                },
                {
                    "metric_id": "ic_std",
                    "name": "IC Std",
                    "description": "IC standard deviation",
                    "tier": "core",
                    "higher_is_better": False,
                },
                {
                    "metric_id": "turnover",
                    "name": "Turnover",
                    "description": "Average turnover",
                    "tier": "core",
                    "higher_is_better": False,
                },
                {
                    "metric_id": "sharpe",
                    "name": "Sharpe Ratio",
                    "description": "Information Sharpe ratio",
                    "tier": "advanced",
                    "higher_is_better": True,
                },
            ]

            if tier:
                return [m for m in all_metrics if m["tier"] == tier]

            return all_metrics

    return MockQEAdapter()
