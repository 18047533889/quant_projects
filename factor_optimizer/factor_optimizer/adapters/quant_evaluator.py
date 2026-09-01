"""QuantEvaluatorAdapter: protocol for QE integration (optional dependency)."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


from factor_optimizer.capabilities import ExecutionMode, require_production_capability


@runtime_checkable
class EvidenceStore(Protocol):
    """Storage boundary for evidence bundles shared across adapter instances."""

    evidence_scope: str

    def put(self, evidence_id: str, evidence: Dict[str, Any]) -> None:
        ...

    def get(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        ...


class InMemoryEvidenceStore:
    """Process-local shared store useful for research and tests."""

    evidence_scope = "process_local_shared_store"

    def __init__(self):
        self._values: Dict[str, Dict[str, Any]] = {}

    def put(self, evidence_id: str, evidence: Dict[str, Any]) -> None:
        self._values[evidence_id] = deepcopy(evidence)

    def get(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        value = self._values.get(evidence_id)
        return deepcopy(value) if value is not None else None


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
            List of metric specifications with registry-authoritative fields:
                - metric_id (str): Unique metric identifier
                - name (str): Display name
                - description (str): Metric description
                - tier (str): Metric tier/category

            Direction metadata is included only when the backing QE registry exposes it;
            callers must not assume ``higher_is_better`` is available.
        """
        ...


class OptionalDependencyMissing(Exception):
    """Raised when optional QE dependency is not available."""

    pass


def create_qe_adapter(*, execution_mode: ExecutionMode = ExecutionMode.RESEARCH_ONLY, evidence_store: Optional[EvidenceStore] = None) -> QuantEvaluatorAdapter:
    """Create a real QE adapter; production remains fail-closed until implemented."""
    if isinstance(execution_mode, str):
        execution_mode = ExecutionMode(execution_mode)
    if execution_mode is ExecutionMode.PRODUCTION:
        require_production_capability()
    try:
        import inspect
        import uuid
        import numpy as np
        import quant_evaluator as qe
        from quant_evaluator.registry import metrics as metric_registry

        evaluate = getattr(qe, "evaluate", None)
        if evaluate is None:
            raise ImportError("quant_evaluator.evaluate is not available")
        parameters = inspect.signature(evaluate).parameters
        required = {"factors", "labels", "context", "metrics"}
        if not required.issubset(parameters):
            raise ImportError(
                "installed quant_evaluator does not expose the current public evaluate API"
            )
        if any(
            parameters[name].kind is inspect.Parameter.POSITIONAL_ONLY
            for name in required
        ):
            raise ImportError("quant_evaluator.evaluate cannot accept the adapter keywords")

        class ConcreteQEAdapter:
            """Adapter for QE's typed public facade and an explicit evidence store."""

            def __init__(self):
                self._evidence_store = evidence_store

            def evaluate(
                self,
                factor_batch: Any,
                labels: Any,
                metrics: Optional[List[str]] = None,
                context: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                """Evaluate through QE and persist a plain evidence snapshot."""
                if self._evidence_store is None:
                    from factor_optimizer.errors import EvidenceUnavailableError
                    raise EvidenceUnavailableError(
                        "an evidence store is required before evaluation"
                    )
                metric_ids = tuple(metrics) if metrics is not None else ("coverage",)
                bundle = evaluate(
                    factor_batch,
                    labels,
                    context=context,
                    metrics=metric_ids,
                )
                evaluation_id = f"qe_adapter_{uuid.uuid4().hex}"
                from dataclasses import asdict, is_dataclass

                def plain(value: Any) -> Any:
                    if is_dataclass(value):
                        return {key: plain(item) for key, item in asdict(value).items()}
                    if isinstance(value, Mapping):
                        return {key: plain(item) for key, item in value.items()}
                    if isinstance(value, (list, tuple)):
                        return [plain(item) for item in value]
                    if isinstance(value, np.ndarray):
                        return value.tolist()
                    if isinstance(value, np.generic):
                        return value.item()
                    return value

                metric_values = {
                    metric_id: plain(metric.value)
                    for metric_id, metric in bundle.metric_values.items()
                }
                diagnostics = {
                    factor_id: plain(diagnosis)
                    for factor_id, diagnosis in bundle.diagnostics.items()
                }
                evidence = {
                    "evaluation_id": evaluation_id,
                    "evidence_scope": self._evidence_store.evidence_scope,
                    "metrics": metric_values,
                    "grouped_metrics": plain(bundle.grouped_metrics),
                    "diagnostics": diagnostics,
                    "metadata": plain(bundle.metadata),
                }
                self._evidence_store.put(evaluation_id, evidence)
                return {
                    "evaluation_id": evaluation_id,
                    "metrics": evidence["metrics"],
                    "diagnostics": evidence["diagnostics"],
                    "evidence_ref": evaluation_id,
                    "evidence_scope": self._evidence_store.evidence_scope,
                }

            def get_evidence(self, evaluation_id: str) -> Dict[str, Any]:
                """Retrieve evidence from the configured shared store."""
                if self._evidence_store is None:
                    from factor_optimizer.errors import EvidenceUnavailableError
                    raise EvidenceUnavailableError(
                        "durable evidence store is required for evidence lookup"
                    )
                evidence = self._evidence_store.get(evaluation_id)
                if evidence is None:
                    from factor_optimizer.errors import EvidenceUnavailableError
                    raise EvidenceUnavailableError(
                        f"evidence not found for evaluation '{evaluation_id}'"
                    )
                return evidence

            def list_metrics(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
                """List QE registry metadata without inventing optimization direction."""
                selected = metric_registry.list_metrics()
                if tier is not None:
                    try:
                        selected = metric_registry.list_metrics_by_tier(
                            metric_registry.MetricTier(tier)
                        )
                    except ValueError as exc:
                        raise ValueError(f"unknown metric tier: {tier}") from exc
                return [
                    {
                        "metric_id": spec.name,
                        "name": spec.display_name,
                        "description": spec.description,
                        "tier": spec.tier.value,
                        "status": spec.status.value,
                    }
                    for name in selected
                    for spec in (metric_registry.get_metric(name),)
                ]

        return ConcreteQEAdapter()

    except ImportError as e:
        raise OptionalDependencyMissing(
            "quant-evaluator is unavailable or incompatible with the current "
            "FactorOptimizer adapter contract; use the explicit research-only "
            "mock adapter only for tests and development."
        ) from e


def create_mock_qe_adapter(*, execution_mode: ExecutionMode = ExecutionMode.RESEARCH_ONLY) -> QuantEvaluatorAdapter:
    """Create the mock adapter only for explicit research use."""
    if isinstance(execution_mode, str):
        execution_mode = ExecutionMode(execution_mode)
    if execution_mode is not ExecutionMode.RESEARCH_ONLY:
        raise ValueError("mock QE adapters are research_only and cannot run in production")
    import uuid
    from datetime import datetime, timezone

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
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                },
                "evidence_ref": eval_id,
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
