"""
Mock adapters for DataAccess and FactorEngine integration testing.

Provides controllable mock implementations that simulate real adapter behavior
without requiring actual DA/FE infrastructure.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
from enum import Enum


class AdapterStatus(Enum):
    """Status codes for adapter responses."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_AVAILABLE = "not_available"


@dataclass
class AdapterResponse:
    """Standard response from mock adapters."""
    status: AdapterStatus
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    latency_ms: float = 0.0


class MockDataAccessAdapter:
    """
    Mock DataAccess adapter for testing integration with data layer.

    Simulates reading factor values, universe definitions, and metadata
    without requiring actual data infrastructure.
    """

    def __init__(
        self,
        fail_rate: float = 0.0,
        latency_ms: float = 10.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            fail_rate: Probability of request failure (0.0 to 1.0)
            latency_ms: Simulated latency in milliseconds
            seed: Random seed for reproducibility
        """
        self.fail_rate = fail_rate
        self.latency_ms = latency_ms
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Simulated storage
        self._factor_cache: Dict[str, np.ndarray] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._universe_cache: Dict[str, np.ndarray] = {}

        # Request tracking
        self.request_count = 0
        self.request_history: List[Dict[str, Any]] = []

    def read_factor(
        self,
        factor_id: str,
        start_date: str,
        end_date: str,
        universe: Optional[str] = None,
    ) -> AdapterResponse:
        """
        Mock factor reading.

        Returns synthetic data or cached data if available.
        """
        self.request_count += 1
        self.request_history.append({
            "method": "read_factor",
            "factor_id": factor_id,
            "start_date": start_date,
            "end_date": end_date,
            "universe": universe,
            "timestamp": datetime.now().isoformat(),
        })

        # Simulate failure
        if self._rng.random() < self.fail_rate:
            return AdapterResponse(
                status=AdapterStatus.FAILED,
                error_message=f"Simulated failure reading {factor_id}",
                latency_ms=self.latency_ms,
            )

        # Return cached or generate synthetic
        if factor_id in self._factor_cache:
            data = self._factor_cache[factor_id]
        else:
            # Generate synthetic panel
            T, N = 100, 50
            data = self._rng.randn(T, N)

        metadata = self._metadata_cache.get(factor_id, {
            "factor_id": factor_id,
            "timing": "daily",
            "frequency": "D",
            "pit_safe": True,
        })

        return AdapterResponse(
            status=AdapterStatus.SUCCESS,
            data=data,
            metadata=metadata,
            latency_ms=self.latency_ms,
        )

    def read_universe(
        self,
        universe_id: str,
        as_of_date: str,
    ) -> AdapterResponse:
        """Mock universe definition reading."""
        self.request_count += 1
        self.request_history.append({
            "method": "read_universe",
            "universe_id": universe_id,
            "as_of_date": as_of_date,
            "timestamp": datetime.now().isoformat(),
        })

        if self._rng.random() < self.fail_rate:
            return AdapterResponse(
                status=AdapterStatus.FAILED,
                error_message=f"Simulated failure reading universe {universe_id}",
                latency_ms=self.latency_ms,
            )

        # Return cached or generate
        if universe_id in self._universe_cache:
            asset_ids = self._universe_cache[universe_id]
        else:
            asset_ids = np.arange(1000, 1050, dtype=np.int64)

        return AdapterResponse(
            status=AdapterStatus.SUCCESS,
            data=asset_ids,
            metadata={"universe_id": universe_id, "as_of_date": as_of_date},
            latency_ms=self.latency_ms,
        )

    def write_factor(
        self,
        factor_id: str,
        data: np.ndarray,
        metadata: Dict[str, Any],
    ) -> AdapterResponse:
        """Mock factor writing."""
        self.request_count += 1
        self.request_history.append({
            "method": "write_factor",
            "factor_id": factor_id,
            "shape": data.shape,
            "timestamp": datetime.now().isoformat(),
        })

        if self._rng.random() < self.fail_rate:
            return AdapterResponse(
                status=AdapterStatus.FAILED,
                error_message=f"Simulated failure writing {factor_id}",
                latency_ms=self.latency_ms,
            )

        # Cache the data
        self._factor_cache[factor_id] = data.copy()
        self._metadata_cache[factor_id] = metadata.copy()

        return AdapterResponse(
            status=AdapterStatus.SUCCESS,
            metadata={"bytes_written": data.nbytes},
            latency_ms=self.latency_ms,
        )

    def get_schema(self, factor_id: str) -> AdapterResponse:
        """Mock schema retrieval."""
        if factor_id in self._metadata_cache:
            return AdapterResponse(
                status=AdapterStatus.SUCCESS,
                data=self._metadata_cache[factor_id],
                latency_ms=self.latency_ms,
            )

        return AdapterResponse(
            status=AdapterStatus.NOT_AVAILABLE,
            error_message=f"Schema not found for {factor_id}",
            latency_ms=self.latency_ms,
        )

    def reset(self):
        """Clear cache and reset counters."""
        self._factor_cache.clear()
        self._metadata_cache.clear()
        self._universe_cache.clear()
        self.request_count = 0
        self.request_history.clear()


class MockFactorEngineAdapter:
    """
    Mock FactorEngine adapter for testing factor computation.

    Simulates operator execution and factor materialization without
    requiring actual FactorEngine infrastructure.
    """

    def __init__(
        self,
        fail_rate: float = 0.0,
        latency_ms: float = 50.0,
        seed: Optional[int] = None,
    ):
        self.fail_rate = fail_rate
        self.latency_ms = latency_ms
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Simulated operator registry
        self._operators: Dict[str, Callable] = {}
        self._operator_metadata: Dict[str, Dict[str, Any]] = {}

        # Execution tracking
        self.execution_count = 0
        self.execution_history: List[Dict[str, Any]] = []

    def register_operator(
        self,
        operator_name: str,
        func: Callable,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a mock operator."""
        self._operators[operator_name] = func
        self._operator_metadata[operator_name] = metadata or {}

    def execute_operator(
        self,
        operator_name: str,
        inputs: Dict[str, np.ndarray],
        params: Dict[str, Any],
    ) -> AdapterResponse:
        """Mock operator execution."""
        self.execution_count += 1
        self.execution_history.append({
            "operator": operator_name,
            "input_shapes": {k: v.shape for k, v in inputs.items()},
            "params": params,
            "timestamp": datetime.now().isoformat(),
        })

        if self._rng.random() < self.fail_rate:
            return AdapterResponse(
                status=AdapterStatus.FAILED,
                error_message=f"Simulated failure executing {operator_name}",
                latency_ms=self.latency_ms,
            )

        # Execute registered operator or generate synthetic
        if operator_name in self._operators:
            try:
                result = self._operators[operator_name](inputs, params)
            except Exception as e:
                return AdapterResponse(
                    status=AdapterStatus.FAILED,
                    error_message=f"Operator execution error: {str(e)}",
                    latency_ms=self.latency_ms,
                )
        else:
            # Generate synthetic result matching first input shape
            first_input = next(iter(inputs.values()))
            result = self._rng.randn(*first_input.shape)

        return AdapterResponse(
            status=AdapterStatus.SUCCESS,
            data=result,
            metadata={"operator": operator_name},
            latency_ms=self.latency_ms,
        )

    def materialize_factor(
        self,
        factor_expr: str,
        start_date: str,
        end_date: str,
        universe: str,
    ) -> AdapterResponse:
        """Mock factor materialization."""
        self.execution_count += 1
        self.execution_history.append({
            "method": "materialize_factor",
            "factor_expr": factor_expr,
            "start_date": start_date,
            "end_date": end_date,
            "timestamp": datetime.now().isoformat(),
        })

        if self._rng.random() < self.fail_rate:
            return AdapterResponse(
                status=AdapterStatus.FAILED,
                error_message=f"Materialization failed for {factor_expr}",
                latency_ms=self.latency_ms * 3,  # Longer latency for materialization
            )

        # Generate synthetic result
        T, N = 100, 50
        data = self._rng.randn(T, N)

        return AdapterResponse(
            status=AdapterStatus.SUCCESS,
            data=data,
            metadata={
                "factor_expr": factor_expr,
                "shape": (T, N),
            },
            latency_ms=self.latency_ms * 3,
        )

    def get_operator_metadata(self, operator_name: str) -> AdapterResponse:
        """Retrieve operator metadata."""
        if operator_name in self._operator_metadata:
            return AdapterResponse(
                status=AdapterStatus.SUCCESS,
                data=self._operator_metadata[operator_name],
                latency_ms=self.latency_ms,
            )

        return AdapterResponse(
            status=AdapterStatus.NOT_AVAILABLE,
            error_message=f"Operator {operator_name} not found",
            latency_ms=self.latency_ms,
        )

    def reset(self):
        """Clear state and reset counters."""
        self.execution_count = 0
        self.execution_history.clear()


class MockEvaluationBackend:
    """
    Mock evaluation backend for testing metric computation.

    Simulates evaluation without requiring full QE infrastructure.
    """

    def __init__(
        self,
        fail_rate: float = 0.0,
        latency_ms: float = 30.0,
        seed: Optional[int] = None,
    ):
        self.fail_rate = fail_rate
        self.latency_ms = latency_ms
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Metric implementations
        self._metrics: Dict[str, Callable] = {}

        # Evaluation tracking
        self.evaluation_count = 0
        self.evaluation_history: List[Dict[str, Any]] = []

    def register_metric(self, metric_name: str, func: Callable):
        """Register a mock metric implementation."""
        self._metrics[metric_name] = func

    def evaluate(
        self,
        factor_values: np.ndarray,
        label_values: np.ndarray,
        metric_ids: Tuple[str, ...],
    ) -> AdapterResponse:
        """Mock evaluation."""
        self.evaluation_count += 1
        self.evaluation_history.append({
            "factor_shape": factor_values.shape,
            "label_shape": label_values.shape,
            "metrics": metric_ids,
            "timestamp": datetime.now().isoformat(),
        })

        if self._rng.random() < self.fail_rate:
            return AdapterResponse(
                status=AdapterStatus.FAILED,
                error_message="Simulated evaluation failure",
                latency_ms=self.latency_ms,
            )

        # Compute metrics
        results = {}
        for metric_id in metric_ids:
            if metric_id in self._metrics:
                try:
                    results[metric_id] = self._metrics[metric_id](
                        factor_values, label_values
                    )
                except Exception as e:
                    results[metric_id] = {
                        "error": str(e),
                        "value": np.nan,
                    }
            else:
                # Generate synthetic metric
                results[metric_id] = {
                    "value": self._rng.uniform(-0.1, 0.1),
                    "std_error": 0.01,
                    "n_obs": factor_values.shape[0] * factor_values.shape[1],
                }

        return AdapterResponse(
            status=AdapterStatus.SUCCESS,
            data=results,
            metadata={"num_metrics": len(metric_ids)},
            latency_ms=self.latency_ms,
        )

    def reset(self):
        """Clear state and reset counters."""
        self.evaluation_count = 0
        self.evaluation_history.clear()


# Pre-configured adapter factories

def create_stable_adapters(seed: int = 42) -> Tuple[MockDataAccessAdapter, MockFactorEngineAdapter]:
    """Create adapters with no failures for happy-path testing."""
    da_adapter = MockDataAccessAdapter(fail_rate=0.0, latency_ms=5.0, seed=seed)
    fe_adapter = MockFactorEngineAdapter(fail_rate=0.0, latency_ms=10.0, seed=seed)
    return da_adapter, fe_adapter


def create_flaky_adapters(seed: int = 42) -> Tuple[MockDataAccessAdapter, MockFactorEngineAdapter]:
    """Create adapters with occasional failures for robustness testing."""
    da_adapter = MockDataAccessAdapter(fail_rate=0.1, latency_ms=20.0, seed=seed)
    fe_adapter = MockFactorEngineAdapter(fail_rate=0.15, latency_ms=50.0, seed=seed)
    return da_adapter, fe_adapter


def create_slow_adapters(seed: int = 42) -> Tuple[MockDataAccessAdapter, MockFactorEngineAdapter]:
    """Create adapters with high latency for performance testing."""
    da_adapter = MockDataAccessAdapter(fail_rate=0.0, latency_ms=500.0, seed=seed)
    fe_adapter = MockFactorEngineAdapter(fail_rate=0.0, latency_ms=1000.0, seed=seed)
    return da_adapter, fe_adapter
