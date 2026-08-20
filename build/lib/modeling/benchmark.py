"""Real-scale benchmark scaffolding and training/runtime responsibility gates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PREDICTION_SIZES = (1_000, 3_000, 5_000, 10_000)
RETRAIN_SIZES = (100_000, 500_000, 1_000_000, 5_000_000)


@dataclass(frozen=True)
class BenchmarkResult:
    rows: int
    fit_seconds: float = 0.0
    peak_memory_bytes: int = 0
    artifact_bytes: int = 0
    preprocessing_seconds: float = 0.0
    resolution_seconds: float = 0.0
    prediction_seconds: float = 0.0
    copy_count: int = 0

    @property
    def rows_per_second(self) -> float:
        elapsed = self.fit_seconds or (self.preprocessing_seconds + self.resolution_seconds + self.prediction_seconds)
        return float("inf") if elapsed == 0 else self.rows / elapsed


class RuntimeBoundary:
    TRAINING_OPERATIONS = frozenset({"dataset_build", "fit", "validate", "test", "select", "artifact_produce", "certify", "retrain"})
    RUNTIME_OPERATIONS = frozenset({"feature_compute", "resolve_artifact", "frozen_score"})

    @classmethod
    def require_runtime(cls, operation: str) -> None:
        if operation not in cls.RUNTIME_OPERATIONS:
            raise RuntimeError(f"FactorEngine Runtime cannot perform {operation}")


def deterministic_resource_parity(runs: dict[str, Any], *, atol: float = 1e-10, rtol: float = 1e-10) -> None:
    """Check thread-count, batch-size, and shard-order metamorphic parity."""
    required = {"single_thread", "multi_thread", "batch_size", "shard_order"}
    missing = required - set(runs)
    if missing:
        raise ValueError(f"missing metamorphic runs: {sorted(missing)}")
    baseline = np.asarray(runs["single_thread"])
    for name, values in runs.items():
        if not np.allclose(baseline, np.asarray(values), atol=atol, rtol=rtol, equal_nan=True):
            raise AssertionError(f"determinism parity failed for {name}")


def seed_stability(scores: dict[int, float], *, max_spread: float) -> bool:
    """Research-only gate for stochastic learners across multiple seeds."""
    if len(scores) < 2 or not all(np.isfinite(list(scores.values()))):
        return False
    return max(scores.values()) - min(scores.values()) <= max_spread


def validate_benchmark_sizes(prediction_sizes=PREDICTION_SIZES, retrain_sizes=RETRAIN_SIZES) -> None:
    if tuple(prediction_sizes) != PREDICTION_SIZES:
        raise ValueError("prediction benchmark must cover 1k/3k/5k/10k rows")
    if tuple(retrain_sizes) != RETRAIN_SIZES:
        raise ValueError("retraining benchmark must cover 100k/500k/1m/5m rows")
