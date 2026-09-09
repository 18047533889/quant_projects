"""Real-scale benchmark scaffolding and training/runtime responsibility gates."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

PREDICTION_SIZES = (1_000, 3_000, 5_000, 10_000)
RETRAIN_SIZES = (100_000, 500_000, 1_000_000, 5_000_000)

BENCHMARK_PHASES = (
    "io", "compile", "transfer", "kernel", "sync", "serialize", "publish", "hash",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    hardware_ref: str
    dependency_versions_ref: str
    t: int
    n: int
    f: int
    dtype: str
    quantiles: int
    metric_profile_ref: str
    data_bytes: int
    cache_state: str
    effective_backend: str

    def __post_init__(self) -> None:
        for name in ("hardware_ref", "dependency_versions_ref", "dtype",
                     "metric_profile_ref", "effective_backend"):
            if not getattr(self, name):
                raise ValueError(f"BenchmarkSpec.{name} is required")
        if min(self.t, self.n, self.f, self.quantiles, self.data_bytes) < 0:
            raise ValueError("benchmark dimensions and data_bytes cannot be negative")
        if self.cache_state not in {"cold", "warm"}:
            raise ValueError("cache_state must be 'cold' or 'warm'")


def measure_synchronized_phase(work, synchronize, *, clock=time.perf_counter):
    """Measure device work through its completion barrier, not kernel launch."""
    if not callable(work) or not callable(synchronize):
        raise TypeError("work and synchronize must be callable")
    started = clock()
    result = work()
    synchronize()
    stopped = clock()
    elapsed = float(stopped - started)
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("benchmark clock produced invalid elapsed time")
    return result, elapsed


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
    spec: BenchmarkSpec | None = None
    phase_seconds: dict[str, float] | None = None
    device_synchronized: bool = False
    correctness_verified: bool = False
    max_abs_error: float | None = None
    missing_state_ref: str = ""

    def require_publishable(self) -> None:
        if self.spec is None:
            raise ValueError("benchmark receipt requires BenchmarkSpec")
        if self.phase_seconds is None or set(self.phase_seconds) != set(BENCHMARK_PHASES):
            raise ValueError("benchmark receipt must contain every end-to-end phase")
        if any(not np.isfinite(value) or value < 0 for value in self.phase_seconds.values()):
            raise ValueError("benchmark phase durations must be finite and non-negative")
        if self.spec.effective_backend.lower() in {"cuda", "gpu", "cupy", "torch-cuda"} and not self.device_synchronized:
            raise ValueError("GPU benchmark cannot publish before device synchronization")
        if not self.correctness_verified or self.max_abs_error is None or not np.isfinite(self.max_abs_error):
            raise ValueError("benchmark speed is not publishable before correctness verification")
        if not self.missing_state_ref:
            raise ValueError("benchmark receipt requires missing-state identity")

    @property
    def end_to_end_seconds(self) -> float:
        if self.phase_seconds is not None:
            return float(sum(self.phase_seconds.values()))
        return self.fit_seconds or (self.preprocessing_seconds + self.resolution_seconds + self.prediction_seconds)

    @property
    def rows_per_second(self) -> float:
        elapsed = self.end_to_end_seconds
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
