# -*- coding: utf-8 -*-
"""Cost model and calibration governance (MB-P2-011).

This module ensures:
- MB-P2-011: static prior is primary, runtime actual only does bounded calibration
- No unbounded drift from measured baseline
- Clear separation of static cost model vs runtime calibration
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StaticCostPrior:
    """Static cost prior for an operator/backend combination (MB-P2-011).

    This is the canonical cost model, derived from:
    - Operator complexity analysis
    - Backend capability/algorithm
    - DataShape metadata
    - Hardware family

    Runtime measurements can only apply bounded calibration.
    """

    canonical: str
    backend: str
    base_cost_ms: float
    row_coefficient: float  # ms per 1M rows
    window_coefficient: float = 0.0  # Additional cost per window size
    feature_dim_coefficient: float = 0.0  # For model operators

    # Static rationale (not runtime-derived)
    complexity_class: str = "medium"  # low, medium, high, very_high
    algorithm_family: str = ""
    requires_sort: bool = False
    requires_full_pass: bool = False

    def estimate_ms(self, rows: int, window: int = 0, feature_dim: int = 0) -> float:
        """Estimate cost from static prior."""
        millions = max(rows / 1_000_000.0, 0.001)
        cost = self.base_cost_ms + self.row_coefficient * millions
        if window > 0:
            cost += self.window_coefficient * (window / 20.0)  # Normalized to window=20
        if feature_dim > 0:
            cost += self.feature_dim_coefficient * (feature_dim / 10.0)
        return max(cost, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "backend": self.backend,
            "base_cost_ms": round(self.base_cost_ms, 3),
            "row_coefficient": round(self.row_coefficient, 6),
            "window_coefficient": round(self.window_coefficient, 6),
            "feature_dim_coefficient": round(self.feature_dim_coefficient, 6),
            "complexity_class": self.complexity_class,
            "algorithm_family": self.algorithm_family,
            "requires_sort": self.requires_sort,
            "requires_full_pass": self.requires_full_pass,
        }


@dataclass
class RuntimeCalibration:
    """Bounded runtime calibration adjustment (MB-P2-011).

    Actual runtime measurements can adjust the static prior by at most
    CALIBRATION_BOUNDS (e.g., ±30%). Prevents unbounded drift.
    """

    canonical: str
    backend: str
    static_estimate_ms: float
    actual_ms: float
    sample_count: int = 1

    # Bounded adjustment
    calibration_factor: float = 1.0  # Bounded to [0.7, 1.3]
    ema_alpha: float = 0.1

    CALIBRATION_LOWER_BOUND = 0.7
    CALIBRATION_UPPER_BOUND = 1.3

    def update_with_actual(self, new_actual_ms: float, new_static_estimate: float) -> RuntimeCalibration:
        """Update calibration with new measurement (bounded).

        Returns new RuntimeCalibration instance (immutable update).
        """
        # Compute new factor from this sample
        if new_static_estimate > 0:
            new_factor = new_actual_ms / new_static_estimate
        else:
            new_factor = 1.0

        # Bound the new factor
        new_factor = max(self.CALIBRATION_LOWER_BOUND, min(self.CALIBRATION_UPPER_BOUND, new_factor))

        # EMA update of calibration factor
        updated_factor = self.ema_alpha * new_factor + (1 - self.ema_alpha) * self.calibration_factor
        updated_factor = max(self.CALIBRATION_LOWER_BOUND, min(self.CALIBRATION_UPPER_BOUND, updated_factor))

        return RuntimeCalibration(
            canonical=self.canonical,
            backend=self.backend,
            static_estimate_ms=new_static_estimate,
            actual_ms=new_actual_ms,
            sample_count=self.sample_count + 1,
            calibration_factor=updated_factor,
            ema_alpha=self.ema_alpha,
        )

    def calibrated_estimate(self, static_ms: float) -> float:
        """Apply bounded calibration to static estimate."""
        return static_ms * self.calibration_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "backend": self.backend,
            "static_estimate_ms": round(self.static_estimate_ms, 3),
            "actual_ms": round(self.actual_ms, 3),
            "sample_count": self.sample_count,
            "calibration_factor": round(self.calibration_factor, 4),
        }


class CostModelStore:
    """Store for static priors and runtime calibrations (MB-P2-011)."""

    def __init__(self, static_priors_path: Path | None = None):
        self.static_priors_path = static_priors_path
        self._static_priors: dict[tuple[str, str], StaticCostPrior] = {}
        self._runtime_calibrations: dict[tuple[str, str], RuntimeCalibration] = {}

        if static_priors_path and static_priors_path.is_file():
            self._load_static_priors()

    def _load_static_priors(self) -> None:
        """Load static priors from file."""
        if not self.static_priors_path or not self.static_priors_path.is_file():
            return

        try:
            data = json.loads(self.static_priors_path.read_text(encoding="utf-8"))
            priors = data.get("priors", [])
            for p in priors:
                prior = StaticCostPrior(
                    canonical=p["canonical"],
                    backend=p["backend"],
                    base_cost_ms=p["base_cost_ms"],
                    row_coefficient=p["row_coefficient"],
                    window_coefficient=p.get("window_coefficient", 0.0),
                    feature_dim_coefficient=p.get("feature_dim_coefficient", 0.0),
                    complexity_class=p.get("complexity_class", "medium"),
                    algorithm_family=p.get("algorithm_family", ""),
                    requires_sort=p.get("requires_sort", False),
                    requires_full_pass=p.get("requires_full_pass", False),
                )
                self._static_priors[(prior.canonical, prior.backend)] = prior
        except Exception:
            pass

    def get_static_prior(self, canonical: str, backend: str) -> StaticCostPrior | None:
        """Get static cost prior (MB-P2-011: static is primary)."""
        return self._static_priors.get((canonical, backend))

    def get_calibration(self, canonical: str, backend: str) -> RuntimeCalibration | None:
        """Get runtime calibration (MB-P2-011: runtime is bounded adjustment only)."""
        return self._runtime_calibrations.get((canonical, backend))

    def estimate_cost(
        self,
        canonical: str,
        backend: str,
        rows: int,
        window: int = 0,
        feature_dim: int = 0,
    ) -> float:
        """Estimate cost with optional bounded calibration (MB-P2-011).

        1. Start with static prior (required)
        2. Apply bounded runtime calibration if available (±30% max)
        """
        prior = self.get_static_prior(canonical, backend)
        if prior is None:
            # Fallback: generic cost model
            return self._generic_fallback_cost(canonical, backend, rows)

        static_estimate = prior.estimate_ms(rows, window, feature_dim)

        # Apply bounded calibration if available
        calibration = self.get_calibration(canonical, backend)
        if calibration:
            return calibration.calibrated_estimate(static_estimate)

        return static_estimate

    def record_actual(
        self,
        canonical: str,
        backend: str,
        actual_ms: float,
        rows: int,
        window: int = 0,
        feature_dim: int = 0,
    ) -> None:
        """Record actual runtime and update bounded calibration (MB-P2-011)."""
        prior = self.get_static_prior(canonical, backend)
        if prior is None:
            return  # No prior, can't calibrate

        static_estimate = prior.estimate_ms(rows, window, feature_dim)

        existing = self.get_calibration(canonical, backend)
        if existing:
            updated = existing.update_with_actual(actual_ms, static_estimate)
        else:
            updated = RuntimeCalibration(
                canonical=canonical,
                backend=backend,
                static_estimate_ms=static_estimate,
                actual_ms=actual_ms,
                sample_count=1,
            )
            # Compute initial factor
            if static_estimate > 0:
                factor = actual_ms / static_estimate
                factor = max(
                    RuntimeCalibration.CALIBRATION_LOWER_BOUND,
                    min(RuntimeCalibration.CALIBRATION_UPPER_BOUND, factor),
                )
                updated = RuntimeCalibration(
                    canonical=canonical,
                    backend=backend,
                    static_estimate_ms=static_estimate,
                    actual_ms=actual_ms,
                    sample_count=1,
                    calibration_factor=factor,
                )

        self._runtime_calibrations[(canonical, backend)] = updated

    def _generic_fallback_cost(self, canonical: str, backend: str, rows: int) -> float:
        """Generic fallback when no static prior exists."""
        millions = max(rows / 1_000_000.0, 0.001)
        base = {"pandas_numpy": 0.5, "polars_panel": 0.3, "polars_long": 0.3,
                "duckdb_sql": 0.2, "q_kx": 0.15}.get(backend, 0.5)
        return base * millions


# Global cost model store
_COST_MODEL_STORE: CostModelStore | None = None


def get_cost_model_store() -> CostModelStore:
    """Get global cost model store singleton."""
    global _COST_MODEL_STORE
    if _COST_MODEL_STORE is None:
        # Try to find static priors file
        from pathlib import Path
        priors_path = Path(__file__).resolve().parents[1] / "benchmarks" / "static_cost_priors.json"
        _COST_MODEL_STORE = CostModelStore(priors_path if priors_path.exists() else None)
    return _COST_MODEL_STORE
