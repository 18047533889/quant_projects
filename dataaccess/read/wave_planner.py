"""
R32-P0-090: Single execution authority for batch factor reads.

Final chain: FactorSourcePlan → BatchDataRequest → ReadWavePlanner → PhysicalFactorDAG
Legacy batch plan is compat/explain only, not independent execution authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from data_access.read.source_binding import FactorSourcePlan


@dataclass(frozen=True)
class BatchDataRequest:
    """Canonical batch data request from FactorEngine.

    R32-P0-090: Single execution authority - this is the only input
    to ReadWavePlanner. Legacy FactorBatchPlan only provides explain/compat.
    """

    source_plan: FactorSourcePlan
    time_range: tuple[Any, Any]
    universe: str | None = None
    execution_mode: str = "batch"  # "batch", "streaming"
    max_concurrency: int | None = None
    resource_budget: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_plan": self.source_plan.to_dict(),
            "time_range": self.time_range,
            "universe": self.universe,
            "execution_mode": self.execution_mode,
            "max_concurrency": self.max_concurrency,
            "resource_budget": self.resource_budget,
        }


@dataclass(frozen=True)
class ReadWave:
    """Single read wave: datasets that can be read concurrently."""

    wave_id: int
    dataset_requests: tuple[tuple[str, tuple[str, ...]], ...]  # (dataset, columns)
    dependencies: tuple[int, ...] = ()  # prior wave IDs

    def get_datasets(self) -> list[str]:
        return [ds for ds, _ in self.dataset_requests]


@dataclass(frozen=True)
class PhysicalFactorDAG:
    """Physical execution plan: ordered waves with dependencies.

    R32-P0-090: This is the final execution authority. Created by
    ReadWavePlanner from BatchDataRequest.
    """

    waves: tuple[ReadWave, ...]
    total_datasets: int
    total_columns: int
    estimated_parallelism: int

    def explain(self) -> dict[str, Any]:
        """Explain execution plan for debugging/logging."""
        return {
            "num_waves": len(self.waves),
            "total_datasets": self.total_datasets,
            "total_columns": self.total_columns,
            "estimated_parallelism": self.estimated_parallelism,
            "waves": [
                {
                    "wave_id": w.wave_id,
                    "datasets": w.get_datasets(),
                    "dependencies": list(w.dependencies),
                }
                for w in self.waves
            ],
        }


class ReadWavePlanner:
    """Plan batch reads into concurrent waves.

    R32-P0-090: Single execution authority for batch factor reads.
    Replaces multiple competing planners.
    """

    def __init__(self, *, max_wave_size: int = 10) -> None:
        self.max_wave_size = max_wave_size

    def plan(self, request: BatchDataRequest) -> PhysicalFactorDAG:
        """Plan BatchDataRequest into PhysicalFactorDAG.

        Args:
            request: Canonical batch data request

        Returns:
            Physical execution DAG with ordered waves
        """
        # Get per-dataset column projections from source plan
        dataset_columns = request.source_plan.get_dataset_columns()

        # Simple wave planning: group datasets into waves respecting max_wave_size
        # Real implementation would consider dependencies and resource constraints
        waves: list[ReadWave] = []
        datasets = sorted(dataset_columns.items())

        wave_id = 0
        i = 0
        while i < len(datasets):
            batch = datasets[i : i + self.max_wave_size]
            wave = ReadWave(
                wave_id=wave_id,
                dataset_requests=tuple(
                    (ds, tuple(cols)) for ds, cols in batch
                ),
                dependencies=tuple(range(wave_id)) if wave_id > 0 else (),
            )
            waves.append(wave)
            wave_id += 1
            i += self.max_wave_size

        total_cols = sum(len(cols) for _, cols in dataset_columns.items())

        return PhysicalFactorDAG(
            waves=tuple(waves),
            total_datasets=len(dataset_columns),
            total_columns=total_cols,
            estimated_parallelism=min(len(dataset_columns), self.max_wave_size),
        )


class LegacyFactorBatchPlan:
    """Legacy batch plan - compat/explain only, NOT execution authority.

    R32-P0-090: This class exists only for backwards compatibility.
    It MUST NOT independently decide execution. All execution goes through
    ReadWavePlanner.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self._is_compat_only = True

    def to_canonical(self) -> BatchDataRequest:
        """Convert legacy plan to canonical BatchDataRequest.

        This is the only valid use of legacy plan - conversion to canonical form.
        """
        from data_access.read.source_binding import (
            ColumnSourceBinding,
            FactorSourcePlan,
            SourceScopeId,
        )

        # Extract bindings from legacy format
        bindings: list[ColumnSourceBinding] = []
        if "sources" in self.data:
            for source_spec in self.data["sources"]:
                ds = source_spec.get("dataset", "")
                for field in source_spec.get("fields", []):
                    bindings.append(
                        ColumnSourceBinding(
                            concept=field,
                            column=field,
                            source_scope=SourceScopeId(dataset=ds),
                        )
                    )

        source_plan = FactorSourcePlan(
            factor_ids=tuple(self.data.get("factor_ids", [])),
            bindings=tuple(bindings),
            time_range=self.data.get("time_range"),
            universe=self.data.get("universe"),
        )

        return BatchDataRequest(
            source_plan=source_plan,
            time_range=self.data.get("time_range", (None, None)),
            universe=self.data.get("universe"),
        )

    def explain(self) -> dict[str, Any]:
        """Explain legacy plan - for debugging only."""
        return {
            "legacy": True,
            "warning": "This plan is compat-only and does not execute directly",
            "conversion": "Use to_canonical() to get executable BatchDataRequest",
            "data": self.data,
        }

    def execute(self, *args: Any, **kwargs: Any) -> None:
        """Explicitly blocked - legacy plans cannot execute.

        Raises:
            RuntimeError: Always - this is compat-only
        """
        raise RuntimeError(
            "LegacyFactorBatchPlan cannot execute directly. "
            "Convert to BatchDataRequest via to_canonical() and use ReadWavePlanner."
        )


__all__ = [
    "BatchDataRequest",
    "ReadWave",
    "PhysicalFactorDAG",
    "ReadWavePlanner",
    "LegacyFactorBatchPlan",
]
