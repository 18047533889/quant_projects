"""Strict model-training reads pinned to one ExperimentDataSnapshot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from data_access.r30.contracts import (
    BackendReadError,
    EmptyResultError,
    SnapshotCompatibilityError,
    SnapshotFidelity,
    SnapshotFidelityError,
)
from data_access.r30.experiment_snapshot import ExperimentDataSnapshot


@dataclass(frozen=True)
class TrainingPanel:
    data: Any
    experiment_snapshot: ExperimentDataSnapshot
    feature_columns: tuple[str, ...]
    label_column: str
    universe: str
    decision_clock: str


@dataclass(frozen=True)
class TrainingReadPlan:
    features: tuple[str, ...]
    label: str
    universe: str
    date_range: tuple[Any, Any]
    decision_clock: str
    datasets: Mapping[str, str]
    experiment_snapshot: ExperimentDataSnapshot
    snapshot_fidelity: SnapshotFidelity = SnapshotFidelity.PROVEN
    financial_revision_mode: str = "point_in_time"
    representation: str = "arrow"
    allow_empty: bool = False

    def __post_init__(self) -> None:
        if not self.features or not self.label or not self.universe:
            raise ValueError("features, label and universe are required")
        if len(self.date_range) != 2:
            raise ValueError("date_range must be a (start, end) pair")
        if not self.decision_clock:
            raise ValueError("decision_clock is required")
        if self.snapshot_fidelity is not SnapshotFidelity.PROVEN:
            raise SnapshotFidelityError(
                "authoritative training requires proven snapshot fidelity, got "
                f"{self.snapshot_fidelity.value}"
            )
        if self.financial_revision_mode not in {
            "point_in_time",
            "retrospective_research",
        }:
            raise ValueError(
                "financial_revision_mode must be point_in_time or retrospective_research"
            )

    @classmethod
    def build(
        cls,
        store: Any,
        *,
        experiment_id: str,
        market: str | None,
        features: Sequence[str],
        label: str,
        universe: str,
        date_range: tuple[Any, Any],
        decision_clock: str,
        datasets: Mapping[str, str],
        **kwargs: Any,
    ) -> "TrainingReadPlan":
        snapshot = ExperimentDataSnapshot.build(
            store,
            experiment_id=experiment_id,
            market=market,
            datasets=datasets,
            universe=universe,
        )
        return cls(
            features=tuple(features),
            label=str(label),
            universe=str(universe),
            date_range=tuple(date_range),
            decision_clock=str(decision_clock),
            datasets=dict(datasets),
            experiment_snapshot=snapshot,
            **kwargs,
        )

    def assert_snapshot_compatible(self, *snapshot_ids: str | None) -> None:
        expected = self.experiment_snapshot.snapshot_id
        incompatible = sorted(
            {str(s) for s in snapshot_ids if s is not None and str(s) != expected}
        )
        if incompatible:
            raise SnapshotCompatibilityError(
                f"training inputs must use snapshot {expected}; incompatible={incompatible}"
            )

    def execute(self, store: Any) -> TrainingPanel:
        """Execute one joined read without converting failures into empty data."""
        read_joined = getattr(store, "read_joined", None)
        if not callable(read_joined):
            raise BackendReadError("store does not provide read_joined")
        fields = {name: [] for name in self.datasets}
        anchor = next(iter(self.datasets))
        fields[anchor] = list(self.features) + [self.label]
        try:
            data = read_joined(
                anchor=anchor,
                fields=fields,
                time_range=self.date_range,
                universe=self.universe,
                decision_clock=self.decision_clock,
                experiment_snapshot=self.experiment_snapshot,
                financial_revision_mode=self.financial_revision_mode,
                representation=self.representation,
            )
        except Exception as exc:
            from data_access.core.exceptions import DataAccessError

            if isinstance(exc, DataAccessError):
                raise
            raise BackendReadError("training joined read failed") from exc
        if not self.allow_empty and _is_empty(data):
            raise EmptyResultError("training read completed with an empty panel")
        observed = getattr(data, "experiment_snapshot_id", None)
        if observed is not None:
            self.assert_snapshot_compatible(str(observed))
        return TrainingPanel(
            data=data,
            experiment_snapshot=self.experiment_snapshot,
            feature_columns=self.features,
            label_column=self.label,
            universe=self.universe,
            decision_clock=self.decision_clock,
        )


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    num_rows = getattr(value, "num_rows", None)
    if num_rows is not None:
        return int(num_rows) == 0
    empty = getattr(value, "empty", None)
    if empty is not None:
        return bool(empty)
    try:
        return len(value) == 0
    except TypeError:
        return False


__all__ = ["TrainingPanel", "TrainingReadPlan"]
