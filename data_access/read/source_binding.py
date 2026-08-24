"""
R32-P0-087: FactorSourcePlan typed Concept/Column→Dataset binding.

Replaces independent `concepts` and `datasets` collections with explicit
ColumnSourceBinding that captures the full semantic contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SourceScopeId:
    """Fully-qualified source identity for a data field.

    Captures market/provider/frequency/grain/timeframe/price_basis/revision
    contract to ensure one source of truth for field resolution.
    """

    dataset: str
    market: str | None = None
    provider: str | None = None
    frequency: str | None = None
    grain: str | None = None
    timeframe: str | None = None
    price_basis: str | None = None
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "market": self.market,
            "provider": self.provider,
            "frequency": self.frequency,
            "grain": self.grain,
            "timeframe": self.timeframe,
            "price_basis": self.price_basis,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceScopeId:
        return cls(
            dataset=str(data["dataset"]),
            market=data.get("market"),
            provider=data.get("provider"),
            frequency=data.get("frequency"),
            grain=data.get("grain"),
            timeframe=data.get("timeframe"),
            price_basis=data.get("price_basis"),
            revision=data.get("revision"),
        )


@dataclass(frozen=True)
class ColumnSourceBinding:
    """Explicit binding: semantic concept/column → dataset → source scope.

    R32-P0-087: No more guessing which dataset provides which field.
    Each column has exactly one authoritative source binding.
    """

    concept: str  # semantic field name (e.g., "close", "revenue")
    column: str   # physical column name in dataset
    source_scope: SourceScopeId
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "column": self.column,
            "source_scope": self.source_scope.to_dict(),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ColumnSourceBinding:
        return cls(
            concept=str(data["concept"]),
            column=str(data["column"]),
            source_scope=SourceScopeId.from_dict(data["source_scope"]),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True)
class FactorSourcePlan:
    """Batch factor read plan with typed source bindings.

    R32-P0-087: Saves explicit Concept/Column→Dataset typed binding.
    R32-P0-088: Each dataset only projects fields it actually needs.
    """

    factor_ids: tuple[str, ...]
    bindings: tuple[ColumnSourceBinding, ...]
    time_range: tuple[Any, Any] | None = None
    universe: str | None = None

    def get_dataset_columns(self) -> dict[str, list[str]]:
        """R32-P0-088: Return per-dataset projection (only required columns)."""
        result: dict[str, list[str]] = {}
        for binding in self.bindings:
            ds = binding.source_scope.dataset
            if ds not in result:
                result[ds] = []
            if binding.column not in result[ds]:
                result[ds].append(binding.column)
        return result

    def get_bindings_for_dataset(self, dataset: str) -> list[ColumnSourceBinding]:
        """Return all bindings that source from the given dataset."""
        return [b for b in self.bindings if b.source_scope.dataset == dataset]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_ids": list(self.factor_ids),
            "bindings": [b.to_dict() for b in self.bindings],
            "time_range": self.time_range,
            "universe": self.universe,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorSourcePlan:
        return cls(
            factor_ids=tuple(data["factor_ids"]),
            bindings=tuple(
                ColumnSourceBinding.from_dict(b) for b in data["bindings"]
            ),
            time_range=data.get("time_range"),
            universe=data.get("universe"),
        )


__all__ = [
    "SourceScopeId",
    "ColumnSourceBinding",
    "FactorSourcePlan",
]
