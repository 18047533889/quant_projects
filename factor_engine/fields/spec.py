"""Immutable field and source-table semantic contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FieldSpec:
    """Semantic description of one logical factor input field.

    ``name`` is the canonical DSL name, while ``source_name`` is the physical
    column in ``table``.  All attributes are deliberately JSON-compatible so
    the registry can provide a deterministic catalog hash.
    """

    name: str
    table: str
    source_name: str
    dtype: str = "float64"
    unit: str = "dimensionless"
    frequency: str = "daily"
    aliases: tuple[str, ...] = ()
    description: str = ""
    nullable: bool = True
    role: str = "feature"
    dataset: str | None = None
    adjustment: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def canonical_name(self) -> str:
        return self.name

    @property
    def physical_name(self) -> str:
        return self.source_name

    @property
    def qualified_name(self) -> str:
        return f"{self.table}.{self.source_name}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["aliases"] = list(self.aliases)
        result["metadata"] = dict(sorted(self.metadata.items()))
        return result


@dataclass(frozen=True)
class TableSpec:
    """Logical table contract used by field resolution and catalog export."""

    name: str
    dataset: str
    time_column: str | None = "TradeDate"
    instrument_column: str | None = "Symbol"
    frequency: str = "daily"
    domain: str = "auxiliary"
    aliases: tuple[str, ...] = ()
    description: str = ""
    fields: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def logical_table(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["aliases"] = list(self.aliases)
        result["fields"] = list(self.fields)
        result["metadata"] = dict(sorted(self.metadata.items()))
        return result
