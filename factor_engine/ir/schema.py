"""IR result schema with optional field semantics."""

from dataclasses import dataclass

from .types import ValueType, normalize_dtype


@dataclass(frozen=True)
class Schema:
    """描述某个 IR 子树输出结果的静态 schema（当前引擎主路径未强依赖）。"""

    value_type: ValueType  # scalar / series / panel
    dtype: str             # 如 float64
    index: tuple[str, ...]  # MultiIndex 层名，如 ("timestamp", "instrument")
    unit: str | None = None
    field_name: str | None = None
    source_table: str | None = None
    source_field: str | None = None
    frequency: str = "daily"
    temporal_model: str = "exact"
    cardinality: str = "many_to_one"
    domain: str = "auxiliary"
    pit_safe: bool = True
    nullable: bool = True

    @classmethod
    def from_field(cls, spec, *, value_type: ValueType = ValueType.PANEL) -> "Schema":
        """Create a panel schema from a ``FieldSpec`` without coupling imports."""

        return cls(
            value_type=value_type,
            dtype=normalize_dtype(spec.dtype),
            index=("timestamp", "instrument"),
            unit=spec.unit,
            field_name=spec.name,
            source_table=spec.table,
            source_field=spec.source_name,
            frequency=spec.frequency,
            temporal_model=spec.temporal_model,
            cardinality=spec.cardinality,
            domain=spec.domain,
            pit_safe=spec.strict_pit_allowed,
            nullable=spec.nullable,
        )


DEFAULT_COLUMN_SCHEMA = Schema(ValueType.PANEL, "float64", ("timestamp", "instrument"))


__all__ = ["DEFAULT_COLUMN_SCHEMA", "Schema"]
