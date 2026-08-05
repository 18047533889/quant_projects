"""Deterministic field registry and resolver."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .spec import FIELD_CATALOG_SCHEMA_VERSION, FieldSpec, TableSpec


class FieldRegistry:
    """Mutable registry with immutable specs and deterministic export."""

    def __init__(self, fields: Iterable[FieldSpec] = (), tables: Iterable[TableSpec] = ()):
        self._fields: dict[str, FieldSpec] = {}
        self._tables: dict[str, TableSpec] = {}
        self._field_aliases: dict[str, set[str]] = {}
        self._table_aliases: dict[str, str] = {}
        for table in tables:
            self.register_table(table)
        for spec in fields:
            self.register(spec)

    @staticmethod
    def _key(value: str) -> str:
        return str(value).strip().lower()

    def register_table(self, spec: TableSpec, *, replace: bool = False) -> TableSpec:
        self._validate_table(spec)
        key = self._key(spec.name)
        if key in self._tables and not replace:
            raise ValueError(f"table already registered: {spec.name}")
        aliases = (spec.name, spec.dataset, *spec.aliases)
        for alias in aliases:
            alias_key = self._key(alias)
            owner = self._table_aliases.get(alias_key)
            if owner is not None and owner != key and not replace:
                raise ValueError(f"table alias already registered: {alias}")
        self._tables[key] = spec
        for alias in aliases:
            self._table_aliases[self._key(alias)] = key
        return spec

    def register(self, spec: FieldSpec, *, replace: bool = False) -> FieldSpec:
        self._validate_field(spec)
        identity = self._identity(spec.table, spec.name)
        if identity in self._fields and not replace:
            raise ValueError(f"field already registered: {spec.table}.{spec.name}")
        self._fields[identity] = spec
        aliases = {
            spec.name, spec.source_name, spec.qualified_name,
            f"{spec.table}.{spec.name}", *(spec.aliases or ()),
        }
        if spec.dataset:
            aliases.update({f"{spec.dataset}.{spec.name}", f"{spec.dataset}.{spec.source_name}"})
        for alias in aliases:
            self._field_aliases.setdefault(self._key(alias), set()).add(identity)
        return spec

    def _identity(self, table: str, field: str) -> str:
        table_key = self._table_aliases.get(self._key(table), self._key(table))
        return f"{table_key}.{self._key(field)}"

    def _validate_table(self, spec: TableSpec) -> None:
        if not spec.name or not spec.dataset:
            raise ValueError("table name and dataset must be non-empty")
        if spec.join_policy in {"financial_pit", "relation_pit"} and not spec.knowledge_time_column:
            raise ValueError(f"{spec.name}: {spec.join_policy} requires knowledge_time_column")
        if spec.join_policy == "financial_pit" and not spec.period_id_column:
            raise ValueError(f"{spec.name}: financial_pit requires period_id_column")
        if spec.cardinality not in {"one_to_one", "many_to_one", "one_to_many"}:
            raise ValueError(f"{spec.name}: unsupported cardinality {spec.cardinality!r}")

    def _validate_field(self, spec: FieldSpec) -> None:
        table = self.resolve_table(spec.table)
        if table is None:
            raise ValueError(f"field {spec.name!r} references unknown table {spec.table!r}")
        if spec.dataset and spec.dataset != table.dataset:
            raise ValueError(
                f"{spec.field_id}: dataset {spec.dataset!r} does not match table dataset {table.dataset!r}"
            )
        if spec.scale_to_canonical is None or spec.scale_to_canonical <= 0:
            raise ValueError(f"{spec.field_id}: scale_to_canonical must be positive")
        if spec.cardinality == "one_to_many" and spec.mining_allowed:
            raise ValueError(
                f"{spec.field_id}: one-to-many fields must be aggregated before mining"
            )

    def resolve_table(self, name: str) -> TableSpec | None:
        key = self._table_aliases.get(self._key(name), self._key(name))
        return self._tables.get(key)

    def get(self, name: str, *, table: str | None = None, strict: bool = False) -> FieldSpec | None:
        candidates: set[str]
        if table is not None:
            table_spec = self.resolve_table(table)
            table_name = table_spec.name if table_spec else table
            allowed_identities = {
                self._identity(spec.table, spec.name) for spec in self.fields(table=table_name)
            }
            candidates = set(self._field_aliases.get(self._key(name), set())) & allowed_identities
            direct = self._fields.get(self._identity(table_name, name))
            if direct is not None:
                candidates.add(self._identity(table_name, direct.name))
        else:
            candidates = set(self._field_aliases.get(self._key(name), set()))
            if "." in str(name):
                maybe_table, maybe_field = str(name).rsplit(".", 1)
                direct = self._fields.get(self._identity(maybe_table, maybe_field))
                if direct is not None:
                    candidates.add(self._identity(direct.table, direct.name))
        specs = {self._fields[item] for item in candidates if item in self._fields}
        if len(specs) == 1:
            return next(iter(specs))
        if not specs:
            if strict:
                qualifier = f" in table {table!r}" if table else ""
                raise KeyError(f"unknown field {name!r}{qualifier}")
            return None
        names = ", ".join(sorted(spec.qualified_name for spec in specs))
        if strict:
            raise KeyError(f"ambiguous field {name!r}; candidates: {names}")
        return None

    resolve = get

    def require(self, name: str, *, table: str | None = None) -> FieldSpec:
        result = self.get(name, table=table, strict=True)
        assert result is not None
        return result

    def fields(self, *, table: str | None = None) -> tuple[FieldSpec, ...]:
        values = self._fields.values()
        if table is not None:
            table_spec = self.resolve_table(table)
            name = table_spec.name if table_spec else table
            values = (item for item in values if self._key(item.table) == self._key(name))
        return tuple(sorted(values, key=lambda item: (item.table, item.name)))

    def tables(self) -> tuple[TableSpec, ...]:
        return tuple(sorted(self._tables.values(), key=lambda item: item.name))

    def export_catalog(self) -> dict[str, object]:
        return {
            "schema_version": FIELD_CATALOG_SCHEMA_VERSION,
            "tables": [item.to_dict() for item in self.tables()],
            "fields": [item.to_dict() for item in self.fields()],
        }

    def catalog_hash(self) -> str:
        payload = json.dumps(
            self.export_catalog(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = ["FieldRegistry"]
