"""Deterministic field registry and resolver."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from .spec import FIELD_CATALOG_SCHEMA_VERSION, FieldSpec, TableSpec


# R10-P0-017: tri-state field resolution.  ``get(..., strict=False)`` used to
# collapse "does not exist" and "ambiguous alias" into the same ``None`` — but
# their production meaning is opposite: an UNKNOWN field may be allowed as a
# raw column, an AMBIGUOUS alias must ALWAYS be rejected.  ``resolve_field``
# returns one of these instead of a bare ``None``.
@dataclass(frozen=True)
class ResolvedField:
    spec: FieldSpec


@dataclass(frozen=True)
class UnknownField:
    name: str


@dataclass(frozen=True)
class AmbiguousField:
    name: str
    candidates: tuple[str, ...]


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

    def register_table(
        self,
        spec: TableSpec,
        *,
        replace: bool = False,
        expected_old_identity: str | None = None,
    ) -> TableSpec:
        self._validate_table(spec)
        key = self._key(spec.name)
        if key in self._tables and not replace:
            raise ValueError(f"table already registered: {spec.name}")
        aliases = (spec.name, spec.dataset, *spec.aliases)
        # R10-P0-016: ``replace=True`` must NOT silently take over an alias that
        # belongs to a DIFFERENT table identity.  The caller has to name the
        # expected old owner; otherwise the third-party alias is stolen and every
        # existing reference silently repoints at the new table.
        expected = self._key(expected_old_identity) if expected_old_identity else None
        for alias in aliases:
            alias_key = self._key(alias)
            owner = self._table_aliases.get(alias_key)
            if owner is not None and owner != key:
                if not replace:
                    raise ValueError(f"table alias already registered: {alias}")
                if expected is None or expected != owner:
                    raise ValueError(
                        f"table alias {alias!r} belongs to table {owner!r}; "
                        f"replace=True cannot take it over without "
                        f"expected_old_identity={owner!r}"
                    )
        # Round-7 WS-E #283: on replace, drop the aliases owned by the OLD table
        # identity so a renamed source/dataset cannot leave stale aliases pointing
        # at the new table.
        if replace and key in self._tables:
            old = self._tables[key]
            for alias in (old.name, old.dataset, *old.aliases):
                alias_key = self._key(alias)
                if self._table_aliases.get(alias_key) == key:
                    del self._table_aliases[alias_key]
        self._tables[key] = spec
        for alias in aliases:
            self._table_aliases[self._key(alias)] = key
        return spec

    @staticmethod
    def _field_alias_set(spec: FieldSpec) -> set[str]:
        """Alias spellings owned by one field identity (round-7 WS-E #283)."""
        aliases = {
            spec.name, spec.source_name, spec.qualified_name,
            f"{spec.table}.{spec.name}", *(spec.aliases or ()),
        }
        if spec.dataset:
            aliases.update({f"{spec.dataset}.{spec.name}", f"{spec.dataset}.{spec.source_name}"})
        return {str(item) for item in aliases if item}

    def register(
        self,
        spec: FieldSpec,
        *,
        replace: bool = False,
        expected_old_identity: str | None = None,
    ) -> FieldSpec:
        self._validate_field(spec)
        identity = self._identity(spec.table, spec.name)
        if identity in self._fields and not replace:
            raise ValueError(f"field already registered: {spec.table}.{spec.name}")
        # R10-P0-016: a field-level replace must not steal an alias owned by a
        # third-party field identity (mirrors the table-level rule).
        if replace:
            expected = self._key(expected_old_identity) if expected_old_identity else None
            for alias in self._field_alias_set(spec):
                owners = set(self._field_aliases.get(self._key(alias), set()))
                third = {o for o in owners if o != identity}
                if third and not (expected and expected in third):
                    raise ValueError(
                        f"field alias {alias!r} belongs to "
                        f"{sorted(third)}; replace=True cannot take it over "
                        f"without expected_old_identity={sorted(third)}"
                    )
        # Round-7 WS-E #283: on replace, remove every alias owned by the OLD
        # identity (old source_name / aliases / dataset-qualified spellings)
        # before registering the new spec's aliases.  Shared aliases keep the
        # other owners — only the replaced identity is dropped from each set.
        if replace and identity in self._fields:
            old = self._fields[identity]
            for alias in self._field_alias_set(old):
                alias_key = self._key(alias)
                owners = self._field_aliases.get(alias_key)
                if owners is None:
                    continue
                owners.discard(identity)
                if not owners:
                    del self._field_aliases[alias_key]
        self._fields[identity] = spec
        for alias in self._field_alias_set(spec):
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

    def _resolve_candidates(self, name: str, *, table: str | None = None) -> set[str]:
        """Candidate field identities matching ``name`` under ``table``."""
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
        return candidates

    def resolve_field(
        self,
        name: str,
        *,
        table: str | None = None,
    ) -> ResolvedField | UnknownField | AmbiguousField:
        """Tri-state resolution (R10-P0-017).

        Production callers distinguish the three outcomes instead of receiving a
        bare ``None``: an UNKNOWN field may be accepted as a raw column when
        explicitly allowed, an AMBIGUOUS alias is ALWAYS rejected (its identity
        is unknowable without the table qualifier).
        """
        candidates = self._resolve_candidates(name, table=table)
        specs = {self._fields[item] for item in candidates if item in self._fields}
        if len(specs) == 1:
            return ResolvedField(spec=next(iter(specs)))
        if not specs:
            return UnknownField(name=name)
        return AmbiguousField(
            name=name,
            candidates=tuple(sorted(spec.qualified_name for spec in specs)),
        )

    def get(self, name: str, *, table: str | None = None, strict: bool = False) -> FieldSpec | None:
        resolved = self.resolve_field(name, table=table)
        if isinstance(resolved, ResolvedField):
            return resolved.spec
        if strict:
            if isinstance(resolved, UnknownField):
                qualifier = f" in table {table!r}" if table else ""
                raise KeyError(f"unknown field {name!r}{qualifier}")
            raise KeyError(
                f"ambiguous field {name!r}; candidates: "
                f"{', '.join(resolved.candidates)}"
            )
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


__all__ = [
    "AmbiguousField",
    "FieldRegistry",
    "ResolvedField",
    "UnknownField",
]
