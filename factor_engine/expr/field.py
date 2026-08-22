"""Semantically resolved factor input reference."""
from __future__ import annotations

from dataclasses import dataclass

from .column import ColumnRef


@dataclass(frozen=True)
class FieldRef(ColumnRef):
    """A catalog-bound field whose source identity survives expression lowering.

    ``name`` remains the runtime transport name for compatibility.  The other
    attributes are canonical catalog identity and are included in expression
    equality/hash, preventing same-named fields from different tables from
    collapsing into one dependency.
    """

    field_id: str = ""
    canonical_name: str = ""
    table: str = ""
    source_name: str = ""
    catalog_hash: str = ""


__all__ = ["FieldRef"]
