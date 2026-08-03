"""Field semantics foundation for the factor DSL and IR."""
from __future__ import annotations

from .catalog import ASHARE_FIELD_SPECS, ASHARE_TABLE_SPECS
from .registry import FieldRegistry
from .spec import FieldSpec, TableSpec
from .units import *  # re-export the small public unit vocabulary

FIELD_REGISTRY = FieldRegistry(ASHARE_FIELD_SPECS, ASHARE_TABLE_SPECS)
DEFAULT_FIELD_REGISTRY = FIELD_REGISTRY


def get_field_registry() -> FieldRegistry:
    return FIELD_REGISTRY


def field_catalog() -> dict[str, object]:
    return FIELD_REGISTRY.export_catalog()


def compute_field_catalog_hash() -> str:
    return FIELD_REGISTRY.catalog_hash()


from .resolver import require_field, resolve_field

__all__ = [
    "ASHARE_FIELD_SPECS", "ASHARE_TABLE_SPECS", "DEFAULT_FIELD_REGISTRY",
    "FIELD_REGISTRY", "FieldRegistry", "FieldSpec", "TableSpec",
    "compute_field_catalog_hash", "field_catalog", "get_field_registry",
    "require_field", "resolve_field",
]
