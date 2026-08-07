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

# ---------------------------------------------------------------------------
# Multi-market layer (additive; the legacy single-market FIELD_REGISTRY above
# is preserved for the existing DSL/IR).  See FACTOR_ENGINE_MULTI_MARKET_PLAN.md
# P0-A..P0-C.
# ---------------------------------------------------------------------------
from .catalog_us import US_FIELD_SPECS, US_TABLE_SPECS
from .concepts import (
    FieldConceptSpec,
    concept_alias_map,
    get_concept,
    list_concepts,
    require_concept,
)
from .market_registry import (
    ASHARE_FIELD_REGISTRY,
    MULTI_MARKET_FIELD_REGISTRY,
    MultiMarketFieldRegistry,
    US_FIELD_REGISTRY,
    multi_market_registry,
)
from .providers import (
    MarketFieldBinding,
    PROVIDER_REGISTRY,
    ProviderRegistry,
    apply_binding_transform,
    binding,
    explain_field_support,
    require_binding,
)
from .units_v2 import *  # noqa: F401,F403  currency-aware unit vocabulary

__all__ = [
    "ASHARE_FIELD_REGISTRY",
    "ASHARE_FIELD_SPECS", "ASHARE_TABLE_SPECS", "DEFAULT_FIELD_REGISTRY",
    "FIELD_REGISTRY", "FieldConceptSpec", "FieldRegistry", "FieldSpec",
    "TableSpec",
    "MULTI_MARKET_FIELD_REGISTRY",
    "MarketFieldBinding", "MultiMarketFieldRegistry",
    "PROVIDER_REGISTRY", "ProviderRegistry",
    "US_FIELD_REGISTRY", "US_FIELD_SPECS", "US_TABLE_SPECS",
    "apply_binding_transform", "binding",
    "compute_field_catalog_hash", "concept_alias_map", "explain_field_support",
    "field_catalog", "get_concept", "get_field_registry", "list_concepts",
    "multi_market_registry", "require_binding", "require_concept",
    "require_field", "resolve_field",
]
