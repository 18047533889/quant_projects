"""MultiMarketFieldRegistry — one physical registry per market + canonical layer.

Per the multi-market plan (§75-§76): same-named tables (``StockValuationDaily``,
``StockIndicator``, ``StockCapitalDaily``...) carry DIFFERENT semantics in each
market, so they must never share one flat registry.  This class keeps an
independent :class:`FieldRegistry` per market and a shared canonical-concept
layer (``fields/providers``) on top.
"""
from __future__ import annotations

from typing import Any

from factor_engine.market.capabilities import MarketSupport
from factor_engine.market.context import ASHARE_CONTEXT, US_CONTEXT, MarketContext

from .catalog import ASHARE_FIELD_SPECS, ASHARE_TABLE_SPECS
from .catalog_us import US_FIELD_SPECS, US_TABLE_SPECS
from .providers import explain_field_support
from .registry import FieldRegistry
from .spec import FieldSpec, TableSpec

ASHARE_FIELD_REGISTRY = FieldRegistry(ASHARE_FIELD_SPECS, ASHARE_TABLE_SPECS)
US_FIELD_REGISTRY = FieldRegistry(US_FIELD_SPECS, US_TABLE_SPECS)


class MultiMarketFieldRegistry:
    """Resolve physical fields and canonical concepts within one market."""

    def __init__(
        self,
        ashare: FieldRegistry | None = None,
        us: FieldRegistry | None = None,
    ) -> None:
        self._registries = {
            "ashare": ashare if ashare is not None else ASHARE_FIELD_REGISTRY,
            "us": us if us is not None else US_FIELD_REGISTRY,
        }

    # -- per-market physical field resolution ------------------------------
    def registry_for(self, market: str) -> FieldRegistry:
        try:
            return self._registries[str(market).strip().lower()]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"no field registry for market {market!r}") from exc

    def resolve_field(
        self,
        market: str,
        name: str,
        *,
        table: str | None = None,
        strict: bool = False,
    ) -> FieldSpec | None:
        """Resolve a physical field name within one market's registry."""
        return self.registry_for(market).get(name, table=table, strict=strict)

    def require_field(self, market: str, name: str, *, table: str | None = None) -> FieldSpec:
        result = self.resolve_field(market, name, table=table, strict=True)
        assert result is not None
        return result

    def field_specs(self, market: str) -> tuple[FieldSpec, ...]:
        return self.registry_for(market).fields()

    def table_specs(self, market: str) -> tuple[TableSpec, ...]:
        return self.registry_for(market).tables()

    # -- canonical-concept layer -------------------------------------------
    def concept_binding(self, market: str, concept_id: str):
        """Return the market's provider binding for a canonical concept."""
        from .providers import require_binding

        return require_binding(concept_id, market)

    def explain_field(self, market: str, concept_id: str, **kwargs: Any) -> MarketSupport:
        """Support verdict for a canonical concept in a market."""
        return explain_field_support(concept_id, market, **kwargs)

    # -- introspection -----------------------------------------------------
    @property
    def markets(self) -> tuple[str, ...]:
        return ("ashare", "us")

    def field_counts(self) -> dict[str, int]:
        return {m: len(self.registry_for(m).fields()) for m in self.markets}

    def table_counts(self) -> dict[str, int]:
        return {m: len(self.registry_for(m).tables()) for m in self.markets}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "factor_engine.fields.multi_market.v1",
            "markets": {
                market: {
                    "field_count": len(reg.fields()),
                    "table_count": len(reg.tables()),
                }
                for market, reg in self._registries.items()
            },
        }


MULTI_MARKET_FIELD_REGISTRY = MultiMarketFieldRegistry()


def multi_market_registry() -> MultiMarketFieldRegistry:
    return MULTI_MARKET_FIELD_REGISTRY


__all__ = [
    "ASHARE_FIELD_REGISTRY",
    "MULTI_MARKET_FIELD_REGISTRY",
    "MultiMarketFieldRegistry",
    "US_FIELD_REGISTRY",
    "multi_market_registry",
]
