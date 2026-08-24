"""Resolution helpers for ColumnRef and encoded SourceRef leaves.

R17-001: the market-aware entry points (:func:`resolve_market_field`,
:func:`require_market_field`) are the SINGLE production field-resolution API.
They require an explicit ``MarketContext`` and never fall back to the A-share
legacy registry.  The legacy :func:`resolve_field` / :func:`require_field`
remain as A-share-bound compatibility wrappers (they bind the legacy
``FIELD_REGISTRY`` explicitly) and are flagged by the R17-001 static audit so
no *production* path reaches them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .registry import FieldRegistry
from .spec import FieldSpec, TableSpec


def _legacy_active(registry: FieldRegistry | None) -> FieldRegistry:
    """The legacy A-share-bound registry used by ``resolve_field`` only.

    R17-001: the legacy wrapper must be *explicitly* A-share, never an
    implicit market.  ``registry`` is honored for callers that construct a
    per-market registry themselves.
    """
    if registry is not None:
        return registry
    from . import FIELD_REGISTRY

    return FIELD_REGISTRY


def _resolve_in(active: FieldRegistry, value: Any, *, table: str | None, strict: bool) -> FieldSpec | None:
    """Shared decode+lookup against one explicit registry."""
    if table is None:
        table = getattr(value, "table", None)
    name = getattr(value, "name", value)
    if not isinstance(name, str):
        if strict:
            raise TypeError(
                f"field reference must be a string or ColumnRef, got {type(value).__name__}"
            )
        return None
    try:
        from factor_engine.api.source_ref import decode_source_ref

        source = decode_source_ref(name)
    except (ImportError, ValueError, TypeError):
        source = None
    if source is not None:
        return active.get(source.field, table=source.table, strict=strict)
    return active.get(name, table=table, strict=strict)


def resolve_field(
    value: Any,
    *,
    table: str | None = None,
    registry: FieldRegistry | None = None,
    strict: bool = False,
) -> FieldSpec | None:
    """Legacy A-share-bound resolution (compat only).

    R17-001: this wrapper MUST NOT be used by production code — it binds the
    legacy single-market A-share ``FIELD_REGISTRY`` and would silently resolve
    US formulas against A-share aliases/units.  Production callers use
    :func:`resolve_market_field` with an explicit ``MarketContext``.
    """
    return _resolve_in(_legacy_active(registry), value, table=table, strict=strict)


def require_field(value: Any, *, table: str | None = None, registry: FieldRegistry | None = None) -> FieldSpec:
    result = resolve_field(value, table=table, registry=registry, strict=True)
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# R17-001: the single production field-resolution API.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedMarketField:
    """A market-scoped resolution result carrying the market it resolved in.

    ``spec`` is the resolved :class:`~fields.spec.FieldSpec`; ``market`` is the
    canonical market id it was resolved under; ``table_spec`` (when the field's
    table is known and resolvable) is the owning :class:`~fields.spec.TableSpec`
    in the SAME market — the two never mix A/US same-named tables.
    """

    spec: FieldSpec
    market: str
    table_spec: TableSpec | None = None

    @property
    def table(self) -> str:
        return self.spec.table


def _canonical_market(market_context: Any) -> str:
    """Canonical market id from a ``MarketContext`` or a plain market string."""
    from factor_engine.market.context import market_context as resolve_ctx

    if market_context is None:
        raise TypeError(
            "resolve_market_field requires an explicit MarketContext; refusing "
            "to guess a market (R17-001)"
        )
    if hasattr(market_context, "market"):
        return resolve_ctx(market_context.market).market
    return resolve_ctx(market_context).market


def resolve_market_field(
    value: Any,
    market_context: Any,
    *,
    table: str | None = None,
    source_context: Any = None,
    strict: bool = False,
) -> ResolvedMarketField | None:
    """Resolve a logical/physical field within ONE explicit market (R17-001).

    ``market_context`` is a :class:`~market.context.MarketContext` (or a plain
    ``ashare``/``us`` string, canonicalized).  Resolution routes through
    ``MULTI_MARKET_FIELD_REGISTRY.registry_for(market)`` — never the legacy
    single-market registry — so a US formula can never be hit by A-share field
    aliases/units/price basis.

    ``source_context`` is accepted for callers that want to thread provider /
    dataset provenance; it does not change field resolution (the market does).
    """
    market = _canonical_market(market_context)
    from .market_registry import MULTI_MARKET_FIELD_REGISTRY

    reg = MULTI_MARKET_FIELD_REGISTRY.registry_for(market)
    spec = _resolve_in(reg, value, table=table, strict=strict)
    if spec is None:
        if strict:
            raise KeyError(
                f"no field {value!r} in market registry {market!r}"
            )
        return None
    table_spec = None
    if table is None:
        table = getattr(spec, "table", None)
    if table:
        table_spec = reg.resolve_table(str(table))
    return ResolvedMarketField(spec=spec, market=market, table_spec=table_spec)


def require_market_field(
    value: Any,
    market_context: Any,
    *,
    table: str | None = None,
    source_context: Any = None,
) -> ResolvedMarketField:
    result = resolve_market_field(
        value, market_context, table=table, source_context=source_context, strict=True
    )
    assert result is not None
    return result


__all__ = [
    "ResolvedMarketField",
    "require_field",
    "require_market_field",
    "resolve_field",
    "resolve_market_field",
]

