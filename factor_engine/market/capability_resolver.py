"""Capability resolver — market support for operators, fields, and expressions.

Implements the multi-market plan P1/P2:

- ``operator_support(canonical, market)`` — per-operator per-market verdict;
- ``explain_expression_support(expr, market)`` — whole-expression propagation with
  compile-time rejection of unsupported nodes (spec §82, §113);
- ``build_search_grammar(market)`` — market-specific mining grammar (spec §83-§86);
- ``build_market_operator_manifest()`` — full 1293-canonical manifest (spec §118).

The verdict is machine-generated: generic math/TS/CS operators default to
"both, input-dependent"; market-mechanism operators use their declared contract
in ``cleaned_operators/operator_market.py``; field-dependent support propagates
from ``fields/providers``.
"""
from __future__ import annotations

from typing import Any, Iterable

from market.capabilities import (
    MarketCapability,
    MarketStatus,
    MarketSupport,
    ProviderQuality,
    capabilities_for,
)
from market.context import ASHARE_CONTEXT, MarketContext, US_CONTEXT


def _market_ctx(market: str) -> MarketContext:
    return ASHARE_CONTEXT if market == "ashare" else US_CONTEXT


def _production_allowlist() -> frozenset[str]:
    try:
        from backend.fastpath_allowlists import production_allowlist

        return frozenset(production_allowlist())
    except Exception:  # pragma: no cover - defensive
        return frozenset()


def operator_support(
    canonical: str,
    market: str,
    *,
    production: bool = True,
    context: MarketContext | None = None,
) -> MarketSupport:
    """Per-operator per-market support verdict.

    ``production=True`` fail-closes: operators not production-certified at all
    become ``RESEARCH_ONLY``; sparse/proxy providers are rejected.  ``research``
    context can open the proxy/sparse gates explicitly.
    """
    from cleaned_operators.operator_market import contract_for
    from cleaned_operators.registry import OperatorRegistry

    ctx = context or _market_ctx(market)
    name = str(canonical).strip()
    if OperatorRegistry.get(name, "pandas_numpy") is None:
        return MarketSupport(
            canonical=name, market=market, status=MarketStatus.UNKNOWN,
            notes="operator not registered",
        )

    contract = contract_for(name)

    # Production certification gate (market-independent).
    if production and ctx.provider_profile == "production":
        prod = _production_allowlist()
        if prod and name not in prod:
            return MarketSupport(
                canonical=name, market=market,
                status=MarketStatus.RESEARCH_ONLY,
                reason_codes=("NOT_PRODUCTION_CERTIFIED",),
                notes="not on the production allowlist; research may opt in explicitly",
            )

    # Market-mechanism gate: contract declares intrinsic markets.
    if contract is not None:
        intrinsic = contract.intrinsic_markets
        if intrinsic != ("ashare", "us") and market not in intrinsic:
            return MarketSupport(
                canonical=name, market=market,
                status=MarketStatus.UNSUPPORTED_MARKET_MECHANISM,
                required_capabilities=contract.required_capabilities,
                reason_codes=("MARKET_MECHANISM",),
                notes=contract.notes,
            )

    # Capability gate: required capabilities missing in this market.
    if contract is not None and contract.required_capabilities:
        market_caps = capabilities_for(market)
        missing = [
            cap for cap in contract.required_capabilities
            if MarketCapability(cap) not in market_caps
        ]
        if missing:
            return MarketSupport(
                canonical=name, market=market,
                status=MarketStatus.PROVIDER_REQUIRED,
                required_capabilities=contract.required_capabilities,
                missing_capabilities=tuple(missing),
                reason_codes=("MISSING_CAPABILITY",),
                notes=contract.notes,
            )

    # Generic mathematical operator: both markets, input-dependent.
    if contract is None or not contract.required_capabilities:
        return MarketSupport(
            canonical=name, market=market,
            status=MarketStatus.CERTIFIED_NATIVE,
            depends_on_inputs=True,
            notes="generic operator; actual support determined by input field providers",
        )

    return MarketSupport(
        canonical=name, market=market,
        status=MarketStatus.CERTIFIED_NATIVE,
        required_capabilities=contract.required_capabilities,
        notes=contract.notes,
    )


def explain_operator_support(
    canonical: str,
    market: str,
    *,
    production: bool = True,
) -> dict[str, Any]:
    """Human/AlphaMiner facing explanation of one operator's market support."""
    support = operator_support(canonical, market, production=production)
    result = support.to_dict()
    if support.status == MarketStatus.PROVIDER_REQUIRED:
        result["suggested_providers"] = _suggested_providers(support)
    return result


def _suggested_providers(support: MarketSupport) -> list[str]:
    suggestion_by_cap = {
        MarketCapability.INDUSTRY_CLASSIFICATION: "GICS / SIC / NAICS adapter",
        MarketCapability.INDEX_WEIGHTS: "index weight provider (e.g. CRSP / index publisher)",
        MarketCapability.TOP_HOLDERS: "institutional ownership / 13F provider",
        MarketCapability.NEWS: "news sentiment provider",
        MarketCapability.FX: "FX rate provider",
    }
    out: list[str] = []
    for cap in support.missing_capabilities:
        try:
            label = suggestion_by_cap[MarketCapability(cap)]
        except KeyError:
            label = f"provider for {cap}"
        out.append(label)
    return out


# ---------------------------------------------------------------------------
# Expression-level support (compile-time gate).
# ---------------------------------------------------------------------------
def _walk(expr: Any, market: str, production: bool, failed: list[dict[str, Any]], warnings: list[str]) -> None:
    try:
        from expr.cleaned_call import CleanedCall
        from expr.column import ColumnRef
        from expr.literal import Literal
    except ImportError:  # pragma: no cover - defensive
        return

    if isinstance(expr, Literal):
        return
    if isinstance(expr, CleanedCall):
        op = expr.op
        support = operator_support(op, market, production=production)
        if support.status == MarketStatus.UNKNOWN:
            warnings.append(f"operator {op!r} not registered")
        elif not support.status.is_supported and not (
            support.status == MarketStatus.RESEARCH_ONLY and not production
        ):
            failed.append(
                {
                    "node": op,
                    "kind": "operator",
                    "reason": support.status.value,
                    "missing_capabilities": list(support.missing_capabilities),
                    "detail": support.notes,
                }
            )
        for child in expr.children():
            _walk(child, market, production, failed, warnings)
        return
    if isinstance(expr, (ColumnRef,)):
        name = getattr(expr, "name", None)
        _check_column(name, market, production, failed, warnings)
        return

    # Fallback: generic Expr with children().
    for child in getattr(expr, "children", lambda: ())():
        _walk(child, market, production, failed, warnings)


def _check_column(name: Any, market: str, production: bool, failed: list[dict[str, Any]], warnings: list[str]) -> None:
    if not isinstance(name, str):
        return
    from fields.concepts import concept_alias_map
    from fields.providers import binding

    aliases = concept_alias_map()
    concept = aliases.get(name)
    if concept is None:
        # Not a canonical concept — legacy column.  Leave to the field registry.
        from fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

        spec = MULTI_MARKET_FIELD_REGISTRY.resolve_field(market, name)
        if spec is None:
            warnings.append(f"field {name!r} not resolved for market {market}")
        return
    b = binding(concept, market)
    if b is None or b.quality == ProviderQuality.UNAVAILABLE:
        failed.append(
            {
                "node": name,
                "kind": "field",
                "reason": "UNSUPPORTED_MARKET_MECHANISM"
                if b is not None
                else "NO_PROVIDER",
                "missing_capabilities": [],
                "detail": f"concept {concept!r} has no usable provider for {market}",
            }
        )
        return
    if production and not b.quality.production_usable:
        failed.append(
            {
                "node": name,
                "kind": "field",
                "reason": "PROVIDER_REQUIRED",
                "missing_capabilities": [],
                "detail": f"concept {concept!r} quality {b.quality.value} below production floor",
            }
        )


def explain_expression_support(
    expr: Any,
    market: str,
    *,
    production: bool = True,
) -> dict[str, Any]:
    """Propagate market support through a DSL expression AST (compile-time).

    Returns ``{"supported": bool, "failed_nodes": [...], "warnings": [...]}``.
    ``expr`` may be a parsed DSL AST or a formula string.
    """
    from api.dsl_parser import parse_expr
    from expr.base import Expr

    if isinstance(expr, str):
        try:
            expr = parse_expr(expr, surface="daily")
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "supported": False,
                "failed_nodes": [
                    {
                        "node": expr,
                        "kind": "parse",
                        "reason": "PARSE_ERROR",
                        "missing_capabilities": [],
                        "detail": str(exc),
                    }
                ],
                "warnings": [],
            }
    failed: list[dict[str, Any]] = []
    warnings: list[str] = []
    _walk(expr, market, production, failed, warnings)
    return {
        "supported": not failed,
        "failed_nodes": failed,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Market-specific search grammar (spec §83-§86).
# ---------------------------------------------------------------------------
def build_search_grammar(market: str) -> dict[str, Any]:
    """Return the legal operator + field-concept grammar for a market.

    Unsupported market-mechanism operators and unavailable field concepts are
    excluded up front, shrinking the mining search space.
    """
    from fields.concepts import list_concepts
    from fields.providers import PROVIDER_REGISTRY, ProviderQuality

    try:
        from api.mining_integration import list_dsl_allowlist

        operators = set(list_dsl_allowlist(surface="daily"))
    except Exception:  # pragma: no cover - defensive
        operators = set()

    allowed_ops: set[str] = set()
    for op in operators:
        support = operator_support(op, market, production=True)
        if support.status.is_supported:
            allowed_ops.add(op)

    concepts: set[str] = set()
    for concept in list_concepts():
        b = PROVIDER_REGISTRY.binding(concept.concept_id, market)
        if b is not None and b.quality != ProviderQuality.UNAVAILABLE:
            concepts.add(concept.concept_id)

    return {
        "market": market,
        "operator_count": len(allowed_ops),
        "operators": sorted(allowed_ops),
        "concept_count": len(concepts),
        "concepts": sorted(concepts),
        "notes": "generic math/TS/CS shared; market-mechanism operators excluded per market",
    }


# ---------------------------------------------------------------------------
# Full-canonical manifest (spec §118, §121).
# ---------------------------------------------------------------------------
def build_market_operator_manifest(
    canonicals: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Generate the per-operator per-market capability manifest for every canonical.

    Every canonical gets an explicit A/US row.  ``UNKNOWN`` and
    ``NOT_REVIEWED`` count is returned so the CI gate can assert it is zero.
    """
    from cleaned_operators.operator_market import contract_for
    from cleaned_operators.registry import OperatorRegistry

    names = (
        sorted(set(canonicals))
        if canonicals is not None
        else sorted(OperatorRegistry.list_canonical())
    )
    rows: dict[str, Any] = {}
    counts = {"total": 0, "unknown": 0, "not_reviewed": 0}
    for name in names:
        counts["total"] += 1
        contract = contract_for(name)
        a = operator_support(name, "ashare", production=False).to_dict()
        u = operator_support(name, "us", production=False).to_dict()
        if a["status"] == MarketStatus.UNKNOWN.value:
            counts["unknown"] += 1
        row: dict[str, Any] = {
            "canonical": name,
            "intrinsic_markets": (
                list(contract.intrinsic_markets) if contract else ["ashare", "us"]
            ),
            "required_capabilities": (
                list(contract.required_capabilities) if contract else []
            ),
            "price_basis": contract.price_basis if contract else "EITHER",
            "ashare": a,
            "us": u,
        }
        rows[name] = row
    counts["not_reviewed"] = counts["unknown"]
    return {
        "schema_version": "factor_engine.operator_market_capabilities.v1",
        "generated_by": "market/capability_resolver.build_market_operator_manifest",
        "counts": counts,
        "operators": rows,
    }


__all__ = [
    "build_market_operator_manifest",
    "build_search_grammar",
    "explain_expression_support",
    "explain_operator_support",
    "operator_support",
]
