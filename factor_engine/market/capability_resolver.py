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
    # Strict market_context (raises on unknown markets) — never a ternary that
    # silently defaults unknown markets to "us" (P1-1).
    from market.context import market_context

    return market_context(market)


# P0-029: production coverage floor.  A provider whose ``coverage_gate`` is below
# this floor is not certified for FULL-UNIVERSE production use: size-neutralized
# cross-sections over a partially-covered universe systematically select names
# with data.  Research keeps a warning; production fails closed to UNSUPPORTED.
COVERAGE_FLOOR = 0.8

# P0-035: market-mechanism operator families with NO explicit/derived contract
# must fail closed to UNSUPPORTED_MARKET_MECHANISM instead of the generic
# INPUT_DEPENDENT (which claims both markets and defers to field providers).
# Pure math/TS/CS operators never start with these prefixes and stay
# INPUT_DEPENDENT.
_MECHANISM_PREFIXES = (
    "holder_", "event_", "minute_", "intra_", "micro_", "session_",
    "fiscal_", "relation_",
)


class ProviderCoverageError(RuntimeError):
    """Production full-universe use of a below-floor-coverage provider.

    Raised (or surfaced as a ``COVERAGE_GATE`` failed node) when a concept whose
    provider covers less than ``COVERAGE_FLOOR`` of the target universe would be
    used in production full-universe mode.  Restricted-universe use is allowed
    only when the universe mask is applied and recorded in lineage (P1-24)."""


class ProductionCertificationUnavailable(RuntimeError):
    """Production certification evidence could not be loaded — fail closed.

    Raised instead of returning an *empty* allowlist: an empty allowlist makes
    the production gate a no-op (``if prod and name not in prod`` never fires),
    silently letting every operator pass.  A build failure must never degrade
    to "nothing is restricted".
    """


def _production_allowlist() -> frozenset[str]:
    try:
        from backend.fastpath_allowlists import production_allowlist

        return frozenset(production_allowlist())
    except Exception as exc:  # pragma: no cover - defensive
        raise ProductionCertificationUnavailable(
            "production allowlist/evidence failed to build; refusing to run an "
            "empty certification gate (fail-closed)"
        ) from exc


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
    raw_name = str(canonical).strip()
    # DSL aliases (e.g. ``industry_neutralize`` -> ``group_neutralize``) resolve
    # to their canonical before contract/production lookups.
    name = raw_name
    if OperatorRegistry.get(raw_name, "pandas_numpy") is None:
        try:
            resolved = OperatorRegistry.resolve_canonical(raw_name)
            if resolved:
                name = resolved
        except Exception:  # pragma: no cover - defensive
            pass
    if OperatorRegistry.get(name, "pandas_numpy") is None:
        return MarketSupport(
            canonical=raw_name, market=market, status=MarketStatus.UNKNOWN,
            notes="operator not registered",
        )

    contract = contract_for(name)

    # Production certification gate (market-independent).
    if production and ctx.provider_profile == "production":
        prod = _production_allowlist()
        if prod and name not in prod:
            return MarketSupport(
                canonical=raw_name, market=market,
                status=MarketStatus.RESEARCH_ONLY,
                reason_codes=("NOT_PRODUCTION_CERTIFIED",),
                notes="not on the production allowlist; research may opt in explicitly",
            )

    # P0-015: ``ashare_*`` prefix is a HARD A-share lock unless an explicit
    # contract opts the operator into US.  This is a safety net for new ashare_*
    # ops (e.g. ashare_suspension_episode_length) that are not enumerated in the
    # price-limit family set — we should never rely on remembering to add each one.
    if name.startswith("ashare_") and market != "ashare":
        if contract is None or "us" not in contract.intrinsic_markets:
            return MarketSupport(
                canonical=raw_name, market=market,
                status=MarketStatus.UNSUPPORTED_MARKET_MECHANISM,
                required_capabilities=contract.required_capabilities if contract else (),
                reason_codes=("MARKET_MECHANISM",),
                notes="ashare_* prefix binds to the A-share market mechanism",
            )

    # Market-mechanism gate: contract declares intrinsic markets.
    if contract is not None:
        intrinsic = contract.intrinsic_markets
        if intrinsic != ("ashare", "us") and market not in intrinsic:
            return MarketSupport(
                canonical=raw_name, market=market,
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
                canonical=raw_name, market=market,
                status=MarketStatus.PROVIDER_REQUIRED,
                required_capabilities=contract.required_capabilities,
                missing_capabilities=tuple(missing),
                reason_codes=("MISSING_CAPABILITY",),
                notes=contract.notes,
            )

    # Generic mathematical operator: both markets, input-dependent.  NOT
    # "CERTIFIED_NATIVE": many generic operators have hidden requirements
    # (minute session, group semantics, event clock, price basis...), so the
    # verdict is explicitly INPUT_DEPENDENT and the real gate is the expression
    # walk over the input field providers (P1-9).
    #
    # P0-035: a market-mechanism family (holder/event/minute/fiscal/relation)
    # with NO explicit/derived contract must fail closed to
    # UNSUPPORTED_MARKET_MECHANISM instead of INPUT_DEPENDENT — claiming "both
    # markets, input-dependent" for a mechanism op would silently grant US (or
    # any future market) capabilities it does not have.  Pure math/TS/CS ops
    # (add/ts_mean/rank/...) do not match the mechanism prefixes and stay
    # INPUT_DEPENDENT.
    if contract is None:
        if name.startswith(_MECHANISM_PREFIXES):
            return MarketSupport(
                canonical=raw_name, market=market,
                status=MarketStatus.UNSUPPORTED_MARKET_MECHANISM,
                reason_codes=("NO_MARKET_CONTRACT",),
                notes=(
                    "market-mechanism operator has no explicit market/capability "
                    "contract; fail-closed to UNSUPPORTED (unclassified) instead "
                    "of INPUT_DEPENDENT"
                ),
            )
        return MarketSupport(
            canonical=raw_name, market=market,
            status=MarketStatus.INPUT_DEPENDENT,
            depends_on_inputs=True,
            notes="generic operator; actual support determined by input field providers",
        )
    if not contract.required_capabilities:
        return MarketSupport(
            canonical=raw_name, market=market,
            status=MarketStatus.INPUT_DEPENDENT,
            depends_on_inputs=True,
            notes="typed contract without capability requirements; input-dependent",
        )

    return MarketSupport(
        canonical=raw_name, market=market,
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


def _decode_field_name(name: str) -> str:
    """Decode an encoded ``SourceRef`` back to its physical field spelling."""
    try:
        from api.source_ref import decode_source_ref

        source = decode_source_ref(name)
        if source is not None:
            return f"{source.table}.{source.field}"
    except (ImportError, ValueError, TypeError):
        pass
    return name


def _concept_for_physical(table: str, physical: str) -> str | None:
    """Map a decoded (table, physical) reference to a canonical concept.

    A-share and US may store the same concept under different physical names
    (A ``StockValuationDaily.MarketCap`` vs US ``StockValuationDaily.market_cap``).
    This scans provider bindings across both markets so an A-share-sourced DSL
    reference resolves to the concept, then the concept resolves in the target
    market.
    """
    from fields.providers import PROVIDER_REGISTRY

    qualified = f"{table}.{physical}"
    for binding in PROVIDER_REGISTRY.for_market("ashare"):
        if qualified in binding.physical_fields:
            return binding.concept_id
    for binding in PROVIDER_REGISTRY.for_market("us"):
        if qualified in binding.physical_fields:
            return binding.concept_id
    return None


def _check_column(name: Any, market: str, production: bool, failed: list[dict[str, Any]], warnings: list[str]) -> None:
    if not isinstance(name, str):
        return
    from fields.concepts import concept_alias_map
    from fields.market_registry import MULTI_MARKET_FIELD_REGISTRY
    from fields.providers import binding

    plain = _decode_field_name(name)
    aliases = concept_alias_map()
    concept = aliases.get(name) or aliases.get(plain)
    if concept is None and "." in plain:
        _table, _phys = plain.split(".", 1)
        concept = _concept_for_physical(_table, _phys)
    if concept is not None:
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
            return
        # P0-029 / P1-002 coverage gate: a provider that only covers part of the
        # target universe (e.g. US market cap at ~42% via TickerSharesSnapshot)
        # must NOT certify FULL-UNIVERSE production use.  Below the floor the
        # production path is UNSUPPORTED (a failed node -> the expression fails
        # compile-time), NOT a warning: size-neutralization over a partially-
        # covered universe systematically selects names with shares data.
        # Research keeps a warning.  Restricted-universe use is allowed only when
        # the coverage mask is applied and recorded in lineage (P0-034/P1-24).
        if b.coverage_gate is not None and b.coverage_gate < COVERAGE_FLOOR:
            detail = (
                f"concept {concept!r} provider {b.provider_id!r} covers "
                f"~{b.coverage_gate:.0%} of the target universe"
            )
            if production:
                failed.append(
                    {
                        "node": name,
                        "kind": "field",
                        "reason": "COVERAGE_GATE",
                        "missing_capabilities": [],
                        "detail": (
                            f"{detail}; production FULL-UNIVERSE use is not certified. "
                            "Restricted-universe use must apply the universe mask and "
                            "record coverage in lineage (P0-034/P1-24)."
                        ),
                    }
                )
                return
            warnings.append(
                f"COVERAGE_GATE: {detail} (research: restricted-universe use must "
                "carry the coverage mask in lineage)"
            )
        return

    # Not a canonical concept — check the per-market physical field registry.
    # Encoded SourceRefs are matched by their physical field within its table.
    table = None
    lookup = plain
    if "." in plain:
        table, _, lookup = plain.partition(".")
    spec = MULTI_MARKET_FIELD_REGISTRY.resolve_field(market, lookup, table=table)
    if spec is None:
        if production:
            failed.append(
                {
                    "node": name,
                    "kind": "field",
                    "reason": "UNKNOWN_FIELD",
                    "missing_capabilities": [],
                    "detail": f"field {plain!r} not resolvable for market {market}",
                }
            )
        else:
            warnings.append(f"field {plain!r} not resolved for market {market} (research: raw column allowed)")
        return
    if table is not None and spec.table != table:
        if production:
            failed.append(
                {
                    "node": name,
                    "kind": "field",
                    "reason": "TABLE_COLLISION",
                    "missing_capabilities": [],
                    "detail": f"{plain!r} resolves to {spec.qualified_name}, not {table!r}",
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
    from fields.providers import explain_field_support

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

    # Field concepts are filtered by the *production* eligibility gate, not by
    # "a binding exists and quality != UNAVAILABLE": PROXY_RESEARCH / SPARSE /
    # PIT_BLOCKED / SOURCE_UNCERTIFIED providers must not leak into the
    # production grammar.
    concepts: set[str] = set()
    for concept in list_concepts():
        support = explain_field_support(concept.concept_id, market, production=True)
        if support.status.is_supported:
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
    counts = {
        "total": 0, "unknown": 0, "not_reviewed": 0,
        "explicit_contract": 0, "fallback_default": 0,
    }
    from cleaned_operators.operator_market import OPERATOR_MARKET_CONTRACTS

    for name in names:
        counts["total"] += 1
        contract = contract_for(name)
        a = operator_support(name, "ashare", production=False).to_dict()
        u = operator_support(name, "us", production=False).to_dict()
        if a["status"] == MarketStatus.UNKNOWN.value:
            counts["unknown"] += 1
        # Honest review accounting (P1-8): ``contract_origin`` records whether
        # this operator's market status came from an explicit hand-written
        # contract, a typed/prefix-derived fallback, or nothing at all.  The
        # old ``not_reviewed = unknown`` counted every registered operator as
        # "reviewed" even when its A/US status was only the generic default.
        if OPERATOR_MARKET_CONTRACTS.get(name) is not None:
            origin = "explicit"
            counts["explicit_contract"] += 1
        elif contract is not None:
            origin = "typed_derived"
        else:
            origin = "fallback_default"
            counts["fallback_default"] += 1
        row: dict[str, Any] = {
            "canonical": name,
            "contract_origin": origin,
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
    # ``not_reviewed`` is the CI gate and counts ONLY truly unclassified ops.
    # ``fallback_default`` is informational: a generic math/TS/CS operator with no
    # explicit contract is *reviewed by default* — both-markets input-dependent is
    # its correct, intentional status (its per-op ``contract_origin`` row records
    # that), so it must not fail the "every canonical has an explicit A/US status"
    # gate.  ``contract_origin`` keeps the honest accounting (P1-8) while the gate
    # stays meaningful.
    counts["not_reviewed"] = counts["unknown"]
    return {
        "schema_version": "factor_engine.operator_market_capabilities.v2",
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
