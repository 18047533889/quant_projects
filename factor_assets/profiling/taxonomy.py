"""Deterministic factor taxonomy classifier (R61-FI-013 / plan §7 C2–C4).

This module is a **pure rule engine**.  Given a DSL factor definition it
produces a frozen :class:`FactorTaxonomyArtifact` classifying:

- ``data_domains`` — a **set** of data domains the factor consumes (mixed
  domains are a multiple-element set, never a mutually-exclusive enum);
- ``display_family`` — a derived label (``PV_ONLY`` / ``PV_FUND`` /
  ``PV_LIQ_FUND`` ...) derived from the domain set, never an authority;
- ``mechanism_tags`` — deterministic AST-style rules over the FE static-analysis
  operator usages (canonical operator ids) with provenance + policy-versioned
  confidence;
- ``structure_tags`` — derived from operator usage classes
  (``TIME_SERIES`` / ``CROSS_SECTIONAL`` / ``RANKED`` ...);
- ``frequency_tags`` — from the underlying data fields' frequency class.

Determinism contract
--------------------
The engine is fully deterministic: rule order is fixed, no LLM, no randomness,
no environment dependence.  The same DSL input always produces the same
artifact (including ``content_hash``).  On the classification side the engine
consumes *exclusively* the FE static-analysis artifact's ``operator_usages``
and ``field_usages`` (plus an optional consumer-provided ``field_taxonomy``
domain mapping); the FE layer — not this module — parses the DSL.

Unknown inputs are explicit, never dropped: unknown fields map to the
``UNKNOWN`` domain (added exactly once) and unknown mechanism patterns produce
``UNKNOWN_MECHANISM`` carrying the operator ids that matched no rule.
Malformed inputs never raise and never silently discard information.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from factor_assets.profiling.policies import (
    TagEvidence,
    TagSource,
    TaxonomyPolicy,
    get_taxonomy_policy,
)

__all__ = [
    "DomainTagSet",
    "FactorTaxonomyArtifact",
    "classify_factor_taxonomy",
    "derive_data_domains",
    "derive_structure_tags",
    "derive_mechanism_tags",
    "derive_frequency_tags",
    "derive_display_family",
]

_TAXONOMY_NS = "factor_taxonomy_v1"

#: Canonical display order for data-domain tokens (stable artifact hashes).
_DOMAIN_ORDER = (
    "PRICE", "VOLUME", "LIQUIDITY", "VALUATION", "FUNDAMENTAL",
    "FUNDAMENTAL.VALUE", "FUNDAMENTAL.QUALITY", "FUNDAMENTAL.GROWTH",
    "FUNDAMENTAL.INVESTMENT", "FUNDAMENTAL.CASHFLOW",
    "FUNDAMENTAL.LEVERAGE", "SIZE", "EVENT", "FLOW_SENTIMENT",
    "MICROSTRUCTURE", "RISK", "CALENDAR", "ALTERNATIVE", "UNKNOWN",
)


# ---------------------------------------------------------------------------
# Public contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainTagSet:
    """A set of data-domain tags with set semantics.

    Mixed-domain factors hold multiple members (``{'PRICE', 'VOLUME'}``) —
    deliberately **not** a mutually-exclusive enum.  ``UNKNOWN`` is kept
    explicit (added exactly once) when any field cannot be mapped to a known
    domain.  Tags are stored in canonical display order so artifact hashes are
    stable.
    """

    tags: tuple[str, ...]

    def __init__(self, tags: Sequence[str]) -> None:
        object.__setattr__(self, "tags", _canonical_domain_order(tags))

    @property
    def members(self) -> tuple[str, ...]:
        """The domain tags, canonical order, deduplicated."""
        return self.tags

    @property
    def has_unknown(self) -> bool:
        return "UNKNOWN" in self.tags

    def __contains__(self, tag: str) -> bool:
        return tag in self.tags

    def __len__(self) -> int:
        return len(self.tags)

    def __iter__(self):
        return iter(self.tags)


@dataclass(frozen=True)
class FactorTaxonomyArtifact:
    """Frozen taxonomy classification of one factor definition.

    Attributes
    ----------
    factor_definition_id:
        The FE identity's factor id (``F_...``).
    data_domains:
        Tuple view of a :class:`DomainTagSet` — set semantics, canonical order.
    display_family:
        **Derived** display label (``PV_ONLY`` / ``PV_FUND`` ...); deliberately
        redundant with ``data_domains`` and never an authority.
    mechanism_tags:
        :class:`~factor_assets.profiling.policies.TagEvidence` records with
        source/confidence/evidence refs.
    structure_tags:
        Structure tokens derived from operator usage.
    frequency_tags:
        Frequency tokens derived from the underlying data fields.
    field_usage_ref:
        Stable content-based ref over the consumed field ids.
    operator_usage_ref:
        Stable content-based ref over the consumed (non-mechanical) operator
        ids.
    taxonomy_policy_id / taxonomy_policy_version:
        The policy this classification ran under.
    content_hash:
        Deterministic hash of everything else in this artifact (same DSL input
        ⇒ same hash).
    """

    factor_definition_id: str
    data_domains: tuple[str, ...]
    display_family: str
    mechanism_tags: tuple[TagEvidence, ...]
    structure_tags: tuple[str, ...]
    frequency_tags: tuple[str, ...]
    field_usage_ref: str
    operator_usage_ref: str
    taxonomy_policy_id: str
    taxonomy_policy_version: str
    content_hash: str

    def __post_init__(self) -> None:
        if not self.factor_definition_id:
            raise ValueError("factor_definition_id is required")
        if len(set(self.data_domains)) != len(self.data_domains):
            raise ValueError("data_domains must be deduplicated")

    @property
    def domain_set(self) -> DomainTagSet:
        """The domain set view with set semantics."""
        return DomainTagSet(self.data_domains)

    @property
    def has_unknown_domain(self) -> bool:
        return self.domain_set.has_unknown

    @property
    def mechanism_tag_names(self) -> tuple[str, ...]:
        """Tag names only (no evidence) for quick checks."""
        return tuple(t.tag for t in self.mechanism_tags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_definition_id": self.factor_definition_id,
            "data_domains": list(self.data_domains),
            "display_family": self.display_family,
            "mechanism_tags": [
                {
                    "tag": t.tag,
                    "source": t.source,
                    "confidence": t.confidence,
                    "evidence_refs": list(t.evidence_refs),
                }
                for t in self.mechanism_tags
            ],
            "structure_tags": list(self.structure_tags),
            "frequency_tags": list(self.frequency_tags),
            "field_usage_ref": self.field_usage_ref,
            "operator_usage_ref": self.operator_usage_ref,
            "taxonomy_policy_id": self.taxonomy_policy_id,
            "taxonomy_policy_version": self.taxonomy_policy_version,
            "content_hash": self.content_hash,
        }


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def classify_factor_taxonomy(
    analysis,
    *,
    policy: TaxonomyPolicy | None = None,
    field_taxonomy: Mapping[str, Sequence[str]] | None = None,
    lineage: Sequence[str] | None = None,
) -> FactorTaxonomyArtifact:
    """Classify a factor definition into a frozen taxonomy artifact.

    Parameters
    ----------
    analysis:
        A value exposing ``operator_usages`` / ``field_usages`` of the FE
        static-analysis API (``factor_engine.api.static_analysis.
        analyze_factor_definition(...)``).  The classifier reads only these
        public fields and never re-parses the DSL.
    policy:
        Optional :class:`TaxonomyPolicy`; defaults to the current policy.
    field_taxonomy:
        Optional ``field_id_or_name -> (domain tokens)`` mapping.  When
        omitted, the classifier uses its built-in deterministic bare-name map
        (an explicit, stable vocabulary).  Consumers that bind the real DA
        semantic catalog (R61-FI-010 ``FieldTaxonomyProvider``) pass the typed
        result here so the classifier never guesses from names.
    lineage:
        Optional sequence of treatment-lineage semantic ids (from the FE static
        analysis ``existing_treatment_semantic_ids``) used for the
        ``SMOOTHED`` / ``NEUTRALIZED*`` structure tags.

    Returns
    -------
    FactorTaxonomyArtifact

    Raises
    ------
    TypeError
        When ``analysis`` does not expose operator usage / field usage
        iterables (a programmer error).  This is the only raise path; unknown
        domains / tags / malformed operator names never raise.
    """
    if policy is None:
        policy = get_taxonomy_policy()

    op_usages = _resolve_operators(analysis)
    field_usages = _resolve_fields(analysis)
    if lineage is None:
        lineage = tuple(getattr(analysis, "existing_treatment_semantic_ids", ()) or ())
    lineage = tuple(str(s) for s in lineage)

    op_ids = tuple(str(getattr(u, "operator_id", "")) for u in op_usages)
    operator_ref_ids = tuple(o for o in op_ids if not _is_mechanical_noise(o))
    field_ids = tuple(str(getattr(u, "canonical_field_id", "")) for u in field_usages)

    data_domains = derive_data_domains(
        field_usages, policy=policy, field_taxonomy=field_taxonomy
    )
    structure_tags = derive_structure_tags(
        op_usages, policy=policy, lineage=lineage, data_domains=data_domains
    )
    mechanism_tags = derive_mechanism_tags(op_ids, data_domains)
    frequency_tags = derive_frequency_tags(field_usages)
    display_family = derive_display_family(data_domains, policy=policy)

    artifact = FactorTaxonomyArtifact(
        factor_definition_id=_factor_definition_id_of(analysis),
        data_domains=data_domains,
        display_family=display_family,
        mechanism_tags=mechanism_tags,
        structure_tags=structure_tags,
        frequency_tags=frequency_tags,
        field_usage_ref=_ref_hash(field_ids),
        operator_usage_ref=_ref_hash(operator_ref_ids),
        taxonomy_policy_id=policy.policy_id,
        taxonomy_policy_version=policy.policy_version,
        content_hash="",
    )
    content_hash = _artifact_hash(artifact)
    return FactorTaxonomyArtifact(
        factor_definition_id=artifact.factor_definition_id,
        data_domains=artifact.data_domains,
        display_family=artifact.display_family,
        mechanism_tags=artifact.mechanism_tags,
        structure_tags=artifact.structure_tags,
        frequency_tags=artifact.frequency_tags,
        field_usage_ref=artifact.field_usage_ref,
        operator_usage_ref=artifact.operator_usage_ref,
        taxonomy_policy_id=artifact.taxonomy_policy_id,
        taxonomy_policy_version=artifact.taxonomy_policy_version,
        content_hash=content_hash,
    )


# ---------------------------------------------------------------------------
# Data domains
# ---------------------------------------------------------------------------


def derive_data_domains(
    field_usages,
    *,
    policy: TaxonomyPolicy | None = None,
    field_taxonomy: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    """Domain tokens of a field-usage sequence (tuple view of a DomainTagSet).

    Resolution precedence (highest first, deterministic):
    1. ``field_taxonomy`` mapping (consumer-provided, e.g. from the DA semantic
       catalog) keyed by canonical field id or bare name;
    2. the built-in bare-name vocabulary;
    3. the FE ``FieldUsage.domain`` coarse attribute (whole-token only);
    4. explicit ``UNKNOWN`` (never silently dropped — any unmapped field adds
       the unknown marker to the set alongside known domains).

    An empty field-usage sequence yields ``("UNKNOWN",)``.
    """
    del policy  # reserved for venue-policy domain rewrites (determinism kept)
    domains: list[str] = []
    has_unmapped = False
    for u in field_usages:
        cid = str(getattr(u, "canonical_field_id", "") or "")
        bare = _bare_field_name(cid)
        tokens = _field_domains(u, cid, bare, field_taxonomy)
        if tokens:
            for token in tokens:
                if token and token not in domains:
                    domains.append(token)
        elif cid:
            has_unmapped = True
    if not domains:
        domains.append("UNKNOWN")
    elif has_unmapped and "UNKNOWN" not in domains:
        domains.append("UNKNOWN")
    return _canonical_domain_order(domains)


def _field_domains(
    u, cid: str, bare: str, field_taxonomy: Mapping[str, Sequence[str]] | None
) -> tuple[str, ...]:
    if field_taxonomy is not None:
        for key in (cid, bare):
            val = field_taxonomy.get(key)
            if val is not None:
                return tuple(val)
        return ()
    # built-in deterministic bare-name vocabulary
    hit = _DEFAULT_FIELD_DOMAIN_MAP.get(cid) or _DEFAULT_FIELD_DOMAIN_MAP.get(bare)
    if hit is not None:
        return tuple(hit)
    # coarse attribute fallback (whole token only — no substring inference)
    attr = str(getattr(u, "domain", "") or "").lower()
    mapped = _COARSE_FIELD_DOMAIN.get(attr)
    if mapped is not None:
        return tuple(mapped)
    return ()


def _bare_field_name(cid: str) -> str:
    """Strip any table prefix off a canonical field id.

    ``StockDailyBarAdj.close`` -> ``close``, ``StockIndicator.roe`` -> ``roe``.
    """
    return cid.rsplit(".", 1)[1] if "." in cid else cid


#: Default bare-name -> data-domain vocabulary.  This is the classifier's
#: hermetic built-in table (explicit, deterministic, no substring guessing).
#: Consumers that have the DA semantic catalog (R61-FI-010) should pass the
#: typed ``field_taxonomy`` mapping so the classifier stays fully
#: data-augmented.
_DEFAULT_FIELD_DOMAIN_MAP: Mapping[str, tuple[str, ...]] = {
    # price / return
    "open": ("PRICE",),
    "high": ("PRICE",),
    "low": ("PRICE",),
    "close": ("PRICE",),
    "vwap": ("PRICE",),
    "prev_close": ("PRICE",),
    "pre_close": ("PRICE",),
    "return_bp": ("PRICE",),
    "ret": ("PRICE",),
    "ret_intra": ("PRICE",),
    "ret_overnight": ("PRICE",),
    "open_close_return": ("PRICE",),
    "overnight_return": ("PRICE",),
    # liquidity / volume
    "volume": ("VOLUME",),
    "turnover_ratio": ("LIQUIDITY",),
    "turnover": ("LIQUIDITY",),
    "amount": ("VOLUME", "LIQUIDITY"),
    "money": ("VOLUME", "LIQUIDITY"),
    "dollar_volume": ("VOLUME", "LIQUIDITY"),
    # size
    "market_cap": ("SIZE",),
    "mktcap": ("SIZE",),
    "circulating_market_cap": ("SIZE",),
    "free_float_mktcap": ("SIZE",),
    "shares_outstanding": ("SIZE",),
    # fundamental — value
    "pe_ratio": ("FUNDAMENTAL.VALUE",),
    "pe": ("FUNDAMENTAL.VALUE",),
    "pe_ttm": ("FUNDAMENTAL.VALUE",),
    "price_to_earnings": ("FUNDAMENTAL.VALUE",),
    "pb_ratio": ("FUNDAMENTAL.VALUE",),
    "pb": ("FUNDAMENTAL.VALUE",),
    "price_to_book": ("FUNDAMENTAL.VALUE",),
    "book_to_price": ("FUNDAMENTAL.VALUE",),
    "dividend_yield": ("FUNDAMENTAL.VALUE",),
    "earnings_yield": ("FUNDAMENTAL.VALUE",),
    "eps": ("FUNDAMENTAL.VALUE", "FUNDAMENTAL.QUALITY"),
    # fundamental — quality / profitability
    "roe": ("FUNDAMENTAL.QUALITY",),
    "roa": ("FUNDAMENTAL.QUALITY",),
    "return_on_equity": ("FUNDAMENTAL.QUALITY",),
    "return_on_assets": ("FUNDAMENTAL.QUALITY",),
    "net_profit": ("FUNDAMENTAL.PROFITABILITY",),
    "net_profit_attributable": ("FUNDAMENTAL.PROFITABILITY",),
    "net_income": ("FUNDAMENTAL.PROFITABILITY",),
    "profit": ("FUNDAMENTAL.PROFITABILITY",),
    "income": ("FUNDAMENTAL.PROFITABILITY",),
    # fundamental — growth
    "operating_revenue": ("FUNDAMENTAL.GROWTH", "FUNDAMENTAL.VALUE"),
    "revenue": ("FUNDAMENTAL.GROWTH", "FUNDAMENTAL.VALUE"),
    "sales_growth": ("FUNDAMENTAL.GROWTH",),
    "revenue_growth": ("FUNDAMENTAL.GROWTH",),
    "net_income_growth": ("FUNDAMENTAL.GROWTH",),
    "operating_profit_growth": ("FUNDAMENTAL.GROWTH",),
    "gross_profit_growth": ("FUNDAMENTAL.GROWTH",),
    # fundamental — leverage
    "total_assets": ("SIZE", "FUNDAMENTAL.LEVERAGE"),
    "total_liabilities": ("FUNDAMENTAL.LEVERAGE",),
    "total_liability": ("FUNDAMENTAL.LEVERAGE",),
    "debt": ("FUNDAMENTAL.LEVERAGE",),
    "debt_to_equity": ("FUNDAMENTAL.LEVERAGE",),
    "leverage": ("FUNDAMENTAL.LEVERAGE",),
    # fundamental — investment
    "capex": ("FUNDAMENTAL.INVESTMENT",),
    "capital_expenditure": ("FUNDAMENTAL.INVESTMENT",),
    "asset_growth": ("FUNDAMENTAL.INVESTMENT",),
    "total_asset_growth": ("FUNDAMENTAL.INVESTMENT",),
    # fundamental — cashflow
    "free_cash_flow": ("FUNDAMENTAL.CASHFLOW",),
    "fcf": ("FUNDAMENTAL.CASHFLOW",),
    "cash_flow": ("FUNDAMENTAL.CASHFLOW",),
    "operating_cash_flow": ("FUNDAMENTAL.CASHFLOW",),
    "accrual": ("FUNDAMENTAL.CASHFLOW",),
    # event / structure
    "index_weight": ("EVENT",),
    "event": ("EVENT",),
    "limit_up": ("EVENT",),
    "limit_down": ("EVENT",),
    "stock_industry": ("EVENT",),
}


#: Coarse ``FieldUsage.domain`` attribute -> canonical domain tokens (whole
#: token only; ``price_volume`` is a coarse FE attribute, not the DA catalog).
_COARSE_FIELD_DOMAIN: Mapping[str, tuple[str, ...]] = {
    "price_volume": ("PRICE", "VOLUME"),
    "fundamental": ("FUNDAMENTAL",),
    "valuation": ("FUNDAMENTAL.VALUE",),
}


# ---------------------------------------------------------------------------
# Structure tags
# ---------------------------------------------------------------------------

_STRUCTURE_ORDER = (
    "TIME_SERIES", "CROSS_SECTIONAL", "GROUPED", "RANKED", "ZSCORED",
    "WINSORIZED", "SMOOTHED", "NEUTRALIZED", "NEUTRALIZED_SIZE",
    "NEUTRALIZED_INDUSTRY", "SPARSE", "EVENT_DRIVEN", "BINARY", "DISCRETE",
    "MIXED_DOMAIN", "HIGH_TURNOVER", "LOW_FREQUENCY",
)


def derive_structure_tags(
    op_usages,
    *,
    policy: TaxonomyPolicy | None = None,
    lineage: Sequence[str] | None = None,
    data_domains: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Structure tags from operator usage + treatment lineage + domain set.

    Deterministic rule order: ``TIME_SERIES`` / ``CROSS_SECTIONAL`` from
    ``axis_effect``; ``GROUPED`` / ``RANKED`` / ``ZSCORED`` / ``WINSORIZED`` /
    ``SMOOTHED`` / ``NEUTRALIZED*`` from operator names and lineage semantic
    ids; ``SPARSE`` from missingness ops; ``EVENT_DRIVEN`` from event domain /
    ops; ``BINARY`` / ``DISCRETE`` from value-class ops; ``MIXED_DOMAIN`` from
    a multi-domain field set.
    """
    if policy is None:
        policy = get_taxonomy_policy()
    lineage = tuple(str(s) for s in (lineage or ()))
    op_names = tuple(str(getattr(u, "operator_id", "")) for u in op_usages)
    axes = tuple(str(getattr(u, "axis_effect", "") or "") for u in op_usages)
    name_set = set(op_names)
    lineage_set = set(lineage)
    domains = frozenset(data_domains or ())

    tags: list[str] = []
    if "time_series" in axes:
        tags.append("TIME_SERIES")
    if "cross_section" in axes:
        tags.append("CROSS_SECTIONAL")
    if any(n.startswith("group_") for n in op_names):
        tags.append("GROUPED")
    if name_set & {"rank", "cs_rank", "group_rank", "cs_pct_rank", "rank_pct",
                   "expanding_rank", "panel_rank"}:
        tags.append("RANKED")
    if name_set & {"zscore", "cs_zscore", "cs_standardize", "standardize",
                   "group_zscore", "expanding_zscore", "cs_mad_zscore"}:
        tags.append("ZSCORED")
    if name_set & {"winsorize", "clip", "bound", "cap", "clamp", "saturate",
                   "group_winsorize"}:
        tags.append("WINSORIZED")
    if name_set & {"ts_ema", "ewm", "ewm_mean", "ts_median", "running_std"} \
            or any(s.startswith("SMOOTH:") for s in lineage):
        tags.append("SMOOTHED")
    if name_set & {"neutralize", "cs_neutralize", "cs_resid", "cs_regression",
                   "group_neutralize", "industry_neutralize", "ind_neutralize",
                   "size_neutralize", "market_cap_neutralize", "cap_neutralize",
                   "industry_size_neutralize", "dual_neutral", "panel_neutralize"} \
            or any(s.startswith("NEUTRAL:") or s.startswith("INDUSTRY_NEUTRAL:")
                   or s.startswith("SIZE_NEUTRAL:") or s.startswith("DUAL_NEUTRAL:")
                   for s in lineage):
        tags.append("NEUTRALIZED")
    if name_set & {"size_neutralize", "market_cap_neutralize", "cap_neutralize"} \
            or any(s.startswith("SIZE_NEUTRAL:") for s in lineage):
        tags.append("NEUTRALIZED_SIZE")
    if name_set & {"group_neutralize", "industry_neutralize", "ind_neutralize",
                   "industry_size_neutralize", "dual_neutral"} \
            or any(s.startswith("INDUSTRY_NEUTRAL:") or s.startswith("DUAL_NEUTRAL:")
                   for s in lineage):
        tags.append("NEUTRALIZED_INDUSTRY")
    if name_set & {"fillna", "ffill", "nan_to_num", "dropna", "is_null",
                   "last_not_null", "first_not_null", "ifnan", "coalesce",
                   "nonfinite_to_num", "log_fill_invalid"}:
        tags.append("SPARSE")
    if "EVENT" in domains or name_set & {
        "limit_up", "limit_down", "open_gap", "close_gap", "event_decay",
    }:
        tags.append("EVENT_DRIVEN")
    if name_set & {"is_null", "is_not_null", "is_finite", "is_inf",
                   "is_nan", "sign", "and_", "or_", "not_", "eq", "gt", "lt",
                   "ge", "le", "ne"}:
        tags.append("BINARY")
    if name_set & {"ceil", "floor", "fix", "round", "digital_count",
                   "decimate"}:
        tags.append("DISCRETE")
    if len([d for d in domains if d != "UNKNOWN"]) > 1:
        tags.append("MIXED_DOMAIN")
    return _canonical_structure_order(tags, policy=policy)


def _canonical_structure_order(
    tags: Iterable[str], *, policy: TaxonomyPolicy
) -> tuple[str, ...]:
    vocab = set(policy.structure_tag_vocab)
    order = {t: i for i, t in enumerate(_STRUCTURE_ORDER)}
    known = sorted((t for t in tags if t in vocab), key=lambda t: order.get(t, 10_000))
    extra = sorted(t for t in tags if t not in vocab)
    return tuple(known) + tuple(extra)


# ---------------------------------------------------------------------------
# Mechanism tags
# ---------------------------------------------------------------------------


def derive_mechanism_tags(
    op_ids: Sequence[str],
    data_domains: Sequence[str],
) -> tuple[TagEvidence, ...]:
    """Deterministic mechanism rules in fixed rule-table order.

    Rules are evaluated in ``_MECHANISM_RULES`` order; a tag dedupes if it
    fires more than once.  Every emitted tag carries evidence refs to the exact
    operator ids (or domains) that fired it.  If nothing fires, the explicit
    ``UNKNOWN_MECHANISM`` tag is emitted with the unmatched (non-mechanical)
    operator ids as evidence.
    """
    op_set = set(op_ids)
    domain_set = set(data_domains)

    tag_names: list[str] = []
    matched: list[str] = []

    def fire(tag: str, ops: Iterable[str] = ()) -> None:
        for o in ops:
            if o in op_set and o not in matched:
                matched.append(o)
        if tag not in tag_names:
            tag_names.append(tag)

    # --- domain-driven fundamental mechanisms -----------------------------
    if domain_set & {"FUNDAMENTAL.VALUE", "VALUATION"}:
        fire("VALUE")
    if domain_set & {"FUNDAMENTAL.QUALITY"}:
        fire("QUALITY")
    if domain_set & {"FUNDAMENTAL.GROWTH"}:
        fire("GROWTH")
    if domain_set & {"FUNDAMENTAL.INVESTMENT"}:
        fire("INVESTMENT")
    if domain_set & {"FUNDAMENTAL.LEVERAGE"}:
        fire("LEVERAGE")
    if domain_set & {"FUNDAMENTAL.CASHFLOW"}:
        fire("CASHFLOW")
    if domain_set & {"FUNDAMENTAL.PROFITABILITY"}:
        fire("PROFITABILITY")
    if "SIZE" in domain_set and not (domain_set & {"FUNDAMENTAL.VALUE", "VALUATION"}):
        fire("VALUE")
    if "LIQUIDITY" in domain_set:
        fire("LIQUIDITY")
    if "EVENT" in domain_set:
        fire("EVENT_DECAY")
    if "MICROSTRUCTURE" in domain_set:
        fire("MICROSTRUCTURE")
    if "FLOW_SENTIMENT" in domain_set:
        fire("CROWDING")

    # --- operator-driven mechanisms (fixed rule-table order) --------------
    for rule in _MECHANISM_RULES:
        fired_ops = rule.fires(op_set, domain_set=frozenset(data_domains))
        if fired_ops:
            fire(rule.tag, fired_ops)

    if not tag_names:
        unmatched = [
            o for o in dict.fromkeys(op_ids) if o and not _is_mechanical_noise(o)
        ]
        fire("UNKNOWN_MECHANISM", unmatched)

    evidence = tuple(
        TagEvidence(
            tag=tag,
            source=TagSource.DETERMINISTIC_RULE,
            confidence=_mechanism_confidence(tag, data_domains),
            evidence_refs=tuple(
                sorted(f"operator:{o}" for o in matched if o)
            ) or tuple(f"domain:{d}" for d in sorted(domain_set)),
        )
        for tag in tag_names
    )
    return evidence


class _MechanismRule:
    """A deterministic mechanism rule.

    ``triggers`` are exact canonical operator ids; ``prefixes`` match by
    ``startswith`` (e.g. ``micro_``).  ``domains`` optionally constrain a
    trigger to only fire when the factor's domain set intersects it (used to
    keep pure-price patterns from firing the liquidity rule).  ``predicate``
    is an optional extra boolean gate over the operator id set (used for
    combination patterns such as momentum-vs-reversal sign detection).
    Evaluation order is the module rule-table order — fixed, not dynamic.
    """

    __slots__ = ("rule_id", "tag", "triggers", "prefixes", "domains", "predicate")

    def __init__(
        self,
        rule_id: str,
        tag: str,
        triggers: Sequence[str] = (),
        prefixes: Sequence[str] = (),
        domains: Sequence[str] = (),
        predicate: Callable[[frozenset[str]], bool] | None = None,
    ) -> None:
        self.rule_id = rule_id
        self.tag = tag
        self.triggers = tuple(triggers)
        self.prefixes = tuple(prefixes)
        self.domains = frozenset(domains)
        self.predicate = predicate

    def fires(
        self,
        op_set: frozenset[str],
        *,
        domain_set: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        if self.domains and not (domain_set & self.domains):
            return ()
        hit = [o for o in self.triggers if o in op_set]
        hit.extend(
            o for o in sorted(op_set) for p in self.prefixes if o.startswith(p)
        )
        if not hit and self.predicate is not None and self.predicate(op_set):
            # predicate-only fire: evidence is the predicate rule id
            hit = [f"rule:{self.rule_id}"]
        return tuple(hit)


def _op_has_any(op_set: frozenset[str], *candidates: str) -> bool:
    return bool(op_set & set(candidates))


_PRICE_RETURN_PRIMITIVES = frozenset({
    "ts_delta", "deltas", "cum_delta", "ts_mean", "ts_log_return",
    "ts_pct_change", "log_returns", "ts_decay_exp_window", "period_change",
})


def _pred_momentum_scaled(op_set: frozenset[str]) -> bool:
    """Positive scaled short-term return: mean/vol ratio without negation."""
    return (
        _op_has_any(op_set, "ts_mean", "ts_delta", "ts_log_return")
        and _op_has_any(op_set, "ts_std", "ts_var")
        and "neg" not in op_set
        and "subtract" not in op_set
    )


def _pred_reversal_negated(op_set: frozenset[str]) -> bool:
    """Negated return primitive: -ret / -mean or -delta → reversal."""
    return "neg" in op_set and bool(op_set & _PRICE_RETURN_PRIMITIVES)


def _pred_mean_reversion(op_set: frozenset[str]) -> bool:
    """Price deviation from trailing mean (unscaled, no trend op)."""
    return (
        _op_has_any(op_set, "ts_mean")
        and _op_has_any(op_set, "divide", "subtract")
        and not _op_has_any(op_set, "ts_std", "ts_var", "ts_skew", "ts_kurt",
                            "slope", "ts_time_slope", "MACD")
    )


_MECHANISM_RULES: tuple[_MechanismRule, ...] = (
    _MechanismRule("MOMENTUM_T1", "MOMENTUM", triggers=(
        "mom", "roc", "log_returns", "cumulative_returns", "ts_log_return",
        "ts_pct_change", "period_change", "period_cagr", "MOM", "ROC", "TRIX",
    )),
    _MechanismRule("MOMENTUM_T2", "MOMENTUM", triggers=(
        "ts_delta", "deltas", "cum_delta",
    ), predicate=_pred_momentum_scaled),
    _MechanismRule("REVERSAL_T1", "REVERSAL", predicate=_pred_reversal_negated),
    _MechanismRule("BREAKOUT_T1", "BREAKOUT", triggers=(
        "ts_max", "ts_min", "cum_max", "cum_min", "expanding_max",
        "expanding_min", "ts_argmax", "ts_argmin",
    )),
    _MechanismRule("TREND_T1", "TREND", triggers=(
        "slope", "ts_time_slope", "ts_linreg", "MACD", "MACD_line",
        "MACD_signal", "ADX", "AROON", "AROON_up", "AROON_down", "KAMA",
        "RSI", "RSI_WILDER", "DPO",
    )),
    _MechanismRule("MEAN_REVERSION_T1", "MEAN_REVERSION", triggers=(
        "BollingerBands", "BollingerUpper", "BollingerLower", "BB",
        "StochasticK", "StochasticD", "WilliamsR", "CCI",
    )),
    _MechanismRule("MEAN_REVERSION_T2", "MEAN_REVERSION",
                   predicate=_pred_mean_reversion),
    _MechanismRule("VOLATILITY_T1", "VOLATILITY", triggers=(
        "ts_std", "ts_var", "ts_skew", "ts_kurt", "expanding_std", "ATR",
        "ATR_WILDER", "running_std", "idio_vol", "micro_bipower_var",
        "micro_realized_vol",
    )),
    _MechanismRule("INTERACTION_T1", "INTERACTION", triggers=(
        "ts_corr", "ts_cov", "ewm_corr", "ewm_cov", "Corr", "Cov",
        "correlate", "kendall_corr_test", "row_corr",
    )),
    _MechanismRule("LIQUIDITY_T1", "LIQUIDITY", triggers=(
        "amihud", "kyle_lambda", "micro_amihud_hf", "micro_kyle_lambda",
        "micro_spread", "micro_trade_imbalance", "micro_vpin",
        "micro_mid_return", "micro_jump_indicator",
    ), domains=("LIQUIDITY", "VOLUME", "MICROSTRUCTURE")),
    _MechanismRule("EVENT_DECAY_T1", "EVENT_DECAY", triggers=(
        "hump_decay", "ts_decay_linear", "group_decay_linear",
        "group_ts_decay_linear", "event_decay",
    )),
    _MechanismRule("TAIL_T1", "TAIL", triggers=(
        "max_drawdown", "downside_beta", "var", "cvar",
    )),
    _MechanismRule("SEASONALITY_T1", "SEASONALITY", triggers=(
        "fin_seasonal_zscore", "fin_seasonal_percentile", "period_average",
    )),
    _MechanismRule("CROWDING_T1", "CROWDING", triggers=(
        "herfindahl", "hhi",
    )),
    _MechanismRule("MICROSTRUCTURE_T1", "MICROSTRUCTURE", prefixes=("micro_",)),
)


#: Operational / mechanical operators never signal a mechanism on their own.
_MECHANICAL_NOISE = frozenset({
    "rank", "zscore", "cs_zscore", "cs_demean", "scale", "normalize",
    "winsorize", "group_rank", "group_zscore", "group_winsorize",
    "cs_quantile", "cs_pct_rank", "cs_scale", "cs_fill_mean", "cs_neutralize",
    "group_neutralize", "group_demean", "group_normalize", "group_std",
    "group_mean", "clip", "bound", "cap", "clamp", "saturate", "fillna",
    "ffill", "nan_to_num", "dropna", "is_null", "is_not_null", "ifnan",
    "if_else", "add", "multiply", "subtract", "divide", "neg", "abs", "sign",
    "max", "min", "mod", "power", "exp", "log", "sqrt", "protected_div",
    "div_or_default", "div_or_null", "le", "lt", "ge", "gt", "eq", "ne",
    "and_", "or_", "not_", "constant", "identity", "latency", "coalesce",
    "first", "last", "first_not_null", "last_not_null", "is_finite",
    "is_inf", "is_nan", "ifnan", "nonfinite_to_num", "log_fill_invalid",
})


def _is_mechanical_noise(op: str) -> bool:
    return op in _MECHANICAL_NOISE


def _mechanism_confidence(tag: str, data_domains: Sequence[str]) -> float:
    """Policy-versioned confidence weight for a tag.

    Domain-driven fundamental tags are the most certain (0.9); operator-driven
    signal tags 0.85; the explicit unknown fallback carries 0.4.
    """
    del data_domains
    if tag == "UNKNOWN_MECHANISM":
        return 0.4
    if tag in {
        "VALUE", "QUALITY", "PROFITABILITY", "GROWTH", "INVESTMENT",
        "LEVERAGE", "CASHFLOW", "LIQUIDITY",
    }:
        return 0.9
    return 0.85


# ---------------------------------------------------------------------------
# Frequency tags
# ---------------------------------------------------------------------------

_FREQ_FROM_TABLE: Mapping[str, str] = {
    "StockDailyBar": "daily",
    "StockDailyBarAdj": "daily",
    "StockIndicator": "daily",
    "StockValuationDaily": "daily",
    "StockBalance": "quarterly",
    "StockIncome": "quarterly",
    "StockCashFlow": "quarterly",
    "StockFinancialIndicator": "quarterly",
}


def derive_frequency_tags(field_usages) -> tuple[str, ...]:
    """Frequency tokens from the field usages' ``frequency_class`` (or table
    fallback).  Never guesses on unknowns — a field with neither contributes
    nothing."""
    freqs: list[str] = []
    for u in field_usages:
        f = str(getattr(u, "frequency_class", "") or "")
        if not f:
            table = str(getattr(u, "table", "") or "")
            f = _FREQ_FROM_TABLE.get(table, "")
        if f and f not in freqs:
            freqs.append(f)
    return tuple(sorted(freqs))


# ---------------------------------------------------------------------------
# Display family
# ---------------------------------------------------------------------------


def derive_display_family(
    data_domains: Sequence[str],
    *,
    policy: TaxonomyPolicy | None = None,
) -> str:
    """Derived display label for a domain set (never an authority).

    Examples: ``PV_ONLY`` (price-only), ``PV_FUND`` (price+volume+fundamental),
    ``PV_LIQ_FUND`` (price+volume+liquidity+fundamental).  The label is a
    compressed, human-friendly view of ``data_domains`` and must never be
    parsed back into an authoritative enum.
    """
    del policy
    domains = set(data_domains)
    has_price = "PRICE" in domains
    has_vol = "VOLUME" in domains
    has_liq = "LIQUIDITY" in domains
    has_fund = bool(domains & {
        "FUNDAMENTAL", "FUNDAMENTAL.VALUE", "FUNDAMENTAL.QUALITY",
        "FUNDAMENTAL.GROWTH", "FUNDAMENTAL.INVESTMENT",
        "FUNDAMENTAL.CASHFLOW", "FUNDAMENTAL.LEVERAGE",
        "FUNDAMENTAL.PROFITABILITY", "VALUATION", "SIZE",
    })
    has_micro = "MICROSTRUCTURE" in domains
    has_unknown = "UNKNOWN" in domains

    parts: list[str] = []
    if has_price or has_vol:
        parts.append("PV")
    if has_liq:
        parts.append("LIQ")
    if has_fund:
        parts.append("FUND")
    if has_micro:
        parts.append("MICRO")

    if not parts:
        return "UNKNOWN" if has_unknown else "OTHER"

    base = "_".join(parts)
    if has_unknown:
        return f"{base}_PLUS_UNKNOWN"
    if len(parts) == 1:
        return f"{base}_ONLY"
    return base


# ---------------------------------------------------------------------------
# Helpers / hashing
# ---------------------------------------------------------------------------


def _resolve_operators(analysis):
    usages = getattr(analysis, "operator_usages", None)
    if usages is None:
        raise TypeError(
            "classify_factor_taxonomy requires analysis.operator_usages (from "
            "factor_engine.api.static_analysis.analyze_factor_definition)"
        )
    return tuple(usages)


def _resolve_fields(analysis):
    usages = getattr(analysis, "field_usages", None)
    if usages is None:
        raise TypeError(
            "classify_factor_taxonomy requires analysis.field_usages (from "
            "factor_engine.api.static_analysis.analyze_factor_definition)"
        )
    return tuple(usages)


def _factor_definition_id_of(analysis) -> str:
    fid = getattr(analysis, "factor_definition_id", None)
    if fid:
        return str(fid)
    h = getattr(analysis, "canonical_dsl_hash", None)
    return str(h or "unspecified")


def _ref_hash(ids: Sequence[str]) -> str:
    return _sha256(json.dumps(sorted(set(ids)), sort_keys=True, ensure_ascii=True))


def _canonical_domain_order(tags: Iterable[str]) -> tuple[str, ...]:
    order = {t: i for i, t in enumerate(_DOMAIN_ORDER)}
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        t = str(t)
        if t not in seen:
            seen.add(t)
            out.append(t)
    out.sort(key=lambda t: order.get(t, 10_000))
    return tuple(out)


def _sha256(data: str) -> str:
    return hashlib.sha256(
        (_TAXONOMY_NS + data).encode("utf-8"), usedforsecurity=False
    ).hexdigest()


def _artifact_hash(artifact: FactorTaxonomyArtifact) -> str:
    payload = json.dumps(
        {
            "factor_definition_id": artifact.factor_definition_id,
            "data_domains": list(artifact.data_domains),
            "display_family": artifact.display_family,
            "mechanism_tags": [
                {
                    "tag": t.tag,
                    "source": t.source,
                    "confidence": t.confidence,
                    "evidence_refs": list(t.evidence_refs),
                }
                for t in artifact.mechanism_tags
            ],
            "structure_tags": list(artifact.structure_tags),
            "frequency_tags": list(artifact.frequency_tags),
            "field_usage_ref": artifact.field_usage_ref,
            "operator_usage_ref": artifact.operator_usage_ref,
            "taxonomy_policy_id": artifact.taxonomy_policy_id,
            "taxonomy_policy_version": artifact.taxonomy_policy_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _sha256(payload)
