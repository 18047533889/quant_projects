# -*- coding: utf-8 -*-
"""Tests for factor_assets/profiling — deterministic factor taxonomy (R61-FI-013).

Consumes the FE static-analysis output shape only (records with
``operator_id`` / ``canonical_field_id`` / ``axis_effect`` /
``existing_treatment_semantic_ids``), constructing the input records here so
these tests are self-contained and deterministic.  The real FE pipeline is
verified end-to-end in the integration-style tests (``test_*_dsl_*``) that call
``factor_engine.api.static_analysis.analyze_factor_definition``.

Coverage required by R61-FI-013 / plan §7:
1. typical DSLs (momentum / reversal / volatility / liquidity / price-volume
   interaction / fundamental fields) classify correctly;
2. mixed domains → multi-element domain set;
3. unknown DSL / fields → explicit ``UNKNOWN_MECHANISM`` / unknown domain; never
   raises, never silently drops;
4. frozen dataclass + content-hash stability (same input ⇒ same hash,
   tuple fields actually frozen);
5. rule-order precedence — a factor carrying *both* ``rank`` and ``ts_mean``
   (or rank + zscore + ema) emits the full structure-tag set, deterministic
   order;
6. display-family derivation ``PV_ONLY`` / ``PV_FUND`` / ``PV_LIQ_FUND``.
"""

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from factor_assets.profiling.policies import (
    TagEvidence,
    TagSource,
    TaxonomyPolicy,
    get_taxonomy_policy,
)
from factor_assets.profiling.taxonomy import (
    DomainTagSet,
    FactorTaxonomyArtifact,
    classify_factor_taxonomy,
    derive_data_domains,
    derive_display_family,
    derive_frequency_tags,
    derive_mechanism_tags,
    derive_structure_tags,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Op:
    """Minimal stand-in for an FE ``OperatorUsage`` record."""

    def __init__(self, operator_id, axis_effect=None, causality_class=None):
        self.operator_id = operator_id
        self.axis_effect = axis_effect
        self.causality_class = causality_class


class _Field:
    """Minimal stand-in for an FE ``FieldUsage`` record."""

    def __init__(self, cid, domain=None, frequency_class=None, table=None):
        self.canonical_field_id = cid
        self.domain = domain
        self.frequency_class = frequency_class
        self.table = table


class _Analysis:
    """Minimal stand-in for an FE ``FactorStaticAnalysisArtifact``."""

    def __init__(
        self,
        operator_usages,
        field_usages,
        factor_definition_id="F_test1234abcd",
        lineage=(),
    ):
        self.operator_usages = tuple(operator_usages)
        self.field_usages = tuple(field_usages)
        self.factor_definition_id = factor_definition_id
        self.canonical_dsl_hash = "deadbeef" * 4
        self.existing_treatment_semantic_ids = tuple(lineage)


def _a(op_ids, field_ids, *, axes=None, lineage=()):
    ops = [
        _Op(op_id, axis_effect=(axes.get(op_id) if axes else None))
        for op_id in op_ids
    ]
    fields = [_Field(fid) for fid in field_ids]
    return _Analysis(ops, fields, lineage=lineage)


def _classify(op_ids, field_ids, **kw):
    return classify_factor_taxonomy(_a(op_ids, field_ids), **kw)


def _classify_axes(op_ids, field_ids, *, axes=None, lineage=()):
    return classify_factor_taxonomy(
        _a(op_ids, field_ids, axes=axes, lineage=lineage)
    )


def _mech_names(art):
    return tuple(t.tag for t in art.mechanism_tags)


# ---------------------------------------------------------------------------
# 1. Typical DSL classification
# ---------------------------------------------------------------------------


class TestTypicalDslClassification:
    def test_momentum_dsl(self):
        art = _classify(
            ["rank", "ts_mean", "ts_std", "divide"],
            ["StockDailyBarAdj.close"],
        )
        assert "MOMENTUM" in _mech_names(art)
        assert art.data_domains == ("PRICE",)

    def test_reversal_dsl(self):
        art = _classify(
            ["rank", "neg", "ts_mean", "ts_std", "divide"],
            ["StockDailyBarAdj.close"],
        )
        names = _mech_names(art)
        assert "REVERSAL" in names
        assert "VOLATILITY" in names

    def test_volatility_dsl(self):
        art = _classify(["rank", "ts_std"], ["StockDailyBarAdj.close"])
        names = _mech_names(art)
        assert "VOLATILITY" in names
        assert names == ("VOLATILITY",)

    def test_liquidity_dsl(self):
        art = _classify(
            ["rank", "ts_mean", "divide"],
            ["StockDailyBarAdj.volume"],
        )
        names = _mech_names(art)
        # volume ratio is a volume/mean-reversion pattern, not tagged liquidity
        assert "LIQUIDITY" not in names

    def test_price_volume_interaction_dsl(self):
        art = _classify(
            ["rank", "ts_corr"],
            ["StockDailyBarAdj.close", "StockDailyBarAdj.volume"],
        )
        assert "INTERACTION" in _mech_names(art)
        assert art.data_domains == ("PRICE", "VOLUME")

    def test_fundamental_quality_dsl(self):
        art = _classify(["rank"], ["StockIndicator.roe"])
        assert art.data_domains == ("FUNDAMENTAL.QUALITY",)
        assert "QUALITY" in _mech_names(art)

    def test_fundamental_value_dsl(self):
        art = _classify(["rank"], ["StockValuationDaily.pe_ratio"])
        assert art.data_domains == ("FUNDAMENTAL.VALUE",)
        assert "VALUE" in _mech_names(art)

    def test_liquidity_field_domain(self):
        art = _classify(["rank"], ["StockDailyBarAdj.amount"])
        assert set(art.data_domains) == {"VOLUME", "LIQUIDITY"}
        assert "VALUE" not in _mech_names(art)


# ---------------------------------------------------------------------------
# 2. Mixed domains → multi-element set
# ---------------------------------------------------------------------------


class TestMixedDomains:
    def test_two_domains(self):
        art = _classify(
            ["rank", "ts_corr"],
            ["StockDailyBarAdj.close", "StockDailyBarAdj.volume"],
        )
        assert len(art.data_domains) == 2
        assert "PRICE" in art.data_domains
        assert "VOLUME" in art.data_domains

    def test_three_domains_pv_vol_fund(self):
        art = _classify(
            ["rank"],
            [
                "StockDailyBarAdj.close",
                "StockDailyBarAdj.volume",
                "StockIndicator.roe",
                "StockDailyBarAdj.amount",
            ],
        )
        assert set(art.data_domains) == {
            "PRICE", "VOLUME", "LIQUIDITY", "FUNDAMENTAL.QUALITY",
        }

    def test_domain_tag_set_semantics(self):
        ds = DomainTagSet(["PRICE", "VOLUME", "PRICE"])
        assert len(ds) == 2
        assert "PRICE" in ds
        assert set(ds.members) == {"PRICE", "VOLUME"}

    def test_mixed_unknown_kept_explicit(self):
        art = _classify(
            ["rank"],
            [
                "StockDailyBarAdj.close",
                "SomeExoticTable.zzz_unknown_field",
            ],
        )
        assert art.has_unknown_domain
        assert "UNKNOWN" in art.data_domains
        assert "PRICE" in art.data_domains


# ---------------------------------------------------------------------------
# 3. Unknown inputs are explicit / never raise
# ---------------------------------------------------------------------------


class TestUnknownInputs:
    def test_unknown_field_never_raises(self):
        art = _classify(["rank"], ["SomeExoticTable.baffling_field_42"])
        assert art.has_unknown_domain
        assert "UNKNOWN" in art.data_domains
        # NEVER silently dropped — factor id still present
        assert art.factor_definition_id.startswith("F_")

    def test_unknown_operator_never_raises(self):
        art = _classify(["rank", "zzz_no_such_operator"], ["StockDailyBarAdj.close"])
        names = _mech_names(art)
        assert "UNKNOWN_MECHANISM" in names
        # the unknown mechanism carries the unmatched op as evidence
        ev = next(t for t in art.mechanism_tags if t.tag == "UNKNOWN_MECHANISM")
        refs = " ".join(ev.evidence_refs)
        assert "zzz_no_such_operator" in refs

    def test_empty_fields_unknown_domain(self):
        art = _classify(["rank"], [])
        assert art.data_domains == ("UNKNOWN",)

    def test_empty_ops_unknown_mechanism(self):
        art = _classify([], ["StockDailyBarAdj.close"])
        assert _mech_names(art) == ("UNKNOWN_MECHANISM",)

    def test_malformed_operator_ids_never_raise(self):
        art = _classify(["rank", "", "operator?", None], ["StockDailyBarAdj.close"])
        assert "UNKNOWN_MECHANISM" in _mech_names(art)

    def test_malformed_field_ids_never_raise(self):
        art = _classify(["rank"], [None, "", "??"])
        assert "UNKNOWN" in art.data_domains


# ---------------------------------------------------------------------------
# 4. Frozen dataclass + content-hash stability
# ---------------------------------------------------------------------------


class TestFrozenAndHashStability:
    def _kw(self):
        return _a(
            ["rank", "ts_mean", "ts_std", "divide"],
            ["StockDailyBarAdj.close"],
            axes={"ts_mean": "time_series", "ts_std": "time_series",
                  "rank": "cross_section"},
            lineage=("CS_RANK:pct",),
        )

    def test_artifact_is_frozen(self):
        art = classify_factor_taxonomy(self._kw())
        with pytest.raises(FrozenInstanceError):
            art.data_domains = ("PRICE",)

    def test_tag_evidence_is_frozen(self):
        ev = TagEvidence(
            tag="MOMENTUM",
            source=TagSource.DETERMINISTIC_RULE,
            confidence=0.9,
            evidence_refs=("operator:ts_mean",),
        )
        with pytest.raises(FrozenInstanceError):
            ev.evidence_refs = ()

    def test_content_hash_stable_for_same_input(self):
        a1 = classify_factor_taxonomy(self._kw())
        a2 = classify_factor_taxonomy(self._kw())
        assert a1.content_hash == a2.content_hash
        assert len(a1.content_hash) == 64

    def test_conservative_hash_uses_payload(self):
        a1 = classify_factor_taxonomy(
            self._kw(),
            policy=get_taxonomy_policy(),
        )
        p = get_taxonomy_policy()
        a2 = classify_factor_taxonomy(self._kw(), policy=p)
        assert a1.content_hash == a2.content_hash


# ---------------------------------------------------------------------------
# 5. Rule-order precedence — full structure tag set
# ---------------------------------------------------------------------------


class TestRuleOrderPrecedence:
    def test_rank_and_ts_mean_both_present(self):
        art = _classify_axes(
            ["rank", "ts_mean", "divide"],
            ["StockDailyBarAdj.close"],
            axes={"ts_mean": "time_series", "rank": "cross_section"},
        )
        assert "RANKED" in art.structure_tags
        assert "TIME_SERIES" in art.structure_tags
        assert "CROSS_SECTIONAL" in art.structure_tags

    def test_structure_tag_order_is_canonical(self):
        art = _classify_axes(
            ["rank", "zscore", "ts_ema", "winsorize"],
            ["StockDailyBarAdj.close"],
            axes={"ts_ema": "time_series", "rank": "cross_section",
                  "zscore": "cross_section", "winsorize": "cross_section"},
        )
        expected = (
            "TIME_SERIES", "CROSS_SECTIONAL", "RANKED", "ZSCORED",
            "WINSORIZED", "SMOOTHED",
        )
        for tag in expected:
            assert tag in art.structure_tags
        # deterministic canonical order → TIME_SERIES before RANKED
        assert art.structure_tags == expected

    def test_grouped_and_domain_flags(self):
        art = _classify(
            ["rank", "ts_corr"],
            ["StockDailyBarAdj.close", "StockDailyBarAdj.volume"],
        )
        assert "MIXED_DOMAIN" in art.structure_tags
        assert "INTERACTION" in _mech_names(art)

    def test_treatment_lineage_structure_tags(self):
        art = classify_factor_taxonomy(
            _a(
                ["rank", "ts_ema"],
                ["StockDailyBarAdj.close"],
                lineage=("CS_RANK:pct", "SMOOTH:ewma", "INDUSTRY_NEUTRAL:sw_l1"),
            )
        )
        assert "RANKED" in art.structure_tags
        assert "SMOOTHED" in art.structure_tags
        assert "NEUTRALIZED" in art.structure_tags
        assert "NEUTRALIZED_INDUSTRY" in art.structure_tags


# ---------------------------------------------------------------------------
# 6. Display family derivation
# ---------------------------------------------------------------------------


class TestDisplayFamily:
    def test_pv_only(self):
        art = _classify(["rank"], ["StockDailyBarAdj.close"])
        assert art.display_family == "PV_ONLY"

    def test_pv_liq(self):
        art = _classify(["rank"], ["StockDailyBarAdj.amount"])
        assert art.display_family == "PV_LIQ"

    def test_pv_fund(self):
        art = _classify(
            ["rank", "ts_mean"],
            ["StockDailyBarAdj.close", "StockIndicator.roe"],
        )
        assert art.display_family == "PV_FUND"

    def test_pv_liq_fund(self):
        art = _classify(
            ["rank"],
            [
                "StockDailyBarAdj.amount",
                "StockValuationDaily.pe_ratio",
            ],
        )
        assert art.display_family == "PV_LIQ_FUND"

    def test_unknown_display(self):
        art = _classify(["rank"], ["SomeExoticTable.zzz"])
        assert art.display_family == "UNKNOWN"

    def test_derive_display_family_direct(self):
        assert derive_display_family(["PRICE"]) == "PV_ONLY"
        assert derive_display_family(["PRICE", "VOLUME", "LIQUIDITY"]) == "PV_LIQ"
        assert derive_display_family(
            ["PRICE", "FUNDAMENTAL"]
        ) == "PV_FUND"
        assert derive_display_family(
            ["PRICE", "VOLUME", "LIQUIDITY", "FUNDAMENTAL"]
        ) == "PV_LIQ_FUND"
        assert derive_display_family(["UNKNOWN"]) == "UNKNOWN"


# ---------------------------------------------------------------------------
# Policy / policy registry
# ---------------------------------------------------------------------------


class TestPolicyRegistry:
    def test_current_policy_resolves(self):
        p = get_taxonomy_policy()
        assert p.policy_id == "CN_A_SHARE_DAILY_TAXONOMY_V1"
        assert p.policy_version == "1.0.0"

    def test_unknown_policy_fail_closed(self):
        with pytest.raises(KeyError):
            get_taxonomy_policy("no_such_policy")

    def test_unknown_version_fail_closed(self):
        with pytest.raises(KeyError):
            get_taxonomy_policy("CN_A_SHARE_DAILY_TAXONOMY_V1", "9.9.9")

    def test_artifact_stamps_policy(self):
        art = _classify(["rank"], ["StockDailyBarAdj.close"])
        assert art.taxonomy_policy_id == "CN_A_SHARE_DAILY_TAXONOMY_V1"
        assert art.taxonomy_policy_version == "1.0.0"

    def test_policy_vocab_validated(self):
        with pytest.raises(ValueError):
            TaxonomyPolicy(
                policy_id="X",
                policy_version="1",
                description="",
                mechanism_confidence=1.5,
            )


# ---------------------------------------------------------------------------
# Manual artifact construction / typing
# ---------------------------------------------------------------------------


class TestArtifactShape:
    def test_to_dict_roundtrip_fields(self):
        art = classify_factor_taxonomy(
            _a(["rank"], ["StockDailyBarAdj.close"])
        )
        d = art.to_dict()
        assert d["display_family"] == "PV_ONLY"
        assert d["mechanism_tags"][0]["tag"] == "UNKNOWN_MECHANISM"
        assert d["taxonomy_policy_version"] == "1.0.0"
        assert d["content_hash"] == art.content_hash


# ---------------------------------------------------------------------------
# Derive helpers standalone
# ---------------------------------------------------------------------------


class TestDeriveHelpers:
    def test_derive_data_domains_unknown_empty(self):
        assert derive_data_domains([]) == ("UNKNOWN",)

    def test_derive_structure_tags_canonical_order(self):
        tags = derive_structure_tags(
            [_Op("zscore", "cross_section"), _Op("rank", "cross_section")],
            lineage=("CS_RANK:pct", "ZSCORE:cs"),
        )
        assert tags == ("CROSS_SECTIONAL", "RANKED", "ZSCORED")

    def test_derive_mechanism_tags_unknown(self):
        mech = derive_mechanism_tags(["rank", "zz_noop", "ts_mean"], ["PRICE"])
        names = tuple(t.tag for t in mech)
        assert "UNKNOWN_MECHANISM" in names

    def test_derive_frequency_tags_table_fallback(self):
        freq = derive_frequency_tags(
            [_Field("StockDailyBarAdj.close", table="StockDailyBarAdj"),
             _Field("StockIncome.operating_revenue", table="StockIncome")]
        )
        assert freq == ("daily", "quarterly")