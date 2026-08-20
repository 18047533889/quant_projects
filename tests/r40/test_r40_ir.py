# -*- coding: utf-8 -*-
"""R40 IR items #174/#176/#183."""
from __future__ import annotations

import datetime

import pytest


class TestProductionAnalyzerMarket:
    def test_production_analyzer_rejects_missing_market(self):
        from ir.analyzer import Analyzer, ProductionMarketContextRequiredError

        with pytest.raises(ProductionMarketContextRequiredError):
            Analyzer(production=True)
        # explicit market is fine
        a = Analyzer(production=True, market="ashare")
        assert a._production is True
        assert a._market == "ashare"
        # research keeps the legacy fallback
        Analyzer(production=False)
        Analyzer(production=False, market=None)


class TestAxisEffectContract:
    def test_declared_axis_effect_contract(self):
        from ir.types import (
            AXIS_EFFECT_CONTRACTS,
            AxisEffectKind,
            check_axis_effect_declared,
            register_axis_effect,
        )

        register_axis_effect("test_axis_ts_op", AxisEffectKind.TIME_SERIES)
        register_axis_effect("test_axis_cs_op", AxisEffectKind.CROSS_SECTION)
        contract = check_axis_effect_declared("test_axis_ts_op", production=True)
        assert contract.has_time_series_effect
        assert not contract.has_cross_section_effect
        cs = check_axis_effect_declared("test_axis_cs_op", production=True)
        assert cs.has_cross_section_effect

    def test_undeclared_axis_effect_rejected_in_production(self):
        from ir.types import check_axis_effect_declared

        with pytest.raises(ValueError):
            check_axis_effect_declared("test_axis_undeclared_op", production=True)
        # research degrades (returns None)
        assert check_axis_effect_declared("test_axis_undeclared_op", production=False) is None

    def test_bulk_registration_covers_daily_surface(self):
        """R40 #176 收尾：daily 表面每个算子都声明了 AxisEffectContract——
        静态门 ``PRODUCTION_AXIS_EFFECT_UNDECLARED == 0``。"""
        from cleaned_operators import load_all
        from cleaned_operators import operator_surface
        from ir.types import AXIS_EFFECT_CONTRACTS

        load_all()
        daily = operator_surface.DAILY_CANONICALS
        undeclared = sorted(c for c in daily if c not in AXIS_EFFECT_CONTRACTS)
        assert undeclared == [], f"undeclared axis effects on daily surface: {undeclared}"
        # 代表性分类：ts_mean→TS、rank→CS、abs→ELEMENTWISE
        from ir.types import axis_effect_contract_for

        assert axis_effect_contract_for("ts_mean").has_time_series_effect
        assert axis_effect_contract_for("rank").has_cross_section_effect
        assert not axis_effect_contract_for("abs").has_time_series_effect
        assert not axis_effect_contract_for("abs").has_cross_section_effect


class TestSemanticIdentityDigest:
    def test_supported_values(self):
        from ir.types import SemanticIdentityDigest

        d = SemanticIdentityDigest.from_attrs({
            "market": "ashare",
            "period_duration": "quarter",
            "availability_expr": None,
        })
        assert d.value  # non-empty

    def test_datetime_and_enum_supported(self):
        from ir.types import SemanticIdentityDigest, SemanticType

        d = SemanticIdentityDigest.from_attrs({
            "source_vintage": datetime.datetime(2026, 1, 1),
            "semantic_kind": SemanticType.PRICE_RAW,
        })
        assert d.value

    def test_unsupported_object_rejected(self):
        from ir.types import SemanticIdentityDigest

        class Opaque:
            def __str__(self):
                return "same-for-everything"

        with pytest.raises(TypeError):
            SemanticIdentityDigest.from_attrs({"market": Opaque()})

    def test_no_str_fallback_collision(self):
        from ir.types import SemanticIdentityDigest

        a = SemanticIdentityDigest.from_attrs({"market": "us", "universe_id": "SPX"})
        b = SemanticIdentityDigest.from_attrs({"market": "us", "universe_id": "SPY"})
        assert a.value != b.value
