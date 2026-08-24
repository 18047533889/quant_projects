# -*- coding: utf-8 -*-
"""R40 units items #184/#185/#186."""
from __future__ import annotations

import pytest

from factor_engine.fields import units as U
from factor_engine.fields import units_v2 as U2


class TestCanonicalUnitKnown:
    def test_known_unit_spelling_returns_known_id(self):
        u = U.canonical_unit_known("cny/share")
        assert isinstance(u, U.KnownUnitId)
        assert u == U.UNIT_CNY_PER_SHARE
        # None -> dimensionless
        assert U.canonical_unit_known(None) == U.UNIT_DIMENSIONLESS

    def test_unknown_unit_spelling_rejected_in_production(self):
        with pytest.raises(U.UnknownUnitError):
            U.canonical_unit_known("cny/shar")  # typo
        with pytest.raises(U.UnknownUnitError):
            U.canonical_unit_known("dollar_per_soup")

    def test_opaque_unit_research_only(self):
        o = U.OpaqueUnit("cny/shar")
        assert o.label == "cny/shar"
        assert isinstance(o, U.OpaqueUnit)


class TestUnitConstantsExported:
    def test_all_registered_unit_ids_publicly_exported(self):
        # R40 #185: every canonical unit id constant must be in __all__.
        exported = set(U.__all__)
        for name in (
            "UNIT_CNY_PER_SHARE",
            "UNIT_SHARE_RATIO",
            "UNIT_DIMENSIONLESS",
            "UNIT_BOOLEAN",
            "UNIT_CNY",
            "UNIT_SHARE",
            "UNIT_RATIO",
            "UNIT_PERCENT",
            "UNIT_BASIS_POINT",
            "UNIT_CNY_10K",
            "UNIT_SHARE_10K",
            "UNIT_DATE",
            "UNIT_DATETIME",
            "UNIT_IDENTIFIER",
            "UNIT_TEXT",
        ):
            assert name in exported, name
            assert hasattr(U, name), name

    def test_alias_table_keys_cover_known_constants(self):
        for const in ("UNIT_CNY_PER_SHARE", "UNIT_SHARE_RATIO", "UNIT_BOOLEAN", "UNIT_CNY_10K"):
            value = getattr(U, const)
            assert U.canonical_unit_known(value) == value


class TestUnitExprAlgebra:
    def test_mul_div_pow(self):
        price = U2.UnitExpr.from_spec(U2.CNY_PER_SHARE)
        shares = U2.UnitExpr.from_spec(U2.SHARES)
        # price * shares -> money
        money = price * shares
        assert money.factors == {"money": 1.0}
        # price / shares -> money * count^-2
        q = price / shares
        assert q.factors == {"money": 1.0, "count": -2.0}
        # price ** 2 -> money^2 count^-2
        p2 = price ** 2
        assert p2.factors["money"] == 2.0
        # sqrt(price) -> money^0.5 count^-0.5
        p05 = price ** 0.5
        assert p05.factors["money"] == 0.5

    def test_unit_algebra_kind(self):
        assert U2.unit_algebra_kind(U2.UNIT_EXPR_DIMENSIONLESS) == "dimensionless"
        assert U2.unit_algebra_kind(U2.UNIT_EXPR_MONEY) == "money"
        price = U2.UnitExpr.from_spec(U2.CNY_PER_SHARE)
        assert U2.unit_algebra_kind(price) == "price"

    def test_log_of_price_rejected(self):
        price = U2.UnitExpr.from_spec(U2.CNY_PER_SHARE)
        with pytest.raises(U2.UnitAlgebraError):
            U2.assert_log_operand_dimensionless(price, operator="log")
        # dimensionless passes
        U2.assert_log_operand_dimensionless(U2.UNIT_EXPR_RATIO, operator="log")

    def test_rank_produces_dimensionless(self):
        out = U2.assert_rank_produces_dimensionless()
        assert out.is_dimensionless

    def test_price_is_not_dimensionless(self):
        price = U2.UnitExpr.from_spec(U2.CNY_PER_SHARE)
        assert not price.is_dimensionless
