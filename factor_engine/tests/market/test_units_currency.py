# -*- coding: utf-8 -*-
"""Currency/denominator-aware UnitSpec tests."""
from __future__ import annotations

import pytest

from fields.units_v2 import (
    CNY,
    CNY_PER_SHARE,
    RATIO,
    SHARES,
    USD,
    USD_PER_SHARE,
    UnitSpec,
    legacy_unit_string,
    unit_spec_from_legacy,
)


def test_ratio_scale_for_legacy() -> None:
    # percent/bp collapse to scale-1 ratio at the canonical layer
    assert unit_spec_from_legacy("percent", canonical=True) == RATIO
    assert unit_spec_from_legacy("basis_point", canonical=True) == RATIO
    # source-unit spelling keeps its scale
    assert unit_spec_from_legacy("percent").scale == pytest.approx(0.01)
    assert unit_spec_from_legacy("basis_point").scale == pytest.approx(0.0001)


def test_legacy_round_trip() -> None:
    for unit in ("CNY", "share", "ratio", "percent", "basis_point", "date", "boolean"):
        spec = unit_spec_from_legacy(unit)
        assert legacy_unit_string(spec) in (unit, "CNY", "share", "ratio", "date", "boolean")


def test_money_requires_currency() -> None:
    with pytest.raises(ValueError):
        UnitSpec(dimension="money")
    with pytest.raises(ValueError):
        UnitSpec(dimension="price", currency="USD")  # missing denominator


def test_cross_currency_arithmetic_blocked() -> None:
    with pytest.raises(ValueError):
        CNY.assert_compatible_with(USD)
    with pytest.raises(ValueError):
        CNY_PER_SHARE.assert_compatible_with(USD_PER_SHARE)


def test_same_currency_compatible() -> None:
    CNY.assert_compatible_with(CNY)
    CNY_PER_SHARE.assert_compatible_with(CNY_PER_SHARE)
    RATIO.assert_compatible_with(RATIO)
    SHARES.assert_compatible_with(SHARES)


def test_dimension_mismatch_blocked() -> None:
    with pytest.raises(ValueError):
        RATIO.assert_compatible_with(CNY)
    with pytest.raises(ValueError):
        CNY.assert_compatible_with(SHARES)


def test_unit_serialization() -> None:
    for spec in (CNY, USD_PER_SHARE, RATIO, SHARES):
        assert UnitSpec.from_dict(spec.to_dict()) == spec


def test_string_forms() -> None:
    assert str(CNY) == "CNY"
    assert str(USD) == "USD"
    assert str(CNY_PER_SHARE) == "CNY/share"
    assert str(RATIO) == "ratio"
