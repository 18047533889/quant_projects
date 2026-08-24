# -*- coding: utf-8 -*-
"""Review-10 field / registry contract fixes (R10-P0-014..017).

* R10-P0-014 — un-declared PIT / mining eligibility defaults to UNKNOWN (None)
  and production fails closed; only the explicit catalog True/False is trusted.
* R10-P0-015 — a ``status`` role is automatically NOT mining-eligible.
* R10-P0-016 — ``replace=True`` cannot steal an alias owned by another table /
  field identity without declaring ``expected_old_identity``.
* R10-P0-017 — field resolution is tri-state: ``ResolvedField`` /
  ``UnknownField`` / ``AmbiguousField``, never a bare ``None``.
"""
from __future__ import annotations

import pytest

from factor_engine.fields.registry import (
    AmbiguousField,
    FieldRegistry,
    ResolvedField,
    UnknownField,
)
from factor_engine.fields.spec import FieldSpec, TableSpec


# ---------------------------------------------------------------------------
# R10-P0-014: UNKNOWN defaults
# ---------------------------------------------------------------------------

def test_bare_field_defaults_to_unknown():
    spec = FieldSpec(name="x", table="StockIncome", source_name="x")
    assert spec.mining_allowed is None
    assert spec.strict_pit_allowed is None


def test_bare_table_defaults_to_unknown():
    table = TableSpec(name="StockIncome", dataset="inc")
    assert table.strict_pit_allowed is None


def test_explicit_catalog_values_still_win():
    spec = FieldSpec(
        name="x", table="StockIncome", source_name="x",
        mining_allowed=True, strict_pit_allowed=False,
    )
    assert spec.mining_allowed is True
    assert spec.strict_pit_allowed is False


# ---------------------------------------------------------------------------
# R10-P0-015: status role auto-blocks mining
# ---------------------------------------------------------------------------

def test_status_role_auto_blocks_mining():
    spec = FieldSpec(
        name="listing_status", table="StockStatus", source_name="ListingStatus",
        role="status",
    )
    assert spec.mining_allowed is False


# ---------------------------------------------------------------------------
# R10-P0-016: replace must not steal third-party aliases
# ---------------------------------------------------------------------------

def test_table_replace_steals_alias_without_expected_identity():
    reg = FieldRegistry()
    reg.register_table(TableSpec(name="StockIncome", dataset="inc"))
    reg.register_table(TableSpec(name="StockBalance", dataset="bal"))
    # StockIncome tries to take over the "bal" alias owned by StockBalance
    with pytest.raises(ValueError, match="belongs to table"):
        reg.register_table(TableSpec(name="StockIncome", dataset="bal"), replace=True)


def test_table_replace_with_expected_identity_ok():
    reg = FieldRegistry()
    reg.register_table(TableSpec(name="StockIncome", dataset="inc"))
    reg.register_table(TableSpec(name="StockBalance", dataset="bal"))
    reg.register_table(
        TableSpec(name="StockIncome", dataset="bal"),
        replace=True,
        expected_old_identity="StockBalance",
    )
    assert reg.resolve_table("bal") is not None


def test_field_replace_steals_alias_without_expected_identity():
    reg = FieldRegistry()
    reg.register_table(TableSpec(name="StockIncome", dataset="inc"))
    reg.register_table(TableSpec(name="StockBalance", dataset="bal"))
    reg.register(
        FieldSpec(name="net_profit", table="StockIncome", source_name="net_profit")
    )
    # StockBalance.net_profit would steal the "net_profit" alias from
    # StockIncome.net_profit
    with pytest.raises(ValueError, match="belongs to"):
        reg.register(
            FieldSpec(name="net_profit", table="StockBalance", source_name="net_profit"),
            replace=True,
        )


# ---------------------------------------------------------------------------
# R10-P0-017: tri-state resolution
# ---------------------------------------------------------------------------

def _registry_with_aliases() -> FieldRegistry:
    reg = FieldRegistry()
    reg.register_table(TableSpec(name="StockIncome", dataset="inc"))
    reg.register(FieldSpec(name="net_profit", table="StockIncome", source_name="net_profit"))
    reg.register(FieldSpec(name="a", table="StockIncome", source_name="a", aliases=("np",)))
    reg.register(FieldSpec(name="b", table="StockIncome", source_name="b", aliases=("np",)))
    return reg


def test_resolve_field_tristate():
    reg = _registry_with_aliases()
    assert isinstance(reg.resolve_field("net_profit"), ResolvedField)
    assert isinstance(reg.resolve_field("does_not_exist"), UnknownField)
    amb = reg.resolve_field("np")
    assert isinstance(amb, AmbiguousField)
    lowered = tuple(c.lower() for c in amb.candidates)
    assert "stockincome.a" in lowered and "stockincome.b" in lowered


def test_get_strict_false_still_none_for_both_but_require_is_strict():
    reg = _registry_with_aliases()
    # backward-compatible: get(strict=False) returns None for unknown AND ambiguous
    assert reg.get("does_not_exist") is None
    assert reg.get("np") is None
    # strict get / require raise for both, with distinct messages
    with pytest.raises(KeyError, match="unknown field"):
        reg.require("does_not_exist")
    with pytest.raises(KeyError, match="ambiguous field"):
        reg.require("np")
