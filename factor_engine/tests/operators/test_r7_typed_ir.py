"""Round-7 WS-C: semantic lattice + per-argument typed contracts (#265-#273).

Covers:
* semantic lattice join replaces first-input inheritance (#265-#268);
* per-parameter ``input_types`` gate rejects Volume<->Return swap (#269-#272);
* production mode rejects unknown raw ColumnRef (#273);
* FieldSpec ``semantic_kind`` mapping helper (Volume/UniverseMask/GroupId).
"""
from __future__ import annotations

import pytest

from ir.nodes import IRNode
from ir.types import (
    OPERATOR_INPUT_TYPE_CONTRACTS,
    SemanticLattice,
    lattice_join_semantic_attrs,
)


def _column(name: str, kind: str | None) -> IRNode:
    attrs = {"name": name, "field": name}
    semantic = {"semantic_kind": kind} if kind else {}
    return IRNode(op="column", attrs=attrs, semantic_attrs=semantic)


# ---------------------------------------------------------------------------
# #265-#268  semantic lattice (replaces first-input inheritance).
# ---------------------------------------------------------------------------
def test_lattice_join_merges_dimensions():
    merged = lattice_join_semantic_attrs([
        {"domain": "price_volume", "frequency": "daily", "semantic_kind": "PriceRaw"},
        {"domain": "fundamental", "frequency": "daily",
         "semantic_kind": "FinancialCumulativeYTDFlow"},
    ])
    # Unambiguous dimension stays scalar.
    assert merged["frequency"] == "daily"
    # Conflicting dimensions are recorded as mixed_ — not silently inherited.
    assert merged["mixed_domain"] == ("price_volume", "fundamental")
    assert merged["mixed_semantic_kind"] == ("PriceRaw", "FinancialCumulativeYTDFlow")
    # pit_safe ANDs across inputs.
    assert merged["pit_safe"] is True


def test_lattice_join_single_child_propagates_scalar():
    merged = lattice_join_semantic_attrs([
        {"domain": "price_volume", "frequency": "daily", "price_basis": "RAW"},
        {"domain": "price_volume", "frequency": "daily", "price_basis": "RAW"},
    ])
    assert merged["domain"] == "price_volume"
    assert merged["frequency"] == "daily"
    assert merged["price_basis"] == "RAW"
    assert "mixed_domain" not in merged


def test_semantic_lattice_dataclass():
    lattice = SemanticLattice(domains=("price_volume",), frequencies=("daily",))
    other = SemanticLattice(domains=("fundamental",), frequencies=("daily",))
    joined = lattice.join(other)
    assert joined.domains == ("price_volume", "fundamental")
    assert joined.frequencies == ("daily",)


# ---------------------------------------------------------------------------
# #269-#272  per-parameter input_types gate.
# ---------------------------------------------------------------------------
def test_input_types_contract_registered():
    contract = OPERATOR_INPUT_TYPE_CONTRACTS["dollar_volume_zscore"]
    assert contract[0].parameter == "close"
    assert "PriceRaw" in contract[0].allowed_semantic_kinds
    assert contract[1].parameter == "volume"
    assert "NonNegativeActivity" in contract[1].allowed_semantic_kinds


def test_input_types_valid_close_volume_passes():
    from ir.analyzer import validate_input_type_contracts

    node = IRNode(
        op="dollar_volume_zscore",
        inputs=(
            _column("close", "PriceRaw"),
            _column("volume", "NonNegativeActivity"),
            IRNode(op="literal", attrs={"value": 20}),
        ),
    )
    assert validate_input_type_contracts(node) == []


def test_input_types_volume_in_close_slot_rejects():
    from ir.analyzer import validate_input_type_contracts

    # Swap: Volume goes into the close slot -> the close contract rejects it.
    node = IRNode(
        op="dollar_volume_zscore",
        inputs=(
            _column("volume", "NonNegativeActivity"),
            _column("close", "PriceRaw"),
            IRNode(op="literal", attrs={"value": 20}),
        ),
    )
    errors = validate_input_type_contracts(node)
    assert any("close" in e and "NonNegativeActivity" in e for e in errors)


def test_input_types_return_in_volume_slot_rejects():
    from ir.analyzer import validate_input_type_contracts

    # Return goes into the volume slot -> the volume contract rejects it.
    node = IRNode(
        op="dollar_volume_zscore",
        inputs=(
            _column("close", "PriceRaw"),
            _column("return", "ReturnDecimal"),
            IRNode(op="literal", attrs={"value": 20}),
        ),
    )
    errors = validate_input_type_contracts(node)
    assert any("volume" in e and "ReturnDecimal" in e for e in errors)


def test_input_types_group_mean_group_contract():
    from ir.analyzer import validate_input_type_contracts

    ok = IRNode(
        op="group_mean",
        inputs=(_column("x", None), _column("industry", "GroupKey")),
    )
    assert validate_input_type_contracts(ok) == []
    bad = IRNode(
        op="group_mean",
        inputs=(_column("x", None), _column("industry", "NonNegativeActivity")),
    )
    errors = validate_input_type_contracts(bad)
    assert any("group" in e and "NonNegativeActivity" in e for e in errors)


# ---------------------------------------------------------------------------
# #273  production mode rejects unknown raw ColumnRef.
# ---------------------------------------------------------------------------
def test_production_rejects_unknown_raw_column():
    from api.columns import col
    from ir.analyzer import Analyzer, UnknownRawColumnError

    with pytest.raises(UnknownRawColumnError):
        Analyzer(production=True).lower(col("custom_alpha_input"))


def test_research_allows_raw_column_opt_in():
    from api.columns import col
    from ir.analyzer import Analyzer

    analysis = Analyzer().lower(col("custom_alpha_input"))
    assert analysis.ir.op == "column"


# ---------------------------------------------------------------------------
# #269  FieldSpec.semantic_kind mapping helper (declared, not guessed).
# ---------------------------------------------------------------------------
def test_semantic_kind_mapping_volume():
    from fields.spec import FieldSpec, semantic_kind_of_field

    assert semantic_kind_of_field(
        FieldSpec(name="volume", table="StockDailyBar", source_name="volume")
    ) == "NonNegativeActivity"
    assert semantic_kind_of_field(
        FieldSpec(name="amount", table="StockDailyBar", source_name="amount")
    ) == "NonNegativeActivity"


def test_semantic_kind_mapping_universe_mask():
    from fields.spec import semantic_kind_of_field

    assert semantic_kind_of_field("universe_mask") == "MaskBool"


def test_semantic_kind_mapping_group_id():
    from fields.spec import semantic_kind_of_field

    assert semantic_kind_of_field("group_id") == "GroupKey"


def test_semantic_kind_explicit_wins_over_name():
    from fields.spec import FieldSpec, semantic_kind_of_field

    # A declared semantic_kind is authoritative — never overridden by a name guess.
    spec = FieldSpec(
        name="volume", table="t", source_name="v", semantic_kind="PriceContinuous"
    )
    assert semantic_kind_of_field(spec) == "PriceContinuous"
