# -*- coding: utf-8 -*-
"""R40 concept/fields items #218/#219/#180/#181/#182."""
from __future__ import annotations

import pytest

from fields.concepts import (
    FieldConceptSpec,
    FieldLegalityError,
    assert_field_legality,
    check_field_legality,
    validate_concept_metadata,
)
from fields.units_v2 import RATIO, UnitSpec


class TestSemanticExtensions:
    def test_semantic_key_not_hidden_in_loose_metadata(self):
        with pytest.raises(ValueError):
            FieldConceptSpec(
                concept_id="x",
                domain="fundamental",
                value_kind="ratio",
                canonical_unit=RATIO,
                descriptive_metadata={"price_basis": "RAW"},  # hidden semantic key
            )

    def test_descriptive_metadata_allowed_when_not_semantic(self):
        spec = FieldConceptSpec(
            concept_id="x",
            domain="fundamental",
            value_kind="ratio",
            canonical_unit=RATIO,
            descriptive_metadata={"display_label": "ROE", "source_note": "from 10-K"},
        )
        assert spec.descriptive_metadata["display_label"] == "ROE"

    def test_semantic_extensions_enter_identity(self):
        # FieldConceptSpec is unhashable (dict fields), so equality is the
        # identity comparison — semantic_extensions participate in __eq__.
        a = FieldConceptSpec(
            concept_id="x", domain="f", value_kind="ratio", canonical_unit=RATIO,
            semantic_extensions={"missing_policy": "forbid_forward_fill"},
        )
        b = FieldConceptSpec(
            concept_id="x", domain="f", value_kind="ratio", canonical_unit=RATIO,
            semantic_extensions={"missing_policy": "allow_forward_fill"},
        )
        assert a != b
        # descriptive_metadata does NOT enter equality
        c = FieldConceptSpec(
            concept_id="x", domain="f", value_kind="ratio", canonical_unit=RATIO,
            semantic_extensions={"missing_policy": "forbid_forward_fill"},
            descriptive_metadata={"display_label": "A"},
        )
        d = FieldConceptSpec(
            concept_id="x", domain="f", value_kind="ratio", canonical_unit=RATIO,
            semantic_extensions={"missing_policy": "forbid_forward_fill"},
            descriptive_metadata={"display_label": "B"},
        )
        assert c == d

    def test_validate_concept_metadata_rejects_semantic_keys(self):
        with pytest.raises(ValueError):
            validate_concept_metadata({"canonical_unit": "CNY"})
        # allowed
        validate_concept_metadata({"display_label": "ROE"})


class TestFieldLegalityPass:
    def _spec(self, families=(), **kw):
        return FieldConceptSpec(
            concept_id="test_legality",
            domain="fundamental",
            value_kind="ratio",
            canonical_unit=RATIO,
            allowed_operator_families=families,
            **kw,
        )

    def test_rejects_disallowed_operator_family(self):
        spec = self._spec(families=("ratio_ops",))
        errors = check_field_legality(spec, operator_family="price_ops")
        assert errors
        assert "not allowed" in errors[0]
        with pytest.raises(FieldLegalityError):
            assert_field_legality(spec, operator_family="price_ops")

    def test_allowed_operator_family_passes(self):
        spec = self._spec(families=("ratio_ops",))
        assert check_field_legality(spec, operator_family="ratio_ops") == []

    def test_ratio_unit_vs_price_level(self):
        spec = self._spec(families=("ratio_ops",))
        # a ratio concept fed a price-level unit is illegal
        price_unit = UnitSpec.price("CNY")
        errors = check_field_legality(spec, operator_family="ratio_ops", unit=price_unit)
        assert errors

    def test_undeclared_families_allow_all(self):
        spec = self._spec(families=())
        assert check_field_legality(spec, operator_family="anything") == []


class TestPeriodDurationAndBundle:
    def test_period_duration_normalized(self):
        from ir.types import PeriodDuration, normalize_period_duration

        assert normalize_period_duration("QUARTER") == PeriodDuration.QUARTER.value
        assert normalize_period_duration("ttm") == "ttm"
        assert normalize_period_duration(None) == "unknown"
        with pytest.raises(ValueError):
            normalize_period_duration("weekly")

    def test_period_duration_in_factor_identity(self):
        from ir.types import SemanticIdentityDigest

        d1 = SemanticIdentityDigest.from_attrs({"period_duration": "quarter"})
        d2 = SemanticIdentityDigest.from_attrs({"period_duration": "annual"})
        assert d1.value != d2.value

    def test_semantic_bundle_covers_economic_dimensions(self):
        from ir.types import SemanticTypeBundle

        b = SemanticTypeBundle.from_field_like(
            type("S", (), {"period_duration": "annual", "frequency": "daily", "value_kind": "ratio"})()
        )
        assert b.period_duration == "annual"
        assert b.temporal_type == "daily"
        assert b.value_semantics == "ratio"
