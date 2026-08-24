# -*- coding: utf-8 -*-
"""R40 #143/#144/#145: declassification approval strictness, empty-tag
fail-closed sensitivity, and production factor_id homoglyph rejection."""
from __future__ import annotations

import pytest

from factor_engine.security.access import (
    DeclassificationApproval,
    derive_derived_access_tags,
    max_sensitivity,
    require_declassification_approval,
)
from factor_engine.security.factor_id import FactorIdError, validate_factor_id


class TestDeclassificationApproval:
    def test_declassification_requires_reviewer_and_policy_when_approved(self):
        # approved but no reviewer/policy/timestamp/approval_id -> fail closed
        with pytest.raises(ValueError, match="reviewer"):
            DeclassificationApproval(approved=True).validate()
        with pytest.raises(ValueError, match="approval_id"):
            DeclassificationApproval(
                approved=True, reviewer="r", policy_version="p1", timestamp="t1"
            ).validate()

    def test_complete_approval_passes(self):
        appr = DeclassificationApproval(
            approved=True, reviewer="r", policy_version="p1",
            timestamp="2026-08-11T00:00:00Z", approval_id="APP-001",
        )
        appr.validate()  # must not raise
        assert appr.to_dict()["approval_id"] == "APP-001"

    def test_from_dict_roundtrip(self):
        raw = {"approved": True, "reviewer": "r", "policy_version": "p1",
               "timestamp": "t1", "approval_id": "a1"}
        appr = DeclassificationApproval.from_dict(raw)
        assert appr.approval_id == "a1"
        assert appr.timestamp == "t1"


class TestEmptyTagsFailClosed:
    def test_empty_access_tags_fail_closed_in_production(self):
        # empty tags -> UNKNOWN sensitivity (100), never 0 (public).
        assert max_sensitivity([]) == 100
        assert max_sensitivity(()) == 100
        assert max_sensitivity(None) == 100
        # public is 10, so an empty source set must NOT be treated as public.
        assert max_sensitivity([]) > max_sensitivity(["public"])

    def test_unknown_tag_rejected_in_production_derive(self):
        with pytest.raises(PermissionError, match="UNKNOWN_CLASSIFICATION"):
            derive_derived_access_tags(["brand_new_tag"], production=True)

    def test_unknown_tag_allowed_in_research_derive(self):
        tags = derive_derived_access_tags(["brand_new_tag"], production=False)
        assert "brand_new_tag" in tags

    def test_require_declassification_empty_source_fails_closed(self):
        # empty source is UNKNOWN (100) > public derived (10) -> needs approval.
        with pytest.raises(PermissionError):
            require_declassification_approval([], ["public"])


class TestFactorIdHomoglyph:
    def test_factor_id_rejects_homoglyph_in_production(self):
        # Cyrillic 'а' (U+0430) vs Latin 'a'
        with pytest.raises(FactorIdError, match="whitelist"):
            validate_factor_id("alpha_аbc", production=True)

    def test_factor_id_non_production_warns_but_allows(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = validate_factor_id("alpha_аbc", production=False)
            assert out == "alpha_аbc"
            assert any("whitelist" in str(x.message) for x in w)

    def test_factor_id_ascii_ok_in_production(self):
        out = validate_factor_id("alpha_001.b:rev-2", production=True)
        assert out == "alpha_001.b:rev-2"
