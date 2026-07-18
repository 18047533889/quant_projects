# -*- coding: utf-8
"""Fastpath tier 与 PRODUCTION_CORE 关系（文档 + 轻量断言）。"""
from __future__ import annotations


def test_fastpath_p0_subset_of_production_allowed():
    from backend.production_fastpath_tiers import P0_PRODUCTION_FASTPATH_CANONICALS
    from cleaned_operators import load_all
    from cleaned_operators.operator_spec import build_operator_spec

    load_all()
    for canon in sorted(P0_PRODUCTION_FASTPATH_CANONICALS):
        spec = build_operator_spec(canon)
        assert spec is not None, canon
        assert spec.allow_in_production, f"P0 {canon} 应 allow_in_production"


def test_production_core_equals_fastpath_p0_p1_plus_legacy():
    """PRODUCTION_CORE = dual-backend core + deferred allowed。"""
    from backend.production_fastpath_tiers import (
        P0_PRODUCTION_FASTPATH_CANONICALS,
        P1_POLARS_CORE_PRODUCTION_SAFE,
    )
    from cleaned_operators.operator_spec import (
        PRODUCTION_ALLOWED_DEFERRED_CANONICALS,
        PRODUCTION_CORE_CANONICALS,
        PRODUCTION_DUAL_BACKEND_CORE_CANONICALS,
        _FASTPATH_ALIAS_ONLY,
        _LEGACY_PRODUCTION_CORE,
    )

    dual_core = P0_PRODUCTION_FASTPATH_CANONICALS - _FASTPATH_ALIAS_ONLY
    assert PRODUCTION_DUAL_BACKEND_CORE_CANONICALS == dual_core
    assert PRODUCTION_CORE_CANONICALS == dual_core | PRODUCTION_ALLOWED_DEFERRED_CANONICALS
    assert not (PRODUCTION_ALLOWED_DEFERRED_CANONICALS & PRODUCTION_DUAL_BACKEND_CORE_CANONICALS)


def test_production_allowed_equals_core_plus_composites():
    from cleaned_operators import load_all
    from cleaned_operators.operator_spec import (
        PRODUCTION_CORE_CANONICALS,
        production_allowed_canonicals,
        production_allowed_composite_canonicals,
    )

    load_all()
    composites = production_allowed_composite_canonicals()
    expected = PRODUCTION_CORE_CANONICALS | composites
    assert production_allowed_canonicals() == expected
    assert composites <= production_allowed_canonicals()
