# -*- coding: utf-8
"""算子 shape / production 契约 CI。"""

from __future__ import annotations

import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.operator_policy import (
    NON_SHAPE_PRESERVING_CANONICALS,
    RESEARCH_CORE_CANONICALS,
    policy_required_canonicals,
)
from cleaned_operators.operator_spec import (
    PRODUCTION_CORE_CANONICALS,
    build_operator_spec,
    check_production_shape_contracts,
    is_production_denied,
)


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_production_core_subset_of_policy_required(_loaded):
    required = policy_required_canonicals()
    assert PRODUCTION_CORE_CANONICALS <= required


def test_research_core_not_in_production(_loaded):
    for canon in RESEARCH_CORE_CANONICALS:
        spec = build_operator_spec(canon)
        if spec is None:
            continue
        assert not spec.allow_in_production, canon


def test_dropna_not_shape_preserving(_loaded):
    spec = build_operator_spec("dropna")
    assert spec is not None
    assert not spec.shape_preserving
    assert not spec.allow_in_production


def test_removed_fill_operators_have_no_runtime(_loaded):
    from cleaned_operators.registry import OperatorRegistry

    for canon in ("bfill", "causal_bfill", "fillna_interpolate"):
        assert OperatorRegistry.get(canon) is None
        assert is_production_denied(canon)


def test_causal_linear_extrapolate_research_not_production(_loaded):
    spec = build_operator_spec("causal_linear_extrapolate")
    assert spec is not None
    assert spec.status == "research"
    assert not spec.allow_in_production


def test_production_shape_contracts_pass(_loaded):
    errors = check_production_shape_contracts()
    assert not errors, errors[:5]


def test_non_shape_preserving_registry(_loaded):
    assert "dropna" in NON_SHAPE_PRESERVING_CANONICALS
