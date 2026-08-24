# -*- coding: utf-8
"""基本面 field contract CI。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="legacy ratio primitives were removed in favour of explicit recipes")

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.fundamental.field_contract import (
    FUNDAMENTAL_RATIO_FIELD_CONTRACTS,
    check_fundamental_ratio_field_contracts,
)
from factor_engine.cleaned_operators.operator_spec import build_operator_spec


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_ratio_field_contracts_complete(_loaded):
    errors = check_fundamental_ratio_field_contracts()
    assert not errors, errors


def test_fundamental_ratios_not_production(_loaded):
    for canon in FUNDAMENTAL_RATIO_FIELD_CONTRACTS:
        spec = build_operator_spec(canon)
        assert spec is not None
        assert not spec.allow_in_production, canon
