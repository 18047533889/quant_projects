# -*- coding: utf-8 -*-
"""R24-113/114: typed input contracts cover the economically-constrained
operators — relation/shareholder/group operators declare their input semantics."""
from __future__ import annotations

import pytest

from factor_engine.ir.types import OPERATOR_INPUT_TYPE_CONTRACTS


@pytest.mark.parametrize("canon", [
    "relation_category_share",
    "relation_peer_weighted_mean_ex_self",
    "group_demean",
    "group_mean",
    "group_sum",
    "group_rank_weighted_value",
    "group_rank",
    "dollar_volume_zscore",
    "event_historical_response_mean",
])
def test_economically_constrained_operator_has_typed_contract(canon: str) -> None:
    assert canon in OPERATOR_INPUT_TYPE_CONTRACTS, canon


def test_relation_category_share_requires_non_negative_value() -> None:
    contract = OPERATOR_INPUT_TYPE_CONTRACTS["relation_category_share"]
    value_contract = contract[0]
    assert value_contract.allowed_semantic_kinds == frozenset({
        "NonNegativeWeight", "NonNegativeActivity", "NonNegativeAmount",
    })
    assert value_contract.accepts("NonNegativeWeight")
    assert value_contract.accepts("ReturnDecimal") is False


def test_group_operators_require_group_key() -> None:
    for canon in ("group_mean", "group_sum", "group_rank_weighted_value"):
        group_contract = OPERATOR_INPUT_TYPE_CONTRACTS[canon][1]
        assert group_contract.allowed_semantic_kinds == frozenset({"GroupKey"})
        assert group_contract.accepts("GroupKey")
        assert group_contract.accepts("PriceRaw") is False
