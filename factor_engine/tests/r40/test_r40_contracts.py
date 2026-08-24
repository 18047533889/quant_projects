# -*- coding: utf-8 -*-
"""R40 contract items #178/#190/#193."""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators.common.data_cleaning import (
    CarryForwardPolicy,
    check_carry_forward_policy,
)
from factor_engine.cleaned_operators.common.time_series import (
    SupportPolicy,
    check_support_policy,
    support_policy_for,
)
from factor_engine.ir.types import (
    HistoryContract,
    HistoryKind,
    check_history_contract_declared,
    register_history_contract,
)


class TestHistoryContract:
    def test_declared_history_contract(self):
        register_history_contract(
            "test_ts_op",
            HistoryContract(kind=HistoryKind.ROLLING_PLUS_LAG, rows=19, params=("window", "lag")),
        )
        c = check_history_contract_declared("test_ts_op", production=True)
        assert c is not None
        assert c.kind is HistoryKind.ROLLING_PLUS_LAG
        assert c.rows == 19

    def test_full_history_contract(self):
        register_history_contract(
            "test_full_hist_op", HistoryContract(kind=HistoryKind.FULL_HISTORY)
        )
        c = check_history_contract_declared("test_full_hist_op", production=True)
        assert c is not None and c.is_full_history

    def test_undeclared_history_contract_rejected_in_production(self):
        with pytest.raises(ValueError):
            check_history_contract_declared("test_no_contract_op", production=True)
        assert check_history_contract_declared("test_no_contract_op", production=False) is None


class TestSupportPolicy:
    def test_support_policy_for_declared_operators(self):
        assert support_policy_for("ts_mean") is not None
        assert support_policy_for("ts_std") is not None

    def test_min_periods_variation_changes_digest(self):
        a = SupportPolicy(min_observations=1, ddof=0)
        b = SupportPolicy(min_observations=3, ddof=0)
        assert a.digest() != b.digest()

    def test_undeclared_support_policy_rejected(self):
        with pytest.raises(ValueError):
            check_support_policy(None, canonical="ts_unknown")
        check_support_policy(SupportPolicy(min_observations=1), canonical="ts_mean")


class TestCarryForwardPolicy:
    def test_user_can_only_tighten(self):
        provider_forbid = CarryForwardPolicy(allowed=False, max_gap_sessions=0)
        user_allow = CarryForwardPolicy(allowed=True, max_gap_sessions=5)
        combined = user_allow.tighten(provider_forbid)
        # provider FORBID wins
        assert combined.allowed is False

        provider_gap2 = CarryForwardPolicy(allowed=True, max_gap_sessions=2)
        user_gap5 = CarryForwardPolicy(allowed=True, max_gap_sessions=5)
        assert user_gap5.tighten(provider_gap2).max_gap_sessions == 2

    def test_ffill_rejected_for_non_ffillable_field(self):
        with pytest.raises(ValueError):
            check_carry_forward_policy(None, canonical="ffill")
        p = check_carry_forward_policy(
            CarryForwardPolicy(allowed=True, max_gap_sessions=2), canonical="ffill"
        )
        assert p.max_gap_sessions == 2
