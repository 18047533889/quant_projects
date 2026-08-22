# -*- coding: utf-8 -*-
"""R9-P1-046/047 regression tests: the parameter-aware cost model must be a
search gate, and declared ``cost_model(params, shape)`` contracts must take
precedence over the prefix-name complexity table."""
from __future__ import annotations

import numpy as np

from cleaned_operators.operator_cost_model import (
    default_mining_allowed,
    has_declared_cost_contract,
    memory_cost,
    runtime_cost,
    search_budget_gate,
)


def test_declared_cost_contract_wins_over_prefix():
    # ts_qn_scale has an EXPLICIT cost contract (O(W²)) — the value must be
    # the declared O(W²)/ref, not the default linear-in-window guess.
    rt20 = runtime_cost("ts_qn_scale", {"window": 20})
    rt500 = runtime_cost("ts_qn_scale", {"window": 500})
    assert has_declared_cost_contract("ts_qn_scale")
    # 500² / (120·120) ≈ 17.4, 20²/(120·120) ≈ 0.03 (floored to 1.0)
    assert rt500 > 8.0
    assert rt20 <= rt500
    assert rt500 / rt20 >= 10.0  # parameter-aware: window 25x -> cost ~625x


def test_undeclared_operator_falls_back_but_stays_linear():
    # ts_ema has no declared contract and no prefix kernel -> linear-in-window.
    assert not has_declared_cost_contract("ts_ema")
    assert runtime_cost("ts_ema", {"window": 240}) == 2.0  # 240/120


def test_search_budget_gate_respects_runtime_and_memory():
    # window=20 fits a budget of 5; window=500 does not.
    assert search_budget_gate("ts_qn_scale", {"window": 20}, runtime_budget=5.0)
    assert not search_budget_gate("ts_qn_scale", {"window": 500}, runtime_budget=5.0)
    # memory budget alone can also reject an O(W²) operator
    assert search_budget_gate("ts_qn_scale", {"window": 20}, memory_budget=5.0)
    assert not search_budget_gate("ts_qn_scale", {"window": 500}, memory_budget=5.0)


def test_default_mining_excludes_heavy_undeclared():
    # ts_matrix_profile is heavy at large windows and has NO declared contract:
    # default mining must exclude it rather than let it eat the budget.
    assert not has_declared_cost_contract("ts_matrix_profile")
    assert runtime_cost("ts_matrix_profile", {"window": 200}) > 8.0
    assert not default_mining_allowed("ts_matrix_profile", {"window": 200})
    # a cheap operator stays allowed
    assert default_mining_allowed("ts_ema", {"window": 500})


def test_default_mining_excludes_research_only_by_construction():
    assert not default_mining_allowed("ts_dmd_mode_concentration", {"window": 120})
    assert not default_mining_allowed("ts_persistence_entropy_h0", {"window": 120})


def test_memory_cost_returns_sane_scale():
    # An O(W²)-family operator's working set is rows²-equivalents.
    mem = memory_cost("ts_recurrence_quantification_analysis", {"window": 240})
    assert mem >= 1.0
    assert np.isfinite(mem)
