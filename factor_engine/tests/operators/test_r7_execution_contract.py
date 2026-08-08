# -*- coding: utf-8 -*-
"""WS-D #256-#264: ExecutionContract / HistoryRequirement authority and the
composite contract wiring.

Tests:
(a) ts_ema execution_contract == recursive/checkpoint (with checkpoint schema).
(b) history_requirement returns a HistoryRequirement dataclass — never the
    1e9 integer sentinel — and full-history stateful ops report kind='full_history'.
(c) composite all-branch certification for a multi-branch composite
    (certified_for_all_branches iterates every reachable param branch).
(d) composite replacement refused on a wrong expected old lowering hash.
"""
from __future__ import annotations

import pytest

from planner.composite_lowering import (
    LoweringContract,
    CompositeLoweringDuplicateError,
    _LOWERING_OLD_HASH,
    certified_for_all_branches,
    composite_param_branches,
    declare_lowering_replacement,
    register_lowering,
)
from planner.logical_plan import PlanNode


# ---------------------------------------------------------------------------
# (a) ExecutionContract authority
# ---------------------------------------------------------------------------
def test_execution_contract_ts_ema_is_recursive_checkpoint():
    from runtime.execution_contract import execution_contract

    contract = execution_contract("ts_ema")
    assert contract.state_model == "recursive"
    assert contract.chunking == "checkpoint"
    assert contract.checkpoint_schema == "ema_state.v2"


def test_execution_contract_kama_requires_full_history():
    from runtime.execution_contract import execution_contract

    # KAMA has a checkpoint schema but no segmented restore -> full-history replay.
    contract = execution_contract("KAMA")
    assert contract.state_model == "recursive"
    assert contract.chunking == "required_full_history"
    assert contract.checkpoint_schema == "kama_state.v1"


def test_execution_contract_stateless_default():
    from runtime.execution_contract import execution_contract

    contract = execution_contract("ts_mean")
    assert contract.state_model == "stateless"
    assert contract.chunking == "independent"
    assert contract.checkpoint_schema is None


# ---------------------------------------------------------------------------
# (b) HistoryRequirement — not the 1e9 sentinel
# ---------------------------------------------------------------------------
def test_history_requirement_returns_dataclass_not_sentinel():
    from runtime.execution_contract import (
        FULL_HISTORY_LOOKBACK_SENTINEL,
        HistoryRequirement,
        history_requirement,
    )

    requirement = history_requirement("ts_ema", {"span": 20})
    assert isinstance(requirement, HistoryRequirement)
    assert not isinstance(requirement, int)  # never a raw integer sentinel
    assert requirement.kind in {"finite", "full_history"}
    assert isinstance(requirement.rows, int)
    assert requirement.rows != FULL_HISTORY_LOOKBACK_SENTINEL


def test_history_requirement_full_history_stateful_op():
    from runtime.execution_contract import HistoryRequirement, history_requirement

    kama = history_requirement("KAMA", {"window": 10})
    assert isinstance(kama, HistoryRequirement)
    assert kama.kind == "full_history"


def test_history_requirement_finite_rows_match_analyzer():
    from runtime.execution_contract import history_requirement

    # ts_ema(span=20) warms up to span-1=19 rows (kernel warmup == analyzer).
    assert history_requirement("ts_ema", {"span": 20}).rows == 19
    # MACD_line fast=12/slow=26/signal=9 -> max(fast,slow)-1 + (signal-1).
    macd = history_requirement("MACD_line", {"fast": 12, "slow": 26, "signal": 9})
    assert macd.rows == 33


# ---------------------------------------------------------------------------
# (c) composite all-branch certification
# ---------------------------------------------------------------------------
def _register_branch_composite(canonical: str, fast_values: tuple[float, ...]):
    """Register a synthetic multi-branch composite that lowers to ``ts_mean``
    when ``fast < slow`` and otherwise leaves the node unchanged."""

    @register_lowering(
        canonical,
        contract=LoweringContract(
            deps=("fast", "slow", "signal"),
            min_inputs=1,
            param_branches={
                "fast": fast_values,
                "slow": (26.0,),
                "signal": (9.0,),
            },
        ),
    )
    def _branch_lowering(node: PlanNode) -> PlanNode:
        fast = node.attrs.get("fast", 5)
        slow = node.attrs.get("slow", 26)
        if fast >= slow:
            return node
        return PlanNode(
            op="ts_mean",
            inputs=[node.inputs[0], PlanNode(op="literal", attrs={"value": float(fast)}, inputs=[])],
            attrs={},
        )

    return _branch_lowering


@pytest.fixture
def patch_dual_backend_safe(monkeypatch):
    """Point PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE at a known certified set."""
    import backend.primitive_evidence as evidence

    monkeypatch.setattr(
        evidence,
        "PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE",
        frozenset({"ts_mean"}),
    )


def test_composite_param_branches_enumerates_all_macd_branches():
    branches = composite_param_branches("MACD_line")
    assert "fast" in branches and "slow" in branches and "signal" in branches
    assert 5.0 in branches["fast"] and 12.0 in branches["fast"]
    assert 26.0 in branches["slow"]
    assert 9.0 in branches["signal"]


def test_certified_for_all_branches_iterates_every_branch(patch_dual_backend_safe):
    canonical = "__r7_branch_all_fire__"
    _register_branch_composite(canonical, fast_values=(5.0, 12.0))
    # Every combo (5,26,9) and (12,26,9) fires fast<slow -> ts_mean is certified.
    assert certified_for_all_branches(canonical) is True


def test_certified_for_all_branches_false_when_branch_does_not_fire(patch_dual_backend_safe):
    canonical = "__r7_branch_bad__"
    # fast=26 >= slow=26 is a reachable branch that does NOT lower.
    _register_branch_composite(canonical, fast_values=(5.0, 26.0))
    assert certified_for_all_branches(canonical) is False


def test_certified_for_all_branches_runs_over_real_macd():
    # Returns a bool even though ts_ema is not dual-backend certified here.
    assert certified_for_all_branches("MACD_line") in (True, False)


# ---------------------------------------------------------------------------
# (d) composite replacement identity-pinning
# ---------------------------------------------------------------------------
def test_replacement_refused_on_wrong_expected_old_hash():
    canonical = "__r7_repl_test__"
    _LOWERING_OLD_HASH.pop(canonical, None)

    @register_lowering(canonical)
    def _r7_repl_first(node):  # noqa: ARG001
        return node

    old_hash = _LOWERING_OLD_HASH[canonical]

    # Declare a replacement pinned to the WRONG old hash -> refused.
    declare_lowering_replacement(
        canonical, "wrong old hash", expected_old_hash="deadbeef"
    )
    with pytest.raises(CompositeLoweringDuplicateError, match="hashes to"):

        @register_lowering(canonical)
        def _r7_repl_second(node):  # noqa: ARG001
            return node

    # Declare the replacement pinned to the CORRECT old hash -> allowed, and
    # the recorded old hash advances to the new lowering.
    declare_lowering_replacement(
        canonical,
        "right old hash",
        expected_old_hash=old_hash,
        new_hash="__new__",
        semantic_version="2.0",
    )

    @register_lowering(canonical)
    def _r7_repl_third(node):  # noqa: ARG001
        return node

    assert _LOWERING_OLD_HASH[canonical] != old_hash


def test_replacement_without_hash_pin_still_allowed():
    canonical = "__r7_repl_nohash__"
    _LOWERING_OLD_HASH.pop(canonical, None)

    @register_lowering(canonical)
    def _r7_repl_first(node):  # noqa: ARG001
        return node

    # Legacy declaration (reason only, no hash pin) still authorises re-register.
    declare_lowering_replacement(canonical, "legacy override")

    @register_lowering(canonical)
    def _r7_repl_second(node):  # noqa: ARG001
        return node
