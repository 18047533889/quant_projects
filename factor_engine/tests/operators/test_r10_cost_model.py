# -*- coding: utf-8 -*-
"""R10 cost-model regression tests (#31 #32 #33 #34 #35).

The external review R10 found that:
  #31 a declared cost contract was treated as a "free pass" that let expensive
      operators skip the default-mining budget;
  #32 cost formulas ignored the actual panel shape (T, N, window, ...);
  #33 kernel/cubic operators' memory was estimated at O(W) instead of O(W²);
  #34 research-tier decisions relied on operator-name prefixes;
  #35 an UNRESOLVABLE cost fell back to a cheap default in production search.

These tests lock in the corrected behaviour.  No xfail / skip.
"""
from __future__ import annotations

from factor_engine.cleaned_operators.operator_cost_model import (
    ComplexityClass,
    CostResolution,
    CostShape,
    CostUnknownError,
    complexity_class,
    cost_resolution,
    default_mining_allowed,
    estimate_runtime,
    has_declared_cost_contract,
    is_research_only,
    memory_cost,
    runtime_cost,
    search_budget_gate,
)
from factor_engine.cleaned_operators.operator_surface import classify_canonical


def _assert_ratio(actual: float, expected: float, factor: float = 2.0) -> None:
    assert actual > 0.0
    assert expected / factor <= actual <= expected * factor, (
        f"ratio {actual} not within {factor}x of {expected}"
    )


# ---------------------------------------------------------------------------
# #31 — a declared cost contract is NOT a free pass.
# ---------------------------------------------------------------------------
def test_declared_cost_contract_is_not_a_free_pass():
    # ts_qn_scale has a VALID declared cost contract (O(W²) pairwise diffs) ...
    assert has_declared_cost_contract("ts_qn_scale")
    assert cost_resolution("ts_qn_scale") is CostResolution.DECLARED
    # ... but its estimated runtime at window=500 is 1000x a tight campaign
    # budget, so default mining must BLOCK it (R10 #31).
    rt = runtime_cost("ts_qn_scale", {"window": 500})
    assert rt > 1000.0  # 500² / 120 ≈ 2083
    budget = 1.0
    assert rt > 1000.0 * budget
    assert not default_mining_allowed(
        "ts_qn_scale", {"window": 500}, heavy_runtime=budget
    )
    # A cheap instance of the same operator with the SAME declared contract
    # still fits the budget — the contract is not blanket-banned.
    assert default_mining_allowed("ts_qn_scale", {"window": 20})
    # The budget gate enforces the declared estimate the same way.
    assert search_budget_gate("ts_qn_scale", {"window": 20}, runtime_budget=5.0)
    assert not search_budget_gate(
        "ts_qn_scale", {"window": 500}, runtime_budget=5.0
    )


# ---------------------------------------------------------------------------
# #32 — cost depends on ACTUAL shape; N² class scales with N.
# ---------------------------------------------------------------------------
def test_n2_class_scales_with_cross_section():
    # cs_knn_distance builds an O(N²) distance matrix each period.
    assert complexity_class("cs_knn_distance") is ComplexityClass.KNN_GRAPH
    rt500 = estimate_runtime("cs_knn_distance", {}, CostShape(N=500))[0]
    rt5000 = estimate_runtime("cs_knn_distance", {}, CostShape(N=5000))[0]
    # N 500 -> 5000 is a 10x growth; an N² class must grow ~100x.
    _assert_ratio(rt5000 / rt500, 100.0)
    # The distance matrix working set is O(N²) as well.
    mem500 = estimate_runtime("cs_knn_distance", {}, CostShape(N=500))[1]
    mem5000 = estimate_runtime("cs_knn_distance", {}, CostShape(N=5000))[1]
    _assert_ratio(mem5000 / mem500, 100.0)


def test_complexity_class_mapping_wired_into_estimator():
    # R10 #32: per-category classes drive the shape-aware estimate.
    assert complexity_class("ts_ema") is ComplexityClass.TS_ROLLING
    assert complexity_class("cs_pct_rank") is ComplexityClass.CS_SORT_RANK
    assert complexity_class("group_mean") is ComplexityClass.GROUP
    assert complexity_class("ts_residualized_hsic") is ComplexityClass.KERNEL_GRAM
    assert complexity_class("ts_ridge_regression_resid") is ComplexityClass.MATRIX_INV
    # TS rolling runtime depends on window.
    rt_small = estimate_runtime("ts_ema", {"window": 60}, CostShape(T=10, N=100))[0]
    rt_large = estimate_runtime("ts_ema", {"window": 120}, CostShape(T=10, N=100))[0]
    _assert_ratio(rt_large / rt_small, 2.0)


# ---------------------------------------------------------------------------
# #33 — kernel Gram operators are O(W²) memory, never O(W).
# ---------------------------------------------------------------------------
def test_kernel_gram_memory_scales_w2():
    # ts_residualized_hsic is a kernel Gram operator (O(W²) memory).
    assert has_declared_cost_contract("ts_residualized_hsic")
    mem_w = estimate_runtime("ts_residualized_hsic", {"window": 120})[1]
    mem_2w = estimate_runtime("ts_residualized_hsic", {"window": 240})[1]
    _assert_ratio(mem_2w / mem_w, 4.0)
    # An O(W) estimate would give ratio ~2 — reject it loudly.
    assert mem_2w > 3.0 * mem_w
    # Same through the legacy per-column memory API.
    mem_w_legacy = memory_cost("ts_residualized_hsic", {"window": 120})
    mem_2w_legacy = memory_cost("ts_residualized_hsic", {"window": 240})
    _assert_ratio(mem_2w_legacy / mem_w_legacy, 4.0)


def test_other_cubic_cost_contracts_are_w2_memory():
    # ts_persistence_entropy_h0 is a cubic-cost operator (Rips/barcode) — its
    # memory estimate must also be O(W²), not O(W) (R10 #33).
    mem_w = memory_cost("ts_persistence_entropy_h0", {"window": 120})
    mem_2w = memory_cost("ts_persistence_entropy_h0", {"window": 240})
    _assert_ratio(mem_2w / mem_w, 4.0)
    assert mem_2w > 3.0 * mem_w


# ---------------------------------------------------------------------------
# #35 — cost UNKNOWN => unselectable in production search.
# ---------------------------------------------------------------------------
def test_unknown_cost_is_unsearchable_in_production():
    unknown = "ts_no_such_operator_r10_2026_08"
    assert cost_resolution(unknown) is CostResolution.UNKNOWN
    # No matter how generous the budget, an UNKNOWN cost must not be let
    # through by a cheap default (R10 #35).
    assert not default_mining_allowed(unknown, {"window": 10})
    assert not search_budget_gate(unknown, runtime_budget=1e12, memory_budget=1e12)
    # estimate_runtime fails loudly rather than guessing.
    try:
        estimate_runtime(unknown, {"window": 10})
        raised = False
    except CostUnknownError:
        raised = True
    assert raised
    # Research-mode planning may still use the cheap default estimate.
    assert runtime_cost(unknown, {"window": 120}) > 0.0
    assert memory_cost(unknown, {"window": 120}) > 0.0


# ---------------------------------------------------------------------------
# #34 — surface metadata, not name prefix, drives cost-tier decisions.
# ---------------------------------------------------------------------------
def test_surface_metadata_drives_cost_tier_not_name_prefix():
    # ts_dmd_mode_concentration is research-tier via surface metadata even
    # though its name carries no 'research' prefix.
    assert classify_canonical("ts_dmd_mode_concentration") == "research"
    assert is_research_only("ts_dmd_mode_concentration")
    # ts_ema is authored on the daily surface (registered from the
    # research_polars MODULE, but its surface tier is daily) — module/name
    # provenance is irrelevant to the tier decision.
    assert classify_canonical("ts_ema") == "daily"
    assert not is_research_only("ts_ema")
    # A literal 'research' name prefix must never gate a cost tier.
    assert not is_research_only("research_something_that_should_be_cheap")
    # An operator that is NOT research-tier is not blocked by construction —
    # only by its actual cost (checked in the #31 test).
    assert not is_research_only("ts_qn_scale")
