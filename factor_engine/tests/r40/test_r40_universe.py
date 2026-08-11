# -*- coding: utf-8 -*-
"""R40 #221-225: UniverseMembership 双时态区间 / production fail-closed /
digest 区间历史 / CrossSectionEligibility / CoverageDecomposition."""
from __future__ import annotations

import numpy as np
import pytest

from market.universe import (
    OPEN_ENDED,
    NON_PIT_STATIC_UNIVERSE,
    CoverageExitReason,
    CrossSectionEligibility,
    StaticUniverseSnapshot,
    UniverseContractError,
    UniverseKnowledgeUnknownError,
    UniverseMembership,
    UniverseMembershipDigest,
    coverage_decomposed_by_exit_reason,
    universe_membership_identity,
)


def test_membership_bitemporal_interval_boundaries() -> None:
    """#221: valid_from <= trade_time < valid_to，区分 OPEN_ENDED 与已退出。"""
    m = UniverseMembership(
        universe="CSI300",
        instrument="000001.SZ",
        valid_time="2024-01-01",
        valid_to_exclusive="2025-01-01",
        knowledge_time="2023-12-20",
    )
    # before valid_from -> not effective
    assert m.effective_at("2024-06-01", trade_time="2023-12-01") is False
    # at valid_from -> effective
    assert m.effective_at("2024-06-01", trade_time="2024-01-01") is True
    # inside interval -> effective
    assert m.effective_at("2024-06-01", trade_time="2024-06-15") is True
    # at valid_to_exclusive -> NOT effective (exclusive)
    assert m.effective_at("2024-06-01", trade_time="2025-01-01") is False
    # after exit -> not effective
    assert m.effective_at("2025-06-01", trade_time="2025-06-01") is False
    # OPEN_ENDED: no upper bound -> still effective
    opened = UniverseMembership(
        universe="CSI300", instrument="000001.SZ",
        valid_time="2024-01-01", valid_to_exclusive=OPEN_ENDED, knowledge_time="2023-12-20",
    )
    assert opened.effective_at("2025-06-01", trade_time="2025-06-01") is True
    # revision: knowledge_time gate
    rev = UniverseMembership(
        universe="CSI300", instrument="000001.SZ",
        valid_time="2024-01-01", valid_to_exclusive=OPEN_ENDED, knowledge_time="2024-02-01",
    )
    assert rev.effective_at("2024-01-15", trade_time="2024-01-15") is False  # not yet known


def test_effective_at_fails_closed_when_missing_temporal_fields() -> None:
    """#222: production 缺 temporal 契约 -> UniverseKnowledgeUnknownError。"""
    m = UniverseMembership(universe="CSI300", instrument="000001.SZ")
    # research 旧行为：fail-open（静态）
    assert m.effective_at("2024-01-01") is True
    # production fail-closed
    with pytest.raises(UniverseKnowledgeUnknownError):
        m.effective_at("2024-01-01", mode="production")


def test_static_universe_snapshot_lineage_tag() -> None:
    """#222: StaticUniverseSnapshot 必须带 NON_PIT_STATIC_UNIVERSE lineage。"""
    s = StaticUniverseSnapshot(as_of="2024-01-01")
    assert s.lineage == NON_PIT_STATIC_UNIVERSE
    assert s.effective_at("2024-06-01") is True
    with pytest.raises(UniverseKnowledgeUnknownError):
        s.effective_at("2024-06-01", mode="production")


def test_universe_digest_differs_when_intervals_differ() -> None:
    """#223: 相同成员列表但区间历史不同 -> 不同 digest。"""
    d1 = universe_membership_identity(
        "CSI300", ("a", "b"),
        effective_intervals=(("a", "2024-01-01", "2024-06-01"), ("b", "2024-01-01", "2024-06-01")),
    )
    d2 = universe_membership_identity(
        "CSI300", ("a", "b"),
        effective_intervals=(("a", "2024-06-02", "2024-12-31"), ("b", "2024-06-02", "2024-12-31")),
    )
    assert d1 != d2
    # provider snapshot / calendar version 也进 hash
    d3 = universe_membership_identity(
        "CSI300", ("a", "b"),
        effective_intervals=(("a", "2024-01-01", "2024-06-01"), ("b", "2024-01-01", "2024-06-01")),
        provider_snapshot="snap-v2", calendar_version="cal-v2",
    )
    assert d1 != d3

    dig = UniverseMembershipDigest(
        universe="CSI300", members=("a", "b"),
        range_start="2024-01-01", range_end="2024-12-31",
    )
    assert len(dig.intersecting("2024-01-01", "2024-12-31")) == 2
    assert len(dig.intersecting("2025-01-01", "2025-12-31")) == 0


def test_eligibility_masks_decomposed() -> None:
    """#224: CrossSectionEligibility 把 opaque mask 拆成可组合掩码。"""
    listed = np.array([[1, 1], [1, 0]], dtype=float)
    suspension = np.array([[1, 0], [1, 1]], dtype=float)
    ce = CrossSectionEligibility(listed_mask=listed, suspension_mask=suspension)
    mask = ce.to_mask((2, 2))
    assert mask.tolist() == [[True, False], [True, False]]
    # 单独取用每个位
    assert CrossSectionEligibility(suspension_mask=suspension).suspension_mask is not None
    assert ce.ranking_eligibility_policy == "listed_and_membership_and_valid_price"
    assert ce.trading_eligibility_policy == "listed_and_not_suspended_and_valid_price"


def test_coverage_decomposed_by_exit_reason() -> None:
    """#225: 覆盖率按缺失原因分类。"""
    panel = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    mask = np.array([[1, 0], [1, 1]], dtype=float)  # (0,1) out-of-universe
    masked = panel * mask
    comps = {
        "is_suspend": np.array([[0, 1], [0, 0]], dtype=float),
        "close": panel.copy(),
        "public_status": np.array([[1, 1], [1, 1]], dtype=float),
    }
    dec = coverage_decomposed_by_exit_reason(panel, masked, mask, components=comps)
    assert dec.per_reason[CoverageExitReason.OUT_SUSPENDED.value] == 1
    assert dec.total_cells == 4
    assert dec.in_universe == 3
    # 无 components：诚实归为 UNKNOWN
    dec2 = coverage_decomposed_by_exit_reason(panel, masked, mask)
    assert CoverageExitReason.OUT_UNKNOWN_STATUS.value in dec2.per_reason
