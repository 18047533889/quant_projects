# -*- coding: utf-8 -*-
"""Tests for the 2026-08 relation / index / event operator expansion (P1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

P1_CANONICALS = frozenset(
    """relation_hhi relation_entropy relation_topk_sum relation_rank_weighted_sum
    relation_category_share relation_peer_weighted_mean_ex_self
    relation_entry_count relation_exit_count relation_weighted_change
    index_member index_weight_change index_entry_exit_event index_membership_age
    event_cumulative_return_past event_abnormal_return_past
    fin_applicability_mask calendar_day_diff fin_announcement_lag
    """.split()
)


def _panel(rows: int = 60, cols: int = 3, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    names = ["A", "B", "C"][:cols]
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((rows, cols)) + 0.1, index=idx, columns=names)


def _group(rows: int = 60, cols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    names = ["A", "B", "C"][:cols]
    return pd.DataFrame(np.resize(np.array(["X", "Y", "X"]), (rows, cols)), index=idx, columns=names)


def _member(panel: pd.DataFrame) -> pd.DataFrame:
    return (panel > 0.5).astype(float)


@pytest.mark.parametrize("name", sorted(P1_CANONICALS))
def test_p1_operator_registered_and_extended_surface(name: str) -> None:
    assert OperatorRegistry.get(name) is not None
    assert classify_canonical(name) == "extended"


def test_relation_hhi_and_topk() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    cols = ["A"]
    top1 = pd.DataFrame([[0.2]], index=idx, columns=cols)
    top2 = pd.DataFrame([[0.1]], index=idx, columns=cols)
    hhi = OperatorRegistry.get("relation_hhi").calculate(top1, top2)
    expected = (0.2**2 + 0.1**2) / (0.3**2)
    assert hhi.iloc[0, 0] == pytest.approx(expected)
    topk = OperatorRegistry.get("relation_topk_sum").calculate(top1, top2)
    assert topk.iloc[0, 0] == pytest.approx(0.3)


def test_relation_category_share_sums_to_one_per_group() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    value = pd.DataFrame([[1.0, 3.0, 2.0]], index=idx, columns=["A", "B", "C"])
    group = pd.DataFrame([["X", "X", "Y"]], index=idx, columns=["A", "B", "C"])
    out = OperatorRegistry.get("relation_category_share").calculate(value, group)
    # X group: A=1/(1+3)=0.25, B=3/4=0.75 ; Y group: C=1.0
    assert out.iloc[0, 0] == pytest.approx(0.25)
    assert out.iloc[0, 1] == pytest.approx(0.75)
    assert out.iloc[0, 2] == pytest.approx(1.0)


def test_relation_peer_weighted_mean_ex_self() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    value = pd.DataFrame([[1.0, 2.0, 3.0]], index=idx, columns=["A", "B", "C"])
    weight = pd.DataFrame([[1.0, 1.0, 1.0]], index=idx, columns=["A", "B", "C"])
    group = pd.DataFrame([["X", "X", "Y"]], index=idx, columns=["A", "B", "C"])
    out = OperatorRegistry.get("relation_peer_weighted_mean_ex_self").calculate(value, weight, group)
    # A peers: only B -> 2.0 ; B peers: only A -> 1.0 ; C has no peer -> NaN
    assert out.iloc[0, 0] == pytest.approx(2.0)
    assert out.iloc[0, 1] == pytest.approx(1.0)
    assert np.isnan(out.iloc[0, 2])


def test_index_entry_exit_event_marks_transitions() -> None:
    idx = pd.date_range("2024-01-01", periods=4)
    member = pd.DataFrame([[0.0], [1.0], [1.0], [0.0]], index=idx, columns=["A"])
    out = OperatorRegistry.get("index_entry_exit_event").calculate(member)
    assert out.iloc[0, 0] == pytest.approx(0.0)
    assert out.iloc[1, 0] == pytest.approx(1.0)
    assert out.iloc[2, 0] == pytest.approx(0.0)
    assert out.iloc[3, 0] == pytest.approx(-1.0)


def test_index_membership_age_counts_since_entry() -> None:
    idx = pd.date_range("2024-01-01", periods=4)
    member = pd.DataFrame([[0.0], [1.0], [1.0], [1.0]], index=idx, columns=["A"])
    out = OperatorRegistry.get("index_membership_age").calculate(member)
    assert np.isnan(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(0.0)
    assert out.iloc[2, 0] == pytest.approx(1.0)
    assert out.iloc[3, 0] == pytest.approx(2.0)


def test_event_cumulative_return_past_is_causal_and_masked() -> None:
    idx = pd.date_range("2024-01-01", periods=4)
    ret = pd.DataFrame([[1.0], [2.0], [3.0], [4.0]], index=idx, columns=["A"])
    event = pd.DataFrame([[0.0], [1.0], [0.0], [1.0]], index=idx, columns=["A"])
    op = OperatorRegistry.get("event_cumulative_return_past")
    # lag=0：事件当日纳入累计 -> row1=2, row2=2+3=5, row3(新事件)=4
    out = op.calculate(ret, event, window=3, event_effective_lag=0)
    assert np.isnan(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(2.0)
    assert out.iloc[2, 0] == pytest.approx(5.0)
    assert out.iloc[3, 0] == pytest.approx(4.0)
    # lag=1（默认）：事件日收益不纳入（避免公告日收盘后时点泄露）-> row1=NaN, row2=3, row3=NaN
    out_lag = op.calculate(ret, event, window=3)
    assert np.isnan(out_lag.iloc[0, 0])
    assert np.isnan(out_lag.iloc[1, 0])
    assert out_lag.iloc[2, 0] == pytest.approx(3.0)
    assert np.isnan(out_lag.iloc[3, 0])


def test_p1_operators_preserve_shape_and_determinism() -> None:
    panel = _panel()
    member = _member(panel)
    grp = _group()
    calls = {
        "relation_hhi": [panel, panel * 0.5, panel * 0.3],
        "relation_entropy": [panel, panel * 0.5, panel * 0.3],
        "relation_topk_sum": [panel, panel * 0.5, panel * 0.3],
        "relation_rank_weighted_sum": [panel, panel * 0.5, panel * 0.3],
        "relation_category_share": [panel, grp],
        "relation_peer_weighted_mean_ex_self": [panel, panel, grp],
        "relation_entry_count": [member],
        "relation_exit_count": [member],
        "relation_weighted_change": [panel, panel],
        "index_member": [member],
        "index_weight_change": [panel],
        "index_entry_exit_event": [member],
        "index_membership_age": [member],
        "event_cumulative_return_past": [panel - 0.1, member],
        "event_abnormal_return_past": [panel - 0.1, panel - 0.2, member],
        "fin_applicability_mask": [panel],
        "calendar_day_diff": [panel, panel + 1],
        "fin_announcement_lag": [panel, panel + 10],
    }
    for name, args in calls.items():
        op = OperatorRegistry.get(name)
        first = op.calculate(*args)
        second = op.calculate(*args)
        assert first.shape == panel.shape, f"{name}: shape changed"
        pd.testing.assert_frame_equal(first, second, check_dtype=False)
