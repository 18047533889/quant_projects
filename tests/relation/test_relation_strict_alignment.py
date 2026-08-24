# -*- coding: utf-8 -*-
"""R24-004/005/006/007: every multi-panel relation operator fails closed on
misaligned axes.  A shuffled instrument column order, a mismatched index, or a
reindexable subset must RAISE — never silently pair positions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.relation.ops import strict_relation_align

ensure_cleaned_loaded()

MULTI_PANEL = {
    "relation_category_share": (["value", "category"], {}),
    "relation_category_signed_contribution": (["value", "category"], {}),
    "relation_peer_weighted_mean_ex_self": (["value", "weight", "group"], {}),
    "relation_weighted_change": (["value", "weight"], {}),
    "event_cumulative_return_past": (["ret", "event"], {"window": 5}),
    "event_abnormal_return_past": (["ret", "benchmark_ret", "event"], {"window": 5}),
    "relation_overlap_ratio": (["current_ids", "previous_ids"], {}),
    "event_return_since_last": (["ret", "event"], {"window": 5}),
    "relation_hhi": (["s1", "s2", "s3"], {}),
    "holder_observed_topk_hhi": (["s1", "s2", "s3"], {}),
    "holder_company_ownership_hhi": (["s1", "s2", "s3"], {}),
}


def _fixture(rows: int = 5, cols: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=rows)
    cols_names = [f"S{i}" for i in range(cols)]
    panels = []
    for _ in range(cols):
        panels.append(pd.DataFrame(rng.random((rows, cols)) + 0.1, index=idx, columns=cols_names))
    group = pd.DataFrame(
        np.resize(np.array(["A", "B"]), (rows, cols)), index=idx, columns=cols_names
    )
    return panels, group, idx, cols_names


def test_strict_relation_align_accepts_identical() -> None:
    panels, _, idx, cols_names = _fixture()
    assert strict_relation_align(*panels[:2]) is not None


def test_strict_relation_align_rejects_shuffled_columns() -> None:
    panels, _, idx, cols_names = _fixture()
    shuffled = panels[1][list(reversed(cols_names))]
    with pytest.raises(ValueError, match="columns"):
        strict_relation_align(panels[0], shuffled)


def test_strict_relation_align_rejects_mismatched_index() -> None:
    panels, _, idx, cols_names = _fixture()
    wrong_idx = panels[1].copy()
    wrong_idx.index = idx + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="index"):
        strict_relation_align(panels[0], wrong_idx)


def test_strict_relation_align_rejects_duplicate_columns() -> None:
    panels, _, idx, cols_names = _fixture()
    # Both panels carry the SAME duplicated column labels: equality passes but
    # the unique-columns gate must reject the ambiguous axis.
    dup_a = panels[0].copy()
    dup_b = panels[1].copy()
    dup_a.columns = dup_b.columns = ["S0", "S0", "S0"]
    with pytest.raises(ValueError, match="not unique"):
        strict_relation_align(dup_a, dup_b)


def test_strict_relation_align_rejects_non_dataframe() -> None:
    panels, _, idx, cols_names = _fixture()
    with pytest.raises(TypeError, match="DataFrame"):
        strict_relation_align(panels[0], panels[1].to_numpy())


def test_relation_category_share_shuffled_group_raises() -> None:
    # R24-007 mutation: random column shuffle of the secondary panel MUST raise.
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    value = pd.DataFrame(
        np.arange(1.0, 13.0).reshape(4, 3), index=pd.date_range("2024-01-01", periods=4),
        columns=["A", "B", "C"],
    )
    group = pd.DataFrame(
        [["X", "Y", "X"], ["X", "Y", "X"], ["X", "Y", "X"], ["X", "Y", "X"]],
        index=value.index, columns=["A", "B", "C"],
    )
    op = OperatorRegistry.get("relation_category_share")
    assert op is not None
    op.calculate(value, group)  # aligned OK
    with pytest.raises(ValueError, match="columns"):
        op.calculate(value, group[["C", "A", "B"]])  # shuffled -> raise


def test_event_abnormal_return_positional_mismatch_raises() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=8)
    ret = pd.DataFrame(rng.random((8, 3)) - 0.5, index=idx, columns=["A", "B", "C"])
    bench = ret.copy()
    event = pd.DataFrame(np.zeros((8, 3)), index=idx, columns=["A", "B", "C"])
    op = OperatorRegistry.get("event_abnormal_return_past")
    op.calculate(ret, bench, event, window=5)
    # The benchmark panel is paired positionally with ret — a column shuffle is
    # a semantic corruption and must raise, not silently re-pair.
    with pytest.raises(ValueError, match="columns"):
        op.calculate(ret, bench[["C", "B", "A"]], event, window=5)


@pytest.mark.parametrize("canon", sorted(MULTI_PANEL))
def test_operator_registered(canon: str) -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.get(canon) is not None, canon
