# -*- coding: utf-8 -*-
"""R24-030..035: relation snapshots are bitemporal (no full-sample keep-last),
and snapshot features carry age/vintage metadata with a bounded-staleness gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from storage.sources.relation import (
    aggregate_holder_rows,
    relation_snapshot_change,
    top_ten_features_asof,
)


def test_old_snapshot_revision_not_leaked_at_earlier_decision_time() -> None:
    # R24-030..032: a future revision of an old snapshot is invisible to a
    # historical replay (a global keep=last would show it).
    rows = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "snapshot_id": [1, 1, 2],
        "available_at": pd.to_datetime(["2024-01-01", "2024-05-10", "2024-03-01"]),
        "concentration": [0.40, 0.42, 0.45],
        "revision_id": [1, 2, 1],
    })
    # At 2024-02-01 snapshot 1 revision 1 (0.40) is the visible state.
    old = relation_snapshot_change(
        rows, value_column="concentration", decision_time="2024-02-01",
        revision_column="revision_id",
    )
    assert old.loc[old["snapshot_id"] == 1, "concentration"].tolist() == [0.40]
    # Snapshot 2 is NOT visible yet.
    assert (old["snapshot_id"] == 2).sum() == 0
    # At a later decision time the revised snapshot-1 value is visible.
    late = relation_snapshot_change(
        rows, value_column="concentration", decision_time="2024-06-01",
        revision_column="revision_id",
    )
    assert 0.42 in late["concentration"].tolist()


def test_top_ten_features_require_staleness_gate() -> None:
    rows = pd.DataFrame({
        "instrument": ["A"] * 3,
        "period_end": pd.to_datetime(["2024-03-31"] * 3),
        "available_at": pd.to_datetime(["2024-04-30"] * 3),
        "holding_amount": [10.0, 9.0, 8.0],
        "holding_ratio": [0.10, 0.09, 0.08],
    })
    decisions = pd.DataFrame({"decision_timestamp": pd.to_datetime(["2024-06-01"]), "instrument": ["A"]})
    with pytest.raises(ValueError, match="max_age_days"):
        top_ten_features_asof(decisions, rows, ratio_unit="decimal")
    # Explicit bounded staleness is fine.
    out = top_ten_features_asof(decisions, rows, ratio_unit="decimal", max_age_days=90)
    assert out.loc[0, "snapshot_age_days"] <= 90
    # Explicit unbounded opt-in is fine too.
    out2 = top_ten_features_asof(
        decisions, rows, ratio_unit="decimal", allow_unbounded_staleness=True
    )
    assert "snapshot_available_at" in out2.columns


def test_official_rank_used_for_top_ten_selection() -> None:
    # R24-027..029: official ShareholderRank drives the TopTen selection.
    rows = pd.DataFrame({
        "instrument": ["A"] * 4,
        "period_end": pd.to_datetime(["2024-03-31"] * 4),
        "available_at": pd.to_datetime(["2024-04-30"] * 4),
        "holding_amount": [1.0, 2.0, 3.0, 4.0],
        "holding_ratio": [0.10, 0.20, 0.30, 0.40],
        "ShareholderRank": [1, 2, 3, 4],
    })
    agg = aggregate_holder_rows(rows, ratio_unit="decimal", top_n=2, rank_column="ShareholderRank")
    assert agg.loc[0, "top_ten_holder_count"] == 2
    assert np.isclose(agg.loc[0, "top_ten_holding_ratio"], 0.10 + 0.20)


def test_entity_name_fallback_requires_opt_in() -> None:
    rows = pd.DataFrame({
        "instrument": ["A"] * 2,
        "period_end": pd.to_datetime(["2024-03-31"] * 2),
        "available_at": pd.to_datetime(["2024-04-30"] * 2),
        "holding_amount": [10.0, 20.0],
        "holding_ratio": [0.10, 0.20],
        "holder_name": ["同一名字", "同一名字"],
    })
    # R24-025: display-name fallback requires explicit opt-in.
    with pytest.raises(ValueError, match="allow_display_name_fallback"):
        aggregate_holder_rows(rows, ratio_unit="decimal", entity_name_column="holder_name")
    # With the explicit opt-in (static same-snapshot aggregation) it is allowed.
    agg = aggregate_holder_rows(
        rows, ratio_unit="decimal", entity_name_column="holder_name",
        allow_display_name_fallback=True,
    )
    assert agg.loc[0, "top_ten_holder_count"] == 1  # merged by the shared name


def test_entity_id_never_merged_with_same_name() -> None:
    # R24-026: identical display names with distinct entity ids must NOT merge.
    rows = pd.DataFrame({
        "instrument": ["A"] * 2,
        "period_end": pd.to_datetime(["2024-03-31"] * 2),
        "available_at": pd.to_datetime(["2024-04-30"] * 2),
        "holding_amount": [10.0, 20.0],
        "holding_ratio": [0.10, 0.20],
        "holder_name": ["同名", "同名"],
        "holder_entity_id": ["E1", "E2"],
    })
    agg = aggregate_holder_rows(rows, ratio_unit="decimal", entity_id_column="holder_entity_id")
    assert agg.loc[0, "top_ten_holder_count"] == 2  # distinct entities preserved
