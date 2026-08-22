from __future__ import annotations

import numpy as np
import pandas as pd

from storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource


def test_benchmark_daily_broadcast_is_exact_not_stale_asof() -> None:
    anchor = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-07-30", "2026-07-31"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    benchmark_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-07-30"), "SPX")],
        names=["timestamp", "instrument"],
    )
    benchmark = pd.Series([0.012], index=benchmark_index, name="ret")

    out = LQTPLogicalDataSource._broadcast_exact_by_date(anchor, benchmark)

    assert np.isclose(float(out.loc[(pd.Timestamp("2026-07-30"), "A")]), 0.012)
    assert np.isnan(out.loc[(pd.Timestamp("2026-07-31"), "A")])
    assert np.isnan(out.loc[(pd.Timestamp("2026-07-31"), "B")])


def test_benchmark_daily_broadcast_normalizes_intraday_anchor_date() -> None:
    anchor = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-07-30 10:15:00"), "A"),
            (pd.Timestamp("2026-07-30 14:30:00"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    benchmark = pd.Series(
        [0.012],
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-07-30"), "SPX")],
            names=["timestamp", "instrument"],
        ),
    )
    out = LQTPLogicalDataSource._broadcast_exact_by_date(anchor, benchmark)
    assert out.tolist() == [0.012, 0.012]


def test_minute_sessions_never_collide_across_lunch_boundary() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-07-31 11:29:00", "2026-07-31 13:00:00"]
            ),
            "instrument": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )

    slotted = LQTPLogicalDataSource._session_slots(frame)

    assert list(slotted["session"]) == ["am", "pm"]
    # A session-local slot can numerically repeat only when the session label is
    # different; the grouping key includes session and therefore cannot collide.
    assert tuple(slotted.iloc[0][["session", "session_slot"]]) != tuple(
        slotted.iloc[1][["session", "session_slot"]]
    )
    assert int(slotted.iloc[1]["session_slot"]) == 0


def test_exact_daily_alignment_does_not_forward_fill_missing_date() -> None:
    anchor = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-07-30"), "A"),
            (pd.Timestamp("2026-07-31"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    source = pd.Series(
        [10.0],
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-07-30"), "A")],
            names=["timestamp", "instrument"],
        ),
    )

    out = LQTPLogicalDataSource._align_exact_by_instrument(anchor, source)

    assert float(out.iloc[0]) == 10.0
    assert np.isnan(out.iloc[1])
