# -*- coding: utf-8 -*-
"""R24-046..048: availability precision is a DECLARED source-metadata attribute.
It is never inferred by inspecting data values (e.g. ``time == midnight``)."""
from __future__ import annotations

import pandas as pd
import pytest

from pit_contract import AvailabilityPrecision, pit_asof_join


def _events(*rows):
    import numpy as np

    out = {
        "instrument": ["A"] * len(rows),
        "period_end": pd.to_datetime([r[0] for r in rows]),
        "available_at": pd.to_datetime([r[1] for r in rows]),
        "value": [float(r[2]) for r in rows],
    }
    return pd.DataFrame(out)


def _decisions(*dates):
    return pd.DataFrame({"decision_timestamp": pd.to_datetime(dates), "instrument": ["A"] * len(dates)})


def test_enum_members_declared() -> None:
    names = {p.value for p in AvailabilityPrecision}
    assert {
        "date", "timestamp_second", "timestamp_millisecond",
        "timestamp_nanosecond", "session_label", "unknown",
    } <= names
    assert AvailabilityPrecision.is_timestamp("timestamp_second")
    assert AvailabilityPrecision.is_timestamp(AvailabilityPrecision.TIMESTAMP_NANOSECOND)
    assert not AvailabilityPrecision.is_timestamp("date")
    assert not AvailabilityPrecision.is_timestamp(None)


def test_same_day_production_rejects_declared_date_precision() -> None:
    events = _events(("2024-03-31", "2024-04-30", 100.0))
    with pytest.raises(ValueError, match="TIMESTAMP-precision"):
        pit_asof_join(
            _decisions("2024-04-30"), events,
            available_policy="same_day", production=True,
            precision=AvailabilityPrecision.DATE,
        )


def test_same_day_production_accepts_declared_timestamp_precision() -> None:
    events = _events(("2024-03-31", "2024-04-30 15:30:00", 100.0))
    out = pit_asof_join(
        _decisions("2024-05-01"), events,
        available_policy="same_day", production=True,
        precision=AvailabilityPrecision.TIMESTAMP_NANOSECOND,
    )
    assert out["value"].iloc[0] == 100.0


def test_research_no_precision_uses_documented_fallback() -> None:
    # Research with no declared precision falls back (warned), not fatal.
    events = _events(("2024-03-31", "2024-04-30 15:30:00", 100.0))
    out = pit_asof_join(
        _decisions("2024-05-01"), events,
        available_policy="same_day", production=False,
    )
    assert out["value"].iloc[0] == 100.0
