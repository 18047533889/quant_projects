from __future__ import annotations

import pandas as pd

from data_access.cos_runtime import _read_cos_events_asof


def test_empty_decision_frame_does_not_scan_data():
    class NoRead:
        def sql(self, *args, **kwargs):
            raise AssertionError("empty decisions must not read data")

    empty = pd.DataFrame({
        "instrument": [],
        "decision_timestamp": [],
    })
    result = _read_cos_events_asof(
        NoRead(),
        "us_stock_balance",
        empty,
        columns=["value"],
        event_filters={"timeframe": "quarterly"},
    )
    assert result.empty
    assert "fundamental_staleness_days" in result.columns
