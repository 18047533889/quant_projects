from __future__ import annotations

import pandas as pd


def test_intraday_anchor_is_detected_without_external_frequency_flag() -> None:
    from storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    class DummyInner:
        pass

    class IntradaySource(LQTPLogicalDataSource):
        def _anchor_index(self):
            return pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-02 09:35:00"), "A"),
                    (pd.Timestamp("2024-01-02 09:40:00"), "A"),
                ],
                names=["timestamp", "instrument"],
            )

    class DailySource(LQTPLogicalDataSource):
        def _anchor_index(self):
            return pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-02"), "A"),
                    (pd.Timestamp("2024-01-03"), "A"),
                ],
                names=["timestamp", "instrument"],
            )

    assert IntradaySource(DummyInner())._anchor_is_intraday() is True
    assert DailySource(DummyInner())._anchor_is_intraday() is False
