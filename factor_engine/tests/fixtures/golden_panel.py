"""Golden panel 回归基准：手算可验证的小面板。"""

from __future__ import annotations

import pandas as pd

GOLDEN_DATES = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
GOLDEN_INSTRUMENTS = ["A", "B"]


def build_golden_close_panel() -> pd.Series:
    """4 日 × 2 标的 close；用于 ts_mean/delay/rank 手算对照。"""
    idx = pd.MultiIndex.from_product(
        [GOLDEN_DATES, GOLDEN_INSTRUMENTS],
        names=["timestamp", "instrument"],
    )
    values = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
    return pd.Series(values, index=idx, name="close")


def expected_ts_mean(close: pd.Series, window: int) -> pd.Series:
    panel = close.unstack("instrument")
    return panel.rolling(window, min_periods=1).mean().stack(future_stack=True)


def expected_ts_delay(close: pd.Series, lag: int) -> pd.Series:
    panel = close.unstack("instrument")
    return panel.shift(lag).stack(future_stack=True)
