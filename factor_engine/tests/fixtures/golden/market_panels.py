"""双市场 golden panel：A 股 Symbol / 美股 Ticker 约定。"""

from __future__ import annotations

import pandas as pd

ASHARE_GOLDEN_DATES = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
ASHARE_GOLDEN_SYMBOLS = ["000001.SZ", "000002.SZ"]

US_GOLDEN_DATES = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
US_GOLDEN_TICKERS = ["AAPL", "MSFT"]


def build_ashare_golden_close_panel() -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [ASHARE_GOLDEN_DATES, ASHARE_GOLDEN_SYMBOLS],
        names=["timestamp", "instrument"],
    )
    values = [10.0, 20.0, 11.0, 21.0, 12.0, 22.0]
    return pd.Series(values, index=idx, name="close")


def build_us_golden_close_panel() -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [US_GOLDEN_DATES, US_GOLDEN_TICKERS],
        names=["timestamp", "instrument"],
    )
    values = [100.0, 200.0, 101.0, 201.0, 102.0, 202.0]
    return pd.Series(values, index=idx, name="close")


def expected_rank(panel: pd.Series) -> pd.Series:
    from cleaned_operators.common.cs_broadcast import cs_rank_01

    wide = panel.unstack("instrument")
    ranked = cs_rank_01(wide)
    return ranked.stack(future_stack=True)
