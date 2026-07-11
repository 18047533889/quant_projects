"""用于 GTJA-191 全 catalog 编译/执行门禁的确定性合成数据源。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class SyntheticGTJADataSource:
    """实现 FactorEngine DataSource 热路径的内存价量面板。"""

    periods: int = 420
    symbols: int = 8
    seed: int = 191

    def __post_init__(self) -> None:
        if self.periods < 300:
            raise ValueError("GTJA synthetic source requires at least 300 periods")
        if self.symbols < 2:
            raise ValueError("GTJA synthetic source requires at least two symbols")
        self.dataset = "synthetic_gtja191"
        self.start_date = "2014-01-01"
        self.end_date = "2015-12-31"
        self.params: dict[str, Any] = {"seed": int(self.seed)}
        self.fields = {
            name: name
            for name in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "amount",
                "ret",
                "preclose",
            )
        }
        self.instrument_filter = None
        self.normalize_timestamp = True
        self.timestamp_unit = None
        self.read_auto = False
        self._lazy_scan = False
        self._data_snapshot_id = f"synthetic-gtja191-{self.periods}-{self.symbols}-{self.seed}"
        self._panels = self._build_panels()
        self._series: dict[str, pd.Series] = {}

    @property
    def data_snapshot_id(self) -> str:
        return self._data_snapshot_id

    def _build_panels(self) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range("2014-01-02", periods=self.periods)
        symbols = [f"S{i:03d}" for i in range(self.symbols)]

        market = rng.normal(0.00035, 0.008, size=(self.periods, 1))
        idio = rng.normal(0.0, 0.012, size=(self.periods, self.symbols))
        returns = market + idio
        close = 50.0 * np.exp(np.cumsum(returns, axis=0))
        preclose = np.vstack([close[0:1], close[:-1]])
        overnight = rng.normal(0.0, 0.004, size=close.shape)
        open_ = preclose * np.exp(overnight)
        spread = np.abs(rng.normal(0.012, 0.006, size=close.shape))
        high = np.maximum(open_, close) * (1.0 + spread)
        low = np.minimum(open_, close) * np.maximum(0.05, 1.0 - spread)
        volume = rng.lognormal(mean=14.0, sigma=0.7, size=close.shape)
        vwap = (open_ + high + low + close) / 4.0
        amount = vwap * volume
        ret = close / preclose - 1.0
        ret[0, :] = np.nan

        for col in range(self.symbols):
            for row in range(37 + col, self.periods, 113):
                close[row, col] = np.nan
                vwap[row, col] = np.nan
            for row in range(71 + col, self.periods, 157):
                volume[row, col] = 0.0
                amount[row, col] = 0.0

        def frame(values: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(values, index=dates, columns=symbols, dtype=float)

        return {
            "open": frame(open_),
            "high": frame(high),
            "low": frame(low),
            "close": frame(close),
            "volume": frame(volume),
            "vwap": frame(vwap),
            "amount": frame(amount),
            "ret": frame(ret),
            "preclose": frame(preclose),
        }

    def load_column_panel(self, name: str) -> pd.DataFrame:
        if name not in self._panels:
            raise KeyError(f"unknown synthetic GTJA column: {name}")
        return self._panels[name]

    def load_column(self, name: str) -> pd.Series:
        if name not in self._series:
            panel = self.load_column_panel(name)
            # pandas 3 的新 stack 实现不允许再显式传 dropna=False，且默认保留 NA 行。
            try:
                series = panel.stack(future_stack=True)
            except TypeError:  # pandas 2.0/2.1 compatibility
                series = panel.stack(dropna=False)
            series.index = series.index.set_names(["timestamp", "instrument"])
            series.name = name
            self._series[name] = series
        return self._series[name]

    def load_columns(self, names: Iterable[str]) -> dict[str, pd.Series]:
        return {name: self.load_column(name) for name in names}

    def prefetch_columns(self, names: Iterable[str]) -> None:
        self.load_columns(names)

    def prefetch_panels(self, names: Iterable[str]) -> None:
        for name in names:
            self.load_column_panel(name)

    def column_cache_stats(self) -> dict[str, int]:
        return {
            "cached_columns": len(self._series),
            "cached_panels": len(self._panels),
        }
