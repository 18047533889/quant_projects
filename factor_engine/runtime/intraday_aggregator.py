# -*- coding: utf-8
"""分钟/小时 bar → 日频面板聚合器（Phase 12 生产地基）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from logging_utils import get_logger
from factor_engine.storage.datasource import DataSource

logger = get_logger("factor_engine.runtime.intraday_aggregator")


def _require_multiindex(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} 期望 pd.Series，得到 {type(series).__name__}")
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels < 2:
        raise ValueError(f"{name} 期望 MultiIndex(timestamp, instrument) Series")
    return series


def _to_daily_key(ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """将 intraday timestamp 归到日历日（UTC normalize）。"""
    return pd.DatetimeIndex(ts).normalize()


@dataclass
class IntradayAggregator:
    """已对齐的 intraday 面板列；按 (trade_date, asset) 聚合为日频。"""

    panels: dict[str, pd.Series] = field(default_factory=dict)

    @classmethod
    def from_panel(cls, panels: dict[str, pd.Series]) -> IntradayAggregator:
        """由列名 → MultiIndex Series 字典构造聚合器。"""
        normalized: dict[str, pd.Series] = {}
        for name, series in panels.items():
            normalized[name] = _require_multiindex(series, name)
        if not normalized:
            raise ValueError("IntradayAggregator 需要至少一列面板数据")
        return cls(panels=normalized)

    def _stack_frame(self) -> pd.DataFrame:
        """MultiIndex Series 字典 → 长表 [datetime, asset, *cols]。"""
        frames: list[pd.DataFrame] = []
        for col, series in self.panels.items():
            df = series.reset_index()
            names = list(df.columns)
            df.columns = ["datetime", "asset", col]
            frames.append(df)
        out = frames[0]
        for extra in frames[1:]:
            out = out.merge(extra, on=["datetime", "asset"], how="outer")
        out["trade_date"] = _to_daily_key(pd.DatetimeIndex(out["datetime"]))
        return out

    def last(self, column: str) -> pd.Series:
        """日末值（last bar of day）。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        df = self._stack_frame()
        daily = (
            df.sort_values(["asset", "datetime"])
            .groupby(["trade_date", "asset"], as_index=False)
            .last()
        )
        idx = pd.MultiIndex.from_arrays(
            [daily["trade_date"], daily["asset"]],
            names=["timestamp", "instrument"],
        )
        return pd.Series(daily[column].values, index=idx, name=column)

    def first(self, column: str) -> pd.Series:
        """日初值（first bar of day）。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        df = self._stack_frame()
        daily = (
            df.sort_values(["asset", "datetime"])
            .groupby(["trade_date", "asset"], as_index=False)
            .first()
        )
        idx = pd.MultiIndex.from_arrays(
            [daily["trade_date"], daily["asset"]],
            names=["timestamp", "instrument"],
        )
        return pd.Series(daily[column].values, index=idx, name=column)

    def sum(self, column: str) -> pd.Series:
        """按 (trade_date, asset) 对列求和。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        df = self._stack_frame()
        grouped = df.groupby(["trade_date", "asset"])[column].sum()
        grouped.index.names = ["timestamp", "instrument"]
        return grouped.rename(column)

    def mean(self, column: str) -> pd.Series:
        """按 (trade_date, asset) 对列求均值。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        df = self._stack_frame()
        grouped = df.groupby(["trade_date", "asset"])[column].mean()
        grouped.index.names = ["timestamp", "instrument"]
        return grouped.rename(column)

    def vwap(
        self,
        *,
        amount: str = "amount",
        volume: str = "volume",
        price: str = "close",
    ) -> pd.Series:
        """成交额加权均价：优先 ``sum(Amount) / sum(Volume)``。

        ``price`` is retained for backward compatibility with callers whose
        minute source has no amount column.  Production multi-minute bars
        should always provide amount because close×volume is not bar VWAP.
        """
        if volume not in self.panels:
            raise KeyError(f"vwap 需要 {volume!r} 列")
        df = self._stack_frame()
        if amount in self.panels:
            numerator = df.groupby(["trade_date", "asset"])[amount].sum()
        elif price in self.panels:
            numerator = (df[price] * df[volume]).groupby(
                [df["trade_date"], df["asset"]]
            ).sum()
        else:
            raise KeyError(f"vwap 需要 {amount!r} 或 {price!r} 列")
        denominator = df.groupby(["trade_date", "asset"])[volume].sum()
        vwap = (numerator / denominator.replace(0, pd.NA)).rename("vwap")
        vwap.index.names = ["timestamp", "instrument"]
        return vwap

    def aggregate_features(
        self,
        *,
        features: tuple[str, ...] = ("last_close", "sum_volume", "vwap"),
    ) -> dict[str, pd.Series]:
        """批量产出常用日频特征。"""
        out: dict[str, pd.Series] = {}
        for feat in features:
            if feat == "last_close":
                out["close"] = self.last("close")
            elif feat == "first_open":
                out["open"] = self.first("open")
            elif feat == "sum_volume":
                out["volume"] = self.sum("volume")
            elif feat == "vwap":
                out["vwap"] = self.vwap()
            elif feat == "mean_close":
                out["close_mean"] = self.mean("close")
            else:
                raise ValueError(f"未知 intraday 日频特征: {feat!r}")
        return out


@dataclass
class IntradayAggregatedDataSource(DataSource):
    """包装 intraday 数据源，对外暴露日频聚合列。"""

    inner: DataSource
    features: tuple[str, ...] = ("last_close", "sum_volume", "vwap")
    bar_freq: str = "1d"
    _cache: dict[str, pd.Series] = field(default_factory=dict, init=False, repr=False)

    def execution_spec(self) -> dict[str, Any] | None:
        """#收官轮 P0：还原 intraday_daily 子源 + features 可重建配置。"""
        from factor_engine.storage.sources.datasource import clean_execution_spec

        fn = getattr(self.inner, "execution_spec", None)
        inner_spec = fn() if callable(fn) else None
        if not isinstance(inner_spec, dict):
            return None
        return clean_execution_spec(
            {
                "type": "intraday_daily",
                "source": dict(inner_spec),
                "features": tuple(self.features),
            }
        )

    def _build_aggregator(self) -> IntradayAggregator:
        load_columns = getattr(self.inner, "load_columns", None)
        base_cols = ["open", "high", "low", "close", "volume", "amount"]
        if callable(load_columns):
            try:
                panels = load_columns(base_cols)
            except (FileNotFoundError, KeyError, NotImplementedError):
                panels = {}
                for col in base_cols:
                    try:
                        panels[col] = self.inner.load_column(col)
                    except (FileNotFoundError, KeyError, NotImplementedError):
                        continue
        else:
            panels = {}
            missing: list[str] = []
            for col in base_cols:
                try:
                    panels[col] = self.inner.load_column(col)
                except (FileNotFoundError, KeyError, NotImplementedError) as exc:
                    missing.append(col)
                    logger.debug("intraday 列 %s 不可用: %s", col, exc)
            if missing:
                logger.warning(
                    "IntradayAggregatedDataSource 缺少列 %s（将跳过）；已加载 %s",
                    missing,
                    sorted(panels.keys()),
                )
        if "close" not in panels:
            raise ValueError("intraday 数据源至少需要 close 列")
        return IntradayAggregator.from_panel(panels)

    def load_column(self, name: str) -> Any:
        """加载日频聚合列（首次访问时批量计算并缓存）。"""
        if name in self._cache:
            return self._cache[name]
        agg = self._build_aggregator()
        features = agg.aggregate_features(features=self.features)
        self._cache.update(features)
        if name not in self._cache:
            raise KeyError(
                f"IntradayAggregatedDataSource 无列 {name!r}；"
                f"可用: {sorted(features.keys())}"
            )
        return self._cache[name]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        """批量加载多个日频聚合列。"""
        return {n: self.load_column(n) for n in names}
