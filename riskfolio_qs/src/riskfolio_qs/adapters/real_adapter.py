"""
生产环境输入适配器：从对齐版 COS parquet 读取真实美股权数据。

数据源对照（对齐版代码验证文档 v4.0）：
- market：StockDailyBar.Close → (date × ticker) pivot
- tradable：StockList（type=CS 且未退市）→ 0/1 矩阵
- F_mcap：StockIndicator.market_cap → (date × ticker) 市值
- benchmark：StockIndicesComponents + StockIndicator → 市值加权基准权重
- G：StockIndustry → 行业分类（美股当前为空表，A 股有数据）

使用方法：
    adapter = RealInputAdapter(
        alpha_df=your_alpha,
        data_root="/data/clean_data/us_stock/massive_data",
        start_date="2026-01-01",
        end_date="2026-06-30",
    )
    bundle = adapter.build_bundle()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import exchange_calendars as xcals

from ..core.contracts import InputBundle, OptimizationContext
from .input_adapter import InputAdapter


@dataclass
class RealInputAdapter(InputAdapter):
    """生产环境输入适配器：读取对齐版 parquet 数据，组装 InputBundle。

    支持三种模式：
    - 传 alpha_df / prev_positions_df：直接注入（推荐，alpha 来自外部）
    - 不传：对应字段为 None，依赖管线 fallback

    Attributes:
        alpha_df: 外部 alpha 信号 DataFrame (date × ticker)
        prev_positions_df: 上期持仓 DataFrame (date × ticker)，默认全零
        data_root: 对齐版数据根目录
        start_date: 起始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        benchmark_index: 基准指数名，默认 "S&P 500"
    """
    # ---- 外部注入 ----
    alpha_df: pd.DataFrame | None = None
    prev_positions_df: pd.DataFrame | None = None

    # ---- 数据路径 ----
    data_root: str = "/data/clean_data/us_stock/massive_data"

    # ---- 日期范围 ----
    start_date: str = "2026-01-01"
    end_date: str = "2026-06-30"

    # ---- 基准配置 ----
    benchmark_index: str = "S&P 500"
    calendar_name: str = "XNYS"
    default_linear_cost_bps: float = 5.0

    # ---- 内部缓存 ----
    _dates: pd.DatetimeIndex = field(init=False, repr=False)
    _ticker_universe: list[str] = field(init=False, repr=False)
    _market_cache: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        calendar = xcals.get_calendar(self.calendar_name)
        sessions = calendar.sessions_in_range(self.start_date, self.end_date)
        if sessions.tz is not None:
            sessions = sessions.tz_localize(None)
        self._dates = pd.DatetimeIndex(sessions).normalize()
        self._ticker_universe = []

    # ================================================================
    # 数据加载
    # ================================================================

    def load_alpha(self, *args, **kwargs) -> pd.DataFrame:
        """外部 alpha 信号，若未注入则返回空 DataFrame。"""
        if self.alpha_df is not None:
            alpha = self.alpha_df.copy()
            index = pd.DatetimeIndex(alpha.index)
            if index.tz is not None:
                index = index.tz_localize(None)
            alpha.index = index.normalize()
            return alpha.reindex(index=self._dates)
        # 回退：空 alpha
        return pd.DataFrame(index=self._dates)

    def load_market(self, *args, **kwargs) -> pd.DataFrame:
        """从 StockDailyBar 读取收盘价，pivot 为 (date × ticker)。

        StockDailyBar 格式：每个日期一个 parquet，列含 Ticker/Close。
        """
        if self._market_cache is not None:
            return self._market_cache

        frames: list[pd.DataFrame] = []
        base = Path(self.data_root) / "StockDailyBar"

        for d in self._dates:
            fpath = base / f"{d.strftime('%Y-%m-%d')}.parquet"
            if not fpath.exists():
                continue
            df = pd.read_parquet(fpath, columns=["Ticker", "Close"])
            df["date"] = d
            frames.append(df)

        if not frames:
            return pd.DataFrame(index=self._dates)

        long = pd.concat(frames, ignore_index=True)
        wide = long.pivot(index="date", columns="Ticker", values="Close")
        wide = wide.reindex(index=self._dates).sort_index(axis=1)
        self._market_cache = wide
        return wide

    def load_industry(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """行业分类（从 StockIndustry 读取）。

        美股：StockIndustry 当前为空表，返回 None。
        A 股：逐日 parquet，取 IndustryCode 做 pivot。
        """
        base = Path(self.data_root) / "StockIndustry"
        frames: list[pd.DataFrame] = []

        for d in self._dates:
            fpath = base / f"{d.strftime('%Y-%m-%d')}.parquet"
            if not fpath.exists():
                continue
            df = pd.read_parquet(fpath, columns=["Symbol", "IndustryCode"])
            df["date"] = d
            frames.append(df)

        if not frames:
            return None

        long = pd.concat(frames, ignore_index=True)
        wide = long.pivot(index="date", columns="Symbol", values="IndustryCode")
        wide = wide.reindex(index=self._dates).sort_index(axis=1)
        if self.alpha_df is not None:
            wide = wide.reindex(columns=self.alpha_df.columns, fill_value=-1)
        return wide

    def load_style(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """风格因子暴露：Barra 外采数据，当前未对接，返回 None。"""
        return None

    def load_benchmark(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """从 StockIndicesComponents + StockIndicator 构建基准权重。

        流程：
        1. 从 StockIndicesComponents 读 S&P500 成分股
        2. 从 StockIndicator 读成分股的 market_cap
        3. 按市值加权计算每只成分股的基准权重
        """
        idx_base = Path(self.data_root) / "StockIndicesComponents"
        ind_base = Path(self.data_root) / "StockIndicator"

        weights: dict = {}
        for d in self._dates:
            idx_path = idx_base / f"{d.strftime('%Y-%m-%d')}.parquet"
            if not idx_path.exists():
                continue
            idx_df = pd.read_parquet(idx_path)
            idx_df = idx_df[idx_df["IndexName"] == self.benchmark_index]
            symbols = idx_df["Symbol"].tolist()
            if not symbols:
                continue

            # 取市值
            ind_path = ind_base / f"{d.strftime('%Y-%m-%d')}.parquet"
            if not ind_path.exists():
                continue
            ind_df = pd.read_parquet(ind_path, columns=["ticker", "market_cap"])
            ind_df = ind_df[ind_df["ticker"].isin(symbols)]
            mcap = ind_df.set_index("ticker")["market_cap"].dropna()
            mcap = mcap.clip(lower=0)

            if mcap.sum() <= 0:
                # 等权回退
                w = pd.Series(1.0 / len(symbols), index=symbols)
            else:
                w = mcap / mcap.sum()
            weights[d] = w

        if not weights:
            return None

        bm = pd.DataFrame(weights).T
        bm.index.name = "date"
        bm = bm.reindex(index=self._dates).sort_index(axis=1)
        # 有效基准日内，非成分股为 0；整日缺数据则保留 NaN。
        observed_dates = bm.notna().any(axis=1)
        bm.loc[observed_dates] = bm.loc[observed_dates].fillna(0.0)
        if self.alpha_df is not None:
            bm = bm.reindex(columns=self.alpha_df.columns)
            bm.loc[observed_dates] = bm.loc[observed_dates].fillna(0.0)
        return bm

    def load_prev_positions(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """上期持仓：外部注入或默认全零。"""
        if self.prev_positions_df is not None:
            prev = self.prev_positions_df.reindex(index=self._dates)
            if self.alpha_df is not None:
                prev = prev.reindex(columns=self.alpha_df.columns)
            return prev
        # 默认空仓起始，用 alpha 的列
        cols = list(self.alpha_df.columns) if self.alpha_df is not None else []
        return pd.DataFrame(0.0, index=self._dates, columns=cols)

    def load_tradable_flags(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """从 StockList 读取可交易股票集合（普通股且未退市）。

        返回 (date × ticker) 的 0/1 矩阵。
        """
        base = Path(self.data_root) / "StockList"
        flags: dict = {}

        for d in self._dates:
            fpath = base / f"{d.strftime('%Y-%m-%d')}.parquet"
            if not fpath.exists():
                continue
            df = pd.read_parquet(fpath, columns=["Symbol", "type", "delisted_utc"])
            # 仅普通股且未退市
            tradable = df[
                (df["type"] == "CS") & (df["delisted_utc"].isna())
            ]["Symbol"].tolist()
            flags[d] = {s: 1 for s in tradable}

        if not flags:
            return None

        fm = pd.DataFrame(flags).T.fillna(0).astype(bool)
        fm.index.name = "date"
        fm = fm.reindex(index=self._dates).sort_index(axis=1)
        if self.alpha_df is not None:
            fm = fm.reindex(columns=self.alpha_df.columns, fill_value=False)
        return fm

    # ---- Barra 外采（未对接） ----

    def load_F(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """因子暴露 F：Barra 外采数据，当前未对接。"""
        return None

    def load_G(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """行业暴露 G：美股用 StockIndustry（空表），A 股可复用 load_industry。"""
        return self.load_industry()

    def load_F_mcap(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """市值暴露 F_mcap：从 StockIndicator.market_cap 读取并 pivot。"""
        base = Path(self.data_root) / "StockIndicator"
        frames: list[pd.DataFrame] = []

        for d in self._dates:
            fpath = base / f"{d.strftime('%Y-%m-%d')}.parquet"
            if not fpath.exists():
                continue
            df = pd.read_parquet(fpath, columns=["ticker", "market_cap"])
            df["date"] = d
            frames.append(df)

        if not frames:
            return None

        long = pd.concat(frames, ignore_index=True)
        wide = long.pivot(index="date", columns="ticker", values="market_cap")
        wide = wide.reindex(index=self._dates).sort_index(axis=1)
        if self.alpha_df is not None:
            wide = wide.reindex(columns=self.alpha_df.columns)
        return wide

    def load_linear_cost_bps(self) -> pd.DataFrame:
        """默认单边线性成本；可由调用方替换为逐日逐资产估计。"""
        columns = list(self.alpha_df.columns) if self.alpha_df is not None else []
        return pd.DataFrame(
            self.default_linear_cost_bps,
            index=self._dates,
            columns=columns,
            dtype=float,
        )

    def load_F_ret(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """因子收益 F_ret：Barra 外采数据，当前未对接。"""
        return None

    def load_F_spec(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """特质收益 F_spec：Barra 外采数据，当前未对接。"""
        return None

    # ================================================================
    # 组装
    # ================================================================

    def build_bundle(self, *args, **kwargs) -> InputBundle:
        """加载所有数据并组装 InputBundle。

        Returns:
            InputBundle，含 alpha/market/tradable/F_mcap/benchmark/G/prev_positions。
            Barra 三件套（F/F_ret/F_spec）当前为 None。
        """
        alpha = self.load_alpha()
        market = self.load_market()

        # 以 alpha 列为基准，统一所有字段的列
        ref_cols = list(alpha.columns)

        def _align(df, fill=0.0):
            if df is None or ref_cols is None:
                return df
            return df.reindex(columns=ref_cols, fill_value=fill)

        return InputBundle(
            alpha=alpha,
            market=_align(market),
            # F 的标准形状是 MultiIndex(date, asset) x factor，不能按
            # alpha 的资产列做普通 reindex。
            F=self.load_F(),                        # Barra 外采：暂缺
            G=_align(self.load_G(), fill=-1),       # 行业暴露
            F_mcap=_align(self.load_F_mcap()),      # 市值暴露
            F_ret=self.load_F_ret(),                # date x factor，不按资产列对齐
            F_spec=_align(self.load_F_spec()),      # Barra 外采：暂缺
            industry=_align(self.load_industry(), fill=-1),
            style=_align(self.load_style()),
            benchmark=_align(self.load_benchmark()),
            prev_positions=_align(self.load_prev_positions()),
            tradable=self.load_tradable_flags(),    # bool 类型不做 fill_value
            linear_cost_bps=_align(self.load_linear_cost_bps()),
            metadata=OptimizationContext(
                alpha_is_absolute_return=False,
                benchmark_name=self.benchmark_index,
                calendar_name=self.calendar_name,
                decision_time="close",
                execution_time="next_open",
                market_data_lag_periods=0,
                exposure_data_lag_periods=0,
                alpha_input_type="score",
                alpha_horizon_days=1,
            ),
        )

    # ================================================================
    # 辅助
    # ================================================================

    def _load_ticker_universe(self) -> list[str]:
        """从 StockList 或市场数据推导股票池。"""
        if self._ticker_universe:
            return self._ticker_universe
        try:
            market = self.load_market()
            return list(market.columns)
        except Exception:
            return []
