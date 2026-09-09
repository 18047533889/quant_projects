"""
信号平滑器：检测 alpha 信号的 topN 换手率，在换手过高时自动触发 EMA 平滑。

工作原理：
1. 计算每期 alpha 的 topN 集合与前一期 topN 集合的重叠率
2. 将重叠率转换为"topN 换手率" = 1 - 重叠率
3. 如果滚动窗口内的平均换手率超过阈值，则对 alpha 做 EMA 平滑
4. 返回平滑后的 alpha 和诊断信息

这是"上游信号平滑与换手门控方案"（参见上游信号平滑与换手门控方案_v0.1.md）
的框架层实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SignalSmoother:
    """信号平滑器：基于 topN 换手率检测，自动决定是否对 alpha 做 EMA 平滑。

    Attributes:
        topn_n: topN 选取数量，默认 50
        turnover_threshold: 换手率阈值，超过此值触发平滑，默认 0.30
        turnover_window: 换手率滚动窗口大小，默认 5
        ema_span: EMA 平滑的 span 参数，默认 5
        mode: 平滑模式——"auto"(自动检测) / "always"(始终平滑) / "never"(不平滑)
    """
    topn_n: int = 50
    turnover_threshold: float = 0.30
    turnover_window: int = 5
    ema_span: int = 5
    mode: str = "auto"

    def _topn_set(self, row: pd.Series) -> set[str]:
        """取一行 alpha 中前 topn_n 大的资产代码集合。

        Args:
            row: 单行 alpha 值

        Returns:
            前 n 大 alpha 值的资产代码集合。
        """
        clean = row.dropna()
        if clean.empty:
            return set()
        n = min(self.topn_n, len(clean))
        return set(clean.nlargest(n).index)

    def compute_topn_turnover(self, alpha: pd.DataFrame) -> pd.Series:
        """计算每期的 topN 换手率。

        topN 换手率 = 1 - |本期 topN ∩ 上期 topN| / min(topn_n, 本期资产数, 上期资产数)

        Args:
            alpha: alpha 信号矩阵（DatetimeIndex）

        Returns:
            Series，index=日期，值=topN 换手率。
        """
        if not isinstance(alpha.index, pd.DatetimeIndex):
            raise ValueError("alpha must use DatetimeIndex")
        turnovers = []
        prev = None
        for dt, row in alpha.iterrows():
            current = self._topn_set(row)
            if prev is None:
                turnovers.append(0.0)
            else:
                overlap = len(current & prev)  # 两期 topN 的交集大小
                denom = max(1, min(self.topn_n, len(current), len(prev)))
                turnovers.append(1.0 - overlap / denom)
            prev = current
        return pd.Series(turnovers, index=alpha.index, name="topn_turnover")

    def should_smooth(self, turnover_series: pd.Series) -> bool:
        """判断当前是否需要触发平滑。

        规则：最近 turnover_window 期平均换手率 > turnover_threshold。

        Args:
            turnover_series: topN 换手率序列

        Returns:
            True 表示需要平滑。
        """
        if turnover_series.empty:
            return False
        window_mean = turnover_series.rolling(self.turnover_window, min_periods=1).mean()
        return bool(window_mean.iloc[-1] > self.turnover_threshold)

    def smooth(self, alpha: pd.DataFrame) -> pd.DataFrame:
        """对 alpha 做 EMA 平滑（指数加权移动平均）。

        Args:
            alpha: 原始 alpha 信号矩阵

        Returns:
            EMA 平滑后的 alpha 矩阵。
        """
        return alpha.ewm(span=self.ema_span, adjust=False, min_periods=1).mean()

    def transform(self, alpha: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """主要的转换入口：检测换手率并决定是否平滑。

        流程：
        1. 计算 topN 换手率序列
        2. 根据 mode 决定是否触发平滑
        3. 返回 (平滑后的 alpha, 诊断 DataFrame)

        Args:
            alpha: 原始 alpha 信号矩阵

        Returns:
            (smoothed_alpha, diagnostics)：
            - smoothed_alpha：平滑后的 alpha（如果不需要平滑则返回副本）
            - diagnostics：含 topn_turnover、turnover_mean、use_ema 三列
        """
        turnover = self.compute_topn_turnover(alpha)
        turnover_mean = turnover.rolling(self.turnover_window, min_periods=1).mean()

        # 逐日决定是否使用 EMA。这里必须保持因果性：
        # t 日的决策只能使用 <=t 的换手诊断，不能用样本末尾的
        # 状态决定是否对整段历史做平滑。
        if self.mode == "always":
            use_ema = pd.Series(True, index=alpha.index, dtype=bool)
        elif self.mode == "never":
            use_ema = pd.Series(False, index=alpha.index, dtype=bool)
        elif self.mode == "auto":
            use_ema = turnover_mean > self.turnover_threshold
        else:
            raise ValueError(f"Unknown smoothing mode: {self.mode}")

        ema = self.smooth(alpha)
        smoothed = alpha.copy()
        smoothed.loc[use_ema] = ema.loc[use_ema]
        diagnostics = pd.DataFrame(
            {
                "topn_turnover": turnover,
                "turnover_mean": turnover_mean,
                "use_ema": use_ema,
            },
            index=alpha.index,
        )
        return smoothed, diagnostics
