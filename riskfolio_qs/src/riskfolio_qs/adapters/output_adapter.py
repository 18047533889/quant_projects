"""
输出适配器：将优化器产出的原始权重矩阵转换为结构化的 OutputBundle。

职责：
1. 复制并校验目标持仓矩阵（build_target_positions）
2. 计算交易明细表 delta_weight = target - prev（build_trade_table）
3. 计算组合摘要指标（gross_exposure / net_exposure / turnover）（build_portfolio_summary）
4. 序列化审计元数据为单行 DataFrame（build_audit_meta）
5. 组装完整的 OutputBundle（build_output_bundle）
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

import pandas as pd

from ..core.contracts import OutputBundle, RunMetadata


class OutputAdapter:
    """输出适配器：将优化结果转换为标准化的 OutputBundle。"""

    def build_target_positions(self, optimized_weights: pd.DataFrame) -> pd.DataFrame:
        """校验并复制目标持仓权重矩阵。

        Args:
            optimized_weights: 优化器输出的权重矩阵（DatetimeIndex）

        Returns:
            经过校验的权重矩阵副本。

        Raises:
            ValueError: index 不是 DatetimeIndex。
        """
        if not isinstance(optimized_weights.index, pd.DatetimeIndex):
            raise ValueError("optimized_weights must use DatetimeIndex")
        return optimized_weights.copy()

    def build_trade_table(
        self,
        prev_positions: Optional[pd.DataFrame],
        target_positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算交易明细表：delta_weight = target - prev。

        将结果 stack 为 (date, asset) 的 MultiIndex 格式，
        列名为 delta_weight。

        Args:
            prev_positions: 上期持仓权重（None 则视为全零）
            target_positions: 本期目标持仓权重

        Returns:
            MultiIndex(date, asset) 的 DataFrame，列=delta_weight。
        """
        # 批量优化的语义是：prev_positions 只提供首日进入优化器的真实持仓，
        # 此后每日的上期持仓必须是前一日实际求解出的目标持仓。
        # 不能用一张全零 prev_positions 逐日相减，否则会把每日整个
        # 组合都误记为新交易。
        previous = target_positions.shift(1).fillna(0.0)
        if prev_positions is not None and not prev_positions.empty:
            first_date = target_positions.index[0]
            if first_date in prev_positions.index:
                initial = prev_positions.loc[first_date]
            else:
                initial = prev_positions.iloc[0]
            previous.loc[first_date] = initial.reindex(target_positions.columns).fillna(0.0)

        delta = target_positions.fillna(0.0) - previous
        trades = delta.stack(future_stack=True).rename("delta_weight").to_frame()
        trades.index.names = ["date", "asset"]
        return trades

    def build_portfolio_summary(
        self,
        target_positions: pd.DataFrame,
        prev_positions: Optional[pd.DataFrame] = None,
        diagnostics: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """计算组合摘要指标。

        包含四列：
        - gross_exposure：多头总敞口（权重绝对值之和）
        - net_exposure：净敞口（权重之和）
        - turnover：单边换手率 = 0.5 * sum(|w_t - w_{t-1}|)
        - feasible_flag：是否可行（当前始终为 True）

        Args:
            target_positions: 目标持仓权重
            diagnostics: 信号平滑诊断信息（可选）

        Returns:
            每行一个日期的摘要 DataFrame。
        """
        gross = target_positions.abs().sum(axis=1)          # Σ|w_i|
        net = target_positions.sum(axis=1)                   # Σ w_i
        previous = target_positions.shift(1).fillna(0.0)
        if prev_positions is not None and not prev_positions.empty:
            first_date = target_positions.index[0]
            initial = (
                prev_positions.loc[first_date]
                if first_date in prev_positions.index
                else prev_positions.iloc[0]
            )
            previous.loc[first_date] = initial.reindex(target_positions.columns).fillna(0.0)
        turnover = (target_positions.fillna(0.0) - previous).abs().sum(axis=1) / 2.0

        summary = pd.DataFrame(
            {
                "gross_exposure": gross,
                "net_exposure": net,
                "turnover": turnover,
                "feasible_flag": True,
            }
        )
        if diagnostics is not None:
            aligned = diagnostics.reindex(summary.index)
            summary = summary.join(aligned)
            if "max_constraint_violation" in summary:
                violation = summary["max_constraint_violation"].fillna(float("inf"))
                summary["feasible_flag"] = violation <= 1e-5
            if "solve_status" in summary:
                failed = summary["solve_status"].fillna("failed").eq("failed")
                summary["feasible_flag"] &= ~failed
        return summary

    def build_audit_meta(self, metadata: RunMetadata) -> pd.DataFrame:
        """将 RunMetadata 序列化为单行审计 DataFrame。

        Args:
            metadata: 运行元数据对象

        Returns:
            单行 DataFrame，index=当前 UTC 时间。
        """
        return pd.DataFrame([asdict(metadata)], index=pd.DatetimeIndex([pd.Timestamp.utcnow()], name="date"))

    def build_output_bundle(
        self,
        target_positions: pd.DataFrame,
        prev_positions: Optional[pd.DataFrame],
        metadata: RunMetadata,
        diagnostics: Optional[pd.DataFrame] = None,
    ) -> OutputBundle:
        """组装完整的 OutputBundle。

        依次调用 build_target_positions → build_trade_table →
        build_portfolio_summary → build_audit_meta，打包为 OutputBundle。

        Args:
            target_positions: 优化器输出的目标权重
            prev_positions: 上期持仓权重
            metadata: 运行元数据
            diagnostics: 信号平滑诊断信息（可选）

        Returns:
            完整的 OutputBundle。
        """
        target_positions = self.build_target_positions(target_positions)
        trades = self.build_trade_table(prev_positions, target_positions)
        summary = self.build_portfolio_summary(target_positions, prev_positions, diagnostics)
        audit_meta = self.build_audit_meta(metadata)
        return OutputBundle(target_positions=target_positions, trades=trades, summary=summary, metadata=audit_meta)
