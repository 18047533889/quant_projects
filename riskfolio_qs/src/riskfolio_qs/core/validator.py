"""
Schema 校验器：在管线关键节点对输入/输出数据包进行格式和完整性校验。

职责：
1. 输入校验：确保 InputBundle 中每个 DataFrame 满足 schema 约束
2. 缺失输入检查：根据优化器映射的 required_inputs，区分强/弱依赖缺失
3. 输出校验：确保 OutputBundle 中四个 DataFrame 结构合法

关联文档：riskfolio_qs_v0.2_优化器映射规范.md（强/弱依赖定义）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .contracts import InputBundle, OutputBundle
from .schema import require_datetime_index, require_matching_columns, require_no_inf, validate_optional_frame


@dataclass(slots=True)
class SchemaValidator:
    """管线数据校验器：在 pipeline.run() 的入口和出口执行校验。"""

    def validate_input_bundle(self, bundle: InputBundle) -> bool:
        """校验输入数据包的格式合法性。

        对必填字段 alpha/market 做严格校验，
        对可选字段（含 Barra 外采因子）做"非 None 则必须合法"校验，
        对有列对齐要求的字段做列名一致性校验。

        Returns:
            始终返回 True（校验失败时直接抛异常）。
        """
        # 必填字段：必须有合法的 DatetimeIndex 且不含非法数值
        require_datetime_index(bundle.alpha, "alpha")
        require_no_inf(bundle.alpha, "alpha")
        if bundle.market is not None:
            require_datetime_index(bundle.market, "market")
            require_no_inf(bundle.market, "market")

        # Barra 暴露 F 需要表示 date x asset x factor，因此使用
        # MultiIndex(date, asset) x factor 的二维载体。
        if bundle.F is not None:
            if not isinstance(bundle.F, pd.DataFrame):
                raise TypeError("F must be a pandas DataFrame")
            if not isinstance(bundle.F.index, pd.MultiIndex) or bundle.F.index.nlevels != 2:
                raise ValueError("F must use MultiIndex(date, asset) rows and factor columns")
            dates = bundle.F.index.get_level_values(0)
            if not isinstance(dates, pd.DatetimeIndex):
                raise ValueError("F first index level must be DatetimeIndex")
            if bundle.F.index.has_duplicates:
                raise ValueError("F index must not contain duplicates")
            if bundle.F.columns.has_duplicates or len(bundle.F.columns) == 0:
                raise ValueError("F factor columns must be non-empty and unique")
        validate_optional_frame(bundle.G, "G")
        validate_optional_frame(bundle.F_mcap, "F_mcap")
        validate_optional_frame(bundle.F_ret, "F_ret")
        validate_optional_frame(bundle.F_spec, "F_spec")
        if bundle.factor_cov is not None:
            if not isinstance(bundle.factor_cov, pd.DataFrame):
                raise TypeError("factor_cov must be a pandas DataFrame")
            if (
                not isinstance(bundle.factor_cov.index, pd.MultiIndex)
                or bundle.factor_cov.index.nlevels != 2
            ):
                raise ValueError(
                    "factor_cov must use MultiIndex(date, factor_i) rows and factor_j columns"
                )
            cov_dates = bundle.factor_cov.index.get_level_values(0)
            if not isinstance(cov_dates, pd.DatetimeIndex):
                raise ValueError("factor_cov first index level must be DatetimeIndex")
            if bundle.factor_cov.index.has_duplicates:
                raise ValueError("factor_cov index must not contain duplicates")
            if bundle.factor_cov.columns.has_duplicates or len(bundle.factor_cov.columns) == 0:
                raise ValueError("factor_cov columns must be non-empty and unique")
            require_no_inf(bundle.factor_cov, "factor_cov")
        validate_optional_frame(bundle.specific_var, "specific_var")

        # 其他可选字段
        validate_optional_frame(bundle.industry, "industry")
        validate_optional_frame(bundle.style, "style")
        validate_optional_frame(bundle.benchmark, "benchmark")
        validate_optional_frame(bundle.prev_positions, "prev_positions")
        validate_optional_frame(bundle.tradable, "tradable")
        validate_optional_frame(bundle.risk_covered, "risk_covered")
        validate_optional_frame(bundle.linear_cost_bps, "linear_cost_bps")
        validate_optional_frame(bundle.impact_cost, "impact_cost")

        # 列对齐校验：benchmark 和 prev_positions 必须与 alpha 列名一致
        if bundle.benchmark is not None:
            require_matching_columns(bundle.alpha, bundle.benchmark, "alpha", "benchmark")
        if bundle.prev_positions is not None:
            require_matching_columns(bundle.alpha, bundle.prev_positions, "alpha", "prev_positions")

        # market 风险模型最终会转为无标签 numpy 矩阵，必须在入口
        # 锁定与 alpha 相同的资产顺序，避免协方差错配到其他股票。
        if bundle.market is not None:
            require_matching_columns(bundle.alpha, bundle.market, "alpha", "market")
        if bundle.market is not None and not bundle.alpha.index.isin(bundle.market.index).all():
            missing_dates = bundle.alpha.index[~bundle.alpha.index.isin(bundle.market.index)]
            raise ValueError(f"market is missing alpha dates: {list(missing_dates[:5])}")

        aligned_asset_frames = {
            "G": bundle.G,
            "F_mcap": bundle.F_mcap,
            "F_spec": bundle.F_spec,
            "specific_var": bundle.specific_var,
            "benchmark": bundle.benchmark,
            "prev_positions": bundle.prev_positions,
            "tradable": bundle.tradable,
            "risk_covered": bundle.risk_covered,
            "linear_cost_bps": bundle.linear_cost_bps,
            "impact_cost": bundle.impact_cost,
        }
        for name, frame in aligned_asset_frames.items():
            if frame is not None:
                require_matching_columns(bundle.alpha, frame, "alpha", name)

        if bundle.specific_var is not None:
            require_no_inf(bundle.specific_var, "specific_var")
            finite = bundle.specific_var.dropna().to_numpy(dtype=float)
            if (finite <= 0).any():
                raise ValueError("specific_var must be strictly positive")

        for name, frame in {
            "linear_cost_bps": bundle.linear_cost_bps,
            "impact_cost": bundle.impact_cost,
        }.items():
            if frame is not None:
                require_no_inf(frame, name)
                if (frame.dropna().to_numpy(dtype=float) < 0).any():
                    raise ValueError(f"{name} must be non-negative")

        context = bundle.metadata
        if context.market_data_lag_periods < 0 or context.exposure_data_lag_periods < 0:
            raise ValueError("data lag periods must be non-negative")
        if context.alpha_horizon_days <= 0:
            raise ValueError("alpha_horizon_days must be positive")
        if context.alpha_input_type not in {"score", "expected_return"}:
            raise ValueError("alpha_input_type must be score or expected_return")
        if context.market_input_type not in {"price", "return"}:
            raise ValueError("market_input_type must be price or return")
        if context.risk_covariance_units not in {
            "daily_variance",
            "horizon_variance",
            "annual_variance",
        }:
            raise ValueError(
                "risk_covariance_units must be daily_variance, "
                "horizon_variance or annual_variance"
            )
        if context.risk_horizon_days <= 0:
            raise ValueError("risk_horizon_days must be positive")
        if context.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")

        return True

    def find_missing_inputs(self, bundle: InputBundle, spec: Any) -> tuple[list[str], list[str]]:
        """根据优化器映射的 required_inputs 检查输入缺失情况。

        从 BenchmarkSpec.required_inputs 中读取强依赖和弱依赖列表，
        逐字段检查 bundle 中是否为 None 或空 DataFrame。

        Args:
            bundle: 输入数据包
            spec: BenchmarkSpec（包含 required_inputs 字段）

        Returns:
            (strong_missing, weak_missing)：两个列表，分别列出缺失的强/弱依赖字段名。
        """
        required = getattr(spec, "required_inputs", {}) or {}
        strong = list(required.get("strong", []))
        weak = list(required.get("weak", []))

        strong_missing = [name for name in strong if self._is_missing(bundle, name)]
        weak_missing = [name for name in weak if self._is_missing(bundle, name)]
        return strong_missing, weak_missing

    @staticmethod
    def _is_missing(bundle: InputBundle, field_name: str) -> bool:
        """判断 bundle 中某个字段是否缺失。

        通过字段名到 bundle 属性的映射表来检查。
        cost_impact 映射到 InputBundle.impact_cost。
        """
        mapping = {
            "alpha": bundle.alpha,
            "market": bundle.market,
            "benchmark": bundle.benchmark,
            "prev_positions": bundle.prev_positions,
            "F": bundle.F,
            "G": bundle.G,
            "F_mcap": bundle.F_mcap,
            "F_ret": bundle.F_ret,
            "F_spec": bundle.F_spec,
            "factor_cov": bundle.factor_cov,
            "specific_var": bundle.specific_var,
            "style": bundle.style,
            "tradable": bundle.tradable,
            "risk_covered": bundle.risk_covered,
            "linear_cost_bps": bundle.linear_cost_bps,
            "cost_impact": bundle.impact_cost,
        }
        value = mapping.get(field_name)
        if value is None:
            return True
        if isinstance(value, pd.DataFrame) and value.empty:
            return True
        return False

    def validate_output_bundle(self, bundle: OutputBundle) -> bool:
        """校验输出数据包的格式合法性。

        确保 target_positions、trades、summary、metadata 四个 DataFrame
        均使用合法的 DatetimeIndex（trades 支持 MultiIndex 首级为日期）。

        Returns:
            始终返回 True（校验失败时直接抛异常）。
        """
        require_datetime_index(bundle.target_positions, "target_positions")

        # trades 支持两种索引格式：MultiIndex(date, asset) 或 DatetimeIndex
        if isinstance(bundle.trades.index, pd.MultiIndex):
            first_level = bundle.trades.index.get_level_values(0)
            if not isinstance(first_level, pd.DatetimeIndex):
                raise ValueError("trades first index level must be DatetimeIndex")
            if not first_level.is_monotonic_increasing:
                raise ValueError("trades first index level must be monotonic increasing")
        else:
            require_datetime_index(bundle.trades, "trades")

        require_datetime_index(bundle.summary, "summary")
        require_datetime_index(bundle.metadata, "metadata")
        return True
