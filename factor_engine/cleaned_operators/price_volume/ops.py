# -*- coding: utf-8 -*-
"""
量价衍生与风险类算子（收益、波动、Beta 等）。

语义
----
基于价量字段构造常见因子构件，例如：
- 收益率、对数收益、超额收益；
- 波动率、下行波动、特质波动；
- 与市场/基准的 beta、协方差衍生。

输入多为 ``close`` / ``volume`` / ``returns`` 等 panel；具体参数见各类 ``OperatorMetadata``。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)

import numpy as np
import pandas as pd

# canonical=cumulative_returns backend=pandas_numpy selected=cumulative_returns source=financial/__init__.py
@register_operator(name="cumulative_returns", category="financial", business_category="price_volume", canonical="cumulative_returns", source="factor_dsl_np")
class CumulativeReturns(SeriesOperator):
    metadata = OperatorMetadata(
        name="cumulative_returns",
        category="financial",
        description="累计收益率",
        examples=["cumulative_returns(close)"],
        param_names=["price"],
        return_type="series",
        tags=["financial", "returns"]
    )

    def _calculate_series(self, price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        returns = price.pct_change().fillna(0)
        return (1 + returns).cumprod() - 1



# canonical=max_drawdown backend=pandas_numpy selected=max_drawdown source=financial/__init__.py
@register_operator(name="max_drawdown", category="financial", business_category="price_volume", canonical="max_drawdown", source="factor_dsl_np")
class MaxDrawdown(SeriesOperator):
    metadata = OperatorMetadata(
        name="max_drawdown",
        category="financial",
        description="最大回撤",
        examples=["max_drawdown(returns)"],
        param_names=["returns"],
        return_type="series",
        tags=["financial", "risk", "drawdown"]
    )

    def _calculate_series(self, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        cum = (1 + returns.fillna(0)).cumprod()
        running_max = cum.expanding(min_periods=1).max()
        drawdown = (cum - running_max) / running_max
        return drawdown.expanding(min_periods=1).min()



# 重复实现：见 ts_pct(x,1)；dedupe 注销，别名 returns
# @register_operator(name="returns", category="financial", business_category="price_volume", canonical="returns", source="factor_dsl_np")
class Returns(SeriesOperator):
    """收益率"""
    metadata = OperatorMetadata(
        name="returns",
        category="financial",
        description="收益率 close/Ref(close,1)-1",
        examples=["returns(close)"],
        param_names=["x"],
        return_type="series",
        tags=["financial", "returns"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.pct_change()


# canonical=log_returns backend=pandas_numpy selected=log_returns source=financial/__init__.py
@register_operator(name="log_returns", category="financial", business_category="price_volume", canonical="log_returns", source="factor_dsl_np")
class LogReturns(SeriesOperator):
    """对数收益率 ln(P_t / P_{t-1})"""
    metadata = OperatorMetadata(
        name="log_returns",
        category="financial",
        description="对数收益率 ln(P_t / P_{t-1})",
        examples=["log_returns(close)"],
        param_names=["x"],
        return_type="series",
        tags=["financial", "returns", "log"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        prev = x.shift(1)
        ratio = x / prev.replace(0, np.nan)
        return np.log(ratio.replace([np.inf, -np.inf], np.nan))


# canonical=sharpe_ratio backend=pandas_numpy selected=sharpe_ratio source=financial/__init__.py
@register_operator(name="sharpe_ratio", category="financial", business_category="price_volume", canonical="sharpe_ratio", source="factor_dsl_np")
class SharpeRatio(SeriesOperator):
    metadata = OperatorMetadata(
        name="sharpe_ratio",
        category="financial",
        description="年化夏普比率",
        examples=["sharpe_ratio(returns, 60)"],
        param_names=["returns", "window"],
        return_type="series",
        tags=["financial", "risk", "sharpe"]
    )

    def _calculate_series(self, returns: pd.DataFrame, window: int = 60, **kwargs) -> pd.DataFrame:
        mean = returns.rolling(window=window, min_periods=2).mean()
        std = returns.rolling(window=window, min_periods=2).std()
        return (mean / std.replace(0, np.nan)) * np.sqrt(252)



# canonical=volatility backend=pandas_numpy selected=volatility source=financial/__init__.py
@register_operator(name="volatility", category="financial", business_category="price_volume", canonical="volatility", source="factor_dsl_np")
class Volatility(SeriesOperator):
    """波动率"""
    metadata = OperatorMetadata(
        name="volatility",
        category="financial",
        description="波动率（年化）",
        examples=["volatility(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "volatility", "std"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).std() * np.sqrt(252)



# canonical=vwap backend=pandas_numpy selected=vwap source=financial/__init__.py
@register_operator(name="vwap", category="financial", business_category="price_volume", canonical="vwap", source="factor_dsl_np")
class VWAP(SeriesOperator):
    metadata = OperatorMetadata(
        name="vwap",
        category="financial",
        description="成交量加权平均价",
        examples=["vwap(close, volume, 20)"],
        param_names=["price", "volume", "window"],
        return_type="series",
        tags=["financial", "vwap"]
    )

    def _calculate_series(self, price: pd.DataFrame, volume: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        pv = price * volume
        return pv.rolling(window=window, min_periods=1).sum() / volume.rolling(window=window, min_periods=1).sum()


@register_operator(name="rolling_beta_to_market", category="price_volume", business_category="price_volume", canonical="rolling_beta_to_market", source="factor_dsl_np")
class LqtpRollingbetatomarketOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="rolling_beta_to_market",
        category="price_volume",
        description="滚动市场 Beta：Cov(r_i, r_m) / Var(r_m)",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import rolling_beta_to_market_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: rolling_beta_to_market_(s.values, **kwargs) if kwargs else rolling_beta_to_market_(s.values))
        return rolling_beta_to_market_(*args, **kwargs)


@register_operator(name="downside_beta", category="price_volume", business_category="price_volume", canonical="downside_beta", source="factor_dsl_np")
class LqtpDownsidebetaOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="downside_beta",
        category="price_volume",
        description="下行 Beta：仅在 r_m < 0 子样本上估计",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import downside_beta_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: downside_beta_(s.values, **kwargs) if kwargs else downside_beta_(s.values))
        return downside_beta_(*args, **kwargs)


@register_operator(name="tail_beta", category="price_volume", business_category="price_volume", canonical="tail_beta", source="factor_dsl_np")
class LqtpTailbetaOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="tail_beta",
        category="price_volume",
        description="尾部 Beta：在市场收益最低 q 分位子样本上估计",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import tail_beta_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: tail_beta_(s.values, **kwargs) if kwargs else tail_beta_(s.values))
        return tail_beta_(*args, **kwargs)


@register_operator(name="residual_momentum_capm", category="price_volume", business_category="price_volume", canonical="residual_momentum_capm", source="factor_dsl_np")
class LqtpResidualmomentumcapmOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="residual_momentum_capm",
        category="price_volume",
        description="CAPM 残差动量：窗口内回归残差之和",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import residual_momentum_capm_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: residual_momentum_capm_(s.values, **kwargs) if kwargs else residual_momentum_capm_(s.values))
        return residual_momentum_capm_(*args, **kwargs)


@register_operator(name="coskewness_to_market", category="price_volume", business_category="price_volume", canonical="coskewness_to_market", source="factor_dsl_np")
class LqtpCoskewnesstomarketOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="coskewness_to_market",
        category="price_volume",
        description="相对市场的协偏度",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import coskewness_to_market_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: coskewness_to_market_(s.values, **kwargs) if kwargs else coskewness_to_market_(s.values))
        return coskewness_to_market_(*args, **kwargs)


@register_operator(name="idio_vol", category="price_volume", business_category="price_volume", canonical="idio_vol", source="factor_dsl_np")
class LqtpIdiovolOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="idio_vol",
        category="price_volume",
        description="特质波动率：CAPM 残差标准差",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import idio_vol_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: idio_vol_(s.values, **kwargs) if kwargs else idio_vol_(s.values))
        return idio_vol_(*args, **kwargs)


@register_operator(name="idio_skew", category="price_volume", business_category="price_volume", canonical="idio_skew", source="factor_dsl_np")
class LqtpIdioskewOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="idio_skew",
        category="price_volume",
        description="特质偏度：CAPM 残差偏度",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import idio_skew_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: idio_skew_(s.values, **kwargs) if kwargs else idio_skew_(s.values))
        return idio_skew_(*args, **kwargs)
