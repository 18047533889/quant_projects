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
@register_operator(name="cumulative_returns", category="financial", business_category="price_volume", canonical="cumulative_returns", source="factor_dsl_np", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class CumulativeReturns(SeriesOperator):
    """累计收益率"""
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
        # Cumulative return along the *contiguous valid price path* only.  A
        # missing price censors that bar to NaN — never a fabricated 0-return
        # day — and the path re-anchors at the next valid price (review P1-124).
        pv = price.to_numpy(dtype=float)
        rows, cols = pv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            anchor: float | None = None
            for t in range(rows):
                if not np.isfinite(pv[t, c]):
                    anchor = None
                    continue
                if anchor is None:
                    anchor = pv[t, c]
                    out[t, c] = 0.0
                else:
                    out[t, c] = pv[t, c] / anchor - 1.0
        return pd.DataFrame(out, index=price.index, columns=price.columns)



# canonical=max_drawdown backend=pandas_numpy selected=max_drawdown source=financial/__init__.py
@register_operator(name="max_drawdown", category="financial", business_category="price_volume", canonical="max_drawdown", source="factor_dsl_np", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class MaxDrawdown(SeriesOperator):
    """最大回撤"""
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
        # Per-bar drawdown along the contiguous valid return path.  A missing
        # return censors that bar to NaN (never a 0-return day) and the
        # cumulative path re-anchors at the next valid bar — no fill0, no
        # dropna/reconnect.  The returned series is the historical worst
        # drawdown up to each bar (expanding min; missing bars do not reset it).
        rv = returns.to_numpy(dtype=float)
        rows, cols = rv.shape
        dd = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            cum = np.nan
            peak = np.nan
            for t in range(rows):
                if not np.isfinite(rv[t, c]):
                    cum = np.nan
                    peak = np.nan
                    continue
                if not np.isfinite(cum):
                    cum = 1.0
                    peak = 1.0
                cum *= 1.0 + rv[t, c]
                if cum > peak:
                    peak = cum
                dd[t, c] = (cum - peak) / peak
        worst = pd.DataFrame(dd, index=returns.index, columns=returns.columns).expanding(min_periods=1).min()
        # Censor bars whose own return is missing (the drawdown state at that bar
        # is unknown) instead of silently reporting the historical record there.
        return worst.where(returns.notna())



# 重复实现：见 ts_pct(x,1)；dedupe 注销，别名 returns
# @register_operator(name="returns", category="financial", business_category="price_volume", canonical="returns", source="factor_dsl_np", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
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
        return x.pct_change(fill_method=None)


# canonical=log_returns backend=pandas_numpy selected=log_returns source=financial/__init__.py
@register_operator(name="log_returns", category="financial", business_category="price_volume", canonical="log_returns", source="factor_dsl_np", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
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
@register_operator(name="sharpe_ratio", category="financial", business_category="price_volume", canonical="sharpe_ratio", source="factor_dsl_np", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class SharpeRatio(SeriesOperator):
    """年化夏普比率"""
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


@register_operator(
    name="open_gap",
    category="price_volume",
    business_category="price_volume",
    canonical="open_gap",
    source="factor_dsl_np",
, replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class OpenGap(SeriesOperator):
    """开盘缺口：open / prev_close - 1。"""

    metadata = OperatorMetadata(
        name="open_gap",
        category="price_volume",
        description="开盘缺口收益率：open / delay(close, 1) - 1",
        param_names=["open", "close"],
        return_type="series",
        tags=["price_volume", "gap", "pit_safe"],
    )

    def _calculate_series(self, open_: pd.DataFrame, close: pd.DataFrame, **kwargs) -> pd.DataFrame:
        prev_close = close.shift(1)
        return open_ / prev_close.replace(0, np.nan) - 1.0


@register_operator(
    name="close_gap",
    category="price_volume",
    business_category="price_volume",
    canonical="close_gap",
    source="factor_dsl_np",
, replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class CloseGap(SeriesOperator):
    """日内缺口：close / open - 1。"""

    metadata = OperatorMetadata(
        name="close_gap",
        category="price_volume",
        description="日内收益率：close / open - 1",
        param_names=["close", "open"],
        return_type="series",
        tags=["price_volume", "gap", "pit_safe"],
    )

    def _calculate_series(self, close: pd.DataFrame, open_: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return close / open_.replace(0, np.nan) - 1.0


# canonical=volatility backend=pandas_numpy selected=volatility source=financial/__init__.py
@register_operator(name="volatility", category="financial", business_category="price_volume", canonical="volatility", source="factor_dsl_np", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class Volatility(SeriesOperator):
    """波动率"""
    metadata = OperatorMetadata(
        name="volatility",
        category="financial",
        description="波动率（年化）",
        examples=["volatility(returns, 20)"],
        param_names=["x", "window", "min_periods"],
        return_type="series",
        tags=["financial", "volatility", "std"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int | None = None, **kwargs) -> pd.DataFrame:
        w = int(window)
        mp = int(min_periods) if min_periods is not None else max(2, w // 2)
        mp = max(2, mp)
        return x.rolling(window=w, min_periods=mp).std() * np.sqrt(252)



# canonical=vwap backend=pandas_numpy selected=vwap source=financial/__init__.py
@register_operator(name="vwap", category="financial", business_category="price_volume", canonical="vwap", source="factor_dsl_np", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class VWAP(SeriesOperator):
    """成交量加权平均价"""
    metadata = OperatorMetadata(
        name="vwap",
        category="financial",
        description="成交量加权平均价",
        examples=["vwap(close, volume, 20)"],
        param_names=["price", "volume", "window", "min_periods"],
        return_type="series",
        tags=["financial", "vwap"]
    )

    def _calculate_series(self, price: pd.DataFrame, volume: pd.DataFrame, window: int = 20, min_periods: int | None = None, **kwargs) -> pd.DataFrame:
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        # Volume is a count/amount and must be non-negative (review P1-126).
        vol_arr = volume.to_numpy(dtype=float)
        if np.any(vol_arr < 0):
            raise ValueError("vwap: volume must be non-negative")
        mp = int(min_periods) if min_periods is not None else 1
        if mp < 1:
            raise ValueError("min_periods must be >= 1")
        valid = price.notna() & volume.notna()
        pv = (price * volume).where(valid)
        vol = volume.where(valid)
        den = vol.rolling(window=w, min_periods=mp).sum().replace(0, np.nan)
        return pv.rolling(window=w, min_periods=mp).sum() / den


@register_operator(
    name="rolling_beta",
    category="price_volume",
    business_category="price_volume",
    canonical="rolling_beta",
    source="factor_dsl_np",
, replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class RollingBetaOp(SeriesOperator):
    """滚动 Beta：Cov(ret, benchmark_ret) / Var(benchmark_ret)。"""

    metadata = OperatorMetadata(
        name="rolling_beta",
        category="price_volume",
        description="滚动 Beta：Cov(ret, benchmark_ret) / Var(benchmark_ret)",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "beta", "pit_safe"],
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        benchmark_ret: pd.DataFrame,
        window: int = 60,
        min_periods: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        from cleaned_operators.price_volume.beta_helpers import compute_rolling_beta

        return compute_rolling_beta(ret, benchmark_ret, window, min_periods=min_periods)


@register_operator(
    name="rolling_beta_to_market",
    category="price_volume",
    business_category="price_volume",
    canonical="rolling_beta_to_market",
    source="factor_dsl_np",
    status="experimental",
, replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class LqtpRollingbetatomarketOp(SeriesOperator):
    """[experimental] 滚动市场 Beta；请优先使用 rolling_beta(ret, benchmark, window)"""
    metadata = OperatorMetadata(
        name="rolling_beta_to_market",
        category="price_volume",
        description="[experimental] 滚动市场 Beta；请优先使用 rolling_beta(ret, benchmark, window)",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
    )

    def _calculate_series(self, ret, benchmark_ret=None, window: int = 60, **kwargs):
        from cleaned_operators.price_volume.beta_helpers import compute_rolling_beta

        if benchmark_ret is None:
            raise ValueError(
                "rolling_beta_to_market 需要 (ret, benchmark_ret, window)；"
                "请改用 rolling_beta(ret, benchmark_ret, window)"
            )
        return compute_rolling_beta(ret, benchmark_ret, window)


@register_operator(name="downside_beta", category="price_volume", business_category="price_volume", canonical="downside_beta", source="factor_dsl_np", status="experimental", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class LqtpDownsidebetaOp(SeriesOperator):
    """下行 Beta：仅在 benchmark_ret < 0 子样本上估计"""
    metadata = OperatorMetadata(
        name="downside_beta",
        category="price_volume",
        description="下行 Beta：仅在 benchmark_ret < 0 子样本上估计",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "capm", "pit_safe"],
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        benchmark_ret: pd.DataFrame | None = None,
        window: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import downside_beta_
        from cleaned_operators.price_volume.capm_helpers import apply_capm_kernel_panel

        return apply_capm_kernel_panel(ret, benchmark_ret, window, downside_beta_)


@register_operator(name="tail_beta", category="price_volume", business_category="price_volume", canonical="tail_beta", source="factor_dsl_np", status="experimental", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class LqtpTailbetaOp(SeriesOperator):
    """尾部 Beta：在 benchmark_ret 最低 q 分位子样本上估计"""
    metadata = OperatorMetadata(
        name="tail_beta",
        category="price_volume",
        description="尾部 Beta：在 benchmark_ret 最低 q 分位子样本上估计",
        param_names=["ret", "benchmark_ret", "window", "q"],
        return_type="series",
        tags=["price_volume", "capm", "pit_safe"],
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        benchmark_ret: pd.DataFrame | None = None,
        window: int = 60,
        q: float = 0.05,
        **kwargs,
    ) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import tail_beta_
        from cleaned_operators.price_volume.capm_helpers import apply_capm_kernel_panel

        return apply_capm_kernel_panel(ret, benchmark_ret, window, tail_beta_, q=float(q))


@register_operator(name="residual_momentum_capm", category="price_volume", business_category="price_volume", canonical="residual_momentum_capm", source="factor_dsl_np", status="experimental", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class LqtpResidualmomentumcapmOp(SeriesOperator):
    """CAPM 残差动量：窗口内回归残差之和"""
    metadata = OperatorMetadata(
        name="residual_momentum_capm",
        category="price_volume",
        description="CAPM 残差动量：窗口内回归残差之和",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "capm", "pit_safe"],
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        benchmark_ret: pd.DataFrame | None = None,
        window: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import residual_momentum_capm_
        from cleaned_operators.price_volume.capm_helpers import apply_capm_kernel_panel

        return apply_capm_kernel_panel(ret, benchmark_ret, window, residual_momentum_capm_)


@register_operator(name="coskewness_to_market", category="price_volume", business_category="price_volume", canonical="coskewness_to_market", source="factor_dsl_np", status="experimental", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class LqtpCoskewnesstomarketOp(SeriesOperator):
    """相对市场的协偏度"""
    metadata = OperatorMetadata(
        name="coskewness_to_market",
        category="price_volume",
        description="相对市场的协偏度",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "capm", "pit_safe"],
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        benchmark_ret: pd.DataFrame | None = None,
        window: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import coskewness_to_market_
        from cleaned_operators.price_volume.capm_helpers import apply_capm_kernel_panel

        return apply_capm_kernel_panel(ret, benchmark_ret, window, coskewness_to_market_)


@register_operator(name="idio_vol", category="price_volume", business_category="price_volume", canonical="idio_vol", source="factor_dsl_np", status="experimental", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class LqtpIdiovolOp(SeriesOperator):
    """特质波动率：CAPM 残差滚动标准差"""
    metadata = OperatorMetadata(
        name="idio_vol",
        category="price_volume",
        description="特质波动率：CAPM 残差滚动标准差",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "capm", "pit_safe"],
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        benchmark_ret: pd.DataFrame | None = None,
        window: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import idio_vol_
        from cleaned_operators.price_volume.capm_helpers import apply_capm_kernel_panel

        return apply_capm_kernel_panel(ret, benchmark_ret, window, idio_vol_)


@register_operator(name="idio_skew", category="price_volume", business_category="price_volume", canonical="idio_skew", source="factor_dsl_np", status="experimental", replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")
class LqtpIdioskewOp(SeriesOperator):
    """特质偏度：CAPM 残差滚动偏度"""
    metadata = OperatorMetadata(
        name="idio_skew",
        category="price_volume",
        description="特质偏度：CAPM 残差滚动偏度",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "capm", "pit_safe"],
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        benchmark_ret: pd.DataFrame | None = None,
        window: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import idio_skew_
        from cleaned_operators.price_volume.capm_helpers import apply_capm_kernel_panel

        return apply_capm_kernel_panel(ret, benchmark_ret, window, idio_skew_)
