# -*- coding: utf-8 -*-
"""量价衍生算子 Polars 实现。

CAPM 簇
-------
``idio_vol`` 使用 Polars 滚动回归残差 + rolling_std（快路径）。
``idio_skew`` / ``residual_momentum_capm`` / ``coskewness_to_market`` 等
窗口内需 OLS + 高阶矩，复用 ``_numpy_kernels`` 保证与 pandas 一致。

``downside_beta`` / ``tail_beta`` 在窗口内按条件子样本估计 Beta。
"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _align_cols(*dfs: pl.DataFrame) -> list[str]:
    cols = _numeric_cols(dfs[0])
    for df in dfs[1:]:
        cols = [c for c in cols if c in df.columns]
    return cols


@register_operator(name="cumulative_returns", category="financial", business_category="price_volume", canonical="cumulative_returns", source="factor_dsl_polars")
class CumulativeReturnsPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cumulative_returns", category="financial", description="累计收益率",
        param_names=["price"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, price: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(price)
        exprs = []
        for c in cols:
            ret = pl.col(c) / pl.col(c).shift(1) - 1.0
            exprs.append(((1.0 + ret.fill_null(0.0)).cum_prod() - 1.0).alias(c))
        return price.with_columns(exprs)


@register_operator(name="volatility", category="financial", business_category="price_volume", canonical="volatility", source="factor_dsl_polars")
class VolatilityPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="volatility", category="financial", description="滚动波动率（年化）",
        param_names=["x", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(x)
        scale = float(np.sqrt(252))
        return x.with_columns([
            (pl.col(c).rolling_std(window_size=w, min_samples=1) * scale).alias(c) for c in cols
        ])


@register_operator(name="vwap", category="financial", business_category="price_volume", canonical="vwap", source="factor_dsl_polars")
class VWAPPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="vwap", category="financial", description="成交量加权平均价",
        param_names=["price", "volume", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, price: pl.DataFrame, volume: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _align_cols(price, volume)
        exprs = []
        for c in cols:
            pv = price[c] * volume[c]
            num = pv.rolling_sum(window_size=w, min_samples=1)
            den = volume[c].rolling_sum(window_size=w, min_samples=1)
            exprs.append((num / den).alias(c))
        return price.with_columns(exprs)


@register_operator(name="m_beta", category="time_series", business_category="time_series", canonical="ts_beta", source="factor_dsl_polars")
class TSBetaPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="m_beta", category="time_series", description="滚动 Beta",
        param_names=["y", "x", "window"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _align_cols(y, x)
        exprs = []
        for c in cols:
            cov = pl.rolling_cov(y[c], x[c], window_size=w, min_samples=1)
            var = x[c].rolling_var(window_size=w, min_samples=1)
            exprs.append((cov / var).alias(c))
        return y.with_columns(exprs)


@register_operator(name="sharpe_ratio", category="financial", business_category="price_volume", canonical="sharpe_ratio", source="factor_dsl_polars")
class SharpeRatioPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="sharpe_ratio", category="financial", description="年化夏普比率",
        param_names=["returns", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, returns: pl.DataFrame, window: int = 60, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 2)
        cols = _numeric_cols(returns)
        scale = float(np.sqrt(252))
        return returns.with_columns([
            (
                pl.col(c).rolling_mean(window_size=w, min_samples=2)
                / pl.col(c).rolling_std(window_size=w, min_samples=2)
                * scale
            ).alias(c)
            for c in cols
        ])


@register_operator(name="max_drawdown", category="financial", business_category="price_volume", canonical="max_drawdown", source="factor_dsl_polars")
class MaxDrawdownPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="max_drawdown", category="financial", description="最大回撤",
        param_names=["returns"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, returns: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(returns)
        exprs = []
        for c in cols:
            cum = (1.0 + pl.col(c).fill_null(0.0)).cum_prod()
            peak = cum.cum_max()
            dd = (cum - peak) / peak
            exprs.append(dd.alias(c))
        return returns.with_columns(exprs)


def _conditional_beta(ret: np.ndarray, mkt: np.ndarray, window: int, *, mode: str, q: float = 0.05) -> np.ndarray:
    """滚动条件 Beta：downside 仅市场收益<0；tail 取市场收益最差 q 分位。"""
    result = np.full_like(ret, np.nan, dtype=float)
    w = max(int(window), 1)
    for i in range(w - 1, len(ret)):
        r = ret[i - w + 1 : i + 1]
        m = mkt[i - w + 1 : i + 1]
        valid = ~(np.isnan(r) | np.isnan(m))
        if mode == "downside":
            mask = valid & (m < 0)
        else:
            threshold = np.nanpercentile(m[valid], q * 100) if valid.any() else 0.0
            mask = valid & (m <= threshold)
        if mask.sum() >= 3:
            cov = np.cov(r[mask], m[mask])[0, 1]
            var = np.var(m[mask])
            result[i] = cov / var if var > 0 else np.nan
    return result


@register_operator(name="downside_beta", category="price_volume", business_category="price_volume", canonical="downside_beta", source="factor_dsl_polars")
class DownsideBetaPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="downside_beta", category="price_volume", description="下行 Beta",
        param_names=["ret", "benchmark_ret", "window"], return_type="series", tags=["price_volume", "polars"],
    )

    def _calculate_series(self, ret: pl.DataFrame, benchmark_ret: pl.DataFrame, window: int = 60, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _align_cols(ret, benchmark_ret)
        out: dict[str, np.ndarray] = {}
        for c in cols:
            out[c] = _conditional_beta(
                ret[c].to_numpy(), benchmark_ret[c].to_numpy(), w, mode="downside"
            )
        result = pl.DataFrame(out)
        if "date" in ret.columns:
            result = result.with_columns(ret["date"])
        return result


@register_operator(name="tail_beta", category="price_volume", business_category="price_volume", canonical="tail_beta", source="factor_dsl_polars")
class TailBetaPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="tail_beta", category="price_volume", description="尾部 Beta",
        param_names=["ret", "benchmark_ret", "window", "q"], return_type="series", tags=["price_volume", "polars"],
    )

    def _calculate_series(self, ret: pl.DataFrame, benchmark_ret: pl.DataFrame, window: int = 60, q: float = 0.05, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        q = float(kwargs.get("q", q))
        cols = _align_cols(ret, benchmark_ret)
        out: dict[str, np.ndarray] = {}
        for c in cols:
            out[c] = _conditional_beta(
                ret[c].to_numpy(), benchmark_ret[c].to_numpy(), w, mode="tail", q=q
            )
        result = pl.DataFrame(out)
        if "date" in ret.columns:
            result = result.with_columns(ret["date"])
        return result


@register_operator(
    name="rolling_beta_to_market",
    category="price_volume",
    business_category="price_volume",
    canonical="rolling_beta_to_market",
    source="factor_dsl_polars",
)
class RollingBetaToMarketPolars(TSBetaPolars):
    metadata = OperatorMetadata(
        name="rolling_beta_to_market",
        category="price_volume",
        description="滚动市场 Beta",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["price_volume", "polars"],
    )


@register_operator(
    name="idio_vol",
    category="price_volume",
    business_category="price_volume",
    canonical="idio_vol",
    source="factor_dsl_polars",
)
class IdioVolPolars(SeriesOperator):
    """CAPM 残差滚动标准差（Polars：ts_regression residual → rolling std）。"""

    metadata = OperatorMetadata(
        name="idio_vol",
        category="price_volume",
        description="特质波动率",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "polars"],
    )

    def _calculate_series(self, ret: pl.DataFrame, benchmark_ret: pl.DataFrame, window: int = 60, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", window)), 3)
        cols = _align_cols(ret, benchmark_ret)
        scale = float(np.sqrt(252))
        exprs = []
        for c in cols:
            cov = pl.rolling_cov(ret[c], benchmark_ret[c], window_size=w, min_samples=3)
            var_x = benchmark_ret[c].rolling_var(window_size=w, min_samples=3)
            slope = cov / var_x
            mean_y = ret[c].rolling_mean(window_size=w, min_samples=3)
            mean_x = benchmark_ret[c].rolling_mean(window_size=w, min_samples=3)
            intercept = mean_y - slope * mean_x
            resid = ret[c] - (slope * benchmark_ret[c] + intercept)
            exprs.append((resid.rolling_std(window_size=w, min_samples=3) * scale).alias(c))
        return ret.with_columns(exprs)


def _apply_numpy_kernel_2d(
    ret: pl.DataFrame,
    index_ret: pl.DataFrame,
    window: int,
    kernel,
) -> pl.DataFrame:
    """宽表逐列调用 ``kernel(ret_col, idx_col, window)``，与 pandas 列方向一致。"""
    w = int(window)
    cols = _align_cols(ret, index_ret)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = kernel(ret[c].to_numpy(), index_ret[c].to_numpy(), w)
    result = pl.DataFrame(out)
    if "date" in ret.columns:
        result = result.with_columns(ret["date"])
    return result


@register_operator(
    name="residual_momentum_capm",
    category="price_volume",
    business_category="price_volume",
    canonical="residual_momentum_capm",
    source="factor_dsl_polars",
)
class ResidualMomentumCapmPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="residual_momentum_capm",
        category="price_volume",
        description="CAPM 残差动量",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "polars"],
    )

    def _calculate_series(
        self, ret: pl.DataFrame, benchmark_ret: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators._numpy_kernels import residual_momentum_capm_

        w = int(kwargs.get("d", window))
        return _apply_numpy_kernel_2d(ret, benchmark_ret, w, residual_momentum_capm_)


@register_operator(
    name="coskewness_to_market",
    category="price_volume",
    business_category="price_volume",
    canonical="coskewness_to_market",
    source="factor_dsl_polars",
)
class CoskewnessToMarketPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="coskewness_to_market",
        category="price_volume",
        description="相对市场的余偏度",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "polars"],
    )

    def _calculate_series(
        self, ret: pl.DataFrame, benchmark_ret: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators._numpy_kernels import coskewness_to_market_

        w = int(kwargs.get("d", window))
        return _apply_numpy_kernel_2d(ret, benchmark_ret, w, coskewness_to_market_)


@register_operator(
    name="idio_skew",
    category="price_volume",
    business_category="price_volume",
    canonical="idio_skew",
    source="factor_dsl_polars",
)
class IdioSkewPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="idio_skew",
        category="price_volume",
        description="CAPM 残差偏度",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "polars"],
    )

    def _calculate_series(
        self, ret: pl.DataFrame, benchmark_ret: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators._numpy_kernels import idio_skew_

        w = int(kwargs.get("d", window))
        return _apply_numpy_kernel_2d(ret, benchmark_ret, w, idio_skew_)
