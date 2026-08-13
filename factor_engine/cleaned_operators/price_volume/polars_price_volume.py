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
from cleaned_operators.base import ParamRole, ParamSpec

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
    """Polars 累计收益率"""
    metadata = OperatorMetadata(
        name="cumulative_returns", category="financial", description="累计收益率",
        param_names=["price"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, price: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(price)
        exprs = []
        for c in cols:
            p = pl.col(c)
            # Re-anchor at each contiguous valid segment: a missing price
            # censors that bar to null rather than a fabricated 0-return day
            # (review P1-124).  ``forward_fill`` of the segment-start anchor is
            # safe because each segment starts at a valid price and the null
            # boundary rows output null regardless.
            is_start = p.is_not_null() & (p.shift(1).is_null().fill_null(True))
            anchor = pl.when(is_start).then(p).otherwise(pl.lit(None)).forward_fill()
            exprs.append((p / anchor - 1.0).alias(c))
        return price.with_columns(exprs)


@register_operator(name="volatility", category="financial", business_category="price_volume", canonical="volatility", source="factor_dsl_polars")
class VolatilityPolars(SeriesOperator):
    """Polars 滚动波动率（年化）"""
    metadata = OperatorMetadata(
        name="volatility", category="financial", description="滚动波动率（年化）",
        param_names=["x", "window", "min_periods"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        window: int = 20,
        min_periods: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        mp = kwargs.get("min_periods", min_periods)
        if mp is None:
            mp = max(2, w // 2)
        mp = max(2, int(mp))
        cols = _numeric_cols(x)
        scale = float(np.sqrt(252))
        return x.with_columns([
            (
                pl.col(c).rolling_std(window_size=w, min_samples=mp, ddof=1) * scale
            ).alias(c)
            for c in cols
        ])


@register_operator(name="vwap", category="financial", business_category="price_volume", canonical="vwap", source="factor_dsl_polars")
class VWAPPolars(SeriesOperator):
    """Polars 成交量加权平均价"""
    metadata = OperatorMetadata(
        name="vwap", category="financial", description="成交量加权平均价",
        param_names=["price", "volume", "window", "min_periods"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self,
        price: pl.DataFrame,
        volume: pl.DataFrame,
        window: int = 20,
        min_periods: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _align_cols(price, volume)
        # Volume is a count/amount and must be non-negative (review P1-126).
        for c in cols:
            if int(volume[c].lt(0).sum()) > 0:
                raise ValueError("vwap: volume must be non-negative")
        mp = int(min_periods) if min_periods is not None else 1
        exprs = []
        for c in cols:
            pv = price[c] * volume[c]
            num = pv.rolling_sum(window_size=w, min_samples=mp)
            den = volume[c].rolling_sum(window_size=w, min_samples=mp)
            # Zero total volume -> NaN (cohort-consistent), matching the pandas
            # reference ``den.replace(0, np.nan)``.
            exprs.append((pl.when(den == 0).then(pl.lit(None)).otherwise(num / den)).alias(c))
        return price.with_columns(exprs)


@register_operator(
    name="m_beta",
    category="time_series",
    business_category="time_series",
    canonical="ts_beta",
    source="factor_dsl_polars",
    backend="polars",
)
class TSBetaPolars(SeriesOperator):
    """Polars 滚动 Beta"""
    metadata = OperatorMetadata(
        name="m_beta", category="time_series", description="滚动 Beta",
        param_names=["y", "x", "window", "min_periods"], return_type="series", tags=["time_series", "polars"],
        # R19-033..035: 与 pandas ``MovingBeta`` 共享同一 authority —— min_periods
        # 默认 5（reviewed rolling_beta default），不是 polars 的 1。
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
            "min_periods": ParamSpec(dtype=int, min=2, default=5, searchable=False,
                                     param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, window: int = 20, min_periods: int = 5, **kwargs) -> pl.DataFrame:
        from cleaned_operators.common.strict_params import strict_int

        w = strict_int(kwargs.get("d", window), "window", minimum=2)
        mp = strict_int(min_periods, "min_periods", minimum=2)
        if mp > w:
            from backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        cols = _align_cols(y, x)
        merged = y
        x_cols: list[str] = []
        for c in cols:
            xname = f"__x_{c}"
            x_cols.append(xname)
            merged = merged.with_columns(x.select(pl.col(c).alias(xname)))
        exprs = []
        for c in cols:
            xn = f"__x_{c}"
            cov = pl.rolling_cov(pl.col(c), pl.col(xn), window_size=w, min_samples=mp, ddof=1)
            var = pl.col(xn).rolling_var(window_size=w, min_samples=mp, ddof=1)
            exprs.append((cov / var).alias(c))
        return merged.with_columns(exprs).drop(x_cols)


@register_operator(name="sharpe_ratio", category="financial", business_category="price_volume", canonical="sharpe_ratio", source="factor_dsl_polars")
class SharpeRatioPolars(SeriesOperator):
    """Polars 年化夏普比率"""
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
    """Polars 最大回撤"""
    metadata = OperatorMetadata(
        name="max_drawdown", category="financial", description="最大回撤",
        param_names=["returns"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, returns: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Per-bar drawdown along the contiguous valid return path (review
        # P1-125): a missing return censors that bar and re-anchors the
        # cumulative path at the next valid bar — no fill-0, no drop-reconnect.
        # The returned series is the historical worst drawdown up to each bar.
        cols = _numeric_cols(returns)
        out: dict[str, np.ndarray] = {}
        for c in cols:
            col = np.asarray(returns[c].to_numpy(), dtype=float)
            rows = col.shape[0]
            dd = np.full(rows, np.nan, dtype=float)
            cum = np.nan
            peak = np.nan
            for t in range(rows):
                if not np.isfinite(col[t]):
                    cum = np.nan
                    peak = np.nan
                    continue
                if not np.isfinite(cum):
                    cum = 1.0
                    peak = 1.0
                cum *= 1.0 + col[t]
                if cum > peak:
                    peak = cum
                dd[t] = (cum - peak) / peak if peak != 0 else np.nan
            worst = np.empty(rows, dtype=float)
            running = 0.0
            for t in range(rows):
                if np.isfinite(dd[t]):
                    running = min(running, dd[t])
                # Censor bars whose own return is missing (drawdown state unknown)
                # instead of reporting the historical record there.
                worst[t] = running if np.isfinite(col[t]) else np.nan
            out[c] = worst
        result = pl.DataFrame(out)
        if "date" in returns.columns:
            result = result.with_columns(returns["date"])
        return result


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
    """Polars 下行 Beta"""
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
    """Polars 尾部 Beta"""
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
    """Polars 滚动市场 Beta"""
    metadata = OperatorMetadata(
        name="rolling_beta_to_market",
        category="price_volume",
        description="滚动市场 Beta",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["price_volume", "polars"],
    )


@register_operator(
    name="rolling_beta",
    category="price_volume",
    business_category="price_volume",
    canonical="rolling_beta",
    source="factor_dsl_polars",
)
class RollingBetaPolars(SeriesOperator):
    """Polars 滚动 Beta：Cov(ret, benchmark_ret) / Var(benchmark_ret)"""
    metadata = OperatorMetadata(
        name="rolling_beta",
        category="price_volume",
        description="滚动 Beta：Cov(ret, benchmark_ret) / Var(benchmark_ret)",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "beta", "polars"],
    )

    def _calculate_series(
        self,
        ret: pl.DataFrame,
        benchmark_ret: pl.DataFrame,
        window: int = 60,
        min_periods: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        w = max(2, int(window))
        mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 3)
        cols = _align_cols(ret, benchmark_ret)
        merged = ret
        y_cols: list[str] = []
        for c in cols:
            yname = f"__bench_{c}"
            y_cols.append(yname)
            merged = merged.with_columns(benchmark_ret[c].alias(yname))
        exprs = []
        for c in cols:
            yn = f"__bench_{c}"
            cov = pl.rolling_cov(pl.col(c), pl.col(yn), window_size=w, min_samples=mp)
            var = pl.col(yn).rolling_var(window_size=w, min_samples=mp)
            exprs.append((cov / var).alias(c))
        return merged.with_columns(exprs).drop(y_cols)


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
            slope = cov / var_x if var_x > 1e-10 else np.nan
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
    """Polars CAPM 残差动量"""
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
    """Polars 相对市场的余偏度"""
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
    """Polars CAPM 残差偏度"""
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
