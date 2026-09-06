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

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

_SKIP = frozenset({"date", "stock_code"})


def _reject_beta_compat_kwargs(kwargs: dict, *, canonical: str) -> None:
    if "min_stop" in kwargs:
        raise OperatorParameterError(
            f"{canonical}: min_stop is not a supported parameter; use min_periods"
        )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise OperatorParameterError(f"{canonical}: unknown parameter(s): {unknown}")


def _beta_delegate_spec(canonical: str) -> PhysicalImplementationSpec:
    return PhysicalImplementationSpec(
        canonical=canonical,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False,
        supports_streaming=False,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=f"price_volume.polars_price_volume:{canonical}:v3",
        emitter_identity="pandas.rolling_beta",
        parameter_domain_hash=f"{canonical}:strict_window_min_periods:v3",
        semantic_contract_hash=f"{canonical}:paired_finite_date_aligned_ols:v3",
        notes="Date-key-aligned full-panel delegate to the canonical pandas rolling-beta kernel.",
    )


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _align_cols(*dfs: pl.DataFrame) -> list[str]:
    cols = _numeric_cols(dfs[0])
    for df in dfs[1:]:
        cols = [c for c in cols if c in df.columns]
    return cols


def _rolling_beta_polars(
    y: pl.DataFrame,
    x: pl.DataFrame,
    *,
    window: int,
    min_periods: int,
) -> pl.DataFrame:
    """Shared paired-finite beta authority for all Polars beta aliases."""
    from factor_engine.cleaned_operators.price_volume.beta_helpers import compute_rolling_beta
    from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge

    y_pd, x_pd = _aligned_beta_pandas(y, x)
    return panel_pandas_bridge(
        y,
        lambda _: compute_rolling_beta(
            y_pd, x_pd, window, min_periods=min_periods
        ),
    )


def _aligned_beta_pandas(
    y: pl.DataFrame,
    x: pl.DataFrame,
):
    """Validate row keys and align a benchmark to the dependent panel."""
    has_y_date = "date" in y.columns
    has_x_date = "date" in x.columns
    if has_y_date != has_x_date:
        raise OperatorParameterError("beta inputs must both carry date keys or neither")
    y_pd = y.select(_numeric_cols(y)).to_pandas()
    x_pd = x.select(_numeric_cols(x)).to_pandas()
    if has_y_date:
        y_idx = y["date"].to_pandas()
        x_idx = x["date"].to_pandas()
        if y_idx.duplicated().any() or x_idx.duplicated().any():
            raise OperatorParameterError("beta inputs require unique date keys")
        if set(y_idx) != set(x_idx):
            raise OperatorParameterError("beta inputs must have identical date key sets")
        y_pd.index = y_idx
        x_pd.index = x_idx
        x_pd = x_pd.reindex(y_pd.index)
    elif len(y_pd) != len(x_pd):
        raise OperatorParameterError("beta inputs without date keys must have equal row counts")
    return y_pd, x_pd


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
    _physical_spec = _beta_delegate_spec("ts_beta")
    metadata = OperatorMetadata(
        name="m_beta", category="time_series", description="滚动 Beta",
        param_names=["y", "x", "window", "min_periods"], return_type="series", tags=["time_series", "polars"],
        # R19-033..035: 与 pandas ``MovingBeta`` 共享同一 authority —— min_periods
        # default is ``None`` (kernel resolves ``min(5, window)``) so the
        # planning-time ``min_periods <= window`` gate never merges default-5
        # against an explicit window<5 (parity with pandas MovingBeta).
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
            "min_periods": ParamSpec(dtype=int, min=2, default=None, searchable=False,
                                     param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, window: int = 20, min_periods: int | None = None, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.common.strict_params import strict_int
        # WINDOW-SEMANTICS PARITY (pandas reference): pandas rolling_beta has
        # three intertwined semantics that polars rolling_cov/var do NOT
        # reproduce exactly (early-window left-padding denominator, pairwise
        # finite masks where either side missing, and a current-row output
        # mask).  Delegate to the SINGLE reference kernel (rolling_beta) through
        # the pandas bridge — the same exact-parity pattern as ts_sum_decay and
        # ts_kurt — so the polars backend cannot diverge from pandas.
        _reject_beta_compat_kwargs(kwargs, canonical="ts_beta")
        w = strict_int(window, "window", minimum=2)
        # Same UNTAXED-default resolution as pandas MovingBeta: min(5, w).
        if min_periods is None:
            mp = min(5, w)
        else:
            mp = strict_int(min_periods, "min_periods", minimum=2)
            if mp > w:
                raise OperatorParameterError("min_periods must be <= window")
        return _rolling_beta_polars(y, x, window=w, min_periods=mp)


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
                dd[t] = (cum - peak) / peak
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
    from factor_engine.cleaned_operators import _numpy_kernels as kernels

    if mode == "downside":
        return kernels.downside_beta_(ret, mkt, window)
    if mode == "tail":
        return kernels.tail_beta_(ret, mkt, window, q=q)
    raise ValueError(f"unknown conditional beta mode: {mode!r}")


@register_operator(name="downside_beta", category="price_volume", business_category="price_volume", canonical="downside_beta", source="factor_dsl_polars")
class DownsideBetaPolars(SeriesOperator):
    """Polars 下行 Beta"""
    metadata = OperatorMetadata(
        name="downside_beta", category="price_volume", description="下行 Beta",
        param_names=["ret", "benchmark_ret", "window"], return_type="series", tags=["price_volume", "polars"],
        param_aliases={"d": "window"},
    )

    def _calculate_series(self, ret: pl.DataFrame, benchmark_ret: pl.DataFrame, window: int = 60, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        _reject_beta_compat_kwargs(kwargs, canonical="downside_beta")
        w = strict_int(window, "window", minimum=1)
        ret_pd, benchmark_pd = _aligned_beta_pandas(ret, benchmark_ret)
        cols = list(ret_pd.columns)
        if len(benchmark_pd.columns) == 1:
            benchmark_pd = benchmark_pd.rename(columns={benchmark_pd.columns[0]: cols[0]})
            for c in cols[1:]:
                benchmark_pd[c] = benchmark_pd.iloc[:, 0]
        else:
            benchmark_pd = benchmark_pd.reindex(columns=cols)
        out: dict[str, np.ndarray] = {}
        for c in cols:
            out[c] = _conditional_beta(
                ret_pd[c].to_numpy(), benchmark_pd[c].to_numpy(), w, mode="downside"
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
        param_aliases={"d": "window"},
    )

    def _calculate_series(self, ret: pl.DataFrame, benchmark_ret: pl.DataFrame, window: int = 60, q: float = 0.05, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.common.strict_params import strict_float, strict_int

        _reject_beta_compat_kwargs(kwargs, canonical="tail_beta")
        w = strict_int(window, "window", minimum=1)
        q = strict_float(q, "q", minimum=0.0, maximum=1.0)
        if not 0.0 < q < 1.0:
            raise OperatorParameterError("q must be strictly between 0 and 1")
        ret_pd, benchmark_pd = _aligned_beta_pandas(ret, benchmark_ret)
        cols = list(ret_pd.columns)
        if len(benchmark_pd.columns) == 1:
            benchmark_pd = benchmark_pd.rename(columns={benchmark_pd.columns[0]: cols[0]})
            for c in cols[1:]:
                benchmark_pd[c] = benchmark_pd.iloc[:, 0]
        else:
            benchmark_pd = benchmark_pd.reindex(columns=cols)
        out: dict[str, np.ndarray] = {}
        for c in cols:
            out[c] = _conditional_beta(
                ret_pd[c].to_numpy(), benchmark_pd[c].to_numpy(), w, mode="tail", q=q
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
    _physical_spec = _beta_delegate_spec("rolling_beta_to_market")
    metadata = OperatorMetadata(
        name="rolling_beta_to_market",
        category="price_volume",
        description="滚动市场 Beta",
        param_names=["ret", "benchmark_ret", "window"],
        return_type="series",
        tags=["price_volume", "polars"],
    )

    def _calculate_series(self, *panels, **kwargs) -> pl.DataFrame:
        # P0 fix (operator-correctness audit P0-3): canonical surface is
        # ``(ret, benchmark_ret, window)``.  The inherited ``TSBetaPolars``
        # kernel declared ``(y, x, window, min_periods=5)`` so a canonical
        # keyword call ``rolling_beta_to_market(ret=..., benchmark_ret=...,
        # window=3)`` was rejected, and the inherited ``min_periods=5`` default
        # raised ``min_periods must be <= window`` on small windows.  Bind the
        # canonical names and default ``min_periods`` to ``window`` so a
        # ``window=3`` call works and output matches the pandas reference.
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        ret = panels[0] if panels else kwargs.pop("ret", None)
        benchmark_ret = panels[1] if len(panels) > 1 else kwargs.pop("benchmark_ret", None)
        if ret is None or benchmark_ret is None:
            raise TypeError(
                "rolling_beta_to_market requires (ret, benchmark_ret) panels"
            )
        window = kwargs.pop("window", 60)
        min_periods = kwargs.pop("min_periods", None)
        _reject_beta_compat_kwargs(kwargs, canonical="rolling_beta_to_market")
        w = strict_int(window, "window", minimum=2)
        # match the pandas reference ``compute_rolling_beta``: default
        # min_periods = max(2, window // 3)
        mp = w // 3 if min_periods is None else strict_int(min_periods, "min_periods", minimum=2)
        mp = max(2, mp)
        if mp > w:
            raise OperatorParameterError("min_periods must be <= window")
        return _rolling_beta_polars(ret, benchmark_ret, window=w, min_periods=mp)


@register_operator(
    name="rolling_beta",
    category="price_volume",
    business_category="price_volume",
    canonical="rolling_beta",
    source="factor_dsl_polars",
)
class RollingBetaPolars(SeriesOperator):
    """Polars 滚动 Beta：Cov(ret, benchmark_ret) / Var(benchmark_ret)"""
    _physical_spec = _beta_delegate_spec("rolling_beta")
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
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        _reject_beta_compat_kwargs(kwargs, canonical="rolling_beta")
        w = strict_int(window, "window", minimum=2)
        mp = (
            max(2, w // 3)
            if min_periods is None
            else strict_int(min_periods, "min_periods", minimum=2)
        )
        if mp > w:
            raise OperatorParameterError("min_periods must be <= window")
        return _rolling_beta_polars(ret, benchmark_ret, window=w, min_periods=mp)


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
        from factor_engine.cleaned_operators._numpy_kernels import residual_momentum_capm_

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
        from factor_engine.cleaned_operators._numpy_kernels import coskewness_to_market_

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
        from factor_engine.cleaned_operators._numpy_kernels import idio_skew_

        w = int(kwargs.get("d", window))
        return _apply_numpy_kernel_2d(ret, benchmark_ret, w, idio_skew_)
