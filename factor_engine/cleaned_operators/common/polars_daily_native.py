# -*- coding: utf-8 -*-
"""Daily-surface native Polars backends (expression and eager-panel variants).

``overhaul.cleanup.finalize`` strips Polars adapters that call ``to_numpy`` / ``np.`` /
``rolling_map``.  These implementations use Polars only, so they survive cleanup.
Cross-sectional/group kernels that unpivot and pivot are explicitly classified as
``polars_eager_native``; they are not advertised as lazy or streaming expressions.
"""
from __future__ import annotations

from typing import Callable

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


def _cs_long_transform(
    x: pl.DataFrame,
    transform: Callable[[pl.DataFrame], pl.DataFrame],
) -> pl.DataFrame:
    """Row-wise cross-section via unpivot → window expr → pivot (no NumPy)."""
    cols = _numeric_cols(x)
    if not cols:
        return x
    long = (
        x.select(cols)
        .with_row_index("_r")
        .unpivot(index="_r", on=cols, variable_name="_c", value_name="_v")
    )
    long = transform(long)
    wide = (
        long.pivot(values="_v", index="_r", on="_c", aggregate_function="first")
        .sort("_r")
        .drop("_r")
        .select(cols)
    )
    return _with_meta(wide, x)


def _group_long_transform(
    x: pl.DataFrame,
    group: pl.DataFrame | None,
    transform: Callable[[pl.DataFrame], pl.DataFrame],
) -> pl.DataFrame:
    cols = _numeric_cols(x)
    if not cols:
        return x
    x_long = (
        x.select(cols)
        .with_row_index("_r")
        .unpivot(index="_r", on=cols, variable_name="_c", value_name="_v")
    )
    if group is None:
        raise ValueError("group is required; use an explicit cs_* operator for full cross-sections")
    g_cols = [c for c in cols if c in group.columns]
    if g_cols != cols:
        missing = sorted(set(cols) - set(g_cols))
        raise ValueError(f"group panel is missing columns: {missing}")
    if group.height != x.height:
        raise ValueError("group panel must have the same row count as the value panel")
    g_long = (
        group.select(g_cols)
        .with_row_index("_r")
        .unpivot(index="_r", on=g_cols, variable_name="_c", value_name="_g")
    )
    long = x_long.join(g_long, on=["_r", "_c"], how="left")
    long = transform(long)
    wide = (
        long.pivot(values="_v", index="_r", on="_c", aggregate_function="first")
        .sort("_r")
        .drop("_r")
        .select(cols)
    )
    return _with_meta(wide, x)


# ---------------------------------------------------------------------------
# Elementwise / time-series (column-wise expressions)
# ---------------------------------------------------------------------------


@register_operator(
    name="signed_sqrt",
    category="math",
    business_category="elementwise_math",
    canonical="signed_sqrt",
    source=_SRC,
    backend="polars",
)
class SignedSqrtNative(SeriesOperator):
    """sign(x) * sqrt(|x|) — expression native."""

    metadata = OperatorMetadata(
        name="signed_sqrt",
        category="math",
        description="保留符号开方",
        param_names=["x"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns(
            [(pl.col(c).sign() * pl.col(c).abs().sqrt()).alias(c) for c in cols]
        )


@register_operator(
    name="ts_log_return",
    category="time_series",
    business_category="time_series",
    canonical="ts_log_return",
    source=_SRC,
    backend="polars",
)
class TSLogReturnNative(SeriesOperator):
    """ln(x_t / x_{t-d}) with non-positive → null."""

    metadata = OperatorMetadata(
        name="ts_log_return",
        category="time_series",
        description="d 期对数收益",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        n = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            prev = pl.col(c).shift(n)
            exprs.append(
                pl.when((pl.col(c) > 0) & (prev > 0))
                .then(pl.col(c).log() - prev.log())
                .otherwise(None)
                .alias(c)
            )
        return x.with_columns(exprs)


@register_operator(
    name="ts_rank",
    category="time_series",
    business_category="time_series",
    canonical="ts_rank",
    source=_SRC,
    backend="polars",
)
class TSRankNative(SeriesOperator):
    """Rolling percentile rank of the current observation in the window."""

    metadata = OperatorMetadata(
        name="ts_rank",
        category="time_series",
        description="滚动百分位排名",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        if "d" in kwargs:
            from backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("ts_rank accepts window; d is not a supported runtime alias")
        w = strict_integer(window, "window", minimum=1)
        min_periods = strict_integer(kwargs.get("min_periods", 1), "min_periods", minimum=1)
        if min_periods > w:
            from backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        cols = _numeric_cols(x)
        return x.with_columns(
            [
                (
                    pl.col(c).rolling_rank(
                        window_size=w, method="average", min_samples=min_periods
                    )
                    / pl.col(c)
                    .is_not_null()
                    .cast(pl.Float64)
                    .rolling_sum(window_size=w, min_samples=min_periods)
                ).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="ts_sharpe",
    category="time_series",
    business_category="time_series",
    canonical="ts_sharpe",
    source=_SRC,
    backend="polars",
)
class TSSharpeNative(SeriesOperator):
    """Rolling Sharpe = mean/std * sqrt(ann_factor); zero std → null."""

    metadata = OperatorMetadata(
        name="ts_sharpe",
        category="time_series",
        description="滚动夏普",
        param_names=["x", "window", "ann_factor"],
        return_type="series",
        tags=["time_series", "polars", "native"],
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        window: int = 60,
        ann_factor: float = 252.0,
        min_periods: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer

        w = strict_integer(window, "window", minimum=2)
        mp = (
            strict_integer(min_periods, "min_periods", minimum=2)
            if min_periods is not None
            else max(2, w // 3)
        )
        if mp > w:
            from backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        scale = strict_finite_scalar(ann_factor, "ann_factor", minimum=0.0) ** 0.5
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w, min_samples=mp)
            std = pl.col(c).rolling_std(window_size=w, min_samples=mp)
            exprs.append(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise(mean / std * scale)
                .alias(c)
            )
        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Cross-section (unpivot window expressions)
# ---------------------------------------------------------------------------


@register_operator(
    name="cs_std",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_std",
    source=_SRC,
    backend="polars",
)
class CSStdNative(SeriesOperator):
    metadata = OperatorMetadata(
        name="cs_std",
        category="cross_sectional",
        description="截面标准差广播",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return _cs_long_transform(
            x,
            lambda long: long.with_columns(
                pl.col("_v").std().over("_r").alias("_v")
            ),
        )


@register_operator(
    name="cs_mad",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_mad",
    source=_SRC,
    backend="polars",
)
class CSMadNative(SeriesOperator):
    metadata = OperatorMetadata(
        name="cs_mad",
        category="cross_sectional",
        description="截面 MAD 广播",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            med = pl.col("_v").median().over("_r")
            return long.with_columns(
                (pl.col("_v") - med).abs().median().over("_r").alias("_v")
            )

        return _cs_long_transform(x, _xform)


@register_operator(
    name="cs_mad_zscore",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_mad_zscore",
    source=_SRC,
    backend="polars",
)
class CSMadZscoreNative(SeriesOperator):
    metadata = OperatorMetadata(
        name="cs_mad_zscore",
        category="cross_sectional",
        description="MAD 稳健 Z-Score",
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            med = pl.col("_v").median().over("_r")
            mad = (pl.col("_v") - med).abs().median().over("_r")
            return long.with_columns(
                pl.when(mad.is_null() | (mad == 0))
                .then(None)
                .otherwise((pl.col("_v") - med) / mad)
                .alias("_v")
            )

        return _cs_long_transform(x, _xform)


@register_operator(
    name="normalize",
    category="math",
    business_category="elementwise_math",
    canonical="normalize",
    source=_SRC,
    backend="polars",
)
class NormalizeNative(SeriesOperator):
    """Row min-max to [0,1]; constant row → 0.5 (matches SQL emitter)."""

    metadata = OperatorMetadata(
        name="normalize",
        category="math",
        description="截面 min-max 归一化",
        param_names=["x"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            lo = pl.col("_v").min().over("_r")
            hi = pl.col("_v").max().over("_r")
            return long.with_columns(
                pl.when(lo.is_null() | hi.is_null())
                .then(None)
                .when(hi == lo)
                .then(pl.lit(0.5))
                .otherwise((pl.col("_v") - lo) / (hi - lo))
                .alias("_v")
            )

        return _cs_long_transform(x, _xform)


@register_operator(
    name="winsorize",
    category="math",
    business_category="elementwise_math",
    canonical="winsorize",
    source=_SRC,
    backend="polars",
)
class WinsorizeNative(SeriesOperator):
    metadata = OperatorMetadata(
        name="winsorize",
        category="math",
        description="截面缩尾",
        param_names=["x", "lower", "upper"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        lower: float = 0.05,
        upper: float = 0.95,
        **kwargs,
    ) -> pl.DataFrame:
        lo_q = float(lower)
        hi_q = float(upper)

        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            lo = pl.col("_v").quantile(lo_q, interpolation="linear").over("_r")
            hi = pl.col("_v").quantile(hi_q, interpolation="linear").over("_r")
            return long.with_columns(pl.col("_v").clip(lo, hi).alias("_v"))

        return _cs_long_transform(x, _xform)


# ---------------------------------------------------------------------------
# Group ops
# ---------------------------------------------------------------------------


@register_operator(
    name="group_neutralize",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_neutralize",
    source=_SRC,
    backend="polars",
)
class GroupNeutralizeNative(SeriesOperator):
    """Group demean (alias of neutralize)."""

    metadata = OperatorMetadata(
        name="group_neutralize",
        category="cross_sectional",
        description="组内去均值",
        param_names=["x", "group"],
        return_type="series",
        tags=["group", "polars", "native"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, group: pl.DataFrame | None = None, **kwargs
    ) -> pl.DataFrame:
        # Wide-panel Polars unpivot is far slower than the vectorized numpy kernel;
        # bridge to pandas_numpy for eager wide frames. Long-table / SQL keep their
        # native emitters in polars_expr_emitter / sql_pushdown.
        from cleaned_operators.common._polars_bridge import bridge_registry

        if group is None:
            return bridge_registry("group_neutralize", x)
        return bridge_registry("group_neutralize", x, group)


@register_operator(
    name="size_neutralize",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="size_neutralize",
    source=_SRC,
    backend="polars",
)
class SizeNeutralizeNative(SeriesOperator):
    """Size neutralize: residual vs log(max(market_cap, 1))."""

    metadata = OperatorMetadata(
        name="size_neutralize",
        category="group_neutralization",
        description="市值中性化（Polars 原生）",
        param_names=["x", "market_cap"],
        return_type="series",
        tags=["group", "size", "polars", "native"],
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        market_cap: pl.DataFrame | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        if market_cap is None:
            raise ValueError("size_neutralize requires market_cap")
        from cleaned_operators.common._polars_bridge import bridge_registry

        return bridge_registry("size_neutralize", x, market_cap)


@register_operator(
    name="industry_size_neutralize",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="industry_size_neutralize",
    source=_SRC,
    backend="polars",
)
class IndustrySizeNeutralizeNative(SeriesOperator):
    """Industry demean then size residual (Polars native)."""

    metadata = OperatorMetadata(
        name="industry_size_neutralize",
        category="group_neutralization",
        description="行业+市值双中性（Polars 原生）",
        param_names=["x", "industry", "market_cap"],
        return_type="series",
        tags=["group", "industry", "size", "polars", "native"],
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        industry: pl.DataFrame | None = None,
        market_cap: pl.DataFrame | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        if industry is None or market_cap is None:
            raise ValueError("industry_size_neutralize requires (x, industry, market_cap)")
        from cleaned_operators.common._polars_bridge import bridge_registry

        return bridge_registry("industry_size_neutralize", x, industry, market_cap)


@register_operator(
    name="group_normalize",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_normalize",
    source=_SRC,
    backend="polars",
)
class GroupNormalizeNative(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_normalize",
        category="cross_sectional",
        description="组内 min-max",
        param_names=["x", "group"],
        return_type="series",
        tags=["group", "polars", "native"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, group: pl.DataFrame | None = None, **kwargs
    ) -> pl.DataFrame:
        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            key = ["_r", "_g"] if "_g" in long.columns else ["_r"]
            lo = pl.col("_v").min().over(key)
            hi = pl.col("_v").max().over(key)
            return long.with_columns(
                pl.when(lo.is_null() | hi.is_null())
                .then(None)
                .when(hi == lo)
                .then(pl.lit(0.5))
                .otherwise((pl.col("_v") - lo) / (hi - lo))
                .alias("_v")
            )

        return _group_long_transform(x, group, _xform)


@register_operator(
    name="group_winsorize",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_winsorize",
    source=_SRC,
    backend="polars",
)
class GroupWinsorizeNative(SeriesOperator):
    metadata = OperatorMetadata(
        name="group_winsorize",
        category="cross_sectional",
        description="组内缩尾",
        param_names=["x", "group", "lower", "upper"],
        return_type="series",
        tags=["group", "polars", "native"],
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        group: pl.DataFrame | None = None,
        lower: float = 0.05,
        upper: float = 0.95,
        **kwargs,
    ) -> pl.DataFrame:
        lo_q = float(lower)
        hi_q = float(upper)

        def _xform(long: pl.DataFrame) -> pl.DataFrame:
            key = ["_r", "_g"] if "_g" in long.columns else ["_r"]
            lo = pl.col("_v").quantile(lo_q, interpolation="linear").over(key)
            hi = pl.col("_v").quantile(hi_q, interpolation="linear").over(key)
            return long.with_columns(pl.col("_v").clip(lo, hi).alias("_v"))

        return _group_long_transform(x, group, _xform)
