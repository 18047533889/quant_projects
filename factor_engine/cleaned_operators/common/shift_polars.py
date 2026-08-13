# -*- coding: utf-8
"""滞后 / 扩展窗口算子 Polars 实现。"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


@register_operator(name="prev", category="time_series", business_category="shift_diff_cum", canonical="prev", source="factor_dsl_polars")
class PrevPolars(SeriesOperator):
    """Polars 滞后 1 期"""
    metadata = OperatorMetadata(
        name="prev", category="time_series", description="滞后 1 期",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).shift(1).alias(c) for c in cols])


@register_operator(name="expanding_mean", category="data_handling", business_category="data_cleaning", canonical="expanding_mean", source="factor_dsl_polars")
class ExpandingMeanPolars(SeriesOperator):
    """Polars 扩展均值"""
    metadata = OperatorMetadata(
        name="expanding_mean", category="data_handling", description="扩展均值",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            cnt = pl.col(c).is_not_null().cast(pl.Float64).cum_sum()
            mu = pl.col(c).cum_sum() / cnt if cnt > 0 else np.nan
            exprs.append(mu.alias(c))
        return x.with_columns(exprs)


@register_operator(name="expanding_std", category="data_handling", business_category="data_cleaning", canonical="expanding_std", source="factor_dsl_polars")
class ExpandingStdPolars(SeriesOperator):
    """Polars 扩展标准差"""
    metadata = OperatorMetadata(
        name="expanding_std", category="data_handling", description="扩展标准差",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            cnt = pl.col(c).is_not_null().cast(pl.Float64).cum_sum()
            mu = pl.col(c).cum_sum() / cnt if cnt > 0 else np.nan
            mean_sq = (pl.col(c).pow(2).cum_sum()) / cnt if cnt > 0 else np.nan
            var_pop = mean_sq - mu.pow(2)
            var_sample = (
                pl.when(cnt <= 1)
                .then(None)
                .otherwise(var_pop * cnt / (cnt - 1))
            )
            exprs.append(var_sample.sqrt().alias(c))
        return x.with_columns(exprs)


@register_operator(name="expanding_zscore", category="time_series", business_category="shift_diff_cum", canonical="expanding_zscore", source="factor_dsl_polars")
class ExpandingZscorePolars(SeriesOperator):
    """Polars 扩展 zscore"""
    metadata = OperatorMetadata(
        name="expanding_zscore", category="time_series", description="扩展 zscore",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            cnt = pl.col(c).is_not_null().cast(pl.Float64).cum_sum()
            mu = pl.col(c).cum_sum() / cnt if cnt > 0 else np.nan
            mean_sq = (pl.col(c).pow(2).cum_sum()) / cnt if cnt > 0 else np.nan
            sd = (mean_sq - mu.pow(2)).sqrt()
            exprs.append(((pl.col(c) - mu) / sd).alias(c))
        return x.with_columns(exprs)
