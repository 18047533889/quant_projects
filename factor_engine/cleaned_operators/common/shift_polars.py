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
    metadata = OperatorMetadata(
        name="prev", category="time_series", description="滞后 1 期",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).shift(1).alias(c) for c in cols])


@register_operator(name="next", category="time_series", business_category="shift_diff_cum", canonical="next", source="factor_dsl_polars")
class NextPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="next", category="time_series", description="前视禁用，输出 NaN",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in cols])


@register_operator(name="Lead", category="time_series", business_category="shift_diff_cum", canonical="Lead", source="factor_dsl_polars")
class LeadPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="Lead", category="time_series", description="前视禁用",
        param_names=["x", "n"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 1, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in cols])


@register_operator(name="expanding_mean", category="data_handling", business_category="data_cleaning", canonical="expanding_mean", source="factor_dsl_polars")
class ExpandingMeanPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="expanding_mean", category="data_handling", description="扩展均值",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            cnt = pl.col(c).is_not_null().cast(pl.Float64).cum_sum()
            mu = pl.col(c).cum_sum() / cnt
            exprs.append(mu.alias(c))
        return x.with_columns(exprs)


@register_operator(name="expanding_std", category="data_handling", business_category="data_cleaning", canonical="expanding_std", source="factor_dsl_polars")
class ExpandingStdPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="expanding_std", category="data_handling", description="扩展标准差",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            cnt = pl.col(c).is_not_null().cast(pl.Float64).cum_sum()
            mu = pl.col(c).cum_sum() / cnt
            mean_sq = (pl.col(c).pow(2).cum_sum()) / cnt
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
    metadata = OperatorMetadata(
        name="expanding_zscore", category="time_series", description="扩展 zscore",
        param_names=["x"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            cnt = pl.col(c).is_not_null().cast(pl.Float64).cum_sum()
            mu = pl.col(c).cum_sum() / cnt
            mean_sq = (pl.col(c).pow(2).cum_sum()) / cnt
            sd = (mean_sq - mu.pow(2)).sqrt()
            exprs.append(((pl.col(c) - mu) / sd).alias(c))
        return x.with_columns(exprs)
