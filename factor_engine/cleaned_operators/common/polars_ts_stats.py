# -*- coding: utf-8 -*-
"""Time-series statistics operators - Polars native implementations.

Rolling statistics (skew, quantiles, tail measures), drawdown metrics,
distance to highs/lows, channel position, and swing analysis.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# Basic rolling statistics
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_skew",
    category="time_series",
    business_category="statistics",
    canonical="ts_skew",
    source=_SRC,
    backend="polars")
class TSSkewNative(SeriesOperator):
    """Rolling skewness."""

    metadata = OperatorMetadata(
        name="ts_skew",
        category="time_series",
        description="滚动偏度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "statistics", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=3)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            # Polars doesn't have rolling_skew, compute manually
            val = x[c]
            mean = val.rolling_mean(window_size=w)
            std = val.rolling_std(window_size=w)
            m3 = ((val - mean) ** 3).rolling_mean(window_size=w)
            skew = pl.when((std.is_null()) | (std == 0)).then(None).otherwise(m3 / (std ** 3))
            exprs.append(skew.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_trimmed_mean",
    category="time_series",
    business_category="statistics",
    canonical="ts_trimmed_mean",
    source=_SRC,
    backend="polars")
class TSTrimmedMeanNative(SeriesOperator):
    """Rolling trimmed mean (exclude top/bottom quantiles)."""

    metadata = OperatorMetadata(
        name="ts_trimmed_mean",
        category="time_series",
        description="滚动截尾均值",
        param_names=["x", "window", "trim_pct"],
        return_type="series",
        tags=["time_series", "statistics", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=5, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "trim_pct": ParamSpec(dtype=float, min=0.0, max=0.5, default=0.1, searchable=False, param_role=ParamRole.TUNING),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, trim_pct: float = 0.1, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=5)
        trim = float(trim_pct)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            lo_q = val.rolling_quantile(quantile=trim, window_size=w, interpolation="linear")
            hi_q = val.rolling_quantile(quantile=1.0 - trim, window_size=w, interpolation="linear")
            # Clip values and compute mean
            clipped = val.clip(lo_q, hi_q)
            trimmed_mean = clipped.rolling_mean(window_size=w)
            exprs.append(trimmed_mean.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_median",
    category="time_series",
    business_category="statistics",
    canonical="ts_median",
    source=_SRC,
    backend="polars")
class TSMedianNative(SeriesOperator):
    """Rolling median."""

    metadata = OperatorMetadata(
        name="ts_median",
        category="time_series",
        description="滚动中位数",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "statistics", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            median = x[c].rolling_median(window_size=w)
            exprs.append(median.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_qn_scale",
    category="time_series",
    business_category="statistics",
    canonical="ts_qn_scale",
    source=_SRC,
    backend="polars")
class TSQnScaleNative(SeriesOperator):
    """Rolling Qn robust scale estimator (IQR / 1.349)."""

    metadata = OperatorMetadata(
        name="ts_qn_scale",
        category="time_series",
        description="滚动Qn稳健尺度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "statistics", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=4, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=4)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            q25 = val.rolling_quantile(quantile=0.25, window_size=w, interpolation="linear")
            q75 = val.rolling_quantile(quantile=0.75, window_size=w, interpolation="linear")
            iqr = q75 - q25
            qn = iqr / pl.lit(1.349)
            exprs.append(qn.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_quantile_range",
    category="time_series",
    business_category="statistics",
    canonical="ts_quantile_range",
    source=_SRC,
    backend="polars")
class TSQuantileRangeNative(SeriesOperator):
    """Rolling quantile range: Q_upper - Q_lower."""

    metadata = OperatorMetadata(
        name="ts_quantile_range",
        category="time_series",
        description="滚动分位数范围",
        param_names=["x", "window", "lower", "upper"],
        return_type="series",
        tags=["time_series", "statistics", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, lower: float = 0.25, upper: float = 0.75, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        lo_q = float(lower)
        hi_q = float(upper)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            q_lo = val.rolling_quantile(quantile=lo_q, window_size=w, interpolation="linear")
            q_hi = val.rolling_quantile(quantile=hi_q, window_size=w, interpolation="linear")
            exprs.append((q_hi - q_lo).alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_quantile_kurtosis",
    category="time_series",
    business_category="statistics",
    canonical="ts_quantile_kurtosis",
    source=_SRC,
    backend="polars")
class TSQuantileKurtosisNative(SeriesOperator):
    """Rolling quantile-based kurtosis: (Q_0.875 - Q_0.125) / (Q_0.75 - Q_0.25)."""

    metadata = OperatorMetadata(
        name="ts_quantile_kurtosis",
        category="time_series",
        description="滚动分位数峰度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "statistics", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=8, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=8)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            q125 = val.rolling_quantile(quantile=0.125, window_size=w, interpolation="linear")
            q25 = val.rolling_quantile(quantile=0.25, window_size=w, interpolation="linear")
            q75 = val.rolling_quantile(quantile=0.75, window_size=w, interpolation="linear")
            q875 = val.rolling_quantile(quantile=0.875, window_size=w, interpolation="linear")
            iqr = q75 - q25
            tail_range = q875 - q125
            kurt = pl.when((iqr.is_null()) | (iqr == 0)).then(None).otherwise(tail_range / iqr)
            exprs.append(kurt.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_quantile_skew",
    category="time_series",
    business_category="statistics",
    canonical="ts_quantile_skew",
    source=_SRC,
    backend="polars")
class TSQuantileSkewNative(SeriesOperator):
    """Rolling quantile-based skew: (Q_0.75 + Q_0.25 - 2*Q_0.5) / (Q_0.75 - Q_0.25)."""

    metadata = OperatorMetadata(
        name="ts_quantile_skew",
        category="time_series",
        description="滚动分位数偏度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "statistics", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=3)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            q25 = val.rolling_quantile(quantile=0.25, window_size=w, interpolation="linear")
            q50 = val.rolling_quantile(quantile=0.50, window_size=w, interpolation="linear")
            q75 = val.rolling_quantile(quantile=0.75, window_size=w, interpolation="linear")
            iqr = q75 - q25
            skew = (pl.when((iqr.is_null()) | (iqr == 0)).then(None).otherwise((q75 + q25 - pl.lit(2.0) * q50)) / iqr)
            exprs.append(skew.alias(c))
        result = x.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Downside/upside deviation
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_downside_deviation",
    category="time_series",
    business_category="risk",
    canonical="ts_downside_deviation",
    source=_SRC,
    backend="polars")
class TSDownsideDeviationNative(SeriesOperator):
    """Rolling downside deviation: sqrt(mean((min(x - target, 0))^2))."""

    metadata = OperatorMetadata(
        name="ts_downside_deviation",
        category="time_series",
        description="滚动下行标准差",
        param_names=["x", "window", "target"],
        return_type="series",
        tags=["time_series", "risk", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "target": ParamSpec(dtype=float, default=0.0, searchable=False, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, target: float = 0.0, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        tgt = float(target)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            downside = pl.when(val < tgt).then((val - tgt) ** 2).otherwise(0.0)
            dd = downside.rolling_mean(window_size=w).sqrt()
            exprs.append(dd.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_upside_deviation",
    category="time_series",
    business_category="risk",
    canonical="ts_upside_deviation",
    source=_SRC,
    backend="polars")
class TSUpsideDeviationNative(SeriesOperator):
    """Rolling upside deviation: sqrt(mean((max(x - target, 0))^2))."""

    metadata = OperatorMetadata(
        name="ts_upside_deviation",
        category="time_series",
        description="滚动上行标准差",
        param_names=["x", "window", "target"],
        return_type="series",
        tags=["time_series", "risk", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "target": ParamSpec(dtype=float, default=0.0, searchable=False, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, target: float = 0.0, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        tgt = float(target)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            upside = pl.when(val > tgt).then((val - tgt) ** 2).otherwise(0.0)
            ud = upside.rolling_mean(window_size=w).sqrt()
            exprs.append(ud.alias(c))
        result = x.with_columns(exprs)
        return result




# ---------------------------------------------------------------------------
# Expected shortfall and partial moments
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_expected_shortfall",
    category="time_series",
    business_category="risk",
    canonical="ts_expected_shortfall",
    source=_SRC,
    backend="polars")
class TSExpectedShortfallNative(SeriesOperator):
    """Rolling expected shortfall (CVaR): mean of losses beyond VaR threshold."""

    metadata = OperatorMetadata(
        name="ts_expected_shortfall",
        category="time_series",
        description="滚动期望损失",
        param_names=["x", "window", "alpha"],
        return_type="series",
        tags=["time_series", "risk", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
            "alpha": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.05, searchable=False, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, alpha: float = 0.05, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=10)
        a = float(alpha)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            var_threshold = val.rolling_quantile(quantile=a, window_size=w, interpolation="linear")
            # Mean of values below VaR
            below_var = pl.when(val <= var_threshold).then(val).otherwise(None)
            es = below_var.rolling_mean(window_size=w)
            exprs.append(es.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_expected_shortfall_asymmetry",
    category="time_series",
    business_category="risk",
    canonical="ts_expected_shortfall_asymmetry",
    source=_SRC,
    backend="polars")
class TSExpectedShortfallAsymmetryNative(SeriesOperator):
    """(ES_upside - |ES_downside|) / (ES_upside + |ES_downside|)."""

    metadata = OperatorMetadata(
        name="ts_expected_shortfall_asymmetry",
        category="time_series",
        description="期望损失不对称度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "risk", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=10)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            var_down = val.rolling_quantile(quantile=0.05, window_size=w, interpolation="linear")
            var_up = val.rolling_quantile(quantile=0.95, window_size=w, interpolation="linear")
            below = pl.when(val <= var_down).then(val).otherwise(None)
            above = pl.when(val >= var_up).then(val).otherwise(None)
            es_down = below.rolling_mean(window_size=w).abs()
            es_up = above.rolling_mean(window_size=w)
            total = es_up + es_down
            asymmetry = (pl.when((total.is_null()) | (total == 0)).then(None).otherwise((es_up - es_down)) / total)
            exprs.append(asymmetry.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_lower_partial_moment",
    category="time_series",
    business_category="risk",
    canonical="ts_lower_partial_moment",
    source=_SRC,
    backend="polars")
class TSLowerPartialMomentNative(SeriesOperator):
    """Rolling lower partial moment: mean((max(target - x, 0))^order)."""

    metadata = OperatorMetadata(
        name="ts_lower_partial_moment",
        category="time_series",
        description="滚动下偏矩",
        param_names=["x", "window", "target", "order"],
        return_type="series",
        tags=["time_series", "risk", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "target": ParamSpec(dtype=float, default=0.0, searchable=False, param_role=ParamRole.THRESHOLD),
            "order": ParamSpec(dtype=int, min=1, default=2, searchable=False, param_role=ParamRole.TUNING),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, target: float = 0.0, order: int = 2, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        ord = strict_integer(order, "order", minimum=1)
        tgt = float(target)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            shortfall = pl.when(val < tgt).then(tgt - val).otherwise(0.0)
            lpm = (shortfall ** ord).rolling_mean(window_size=w)
            exprs.append(lpm.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_upper_partial_moment",
    category="time_series",
    business_category="risk",
    canonical="ts_upper_partial_moment",
    source=_SRC,
    backend="polars")
class TSUpperPartialMomentNative(SeriesOperator):
    """Rolling upper partial moment: mean((max(x - target, 0))^order)."""

    metadata = OperatorMetadata(
        name="ts_upper_partial_moment",
        category="time_series",
        description="滚动上偏矩",
        param_names=["x", "window", "target", "order"],
        return_type="series",
        tags=["time_series", "risk", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "target": ParamSpec(dtype=float, default=0.0, searchable=False, param_role=ParamRole.THRESHOLD),
            "order": ParamSpec(dtype=int, min=1, default=2, searchable=False, param_role=ParamRole.TUNING),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, target: float = 0.0, order: int = 2, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        ord = strict_integer(order, "order", minimum=1)
        tgt = float(target)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            excess = pl.when(val > tgt).then(val - tgt).otherwise(0.0)
            upm = (excess ** ord).rolling_mean(window_size=w)
            exprs.append(upm.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_semivariance_balance",
    category="time_series",
    business_category="risk",
    canonical="ts_semivariance_balance",
    source=_SRC,
    backend="polars")
class TSSemivarianceBalanceNative(SeriesOperator):
    """(upside_variance - downside_variance) / (upside_variance + downside_variance)."""

    metadata = OperatorMetadata(
        name="ts_semivariance_balance",
        category="time_series",
        description="半方差平衡",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "risk", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            downside = pl.when(val < 0).then(val ** 2).otherwise(0.0)
            upside = pl.when(val > 0).then(val ** 2).otherwise(0.0)
            down_var = downside.rolling_mean(window_size=w)
            up_var = upside.rolling_mean(window_size=w)
            total = up_var + down_var
            balance = (pl.when((total.is_null()) | (total == 0)).then(None).otherwise((up_var - down_var)) / total)
            exprs.append(balance.alias(c))
        result = x.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Tail statistics
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_tail_mean",
    category="time_series",
    business_category="tail",
    canonical="ts_tail_mean",
    source=_SRC,
    backend="polars")
class TSTailMeanNative(SeriesOperator):
    """Rolling mean of tail beyond quantile threshold."""

    metadata = OperatorMetadata(
        name="ts_tail_mean",
        category="time_series",
        description="滚动尾部均值",
        param_names=["x", "window", "quantile"],
        return_type="series",
        tags=["time_series", "tail", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=5, default=60, searchable=True, param_role=ParamRole.HORIZON),
            "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.95, searchable=False, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, quantile: float = 0.95, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=5)
        q = float(quantile)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            threshold = val.rolling_quantile(quantile=q, window_size=w, interpolation="linear")
            tail = pl.when(val >= threshold).then(val).otherwise(None)
            tail_mean = tail.rolling_mean(window_size=w)
            exprs.append(tail_mean.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_tail_ratio",
    category="time_series",
    business_category="tail",
    canonical="ts_tail_ratio",
    source=_SRC,
    backend="polars")
class TSTailRatioNative(SeriesOperator):
    """Rolling tail ratio: |Q_0.95| / |Q_0.05|."""

    metadata = OperatorMetadata(
        name="ts_tail_ratio",
        category="time_series",
        description="滚动尾部比率",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "tail", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=10)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            q05 = val.rolling_quantile(quantile=0.05, window_size=w, interpolation="linear")
            q95 = val.rolling_quantile(quantile=0.95, window_size=w, interpolation="linear")
            ratio = (pl.when((q05.is_null()) | (q05 == 0)).then(None).otherwise(q95.abs()) / (q05.abs())) if (q05.abs()) != 0 else np.nan
            exprs.append(ratio.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_tail_imbalance",
    category="time_series",
    business_category="tail",
    canonical="ts_tail_imbalance",
    source=_SRC,
    backend="polars")
class TSTailImbalanceNative(SeriesOperator):
    """Rolling tail imbalance: (Q_0.95 + Q_0.05) / (Q_0.95 - Q_0.05)."""

    metadata = OperatorMetadata(
        name="ts_tail_imbalance",
        category="time_series",
        description="滚动尾部不对称",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "tail", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=10)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            q05 = val.rolling_quantile(quantile=0.05, window_size=w, interpolation="linear")
            q95 = val.rolling_quantile(quantile=0.95, window_size=w, interpolation="linear")
            tail_range = q95 - q05
            imbalance = (pl.when((tail_range.is_null()) | (tail_range == 0)).then(None).otherwise((q95 + q05)) / tail_range)
            exprs.append(imbalance.alias(c))
        result = x.with_columns(exprs)
        return result




# ---------------------------------------------------------------------------
# Drawdown metrics
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_max_drawdown",
    category="time_series",
    business_category="drawdown",
    canonical="ts_max_drawdown",
    source=_SRC,
    backend="polars")
class TSMaxDrawdownNative(SeriesOperator):
    """Rolling maximum drawdown from peak."""

    metadata = OperatorMetadata(
        name="ts_max_drawdown",
        category="time_series",
        description="滚动最大回撤",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "drawdown", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            peak = val.rolling_max(window_size=w)
            drawdown = (pl.when((peak.is_null()) | (peak == 0)).then(None).otherwise((val - peak)) / peak)
            max_dd = drawdown.rolling_min(window_size=w)
            exprs.append(max_dd.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_current_drawdown_duration",
    category="time_series",
    business_category="drawdown",
    canonical="ts_current_drawdown_duration",
    source=_SRC,
    backend="polars")
class TSCurrentDrawdownDurationNative(SeriesOperator):
    """Days since last peak."""

    metadata = OperatorMetadata(
        name="ts_current_drawdown_duration",
        category="time_series",
        description="当前回撤持续天数",
        param_names=["x"],
        return_type="series",
        tags=["time_series", "drawdown", "polars", "native"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            # Check if current value is new peak
            cummax = val.cum_max()
            is_peak = (val == cummax)
            # Count days since last peak
            block = is_peak.cum_sum()
            duration = pl.int_range(0, pl.len()).over(block) - pl.int_range(0, pl.len()).shift(1).fill_null(0).over(block)
            exprs.append(duration.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_current_drawdown_area",
    category="time_series",
    business_category="drawdown",
    canonical="ts_current_drawdown_area",
    source=_SRC,
    backend="polars")
class TSCurrentDrawdownAreaNative(SeriesOperator):
    """Cumulative drawdown area since last peak."""

    metadata = OperatorMetadata(
        name="ts_current_drawdown_area",
        category="time_series",
        description="当前回撤面积",
        param_names=["x"],
        return_type="series",
        tags=["time_series", "drawdown", "polars", "native"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            cummax = val.cum_max()
            is_peak = (val == cummax)
            drawdown = (pl.when((cummax.is_null()) | (cummax == 0)).then(None).otherwise((val - cummax)) / cummax)
            # Sum drawdown since last peak
            block = is_peak.cum_sum()
            area = drawdown.cum_sum() - drawdown.cum_sum().shift(1).fill_null(0).over(block)
            exprs.append(area.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_time_under_water",
    category="time_series",
    business_category="drawdown",
    canonical="ts_time_under_water",
    source=_SRC,
    backend="polars")
class TSTimeUnderWaterNative(SeriesOperator):
    """Rolling fraction of time in drawdown."""

    metadata = OperatorMetadata(
        name="ts_time_under_water",
        category="time_series",
        description="滚动水下时间比例",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "drawdown", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            peak = val.rolling_max(window_size=w)
            under_water = (val < peak).cast(pl.Float64)
            time_uw = under_water.rolling_mean(window_size=w)
            exprs.append(time_uw.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_recovery_fraction",
    category="time_series",
    business_category="drawdown",
    canonical="ts_recovery_fraction",
    source=_SRC,
    backend="polars")
class TSRecoveryFractionNative(SeriesOperator):
    """Current value / rolling max (recovery from drawdown)."""

    metadata = OperatorMetadata(
        name="ts_recovery_fraction",
        category="time_series",
        description="回撤恢复比例",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "drawdown", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            peak = val.rolling_max(window_size=w)
            recovery = pl.when((peak.is_null()) | (peak == 0)).then(None).otherwise(val / peak)
            exprs.append(recovery.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_max_buildup",
    category="time_series",
    business_category="buildup",
    canonical="ts_max_buildup",
    source=_SRC,
    backend="polars")
class TSMaxBuildupNative(SeriesOperator):
    """Rolling maximum buildup from trough."""

    metadata = OperatorMetadata(
        name="ts_max_buildup",
        category="time_series",
        description="滚动最大上涨",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "buildup", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            trough = val.rolling_min(window_size=w)
            buildup = (pl.when((trough.is_null()) | (trough == 0)).then(None).otherwise((val - trough)) / trough)
            max_bu = buildup.rolling_max(window_size=w)
            exprs.append(max_bu.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_realized_quarticity",
    category="time_series",
    business_category="volatility",
    canonical="ts_realized_quarticity",
    source=_SRC,
    backend="polars")
class TSRealizedQuarticityNative(SeriesOperator):
    """Rolling realized quarticity: sum(x^4)."""

    metadata = OperatorMetadata(
        name="ts_realized_quarticity",
        category="time_series",
        description="滚动已实现四次方波动",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "volatility", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            quarticity = (val ** 4).rolling_sum(window_size=w)
            exprs.append(quarticity.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ulcer_index",
    category="time_series",
    business_category="drawdown",
    canonical="ulcer_index",
    source=_SRC,
    backend="polars")
class UlcerIndexNative(SeriesOperator):
    """Ulcer index: sqrt(mean(drawdown^2))."""

    metadata = OperatorMetadata(
        name="ulcer_index",
        category="time_series",
        description="溃疡指数",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "drawdown", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=14, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 14, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            peak = val.rolling_max(window_size=w)
            pct_dd = (pl.when((peak.is_null()) | (peak == 0)).then(0.0).otherwise((val - peak)) / peak * 100.0)
            ulcer = (pct_dd ** 2).rolling_mean(window_size=w).sqrt()
            exprs.append(ulcer.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_staleness",
    category="time_series",
    business_category="staleness",
    canonical="ts_staleness",
    source=_SRC,
    backend="polars")
class TSStalenessNative(SeriesOperator):
    """Rolling staleness: fraction of unchanged values."""

    metadata = OperatorMetadata(
        name="ts_staleness",
        category="time_series",
        description="滚动陈旧度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "staleness", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            unchanged = (val == val.shift(1)).cast(pl.Float64)
            staleness = unchanged.rolling_mean(window_size=w)
            exprs.append(staleness.alias(c))
        result = x.with_columns(exprs)
        return result




# ---------------------------------------------------------------------------
# Distance to highs/lows
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_distance_to_high",
    category="time_series",
    business_category="position",
    canonical="ts_distance_to_high",
    source=_SRC,
    backend="polars")
class TSDistanceToHighNative(SeriesOperator):
    """(rolling_max - current) / rolling_max."""

    metadata = OperatorMetadata(
        name="ts_distance_to_high",
        category="time_series",
        description="距高点距离",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "position", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            high = val.rolling_max(window_size=w)
            dist = (pl.when((high.is_null()) | (high == 0)).then(None).otherwise((high - val)) / high)
            exprs.append(dist.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_distance_to_low",
    category="time_series",
    business_category="position",
    canonical="ts_distance_to_low",
    source=_SRC,
    backend="polars")
class TSDistanceToLowNative(SeriesOperator):
    """(current - rolling_min) / rolling_min."""

    metadata = OperatorMetadata(
        name="ts_distance_to_low",
        category="time_series",
        description="距低点距离",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "position", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            dist = (pl.when((low.is_null()) | (low == 0)).then(None).otherwise((val - low)) / low)
            exprs.append(dist.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_prev_high",
    category="time_series",
    business_category="position",
    canonical="ts_prev_high",
    source=_SRC,
    backend="polars")
class TSPrevHighNative(SeriesOperator):
    """Rolling maximum value."""

    metadata = OperatorMetadata(
        name="ts_prev_high",
        category="time_series",
        description="前期高点",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "position", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            high = x[c].rolling_max(window_size=w)
            exprs.append(high.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_prev_low",
    category="time_series",
    business_category="position",
    canonical="ts_prev_low",
    source=_SRC,
    backend="polars")
class TSPrevLowNative(SeriesOperator):
    """Rolling minimum value."""

    metadata = OperatorMetadata(
        name="ts_prev_low",
        category="time_series",
        description="前期低点",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "position", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            low = x[c].rolling_min(window_size=w)
            exprs.append(low.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_breakout_high",
    category="time_series",
    business_category="breakout",
    canonical="ts_breakout_high",
    source=_SRC,
    backend="polars")
class TSBreakoutHighNative(SeriesOperator):
    """Binary: 1 if current > rolling_max(lookback), 0 otherwise."""

    metadata = OperatorMetadata(
        name="ts_breakout_high",
        category="time_series",
        description="突破前高",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "breakout", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            prev_high = val.shift(1).rolling_max(window_size=w)
            breakout = pl.when(val > prev_high).then(1).otherwise(0)
            exprs.append(breakout.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_breakdown_low",
    category="time_series",
    business_category="breakout",
    canonical="ts_breakdown_low",
    source=_SRC,
    backend="polars")
class TSBreakdownLowNative(SeriesOperator):
    """Binary: 1 if current < rolling_min(lookback), 0 otherwise."""

    metadata = OperatorMetadata(
        name="ts_breakdown_low",
        category="time_series",
        description="跌破前低",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "breakout", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            prev_low = val.shift(1).rolling_min(window_size=w)
            breakdown = pl.when(val < prev_low).then(1).otherwise(0)
            exprs.append(breakdown.alias(c))
        result = x.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Channel position and width
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_channel_position",
    category="time_series",
    business_category="channel",
    canonical="ts_channel_position",
    source=_SRC,
    backend="polars")
class TSChannelPositionNative(SeriesOperator):
    """(x - low) / (high - low); position in [0, 1]."""

    metadata = OperatorMetadata(
        name="ts_channel_position",
        category="time_series",
        description="通道位置",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "channel", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            high = val.rolling_max(window_size=w)
            channel_range = high - low
            pos = (pl.when((channel_range.is_null()) | (channel_range == 0)).then(pl.lit(0.5)).otherwise((val - low)) / channel_range)
            exprs.append(pos.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_channel_width",
    category="time_series",
    business_category="channel",
    canonical="ts_channel_width",
    source=_SRC,
    backend="polars")
class TSChannelWidthNative(SeriesOperator):
    """Absolute channel width: high - low."""

    metadata = OperatorMetadata(
        name="ts_channel_width",
        category="time_series",
        description="通道宽度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "channel", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            high = val.rolling_max(window_size=w)
            width = high - low
            exprs.append(width.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_channel_width_pct",
    category="time_series",
    business_category="channel",
    canonical="ts_channel_width_pct",
    source=_SRC,
    backend="polars")
class TSChannelWidthPctNative(SeriesOperator):
    """Relative channel width: (high - low) / low."""

    metadata = OperatorMetadata(
        name="ts_channel_width_pct",
        category="time_series",
        description="通道宽度百分比",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "channel", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            high = val.rolling_max(window_size=w)
            width_pct = (pl.when((low.is_null()) | (low == 0)).then(None).otherwise((high - low)) / low)
            exprs.append(width_pct.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_channel_width_atr",
    category="time_series",
    business_category="channel",
    canonical="ts_channel_width_atr",
    source=_SRC,
    backend="polars")
class TSChannelWidthATRNative(SeriesOperator):
    """Channel width normalized by ATR."""

    metadata = OperatorMetadata(
        name="ts_channel_width_atr",
        category="time_series",
        description="ATR标准化通道宽度",
        param_names=["x", "window", "atr_window"],
        return_type="series",
        tags=["time_series", "channel", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "atr_window": ParamSpec(dtype=int, min=2, default=14, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, atr_window: int = 14, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        atr_w = strict_integer(atr_window, "atr_window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            high = val.rolling_max(window_size=w)
            width = high - low
            # Simple ATR approximation: rolling std
            atr = val.rolling_std(window_size=atr_w)
            width_atr = pl.when((atr.is_null()) | (atr == 0)).then(None).otherwise(width / atr)
            exprs.append(width_atr.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_channel_width_slope",
    category="time_series",
    business_category="channel",
    canonical="ts_channel_width_slope",
    source=_SRC,
    backend="polars")
class TSChannelWidthSlopeNative(SeriesOperator):
    """Change in channel width over period."""

    metadata = OperatorMetadata(
        name="ts_channel_width_slope",
        category="time_series",
        description="通道宽度斜率",
        param_names=["x", "window", "slope_period"],
        return_type="series",
        tags=["time_series", "channel", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "slope_period": ParamSpec(dtype=int, min=1, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, slope_period: int = 5, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        sp = strict_integer(slope_period, "slope_period", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            high = val.rolling_max(window_size=w)
            width = high - low
            slope = width - width.shift(sp)
            exprs.append(slope.alias(c))
        result = x.with_columns(exprs)
        return result




# ---------------------------------------------------------------------------
# Swing analysis
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_swing_amplitude",
    category="time_series",
    business_category="swing",
    canonical="ts_swing_amplitude",
    source=_SRC,
    backend="polars")
class TSSwingAmplitudeNative(SeriesOperator):
    """Absolute swing amplitude: |high - low| over window."""

    metadata = OperatorMetadata(
        name="ts_swing_amplitude",
        category="time_series",
        description="摆动幅度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "swing", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            high = val.rolling_max(window_size=w)
            amplitude = (high - low).abs()
            exprs.append(amplitude.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_swing_amplitude_pct",
    category="time_series",
    business_category="swing",
    canonical="ts_swing_amplitude_pct",
    source=_SRC,
    backend="polars")
class TSSwingAmplitudePctNative(SeriesOperator):
    """Relative swing amplitude: (high - low) / avg."""

    metadata = OperatorMetadata(
        name="ts_swing_amplitude_pct",
        category="time_series",
        description="摆动幅度百分比",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "swing", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            high = val.rolling_max(window_size=w)
            avg = val.rolling_mean(window_size=w)
            amplitude_pct = (pl.when((avg.is_null()) | (avg == 0)).then(None).otherwise((high - low)) / avg)
            exprs.append(amplitude_pct.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_swing_amplitude_atr",
    category="time_series",
    business_category="swing",
    canonical="ts_swing_amplitude_atr",
    source=_SRC,
    backend="polars")
class TSSwingAmplitudeATRNative(SeriesOperator):
    """Swing amplitude normalized by ATR."""

    metadata = OperatorMetadata(
        name="ts_swing_amplitude_atr",
        category="time_series",
        description="ATR标准化摆动幅度",
        param_names=["x", "window", "atr_window"],
        return_type="series",
        tags=["time_series", "swing", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "atr_window": ParamSpec(dtype=int, min=2, default=14, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, atr_window: int = 14, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        atr_w = strict_integer(atr_window, "atr_window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            high = val.rolling_max(window_size=w)
            amplitude = high - low
            atr = val.rolling_std(window_size=atr_w)
            amplitude_atr = pl.when((atr.is_null()) | (atr == 0)).then(None).otherwise(amplitude / atr)
            exprs.append(amplitude_atr.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_swing_duration",
    category="time_series",
    business_category="swing",
    canonical="ts_swing_duration",
    source=_SRC,
    backend="polars")
class TSSwingDurationNative(SeriesOperator):
    """Days since last swing reversal (simplified: days since change > threshold)."""

    metadata = OperatorMetadata(
        name="ts_swing_duration",
        category="time_series",
        description="摆动持续天数",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["time_series", "swing", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.02, searchable=False, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, threshold: float = 0.02, **kwargs
    ) -> pl.DataFrame:
        thresh = float(threshold)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            change = (val - val.shift(1)) / val.shift(1).abs()
            is_swing = (change.abs() > thresh)
            block = is_swing.cum_sum()
            duration = pl.int_range(0, pl.len()).over(block) - pl.int_range(0, pl.len()).shift(1).fill_null(0).over(block)
            exprs.append(duration.alias(c))
        result = x.with_columns(exprs)
        return result


@register_operator(
    name="ts_swing_velocity",
    category="time_series",
    business_category="swing",
    canonical="ts_swing_velocity",
    source=_SRC,
    backend="polars")
class TSSwingVelocityNative(SeriesOperator):
    """Swing velocity: amplitude / duration."""

    metadata = OperatorMetadata(
        name="ts_swing_velocity",
        category="time_series",
        description="摆动速度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "swing", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, x: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            val = x[c]
            low = val.rolling_min(window_size=w)
            high = val.rolling_max(window_size=w)
            amplitude = high - low
            # Velocity = amplitude / window (time proxy)
            velocity = amplitude / pl.lit(float(w))
            exprs.append(velocity.alias(c))
        result = x.with_columns(exprs)
        return result
