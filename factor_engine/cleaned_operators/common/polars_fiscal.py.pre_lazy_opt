# -*- coding: utf-8 -*-
"""Fiscal period operators - Polars native implementations.

Fiscal operators analyze financial statement data on fiscal period boundaries.
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


@register_operator(
    name="fiscal_pct_change",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_pct_change",
    source=_SRC,
    backend="polars",
)
class FiscalPctChangeNative(SeriesOperator):
    """Fiscal period percentage change."""

    metadata = OperatorMetadata(
        name="fiscal_pct_change",
        category="fiscal",
        description="财报期百分比变化",
        param_names=["x", "d"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(d, "d", minimum=1)

        # TODO: Implement fiscal pct_change
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_acceleration",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_acceleration",
    source=_SRC,
    backend="polars",
)
class FiscalAccelerationNative(SeriesOperator):
    """Fiscal period acceleration (second derivative)."""

    metadata = OperatorMetadata(
        name="fiscal_acceleration",
        category="fiscal",
        description="财报期加速度",
        param_names=["x", "d"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(d, "d", minimum=1)

        # TODO: Implement fiscal acceleration
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_autocorr",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_autocorr",
    source=_SRC,
    backend="polars",
)
class FiscalAutocorrNative(SeriesOperator):
    """Fiscal period autocorrelation."""

    metadata = OperatorMetadata(
        name="fiscal_autocorr",
        category="fiscal",
        description="财报期自相关",
        param_names=["x", "lag", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, lag: int = 1, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        lag_val = strict_integer(lag, "lag", minimum=1)
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement fiscal autocorrelation
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_rolling_std",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_rolling_std",
    source=_SRC,
    backend="polars",
)
class FiscalRollingStdNative(SeriesOperator):
    """Rolling standard deviation over fiscal periods."""

    metadata = OperatorMetadata(
        name="fiscal_rolling_std",
        category="fiscal",
        description="财报期滚动标准差",
        param_names=["x", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement fiscal rolling std
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_reversal_ratio",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_reversal_ratio",
    source=_SRC,
    backend="polars",
)
class FiscalReversalRatioNative(SeriesOperator):
    """Ratio of sign reversals in fiscal period changes."""

    metadata = OperatorMetadata(
        name="fiscal_reversal_ratio",
        category="fiscal",
        description="财报期反转比率",
        param_names=["x", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement fiscal reversal ratio
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_change_direction_agreement",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_change_direction_agreement",
    source=_SRC,
    backend="polars",
)
class FiscalChangeDirectionAgreementNative(SeriesOperator):
    """Agreement of change directions between two metrics."""

    metadata = OperatorMetadata(
        name="fiscal_change_direction_agreement",
        category="fiscal",
        description="财报期变化方向一致性",
        param_names=["x", "y", "d"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None, d: int = 1, **kwargs) -> pl.DataFrame:
        if y is None:
            raise ValueError("fiscal_change_direction_agreement requires y")
        from cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(d, "d", minimum=1)

        # TODO: Implement direction agreement
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_pair_direction_agreement",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_pair_direction_agreement",
    source=_SRC,
    backend="polars",
)
class FiscalPairDirectionAgreementNative(SeriesOperator):
    """Pairwise direction agreement across fiscal periods."""

    metadata = OperatorMetadata(
        name="fiscal_pair_direction_agreement",
        category="fiscal",
        description="财报期配对方向一致性",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None, window: int = 8, **kwargs) -> pl.DataFrame:
        if y is None:
            raise ValueError("fiscal_pair_direction_agreement requires y")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement pair direction agreement
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_direction_consistency",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_direction_consistency",
    source=_SRC,
    backend="polars",
)
class FiscalDirectionConsistencyNative(SeriesOperator):
    """Consistency of change direction over fiscal periods."""

    metadata = OperatorMetadata(
        name="fiscal_direction_consistency",
        category="fiscal",
        description="财报期方向连贯性",
        param_names=["x", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement direction consistency
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_sign_consistency",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_sign_consistency",
    source=_SRC,
    backend="polars",
)
class FiscalSignConsistencyNative(SeriesOperator):
    """Consistency of sign over fiscal periods."""

    metadata = OperatorMetadata(
        name="fiscal_sign_consistency",
        category="fiscal",
        description="财报期符号连贯性",
        param_names=["x", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement sign consistency
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_sign_agreement",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_sign_agreement",
    source=_SRC,
    backend="polars",
)
class FiscalSignAgreementNative(SeriesOperator):
    """Sign agreement between two metrics."""

    metadata = OperatorMetadata(
        name="fiscal_sign_agreement",
        category="fiscal",
        description="财报期符号一致性",
        param_names=["x", "y"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if y is None:
            raise ValueError("fiscal_sign_agreement requires y")

        # TODO: Implement sign agreement
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_true_streak",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_true_streak",
    source=_SRC,
    backend="polars",
)
class FiscalTrueStreakNative(SeriesOperator):
    """Consecutive fiscal periods with same sign."""

    metadata = OperatorMetadata(
        name="fiscal_true_streak",
        category="fiscal",
        description="财报期连续同向期数",
        param_names=["x"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # TODO: Implement true streak
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_accrual_quality",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_accrual_quality",
    source=_SRC,
    backend="polars",
)
class FiscalAccrualQualityNative(SeriesOperator):
    """Accrual quality metric (e.g., Dechow-Dichev)."""

    metadata = OperatorMetadata(
        name="fiscal_accrual_quality",
        category="fiscal",
        description="应计质量",
        param_names=["accruals", "cash_flow", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, accruals: pl.DataFrame, cash_flow: pl.DataFrame | None = None, window: int = 8, **kwargs) -> pl.DataFrame:
        if cash_flow is None:
            raise ValueError("fiscal_accrual_quality requires cash_flow")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=3)

        # TODO: Implement accrual quality
        cols = _numeric_cols(accruals)
        return accruals.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_asymmetric_timeliness",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_asymmetric_timeliness",
    source=_SRC,
    backend="polars",
)
class FiscalAsymmetricTimelinessNative(SeriesOperator):
    """Basu (1997) asymmetric timeliness measure."""

    metadata = OperatorMetadata(
        name="fiscal_asymmetric_timeliness",
        category="fiscal",
        description="非对称及时性",
        param_names=["earnings", "returns", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=8, default=12, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, earnings: pl.DataFrame, returns: pl.DataFrame | None = None, window: int = 12, **kwargs) -> pl.DataFrame:
        if returns is None:
            raise ValueError("fiscal_asymmetric_timeliness requires returns")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=8)

        # TODO: Implement asymmetric timeliness
        cols = _numeric_cols(earnings)
        return earnings.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_asymmetric_elasticity",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_asymmetric_elasticity",
    source=_SRC,
    backend="polars",
)
class FiscalAsymmetricElasticityNative(SeriesOperator):
    """Asymmetric response elasticity (good vs bad news)."""

    metadata = OperatorMetadata(
        name="fiscal_asymmetric_elasticity",
        category="fiscal",
        description="非对称弹性",
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=8, default=12, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None, window: int = 12, **kwargs) -> pl.DataFrame:
        if y is None:
            raise ValueError("fiscal_asymmetric_elasticity requires y")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=8)

        # TODO: Implement asymmetric elasticity
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_regression_resid_std",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_regression_resid_std",
    source=_SRC,
    backend="polars",
)
class FiscalRegressionResidStdNative(SeriesOperator):
    """Standard deviation of regression residuals over fiscal periods."""

    metadata = OperatorMetadata(
        name="fiscal_regression_resid_std",
        category="fiscal",
        description="财报期回归残差标准差",
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame | None = None, window: int = 8, **kwargs) -> pl.DataFrame:
        if x is None:
            raise ValueError("fiscal_regression_resid_std requires x")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=3)

        # TODO: Implement regression resid std
        cols = _numeric_cols(y)
        return y.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_ar_resid_std",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_ar_resid_std",
    source=_SRC,
    backend="polars",
)
class FiscalARResidStdNative(SeriesOperator):
    """Standard deviation of AR(1) residuals over fiscal periods."""

    metadata = OperatorMetadata(
        name="fiscal_ar_resid_std",
        category="fiscal",
        description="财报期AR残差标准差",
        param_names=["x", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=3)

        # TODO: Implement AR resid std
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_perpetual_inventory",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_perpetual_inventory",
    source=_SRC,
    backend="polars",
)
class FiscalPerpetualInventoryNative(SeriesOperator):
    """Perpetual inventory method for capital stock accumulation."""

    metadata = OperatorMetadata(
        name="fiscal_perpetual_inventory",
        category="fiscal",
        description="永续盘存法",
        param_names=["investment", "depreciation_rate"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "depreciation_rate": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, investment: pl.DataFrame, depreciation_rate: float = 0.1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        delta = strict_finite_scalar(depreciation_rate, "depreciation_rate", minimum=0.0, maximum=1.0)

        # TODO: Implement perpetual inventory
        cols = _numeric_cols(investment)
        return investment.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="fiscal_standardized_surprise",
    category="fiscal",
    business_category="fiscal",
    canonical="fiscal_standardized_surprise",
    source=_SRC,
    backend="polars",
)
class FiscalStandardizedSurpriseNative(SeriesOperator):
    """Standardized unexpected earnings (SUE)."""

    metadata = OperatorMetadata(
        name="fiscal_standardized_surprise",
        category="fiscal",
        description="标准化意外盈余",
        param_names=["actual", "expected", "window"],
        return_type="series",
        tags=["fiscal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=4, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, actual: pl.DataFrame, expected: pl.DataFrame | None = None, window: int = 8, **kwargs) -> pl.DataFrame:
        if expected is None:
            raise ValueError("fiscal_standardized_surprise requires expected")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=4)

        # TODO: Implement standardized surprise
        cols = _numeric_cols(actual)
        return actual.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])
