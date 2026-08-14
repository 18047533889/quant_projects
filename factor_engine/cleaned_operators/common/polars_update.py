# -*- coding: utf-8 -*-
"""Update operators - Polars native implementations.

Update operators analyze patterns in data updates and revisions.
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
    name="update_surprise",
    category="update",
    business_category="update",
    canonical="update_surprise",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class UpdateSurpriseNative(SeriesOperator):
    """Surprise in update magnitude relative to history."""

    metadata = OperatorMetadata(
        name="update_surprise",
        category="update",
        description="更新意外度",
        param_names=["x", "window"],
        return_type="series",
        tags=["update", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement update surprise
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="update_acceleration",
    category="update",
    business_category="update",
    canonical="update_acceleration",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class UpdateAccelerationNative(SeriesOperator):
    """Acceleration in update magnitude."""

    metadata = OperatorMetadata(
        name="update_acceleration",
        category="update",
        description="更新加速度",
        param_names=["x", "d"],
        return_type="series",
        tags=["update", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(d, "d", minimum=1)

        # TODO: Implement update acceleration
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="update_direction_persistence",
    category="update",
    business_category="update",
    canonical="update_direction_persistence",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class UpdateDirectionPersistenceNative(SeriesOperator):
    """Persistence of update direction over time."""

    metadata = OperatorMetadata(
        name="update_direction_persistence",
        category="update",
        description="更新方向持续性",
        param_names=["x", "window"],
        return_type="series",
        tags=["update", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement direction persistence
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="update_path_efficiency",
    category="update",
    business_category="update",
    canonical="update_path_efficiency",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class UpdatePathEfficiencyNative(SeriesOperator):
    """Path efficiency: net change / sum of absolute changes."""

    metadata = OperatorMetadata(
        name="update_path_efficiency",
        category="update",
        description="更新路径效率",
        param_names=["x", "window"],
        return_type="series",
        tags=["update", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement path efficiency
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])
