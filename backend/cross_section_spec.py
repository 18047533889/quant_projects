# -*- coding: utf-8
"""截面变换契约：normalize / zscore / scale / group_* 常数截面语义。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import polars as pl

StdDdof = Literal["sample", "population"]
ZscoreZeroStdPolicy = Literal["zero", "nan", "null"]


@dataclass(frozen=True)
class CrossSectionSpec:
    """截面标准化/缩放公开契约。"""

    std_ddof: StdDdof = "sample"
    zscore_zero_std: ZscoreZeroStdPolicy = "zero"
    normalize_single_valid_is_null: bool = True
    normalize_constant_fill: float = 0.5
    scale_sum_abs_target: float = 1.0
    scale_zero_sum_fill: float = 0.0
    inf_participates: bool = False


CROSS_SECTION_SPECS: dict[str, CrossSectionSpec] = {
    "zscore": CrossSectionSpec(zscore_zero_std="zero", std_ddof="sample"),
    "group_zscore": CrossSectionSpec(zscore_zero_std="zero", std_ddof="sample"),
    "group_neutralize": CrossSectionSpec(zscore_zero_std="zero", std_ddof="sample"),
    "ts_zscore": CrossSectionSpec(zscore_zero_std="zero", std_ddof="sample"),
    "normalize": CrossSectionSpec(),
    "group_normalize": CrossSectionSpec(),
    "scale": CrossSectionSpec(scale_sum_abs_target=1.0),
}


def cross_section_spec_for(canon: str) -> CrossSectionSpec:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return CROSS_SECTION_SPECS.get(name, CrossSectionSpec())


def zscore_zero_std_fill(canon: str) -> float | None:
    """std=0 时 zscore 填充值。"""
    policy = cross_section_spec_for(canon).zscore_zero_std
    if policy == "zero":
        return 0.0
    if policy == "nan":
        return float("nan")
    return None


def std_ddof_value(canon: str) -> int:
    return 1 if cross_section_spec_for(canon).std_ddof == "sample" else 0


def normalize_single_valid_is_null() -> bool:
    return cross_section_spec_for("normalize").normalize_single_valid_is_null


def normalize_constant_cross_section_fill() -> float:
    return cross_section_spec_for("normalize").normalize_constant_fill


def scale_zero_sum_fill() -> float:
    return cross_section_spec_for("scale").scale_zero_sum_fill


def scale_default_target() -> float:
    return cross_section_spec_for("scale").scale_sum_abs_target


def polars_scale_expr(
    value_col: str,
    to_val: float,
    *,
    partition_cols: tuple[str, ...],
    order_by: str,
) -> "pl.Expr":
    """scale：当前行 NULL → NULL；sum(abs)=0 → fill；否则 x*to/sum(abs)。"""
    import polars as pl

    v = pl.col(value_col).cast(pl.Float64, strict=False)
    s = v.abs().sum().over(*partition_cols, order_by=order_by)
    fill = scale_zero_sum_fill()
    return (
        pl.when(pl.col(value_col).is_null())
        .then(None)
        .when(s.is_null() | (s == 0))
        .then(fill)
        .otherwise(v / s * to_val)
    )


def zscore_zero_std_epsilon(*, scale: float = 1.0) -> float:
    """std≈0 判定：``max(1e-12, 1e-8 * |scale|)``，zscore/ts_zscore/group_zscore 共用。"""
    return max(1e-12, 1e-8 * abs(scale))


def is_effectively_zero_std(std_val: float, *, scale: float = 1.0) -> bool:
    if std_val != std_val:
        return True
    return abs(std_val) <= zscore_zero_std_epsilon(scale=scale)
