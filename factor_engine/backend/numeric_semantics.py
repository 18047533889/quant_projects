# -*- coding: utf-8
"""跨后端数值语义契约：Pandas / PolarsLong / DuckDB 对齐依据。

各 backend 实现应引用本模块常量，parity 测试覆盖 edge cases。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NullPolicy = Literal["propagate", "ignore", "coerce_to_null"]
InfPolicy = Literal["propagate", "to_nan", "to_null"]
DivZeroPolicy = Literal["inf", "null", "default", "protected"]
StdDdof = Literal["sample", "population"]
RankNullPolicy = Literal["ignore", "bottom", "error"]
ZscoreZeroStdPolicy = Literal["zero", "nan", "null"]


@dataclass(frozen=True)
class NumericSemantics:
    """单算子或全局默认数值语义。"""

    input_nan_to_null: bool = True
    output_inf_to_nan: bool = True
    output_inf_to_null: bool = False
    div_zero: DivZeroPolicy = "protected"
    rank_ignore_nan: bool = True
    std_ddof: StdDdof = "sample"
    zscore_zero_std: ZscoreZeroStdPolicy = "zero"
    quantile_interpolation: str = "linear"
    window_min_periods_default: int = 1


# 全局默认（panel / long-table 通用）
DEFAULT_SEMANTICS = NumericSemantics()

# 算子级覆盖
OPERATOR_SEMANTICS: dict[str, NumericSemantics] = {
    "protected_div": NumericSemantics(div_zero="default"),
    "protected_log": NumericSemantics(input_nan_to_null=True, div_zero="default"),
    "protected_sqrt": NumericSemantics(input_nan_to_null=True),
    "divide": NumericSemantics(div_zero="inf"),
    "rank": NumericSemantics(rank_ignore_nan=True),
    "rank_pct": NumericSemantics(rank_ignore_nan=True),
    "group_rank": NumericSemantics(rank_ignore_nan=True),
    "zscore": NumericSemantics(zscore_zero_std="zero", std_ddof="sample"),
    "group_zscore": NumericSemantics(zscore_zero_std="zero", std_ddof="sample"),
    "group_std": NumericSemantics(std_ddof="sample"),
    "ts_std": NumericSemantics(std_ddof="sample"),
    "ts_var": NumericSemantics(std_ddof="sample"),
    "ts_zscore": NumericSemantics(zscore_zero_std="zero", std_ddof="sample"),
    "winsorize": NumericSemantics(rank_ignore_nan=True),
    "group_winsorize": NumericSemantics(rank_ignore_nan=True),
    "cs_quantile": NumericSemantics(quantile_interpolation="linear"),
    "c_percentile": NumericSemantics(quantile_interpolation="linear"),
    "ts_quantile": NumericSemantics(quantile_interpolation="linear"),
    "group_percentile": NumericSemantics(quantile_interpolation="linear"),
    "nan_to_num": NumericSemantics(output_inf_to_nan=False, output_inf_to_null=False),
    "normalize": NumericSemantics(),
    "ts_pct": NumericSemantics(input_nan_to_null=True),
    "and_": NumericSemantics(),
    "or_": NumericSemantics(),
    "not_": NumericSemantics(),
    "where": NumericSemantics(),
    "log": NumericSemantics(input_nan_to_null=True),
}


def semantics_for(canon: str) -> NumericSemantics:
    """查询算子级数值语义，未覆盖时返回全局默认。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        对应的 ``NumericSemantics`` 配置。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return OPERATOR_SEMANTICS.get(name, DEFAULT_SEMANTICS)


def std_ddof_value(canon: str) -> int:
    """返回标准差/方差算子的 ddof 数值。

    参数:
        canon: 算子 canonical 名称。

    返回:
        sample 策略为 ``1``，population 策略为 ``0``。
    """
    return 1 if semantics_for(canon).std_ddof == "sample" else 0


def protected_epsilon_default() -> float:
    """返回 protected 算子默认 epsilon 常量。

    返回:
        用于除零/对数保护的极小正数。
    """
    return 1e-12


def protected_div_default() -> float:
    """返回 protected_div 除零时的默认填充值。

    返回:
        除零保护策略下的默认输出。
    """
    return 0.0


def rank_ignore_nan(canon: str) -> bool:
    """截面/组内 rank 是否忽略 NaN（输出仍为 null）。"""
    return semantics_for(canon).rank_ignore_nan


def zscore_zero_std_fill(canon: str) -> float | None:
    """std=0 时 zscore 填充值；``null`` 策略返回 ``None``。"""
    policy = semantics_for(canon).zscore_zero_std
    if policy == "zero":
        return 0.0
    if policy == "nan":
        return float("nan")
    return None


def normalize_single_valid_is_null() -> bool:
    """截面 normalize：仅一个有效值时输出 NULL（非常数截面）。"""
    return True


def normalize_constant_cross_section_fill() -> float:
    """截面 normalize：全部有效值相同（span=0）时输出 0.5。"""
    return 0.5


def nan_to_num_replaces_infinite() -> bool:
    """``nan_to_num`` 将 ±Inf 替换为与 NaN/null 相同的填充常数。"""
    return True


def truthy_null_is_false() -> bool:
    """逻辑算子 ``and_``/``or_``/``not_``/``where``：NULL/NaN 视为 false。"""
    return True


def ts_pct_zero_prev_is_null() -> bool:
    """``ts_pct``：滞后值为 0 或 NULL 时输出 NULL（不做 forward-fill）。"""
    return True


def log_zero_returns_negative_infinity() -> bool:
    """``log``：输入为 0 时输出 -Inf；负数输出 NULL。"""
    return True


def sql_stddev_fn_key(canon: str) -> str:
    """DuckDB/CH emitter 用：``stddev`` → sample，``stddev_pop`` → population。"""
    return "stddev" if semantics_for(canon).std_ddof == "sample" else "stddev_pop"
