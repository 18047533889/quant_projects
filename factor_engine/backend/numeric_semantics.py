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
}


def semantics_for(canon: str) -> NumericSemantics:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return OPERATOR_SEMANTICS.get(name, DEFAULT_SEMANTICS)


def std_ddof_value(canon: str) -> int:
    return 1 if semantics_for(canon).std_ddof == "sample" else 0


def protected_epsilon_default() -> float:
    return 1e-12


def protected_div_default() -> float:
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


def sql_stddev_fn_key(canon: str) -> str:
    """DuckDB/CH emitter 用：``stddev`` → sample，``stddev_pop`` → population。"""
    return "stddev" if semantics_for(canon).std_ddof == "sample" else "stddev_pop"
