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
    "protected_div": NumericSemantics(div_zero="null"),
    "safe_div_null": NumericSemantics(div_zero="null"),
    "div_or_default": NumericSemantics(div_zero="default"),
    "div_or_null": NumericSemantics(div_zero="null"),
    "protected_log": NumericSemantics(input_nan_to_null=True),
    "log_fill_invalid": NumericSemantics(input_nan_to_null=False, div_zero="default"),
    "protected_sqrt": NumericSemantics(input_nan_to_null=True),
    "divide": NumericSemantics(div_zero="inf"),
    "rank": NumericSemantics(rank_ignore_nan=True),
    "rank_pct": NumericSemantics(rank_ignore_nan=True),
    "group_rank": NumericSemantics(rank_ignore_nan=True),
    "ts_rank": NumericSemantics(rank_ignore_nan=True),
    "cs_pct_rank": NumericSemantics(rank_ignore_nan=True),
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
    "log": NumericSemantics(input_nan_to_null=True, output_inf_to_nan=False),
    "divide": NumericSemantics(div_zero="inf", output_inf_to_nan=False),
    "exp": NumericSemantics(output_inf_to_nan=True),
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
    from backend.rank_spec import rank_ignore_nan as _rank_ignore_nan

    return _rank_ignore_nan(canon)


def zscore_zero_std_fill(canon: str) -> float | None:
    """std=0 时 zscore 填充值；``null`` 策略返回 ``None``。"""
    from backend.cross_section_spec import zscore_zero_std_fill as _zscore_zero_std_fill

    return _zscore_zero_std_fill(canon)


def normalize_single_valid_is_null() -> bool:
    """截面 normalize：仅一个有效值时输出 NULL（非常数截面）。"""
    from backend.cross_section_spec import normalize_single_valid_is_null as _norm_single

    return _norm_single()


def normalize_constant_cross_section_fill() -> float:
    """截面 normalize：全部有效值相同（span=0）时输出 0.5。"""
    from backend.cross_section_spec import normalize_constant_cross_section_fill as _norm_const

    return _norm_const()


def nan_to_num_replaces_infinite() -> bool:
    """``nan_to_num`` 将 ±Inf 替换为与 NaN/null 相同的填充常数。"""
    return True


def truthy_null_is_false() -> bool:
    """逻辑算子 ``and_``/``or_``/``not_``/``where``：NULL/NaN 视为 false。"""
    return True


def truthy_nan_is_false() -> bool:
    """NaN 在逻辑算子中视为 false（Polars NaN 非 NULL）。"""
    return True


def truthy_inf_is_true() -> bool:
    """±Inf 在逻辑算子中视为 true。"""
    return True


def is_nan_excludes_null() -> bool:
    """``is_nan(NULL)=0``；``is_null(NULL)=1``。"""
    return True


def ts_argmax_empty_window_is_null() -> bool:
    """``ts_argmax/ts_argmin``：窗口全 NULL/无效 → NULL（非 0）。"""
    return True


def ts_argmax_index_origin() -> str:
    """arg 位置从窗口左端计 0（距当前点 w-1 的偏移）。"""
    return "window_left_0"


def ts_argmax_tie_break() -> str:
    """并列极值取 first（``nanargmax`` 默认）。"""
    return "first"


def ts_sharpe_zero_std_is_null() -> bool:
    """``ts_sharpe``：std=0 → NULL（禁止 Inf/1e308 cap 分叉）。"""
    return True


def group_percentile_null_is_null() -> bool:
    """``group_percentile``：输入 NULL → 输出 NULL（非 0）。"""
    return True


def ts_mad_is_nonstandard() -> bool:
    """当前 ``ts_mad`` 为双重滚动近似，非标准 MAD；禁止 production。"""
    return True


def rank_tie_method(canon: str) -> str:
    """截面/组内/时序 rank 并列策略（见 ``rank_spec.RANK_SPECS``）。"""
    if canon in {"ts_argmax", "ts_argmin"}:
        return "first"
    from backend.rank_spec import rank_tie_method as _rank_tie_method

    return _rank_tie_method(canon)


def panel_binary_join_preserves_anchor() -> bool:
    """二元算子以左操作数为 anchor LEFT JOIN，禁止隐式删行。"""
    return True


def chunk_scan_invariance_required() -> bool:
    """production 算子须通过全量 vs 分块 overlap 扫描 parity。"""
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
