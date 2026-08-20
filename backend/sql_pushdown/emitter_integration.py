# -*- coding: utf-8 -*-
"""Emitter 集成补丁：将高级 SQL 算子集成到 emitter.py。

这个模块提供了将 advanced_sql_operators.py 中的算子集成到现有
backend/sql_pushdown/emitter.py 的入口点。

使用方法：
    在 emitter.py 的 _compile_layer_impl 函数中，添加对本模块的调用：

    from backend.sql_pushdown.emitter_integration import try_compile_advanced_operator

    # 在 _compile_layer_impl 函数开头添加：
    advanced = try_compile_advanced_operator(node, dialect)
    if advanced is not None:
        return advanced
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planner.logical_plan import PlanNode

from backend.sql_pushdown.advanced_sql_operators import (
    cs_zscore_sql,
    cs_winsorize_sql,
    cs_rank_normalize_sql,
    ts_seasonal_diff_sql,
    panel_rank_sql,
    ts_regression_beta_sql,
    group_residual_sql,
    cs_industry_neutralize_sql,
    panel_residual_sql,
    cs_factor_score_sql,
    SqlDialect,
)


def _compile_layer(node: PlanNode, dialect: SqlDialect):
    """占位：实际调用 emitter 的 _compile_layer（避免循环导入）。"""
    from backend.sql_pushdown.emitter import _compile_layer as emitter_compile_layer
    return emitter_compile_layer(node, dialect=dialect)


def _float_attr(node: PlanNode, key: str, default: float = 0.0) -> float:
    """从 node.attrs 提取 float 属性。"""
    val = (node.attrs or {}).get(key, default)
    if val is None:
        return default
    return float(val)


def _int_attr(node: PlanNode, key: str, default: int = 0) -> int:
    """从 node.attrs 提取 int 属性。"""
    val = (node.attrs or {}).get(key, default)
    if val is None:
        return default
    return int(val)


def _bool_attr(node: PlanNode, key: str, default: bool = False) -> bool:
    """从 node.attrs 提取 bool 属性。"""
    val = (node.attrs or {}).get(key, default)
    if val is None:
        return default
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def try_compile_advanced_operator(node: PlanNode, dialect: SqlDialect):
    """尝试编译高级 SQL 算子。

    参数：
        node: 逻辑计划节点
        dialect: SQL 方言

    返回：
        _Layer 对象或 None（不支持时）
    """
    # 导入 _Layer（避免循环导入）
    from backend.sql_pushdown.emitter import _Layer

    op = node.op

    # -----------------------------------------------------------------------
    # 1. cs_zscore - 截面 z-score
    # -----------------------------------------------------------------------
    if op == "cs_zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            cs_zscore_sql(inner.sql, dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # -----------------------------------------------------------------------
    # 2. cs_winsorize - 截面 winsorize
    # -----------------------------------------------------------------------
    if op == "cs_winsorize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lower = _float_attr(node, "lower", default=0.01)
        upper = _float_attr(node, "upper", default=0.99)
        return _Layer(
            cs_winsorize_sql(inner.sql, lower=lower, upper=upper, dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # -----------------------------------------------------------------------
    # 3. cs_rank_normalize - 截面 rank 归一化
    # -----------------------------------------------------------------------
    if op == "cs_rank_normalize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            cs_rank_normalize_sql(inner.sql, dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # -----------------------------------------------------------------------
    # 4. ts_seasonal_diff - 时序季节性差分
    # -----------------------------------------------------------------------
    if op == "ts_seasonal_diff":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        period = _int_attr(node, "seasonal_period", default=12)
        # 也支持位置参数
        if len(node.inputs) > 1 and node.inputs[1].op == "literal":
            period = int(node.inputs[1].attrs.get("value", period))
        return _Layer(
            ts_seasonal_diff_sql(inner.sql, seasonal_period=period, dialect=dialect),
            has_inst_window=True,
        )

    # -----------------------------------------------------------------------
    # 5. panel_rank - 面板 rank
    # -----------------------------------------------------------------------
    if op == "panel_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            panel_rank_sql(inner.sql, dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    # -----------------------------------------------------------------------
    # 6. ts_regression_beta - 时序回归 beta
    # -----------------------------------------------------------------------
    if op == "ts_regression_beta":
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None

        # 提取 window 参数
        window = _int_attr(node, "window", default=20)
        if len(node.inputs) > 2 and node.inputs[2].op == "literal":
            window = int(node.inputs[2].attrs.get("value", window))

        min_periods = _int_attr(node, "min_periods", default=max(1, window // 2))

        return _Layer(
            ts_regression_beta_sql(
                left.sql,
                right.sql,
                window=window,
                min_periods=min_periods,
                dialect=dialect,
            ),
            has_inst_window=True,
        )

    # -----------------------------------------------------------------------
    # 7. group_residual - 分组残差
    # -----------------------------------------------------------------------
    if op == "group_residual":
        if len(node.inputs) < 3:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        grp = _compile_layer(node.inputs[2], dialect=dialect)
        if y is None or x is None or grp is None:
            return None

        return _Layer(
            group_residual_sql(y.sql, x.sql, grp.sql, dialect=dialect),
            has_inst_window=(
                y.has_inst_window or x.has_inst_window or grp.has_inst_window
            ),
            has_ts_partition=True,
        )

    # -----------------------------------------------------------------------
    # 8. cs_industry_neutralize - 行业中性化
    # -----------------------------------------------------------------------
    if op == "cs_industry_neutralize":
        if len(node.inputs) < 2:
            return None
        x = _compile_layer(node.inputs[0], dialect=dialect)
        ind = _compile_layer(node.inputs[1], dialect=dialect)
        if x is None or ind is None:
            return None

        return _Layer(
            cs_industry_neutralize_sql(x.sql, ind.sql, dialect=dialect),
            has_inst_window=x.has_inst_window or ind.has_inst_window,
            has_ts_partition=True,
        )

    # -----------------------------------------------------------------------
    # 9. panel_residual - 面板残差
    # -----------------------------------------------------------------------
    if op == "panel_residual":
        if len(node.inputs) < 2:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        if y is None or x is None:
            return None

        return _Layer(
            panel_residual_sql(y.sql, x.sql, dialect=dialect),
            has_inst_window=y.has_inst_window or x.has_inst_window,
            has_ts_partition=True,
        )

    # -----------------------------------------------------------------------
    # 10. cs_factor_score - 因子打分（多输入）
    # -----------------------------------------------------------------------
    if op == "cs_factor_score":
        # 所有输入都是因子
        factor_layers = []
        for inp in node.inputs:
            layer = _compile_layer(inp, dialect=dialect)
            if layer is None:
                return None
            factor_layers.append(layer)

        if not factor_layers:
            return None

        # 提取权重（如果有）
        weights = None
        if "weights" in (node.attrs or {}):
            weights = node.attrs["weights"]
            if isinstance(weights, (list, tuple)):
                weights = [float(w) for w in weights]

        standardize = _bool_attr(node, "standardize", default=True)

        factor_sqls = [layer.sql for layer in factor_layers]
        has_inst = any(layer.has_inst_window for layer in factor_layers)

        return _Layer(
            cs_factor_score_sql(
                factor_sqls,
                weights=weights,
                standardize=standardize,
                dialect=dialect,
            ),
            has_inst_window=has_inst,
            has_ts_partition=True,
        )

    # 不支持的算子
    return None


def register_advanced_operators_to_sql_tiers():
    """将高级算子注册到 SQL_IMPLEMENTED_CANONICALS。

    应该在模块加载时调用一次。
    """
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    new_ops = {
        "cs_zscore",
        "cs_winsorize",
        "cs_rank_normalize",
        "ts_seasonal_diff",
        "panel_rank",
        "ts_regression_beta",
        "group_residual",
        "cs_industry_neutralize",
        "panel_residual",
        "cs_factor_score",
    }

    # 更新全局集合（注意：这是模块级变量，需要重新赋值）
    globals()["SQL_IMPLEMENTED_CANONICALS"] = SQL_IMPLEMENTED_CANONICALS | frozenset(new_ops)


# 自动注册
register_advanced_operators_to_sql_tiers()
