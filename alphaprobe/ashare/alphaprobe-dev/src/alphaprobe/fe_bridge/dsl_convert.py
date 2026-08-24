"""AlphaGen 表达式树 → factor_engine DSL 字符串。"""

from __future__ import annotations

from typing import Type

from shared.alphagen.data.expression import (
    Abs,
    Add,
    BinaryOperator,
    Constant,
    Div,
    Expression,
    Feature,
    Greater,
    Inv,
    Less,
    Log,
    Mul,
    Operator,
    PairRollingOperator,
    Pow,
    Rank,
    Ref,
    RollingOperator,
    Sign,
    SLog1p,
    Sub,
    TsCorr,
    TsCov,
    TsDelta,
    TsEMA,
    TsKurt,
    TsMad,
    TsMax,
    TsMean,
    TsMed,
    TsMin,
    TsPctChange,
    TsRank,
    TsSkew,
    TsStd,
    TsSum,
    TsVar,
    UnaryOperator,
    GetGreater,
    GetLess,
)
from shared.alphagen_qlib.stock_data import FeatureType

FEATURE_TO_FIELD: dict[FeatureType, str] = {
    FeatureType.OPEN: "open",
    FeatureType.CLOSE: "close",
    FeatureType.HIGH: "high",
    FeatureType.LOW: "low",
    FeatureType.VOLUME: "volume",
    FeatureType.VWAP: "vwap",
}

UNARY_DSL: dict[Type[Operator], str] = {
    Abs: "abs",
    Sign: "sign",
    Log: "log",
    Rank: "rank",
    SLog1p: "signed_log",
    Inv: "inv",
}

ROLLING_DSL: dict[Type[Operator], str] = {
    Ref: "ts_delay",
    TsMean: "ts_mean",
    TsSum: "ts_sum",
    TsStd: "ts_std",
    TsVar: "var",
    TsSkew: "ts_skew",
    TsKurt: "ts_kurt",
    TsMax: "ts_max",
    TsMin: "ts_min",
    TsMed: "m_median",
    TsMad: "m_mad",
    TsRank: "ts_rank",
    TsDelta: "ts_delta",
    TsPctChange: "pct_change",
    TsEMA: "ema",
}

PAIR_ROLLING_DSL: dict[Type[Operator], str] = {
    TsCov: "ts_cov",
    TsCorr: "ts_corr",
}


def _wrap(text: str) -> str:
    return f"({text})"


def expression_to_dsl(expr: Expression) -> str:
    """将 AlphaGen ``Expression`` 转为 factor_engine 可 ``parse_expr`` 的 DSL。"""
    from alphaprobe.fe_bridge.dsl_expression import FactorEngineDslExpression

    if isinstance(expr, FactorEngineDslExpression):
        return expr.dsl
    if isinstance(expr, Feature):
        return FEATURE_TO_FIELD[expr._feature]
    if isinstance(expr, Constant):
        value = expr._value
        if value == int(value):
            return str(int(value))
        return repr(float(value))
    if isinstance(expr, UnaryOperator):
        op_name = UNARY_DSL.get(type(expr))
        if op_name is None:
            raise ValueError(f"无 factor_engine 映射的一元算子: {type(expr).__name__}")
        return f"{op_name}({expression_to_dsl(expr._operand)})"
    if isinstance(expr, Add):
        return _wrap(f"{expression_to_dsl(expr._lhs)} + {expression_to_dsl(expr._rhs)}")
    if isinstance(expr, Sub):
        return _wrap(f"{expression_to_dsl(expr._lhs)} - {expression_to_dsl(expr._rhs)}")
    if isinstance(expr, Mul):
        return _wrap(f"{expression_to_dsl(expr._lhs)} * {expression_to_dsl(expr._rhs)}")
    if isinstance(expr, Div):
        rhs = expression_to_dsl(expr._rhs)
        return _wrap(f"{expression_to_dsl(expr._lhs)} / ({rhs} + 1e-9)")
    if isinstance(expr, Pow):
        return f"power({expression_to_dsl(expr._lhs)}, {expression_to_dsl(expr._rhs)})"
    if isinstance(expr, Greater):
        lhs, rhs = expression_to_dsl(expr._lhs), expression_to_dsl(expr._rhs)
        return f"if_else({lhs} > {rhs}, {lhs}, {rhs})"
    if isinstance(expr, Less):
        lhs, rhs = expression_to_dsl(expr._lhs), expression_to_dsl(expr._rhs)
        return f"if_else({lhs} < {rhs}, {lhs}, {rhs})"
    if isinstance(expr, GetGreater):
        lhs, rhs = expression_to_dsl(expr._lhs), expression_to_dsl(expr._rhs)
        return f"if_else({lhs} > {rhs}, {lhs}, {rhs})"
    if isinstance(expr, GetLess):
        lhs, rhs = expression_to_dsl(expr._lhs), expression_to_dsl(expr._rhs)
        return f"if_else({lhs} < {rhs}, {lhs}, {rhs})"
    if isinstance(expr, RollingOperator):
        op_name = ROLLING_DSL.get(type(expr))
        if op_name is None:
            raise ValueError(f"无 factor_engine 映射的时序算子: {type(expr).__name__}")
        window = expr._delta_time
        return f"{op_name}({expression_to_dsl(expr._operand)}, {window})"
    if isinstance(expr, PairRollingOperator):
        op_name = PAIR_ROLLING_DSL.get(type(expr))
        if op_name is None:
            raise ValueError(f"无 factor_engine 映射的双序列算子: {type(expr).__name__}")
        window = expr._delta_time
        lhs, rhs = expression_to_dsl(expr._lhs), expression_to_dsl(expr._rhs)
        return f"{op_name}({lhs}, {rhs}, {window})"
    raise TypeError(f"不支持的表达式类型: {type(expr)!r}")


def validate_expression_dsl(expr: Expression) -> tuple[bool, str]:
    from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

    ensure_factor_engine_importable()
    from factor_engine.api.dsl_parser import DSLParseError, parse_expr

    try:
        parse_expr(expression_to_dsl(expr))
        return True, "OK"
    except (DSLParseError, ValueError, TypeError) as exc:
        return False, str(exc)
