"""与 factor_engine 白名单对齐的挖掘算子集合。"""

from __future__ import annotations

from shared.alphagen.data.expression import (
    Abs,
    Add,
    Div,
    Greater,
    Inv,
    Less,
    Log,
    Mul,
    Pow,
    Rank,
    Ref,
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
    GetGreater,
    GetLess,
)
from shared.alphagen_qlib.stock_data import FeatureType

# 仅保留能映射到 factor_engine DSL 的算子（剔除 TsIr / TsWMA / TsMinMaxDiff 等无对应项）
OPERATORS = [
  # Unary
    Abs, SLog1p, Inv, Sign, Log, Rank,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less, GetGreater, GetLess,
    # Rolling
    Ref, TsMean, TsSum, TsStd, TsVar, TsSkew, TsKurt, TsMax, TsMin,
    TsMed, TsMad, TsRank, TsDelta, TsPctChange, TsEMA,
    # Pair rolling
    TsCov, TsCorr,
]

FEATURES = [
    FeatureType.OPEN,
    FeatureType.CLOSE,
    FeatureType.HIGH,
    FeatureType.LOW,
    FeatureType.VOLUME,
    FeatureType.VWAP,
]

# 与 LLM / 冷启动窗口集合对齐（短到超长）
DELTA_TIMES = [
    1, 2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50,
    60, 70, 80, 90, 100, 120, 150, 180, 200, 220, 250, 300, 500, 800, 1000,
]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256
REWARD_PER_STEP = 0.
