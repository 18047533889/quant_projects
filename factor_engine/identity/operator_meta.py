# -*- coding: utf-8 -*-
"""算子代数属性声明表（NOT 硬编码在 canonicalizer 里）。

``get_operator_meta(op_name)`` 返回一个 ``OperatorAlgebraMeta`` 实例，
覆盖 FE daily surface 常用算子。未知算子默认全 **False**（保守，不化简不排序）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class OperatorAlgebraMeta:
    """算子的代数属性元数据。

    Attributes
    ----------
    commutative : bool
        算子是否满足交换律（a ∘ b = b ∘ a）。
        仅当 FE 算子实现严格 symmetric 时才标 True。
    associative : bool
        算子是否满足结合律（(a ∘ b) ∘ c = a ∘ (b ∘ c)）。
        目前仅用于元信息（化简器未使用结合律重排）。
    role : str
        参数角色分类（参考 ParameterCanonicalizer / ParamRole）。
        可选值：WINDOW / LAG / THRESHOLD / POWER / DECAY / CLIP_BOUND /
        SCALE / CONSTANT / OTHER。
    """

    commutative: bool = False
    associative: bool = False
    role: str = "OTHER"


# ---------------------------------------------------------------------------
# 声明表
# 算子名参考 build_dsl_allowlist(surface="daily") 名单。
# 交换律声明条件：FE 算子实现严格 symmetric（元素级运算，忽略顺序结果相同）。
# ---------------------------------------------------------------------------
_OPERATOR_META: Dict[str, OperatorAlgebraMeta] = {
    # --- 元素级代数 ---
    "add": OperatorAlgebraMeta(commutative=True, associative=True, role="SCALE"),
    "multiply": OperatorAlgebraMeta(commutative=True, associative=True, role="SCALE"),
    "and_": OperatorAlgebraMeta(commutative=True, associative=True, role="OTHER"),
    "or_": OperatorAlgebraMeta(commutative=True, associative=True, role="OTHER"),
    "eq": OperatorAlgebraMeta(commutative=True, role="OTHER"),
    "ne": OperatorAlgebraMeta(commutative=True, role="OTHER"),
    # subtract / divide / lt / le / gt / ge 等有序算子不 commutative
    "subtract": OperatorAlgebraMeta(commutative=False, role="SCALE"),
    "divide": OperatorAlgebraMeta(commutative=False, role="SCALE"),
    "lt": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "le": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "gt": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "ge": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "neg": OperatorAlgebraMeta(commutative=False, role="SCALE"),
    "power": OperatorAlgebraMeta(commutative=False, role="POWER"),
    "abs": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "LOG": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "SQRT": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "SIGN": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "ROUND": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    # --- 窗口聚合 ---
    "ts_mean": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_sum": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_std": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_var": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_max": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_min": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_median": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_rank": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_quantile": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_skew": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_kurt": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_product": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_pct": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_delta": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "ts_corr": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_cov": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_regression_slope": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_time_slope": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ts_decay_linear": OperatorAlgebraMeta(commutative=False, role="DECAY"),
    "ts_topk_sum": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    # --- 横截面 ---
    "rank": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "cs_rank": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "cs_demean": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "cs_zscore": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "zscore": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "scale": OperatorAlgebraMeta(commutative=False, role="SCALE"),
    "cap_neutralize": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "industry_neutralize": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "ind_neutralize": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "neutralize": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "market_cap_neutralize": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "size_neutralize": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    # --- 条件 / 逻辑 ---
    "where": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "iif": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "coalesce": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "fillna": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    # --- 延迟 / 移位 ---
    "delay": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "ref": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "delta": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "diff": OperatorAlgebraMeta(commutative=False, role="LAG"),
    # --- 裁剪 / 边界 ---
    "clip": OperatorAlgebraMeta(commutative=False, role="CLIP_BOUND"),
    "winsorize": OperatorAlgebraMeta(commutative=False, role="CLIP_BOUND"),
    # --- 特殊 ---
    "is_nan": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "is_infinite": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "is_null": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "is_not_null": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "not_": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    # --- 统计对算子 ---
    "corr": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "cov": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "covariance": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "Corr": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "Cov": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    # 注：corr/cov Stricly symmetric (corr(A,B) == corr(B,A)),
    # 见 ``factor_engine.cleaned_operators.base`` 注册，
    # 实现为 DataFrame.corr() / DataFrame.cov() 语义对称。
    "rankcorr": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "rank_corr": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "ts_corr": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "ts_cov": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    # --- 别名算子（对标 canonical）---
    "SMA": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "EMA": OperatorAlgebraMeta(commutative=False, role="DECAY"),
    "KAMA": OperatorAlgebraMeta(commutative=False, role="DECAY"),
    "DEMA": OperatorAlgebraMeta(commutative=False, role="DECAY"),
    "TEMA": OperatorAlgebraMeta(commutative=False, role="DECAY"),
    "MACD": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "MACD_line": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "MACD_signal": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "MACD_hist": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "RSI_WILDER": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_MEAN": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_SUM": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_STD": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_MAX": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_MIN": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_RANK": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_DELTA": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "TS_CORR": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "TS_COV": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "TS_DECAY_LINEAR": OperatorAlgebraMeta(commutative=False, role="DECAY"),
    "RANK": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "ZSCORE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "SCALE": OperatorAlgebraMeta(commutative=False, role="SCALE"),
    "DELAY": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "CLIP": OperatorAlgebraMeta(commutative=False, role="CLIP_BOUND"),
    "WINSORIZE": OperatorAlgebraMeta(commutative=False, role="CLIP_BOUND"),
    "WHERE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "IIF": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "POWER": OperatorAlgebraMeta(commutative=False, role="POWER"),
    "LOG": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "SQRT": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "SIGN": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "ROUND": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "Max": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Min": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Mean": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Median": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Sum": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Std": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Var": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Skew": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Kurt": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Slope": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "Ref": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "Delta": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "Diff": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "Delay": OperatorAlgebraMeta(commutative=False, role="LAG"),
    "RANKCORR": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "RANK_CORR": OperatorAlgebraMeta(commutative=True, role="WINDOW"),
    "Beta": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "ROLLING_BETA": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "COALESCE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "FillNA": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "DECAY_LINEAR": OperatorAlgebraMeta(commutative=False, role="DECAY"),
    "Percentile": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "CS_DEMEAN": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "CS_RANK": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "CS_ZSCORE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "IS_NAN": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "IS_INFINITE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "IS_NULL": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "IS_NOT_NULL": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "CAP_NEUTRALIZE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "INDUSTRY_NEUTRALIZE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "IND_NEUTRALIZE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "NEUTRALIZE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "MARKET_CAP_NEUTRALIZE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "SIZE_NEUTRALIZE": OperatorAlgebraMeta(commutative=False, role="OTHER"),
    "TS_TOPK_SUM": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_QUANTILE": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_KURT": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_SKEW": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_PCT": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_REGRESSION_SLOPE": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_TIME_SLOPE": OperatorAlgebraMeta(commutative=False, role="WINDOW"),
    "TS_DECAY_LINEAR": OperatorAlgebraMeta(commutative=False, role="DECAY"),
}


def get_operator_meta(op_name: str) -> OperatorAlgebraMeta | None:
    """返回算子的代数属性元数据。

    未知算子返回 None（默认全 False，保守不化简）。
    """
    return _OPERATOR_META.get(op_name)


def get_commutative_operators() -> frozenset[str]:
    """返回所有声明为 commutative 的算子名称集合。"""
    return frozenset(
        name for name, meta in _OPERATOR_META.items() if meta.commutative
    )


def get_operators_by_role(role: str) -> list[str]:
    """返回指定角色分类的所有算子名称。"""
    role_upper = role.upper()
    return sorted(
        name for name, meta in _OPERATOR_META.items()
        if meta.role.upper() == role_upper
    )