"""算子复杂度评分表生成。

通过 DeepSeek V4 Flash 为已加载的算子生成临时复杂度评分表。
"""

from __future__ import annotations

import json
from typing import Any

from .operator_loader import ensure_factor_engine_path, get_registered_operators


def build_complexity_table(
    operator_list: list[str] | None = None,
) -> dict[str, float]:
    """通过 DeepSeek 为已加载的算子生成临时复杂度评分表。

    以 1.0 为基准 scale。基础运算符为 0.5，简单时序/截面为 1.0，
    统计/回归为 2.0~5.0，机器学习为 8.0~10.0。

    Args:
        operator_list: 要评分的算子列表。None=从 OperatorRegistry 获取。

    Returns:
        {算子名: 复杂度分数}
    """
    ensure_factor_engine_path()

    if operator_list is None:
        operator_list = get_registered_operators()

    if not operator_list:
        return {}

    # 分块发送（避免超长 prompt）
    chunk_size = 200
    all_weights: dict[str, float] = {}

    for i in range(0, len(operator_list), chunk_size):
        chunk = operator_list[i: i + chunk_size]
        chunk_weights = _query_complexity_chunk(chunk)
        all_weights.update(chunk_weights)

    return all_weights


def _query_complexity_chunk(operators: list[str]) -> dict[str, float]:
    """向 DeepSeek 查询一批算子的复杂度评分。"""
    from utils.deepseek_client import deepseek_chat

    ops_json = json.dumps(operators, ensure_ascii=False)

    prompt = f"""你是一个量化因子引擎的配置专家。请为以下算子列表生成复杂度评分表。

评分规则（以 1.0 为基准 scale）：
- 0.5：基础四则运算、比较、逻辑（add, subtract, multiply, divide, gt, lt, if_else 等）
- 1.0~1.5：简单时序/截面算子（ts_mean, ts_sum, rank, delay, abs, sign, log 等）
- 2.0~3.0：统计类算子（ts_std, ts_cov, ts_corr, ts_skew, ts_kurt, winsorize, zscore 等）
- 3.0~5.0：回归/复杂模型（ts_regression, ts_beta, group_neutralize, lm 等）
- 8.0~10.0：机器学习类（nn, gbdt, rf, xgboost, svm 等）

**重要：所有评分必须以 1.0 为基准 scale，即 ts_mean 应为 1.0，而不是 10 或 100。**

请严格按照 JSON 格式返回，只返回 JSON 对象，不要包含任何其他文字：
{{"op_name": 分数, ...}}

算子列表：
{ops_json}"""

    try:
        result = deepseek_chat([
            {"role": "system", "content": "你是一个精确的 JSON 生成器，只输出 JSON，不包含其他任何内容。"},
            {"role": "user", "content": prompt},
        ])
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        weights = json.loads(result)
        validated: dict[str, float] = {}
        for op, score in weights.items():
            if isinstance(score, (int, float)):
                validated[op] = max(0.1, min(20.0, float(score)))
        return validated
    except Exception as e:
        print(f"[WARNING] 复杂度评分表生成失败 (chunk): {e}")
        return {op: 1.0 for op in operators}
