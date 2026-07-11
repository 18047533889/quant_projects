"""
AutoFactorEvaluation 标准路径规范 — 三基座设计

定义三套独立的路径推导规则:

  1. candidate_pool — COS 候选因子池
     子路径: candidate/ (投放入口), archive/ (已处理归档)

  2. factor_pool — COS 因子持久化基座
     子路径: tier1~tier4 所有基座目录

  3. local_tmp — 本地临时工作区
     子路径: tier0 各阶段临时目录

使用方式:
    from path_convention import resolve_candidate, resolve_factor, resolve_local

    resolve_candidate("cos://bucket/candidate_pool", "candidate")
    # → cos://bucket/candidate_pool/candidate

    resolve_factor("cos://bucket/factor_pool", "gateway_pass_base")
    # → cos://bucket/factor_pool/tier1/gateway_pass_base

    resolve_local("/tmp/afe_tmp", "gateway_temp")
    # → /tmp/afe_tmp/tier0/gateway_temp
"""

from __future__ import annotations

from pathlib import Path

from uri_path import UriPath

# ============================================================
# 基座一: candidate_pool
# 候选因子池 — COS 端
# ============================================================
CANDIDATE_TREE: dict[str, str] = {
    "candidate":   "candidate",    # 封板因子投放入口
    "archive":     "archive",      # 已处理完成的 campaign 归档
}

# ============================================================
# 基座二: factor_pool
# 因子持久化存储 — COS 端
# ============================================================
FACTOR_TREE: dict[str, str] = {
    # tier1: Gateway / Assetization / Purification 输出基座
    "gateway_pass_base":              "tier1/gateway_pass_base",
    "gateway_duplicated_base":        "tier1/gateway_duplicated_base",
    "gateway_anti_sample_base":       "tier1/gateway_anti_sample_base",
    "assetization_raw_factor_base":   "tier1/assetization_raw_factor_base",
    "purification_pure_factor_base":  "tier1/purification_pure_factor_base",

    # tier2: 孵化 / 变异
    "tier2_incubator":                "tier2/2_fix_base",
    "tier2x_optimization_factory":    "tier2/2x_llm_mutation_base",

    # tier3: 生产级因子
    "tier3a_core":                    "tier3/3a_core_production_base",
    "tier3b_satellite":               "tier3/3b_satellite_production_base",
    "tier3c_feature":                 "tier3/3c_feature_matierial_base",
    "tier3d_optimized_reserve":       "tier3/3d_operation_storage_base",

    # tier4: 归档
    "tier4_archive":                  "tier4/anti_sample_base",
}

# ============================================================
# 基座三: local_tmp
# 本地临时工作区
# ============================================================
LOCAL_TREE: dict[str, str] = {
    # tier0: 各阶段临时工作目录
    "gateway_temp":            "tier0/gateway_temp",
    "assetization_temp":       "tier0/assetization_temp",
    "purification_temp":       "tier0/purification_temp",
    "evaluation_temp":         "tier0/evaluation_temp",

    # 计算缓存
    "market_data_cache":                 "cache/market_data",
    "timeseries_forward_return_cache":   "cache/timeseries_forward_return",
    "evaluation_label_cache":            "cache/evaluation_label",
    "gateway_report_cache":              "cache/gateway_report",

    # 日志
    "pipeline_log":          "logs/pipeline",
    "gateway_log":           "logs/gateway",
    "assetization_log":      "logs/assetization",
    "purification_log":      "logs/purification",
    "evaluation_log":        "logs/evaluation",
}

# ============================================================
# 合并映射（用于全量查询）
# ============================================================
_ALL_TREES: dict[str, str] = {**CANDIDATE_TREE, **FACTOR_TREE, **LOCAL_TREE}


# ============================================================
# 解析函数
# ============================================================


def resolve_candidate(base: str | Path, logical_key: str) -> UriPath | Path:
    """从 candidate_pool 基座推导路径。"""
    rel = CANDIDATE_TREE.get(logical_key)
    if rel is None:
        raise KeyError(
            f"未知 candidate_pool 路径键名: {logical_key!r}。"
            f"可用: {', '.join(sorted(CANDIDATE_TREE))}"
        )
    base_str = str(base).rstrip("/")
    if base_str.startswith("cos://") or base_str.startswith("cos:"):
        return UriPath(base_str + "/" + rel)
    return Path(base_str) / rel


def resolve_factor(base: str | Path, logical_key: str) -> UriPath | Path:
    """从 factor_pool 基座推导路径。"""
    rel = FACTOR_TREE.get(logical_key)
    if rel is None:
        raise KeyError(
            f"未知 factor_pool 路径键名: {logical_key!r}。"
            f"可用: {', '.join(sorted(FACTOR_TREE))}"
        )
    base_str = str(base).rstrip("/")
    if base_str.startswith("cos://") or base_str.startswith("cos:"):
        return UriPath(base_str + "/" + rel)
    return Path(base_str) / rel


def resolve_local(base: str | Path, logical_key: str) -> UriPath | Path:
    """从 local_tmp 基座推导路径。"""
    rel = LOCAL_TREE.get(logical_key)
    if rel is None:
        raise KeyError(
            f"未知 local_tmp 路径键名: {logical_key!r}。"
            f"可用: {', '.join(sorted(LOCAL_TREE))}"
        )
    return UriPath(str(base).rstrip("/")) / rel


# ============================================================
# 全量查询与工具函数
# ============================================================


def all_logical_keys() -> list[str]:
    """返回所有逻辑键名列表。"""
    return sorted(_ALL_TREES)


# ============================================================
# 目录树结构查询
# ============================================================


def expected_subdirs(tree: dict[str, str]) -> dict[str, list[str]]:
    """从指定目录树提取 tier → 子目录名 映射。"""
    result: dict[str, list[str]] = {}
    for rel in sorted(set(tree.values())):
        parts = rel.split("/")
        if len(parts) >= 2 and parts[0].startswith("tier"):
            tier = parts[0]
            if tier not in result:
                result[tier] = []
            result[tier].append(parts[1])
    return result
