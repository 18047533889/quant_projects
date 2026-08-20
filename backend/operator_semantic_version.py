# -*- coding: utf-8
"""算子语义版本：定义变更须 bump version 并在 factor metadata 记录。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSemanticVersion:
    canonical: str
    version: int
    note: str = ""


# 已知语义变更历史（新因子应使用最新 version）
OPERATOR_SEMANTIC_VERSIONS: dict[str, int] = {
    "group_percentile": 2,  # v1: NULL→0; v2: NULL→NULL
    "signed_log": 1,  # sign(x)*log(abs(x)+1e-10)
    "compare": 2,  # v2: NULL/NaN propagate
    "maximum": 2,
    "minimum": 2,
    "protected_log": 2,  # NULL preserved
    "protected_div": 2,
    "rank": 1,
    "rank_pct": 1,
    "ts_beta": 2,  # v2: pairwise var/cov sample set
    "ts_corr": 2,
    "ts_cov": 2,
}


def semantic_version(canon: str) -> int:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return OPERATOR_SEMANTIC_VERSIONS.get(name, 1)


def versioned_name(canon: str) -> str:
    return f"{canon}@v{semantic_version(canon)}"


def record_version_in_factor_metadata() -> bool:
    """Factor metadata 应记录 ``operator_semantic_versions`` 映射。"""
    return True
