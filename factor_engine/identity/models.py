# -*- coding: utf-8 -*-
"""冻结 dataclass FactorIdentity 和值对象。

``FactorIdentity`` 是 ``get_factor_identity`` 的返回值类型，
为 frozen dataclass，严格按任务书字段定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FactorIdentity:
    """因子身份的冻结 dataclass。

    全部字段在构造时填充，不可变。
    """

    canonical_dsl: str
    """canonical AST 反序列化的 DSL 文本（稳定序列化形式的 JSON）。"""

    canonical_ast_hash: str
    """64 字符全长的 SHA-256 十六进制哈希。"""

    signal_equivalence_id: str
    """sign-normalized 的信号等价 ID。"""

    parameter_family_id: str | None
    """parameter-family 化的 ID，无参数时可为 None。"""

    orientation: int
    """全局符号方向：+1 或 -1。"""

    subtree_hashes: Tuple[str, ...]
    """全部子树哈希去重列表（后序）。"""

    field_signature: str
    """字段签名（字段名列表的哈希）。"""

    operator_signature: str
    """算子签名（算子名列表的哈希）。"""

    complexity: int
    """AST 节点总数（包含所有算子调用+字段引用+字面量）。"""

    depth: int
    """AST 最大深度（根为 1，每层 +1）。"""

    lookback: int
    """最大窗口参数（启发式，从 WINDOW 角色的字面量参数中取最大值）。"""

    identity_version: str = "factor_identity_v1"
    """身份版本号。"""

    operator_semantics_version: str = "v1"
    """算子语义版本号。"""


@dataclass(frozen=True)
class IdentityLookupResult:
    """身份查询结果的值对象（保留供未来扩展）。"""

    identity: FactorIdentity
    source: str = "parsed"


class FactorIdentityError(ValueError):
    """公式身份计算过程中出错时抛出的异常。

    携带原始公式文本和错误原因。
    """

    def __init__(self, formula: str, reason: str) -> None:
        self.formula = formula
        self.reason = reason
        super().__init__(f"FactorIdentityError: {reason}\n  formula: {formula}")


def orientation_from_int(value: int) -> int:
    """验证 orientation 为 +1 或 -1。"""
    if value not in (-1, 1):
        raise ValueError(f"orientation must be +1 or -1, got {value}")
    return value