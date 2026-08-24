"""挖掘表达式解析：只接受 factor_engine DSL（不再回退 AlphaGen 树语法）。"""

from __future__ import annotations

import re
from typing import Any

from alphaprobe.fe_bridge.dsl_expression import try_parse_factor_engine_dsl

_ALPHAGEN_LEGACY_RE = re.compile(
    r"(?:\$[A-Za-z_]|%(?:d|s)\b|\bTs[A-Z][A-Za-z]*\s*\(|\b(?:Div|Mul|Sub|Add|Ref)\s*\()"
)


def looks_like_legacy_alphagen(text: str) -> bool:
    return bool(_ALPHAGEN_LEGACY_RE.search(str(text)))


def parse_mining_expression(text: str, alphagen_parser: Any = None) -> Any:
    """解析挖掘表达式为 ``FactorEngineDslExpression``。

    ``alphagen_parser`` 保留签名兼容，A 股 factor_engine 模式下不再使用。
    """
    stripped = str(text).strip()
    if not stripped:
        raise ValueError("empty expression")
    if looks_like_legacy_alphagen(stripped):
        raise ValueError(
            "拒绝 AlphaGen 旧语法（$close / TsMean / Div…）；请使用 factor_engine DSL，"
            f"例如 rank(ts_mean(close, 20))。收到: {stripped[:120]}"
        )
    dsl_expr = try_parse_factor_engine_dsl(stripped)
    if dsl_expr is None:
        raise ValueError(
            f"无法解析为 factor_engine DSL（surface=compat）: {stripped[:160]}"
        )
    return dsl_expr
