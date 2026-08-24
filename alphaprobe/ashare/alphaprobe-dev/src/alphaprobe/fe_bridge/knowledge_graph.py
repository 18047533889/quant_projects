"""让知识图谱接受 factor_engine DSL 冷启动因子。"""

from __future__ import annotations

from typing import Optional

from shared.alphagen.data.expression_knowledge_graph import ExpressionKnowledgeGraph, ExpressionPayload
from alphaprobe.fe_bridge.dsl_expression import (
    FactorEngineDslExpression,
    dsl_payload_tokens,
    validate_factor_engine_dsl,
)

_PATCHED = False


def patch_knowledge_graph_for_factor_engine_dsl() -> None:
    global _PATCHED
    if _PATCHED:
        return

    def _build_payload_fe(self: ExpressionKnowledgeGraph, expression: str) -> Optional[ExpressionPayload]:
        from alphaprobe.fe_bridge.expr_parse import looks_like_legacy_alphagen

        text = expression.strip()
        # A 股挖掘：只接受 factor_engine DSL，拒绝 AlphaGen 旧语法
        if looks_like_legacy_alphagen(text):
            return None
        ok, _ = validate_factor_engine_dsl(text)
        if not ok:
            return None
        expr_obj = FactorEngineDslExpression(text)
        tokens = dsl_payload_tokens(text)
        return ExpressionPayload(
            expression=expr_obj,
            length=len(tokens),
            tokens=tokens,
            source=text,
        )

    ExpressionKnowledgeGraph._build_payload = _build_payload_fe
    _PATCHED = True
