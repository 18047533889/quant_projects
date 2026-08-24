"""factor_engine DSL 字符串包装为 AlphaGen Expression（供挖掘池 / 知识图谱评估）。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class FactorEngineDslExpression:
    """持有 ``parse_expr`` 可校验的 factor_engine DSL，评估走 ``FactorEngineStockData``。"""

    __slots__ = ("_dsl", "_source")

    def __init__(self, dsl: str, source: str | None = None) -> None:
        self._dsl = str(dsl).strip()
        self._source = (source or self._dsl).strip()
        if not self._dsl:
            raise ValueError("empty factor_engine DSL")

    @property
    def dsl(self) -> str:
        return self._dsl

    @property
    def is_featured(self) -> bool:
        return True

    def evaluate(self, data: Any, period: slice = slice(0, 1)) -> Any:
        import torch
        from shared.alphagen.data.expression import OutOfDataRangeError
        from alphaprobe.fe_bridge.stock_data import FactorEngineStockData

        if not isinstance(data, FactorEngineStockData):
            raise TypeError("FactorEngineDslExpression 需要 FactorEngineStockData")
        assert period.step == 1 or period.step is None
        if (
            period.start < -data.max_backtrack_days
            or period.stop - 1 > data.max_future_days
        ):
            raise OutOfDataRangeError()
        full = data.evaluate_dsl(self._dsl)
        if full.shape[0] < data.data.shape[0]:
            pad = data.data.shape[0] - full.shape[0]
            full = torch.nn.functional.pad(full, (0, 0, 0, pad))
        elif full.shape[0] > data.data.shape[0]:
            full = full[-data.data.shape[0]:]
        start = period.start + data.max_backtrack_days
        stop = period.stop + data.max_backtrack_days + data.n_days - 1
        return full[start:stop]

    def __str__(self) -> str:
        return self._source

    def __repr__(self) -> str:
        return f"FE_DSL({self._dsl[:80]})"


_DSL_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def extract_dsl_operator_names(dsl: str) -> tuple[str, ...]:
    from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

    ensure_factor_engine_importable()
    from factor_engine.api.operator_registry import build_dsl_allowlist

    allow = build_dsl_allowlist()
    found: set[str] = set()
    for match in _DSL_CALL_RE.finditer(dsl):
        name = match.group(1)
        if name in allow:
            found.add(name)
    return tuple(sorted(found))


def validate_factor_engine_dsl(dsl: str) -> tuple[bool, str]:
    from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

    ensure_factor_engine_importable()
    from factor_engine.api.dsl_parser import DSLParseError, parse_expr

    text = str(dsl or "").strip()
    if not text:
        return False, "empty DSL"
    try:
        # V9 冷启动库使用 compat 算子面（ts_ema / ATR_WILDER 等）
        parse_expr(text, surface="compat")
        return True, "OK"
    except DSLParseError as exc:
        return False, str(exc)


def dsl_payload_tokens(dsl: str) -> tuple[str, ...]:
    """知识图谱用的粗粒度 token（算子名 + 字段名）。"""
    ops = extract_dsl_operator_names(dsl)
    fields = tuple(
        sorted({m.lower() for m in re.findall(r"\b(open|high|low|close|volume|vwap)\b", dsl)})
    )
    tokens: list[str] = [f"OP:{op}" for op in ops]
    tokens.extend(f"FIELD:{f}" for f in fields)
    if not tokens:
        tokens.append("DSL:raw")
    return tuple(tokens)


def try_parse_factor_engine_dsl(expression: str) -> FactorEngineDslExpression | None:
    text = str(expression).strip()
    ok, _ = validate_factor_engine_dsl(text)
    if not ok:
        return None
    return FactorEngineDslExpression(text)

