# -*- coding: utf-8 -*-
"""FE 100k GO P0#12 —— 失败分类（GO prompt §73）。

把 10 万因子生产运行中的所有异常映射到 **稳定、非消息解析** 的分类：

1. 底层 code：``runtime/exceptions.py`` 的既有异常族 + 内建类型按关键字映射；
2. 上层 code：GO §73 固定的 11 个 code（``INVALID_FORMULA`` / ``FIELD_CONTRACT`` /
   ``PIT_VIOLATION`` / ``BACKEND_PARITY`` / ``NUMERIC`` / ``DATA_MISSING`` /
   ``OOM`` / ``TIMEOUT`` / ``IO`` / ``WRITE`` / ``INTERNAL``）；
3. bucket：任务书要求的 4 类高层归口 —— ``data-degeneracy`` /
   ``contract-violation`` / ``resource-exhaustion`` / ``internal-bug``。

任何未知/裸异常都归 ``INTERNAL``（internal-bug），绝不允许在调用方
``except Exception: continue`` 全吞。``FailureReport`` 累加计数并输出 JSON。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from factor_engine.runtime.exceptions import (
    BackendBug,
    CheckpointInvalid,
    DataQualityError,
    DeadlineExceeded,
    FactorEngineError,
    OOMReplanRequired,
    OptimizerMismatch,
    ParameterDomainError,
    PermanentIOError,
    PITViolation,
    ResourceAdmissionError,
    SchemaError,
    SemanticError,
    TransientIOError,
    WriterFatalError,
)


# ---------------------------------------------------------------------------
# §73 稳定分类 code
# ---------------------------------------------------------------------------
class FailureCode(str, Enum):
    INVALID_FORMULA = "INVALID_FORMULA"
    FIELD_CONTRACT = "FIELD_CONTRACT"
    PIT_VIOLATION = "PIT_VIOLATION"
    BACKEND_PARITY = "BACKEND_PARITY"
    NUMERIC = "NUMERIC"
    DATA_MISSING = "DATA_MISSING"
    OOM = "OOM"
    TIMEOUT = "TIMEOUT"
    IO = "IO"
    WRITE = "WRITE"
    INTERNAL = "INTERNAL"


#: 任务书 4 类高层归口（§73 code → bucket）。
CODE_TO_BUCKET: dict[FailureCode, str] = {
    FailureCode.INVALID_FORMULA: "contract-violation",
    FailureCode.FIELD_CONTRACT: "contract-violation",
    FailureCode.PIT_VIOLATION: "contract-violation",
    FailureCode.BACKEND_PARITY: "contract-violation",
    FailureCode.NUMERIC: "data-degeneracy",
    FailureCode.DATA_MISSING: "data-degeneracy",
    FailureCode.OOM: "resource-exhaustion",
    FailureCode.TIMEOUT: "resource-exhaustion",
    FailureCode.IO: "internal-bug",
    FailureCode.WRITE: "internal-bug",
    FailureCode.INTERNAL: "internal-bug",
}

_VALID_BUCKETS = frozenset(CODE_TO_BUCKET.values())


class _TypedRule:
    """把 ``runtime.exceptions`` 的某种具体异常映射到 §73 code。"""

    __slots__ = ("exc_types", "target")

    def __init__(self, exc_types: tuple[type, ...], target: FailureCode) -> None:
        self.exc_types = exc_types
        self.target = target

    def applies(self, exc: BaseException) -> bool:
        return isinstance(exc, self.exc_types)


_TYPED_RULES: tuple[_TypedRule, ...] = (
    _TypedRule((SemanticError, ParameterDomainError), FailureCode.INVALID_FORMULA),
    _TypedRule((PITViolation,), FailureCode.PIT_VIOLATION),
    _TypedRule((SchemaError,), FailureCode.FIELD_CONTRACT),
    _TypedRule((BackendBug, OptimizerMismatch), FailureCode.BACKEND_PARITY),
    _TypedRule((DataQualityError,), FailureCode.DATA_MISSING),
    _TypedRule((OOMReplanRequired,), FailureCode.OOM),
    _TypedRule((DeadlineExceeded,), FailureCode.TIMEOUT),
    _TypedRule((TransientIOError, PermanentIOError), FailureCode.IO),
    _TypedRule((WriterFatalError,), FailureCode.WRITE),
)


# ---------------------------------------------------------------------------
# 关键字（内建 / 第三方异常类型名）→ code
# ---------------------------------------------------------------------------
_KEYWORD_RULES: tuple[tuple[tuple[str, ...], FailureCode], ...] = (
    (("MemoryError",), FailureCode.OOM),
    (("timeout", "Timeout", "TimeoutError", "DeadlineExceeded"), FailureCode.TIMEOUT),
    (("ParseError", "FormulaError", "SyntaxError"), FailureCode.INVALID_FORMULA),
    (("FieldError", "SchemaError", "ContractError", "field_contract"), FailureCode.FIELD_CONTRACT),
    (("BackendParity", "ParityError"), FailureCode.BACKEND_PARITY),
    (("DataMissing", "MissingError", "KeyError__missing", "no data", "empty source"), FailureCode.DATA_MISSING),
    (("OSError",), FailureCode.IO),
    (("ParquetWriteError", "WriteError"), FailureCode.WRITE),
    (("FloatingPointError", "OverflowError", "division by zero"), FailureCode.NUMERIC),
)


def classify_exception(exc: BaseException) -> tuple[FailureCode, str]:
    """返回 ``(code, bucket)``。

    优先级：① 具体类型精确匹配；② 类型名关键字；③ 兜底 ``INTERNAL``。
    """
    for rule in _TYPED_RULES:
        if rule.applies(exc):
            return rule.target, CODE_TO_BUCKET[rule.target]

    type_name = type(exc).__name__
    for keywords, target in _KEYWORD_RULES:
        if type_name in keywords:  # 类型名精确命中
            return target, CODE_TO_BUCKET[target]
        msg = str(exc) or ""
        for kw in keywords:
            if kw in msg:
                return target, CODE_TO_BUCKET[target]

    final_code = FailureCode.INTERNAL
    return final_code, CODE_TO_BUCKET[final_code]


@dataclass
class FailureReport:
    """失败分类汇总（可 dump JSON）。"""

    by_code: Counter = field(default_factory=Counter)
    by_bucket: Counter = field(default_factory=Counter)

    def record(self, exc: BaseException) -> FailureCode:
        code, bucket = classify_exception(exc)
        self.by_code[str(code)] += 1
        self.by_bucket[bucket] += 1
        return code

    def code_counts(self) -> dict[str, int]:
        return {str(c): int(n) for c, n in sorted(self.by_code.items())}

    def bucket_counts(self) -> dict[str, int]:
        return {str(b): int(n) for b, n in sorted(self.by_bucket.items())}

    @property
    def total(self) -> int:
        return int(sum(self.by_code.values()))

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "by_code": self.code_counts(), "by_bucket": self.bucket_counts()}

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
