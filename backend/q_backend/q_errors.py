"""Q Backend Typed Errors (Q2-P0-016, Q2-P0-017, Q2-P0-018).

Production requirements:
- Missing base data → typed DataUnavailableError (not empty DataFrame)
- Schema mismatch → typed QSchemaError (hard fail)
- Runtime failures → typed errors (not silent Pandas fallback)
- Only planning-time capability decision allows fallback

Hard Gates (文档 §85):
- Q_BACKEND_FAIL_CLOSED_DATA: 缺失数据必须抛出 typed error
- Q_BACKEND_FAIL_CLOSED_SCHEMA: schema 不匹配必须 hard fail
- Q_BACKEND_FAIL_CLOSED_RUNTIME: runtime 失败禁止静默回退 Pandas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from runtime.exceptions import (
    DataQualityError,
    FactorEngineError,
    SchemaError,
    SemanticError,
)


# ---------------------------------------------------------------------------
# Q2-P0-017: Semantic-kind → q output dtype / null policy derivation.
#
# The ``value`` column dtype is derived from the canonical semantic contract
# (``IRNode.semantic_attrs["semantic_kind"]`` — vocabulary in
# ``ir.types.SemanticType``), never hardcoded to float64.  A Condition /
# Event / Mask output is bool, a State / Group / Period / Status output is an
# integer categorical code, and a timestamp-role output is datetime — only
# numeric-derived series fall back to float64.
# ---------------------------------------------------------------------------

# Canonical semantic kind -> q ``value`` output dtype.  Kinds absent here are
# numeric-derived / generic and default to float64.
_SEMANTIC_KIND_TO_DTYPE: Mapping[str, str] = {
    # Condition / Event / Mask outputs are boolean.
    "EventBool": "bool",
    "MaskBool": "bool",
    "ConditionBool": "bool",
    # State / categorical / group / period / status outputs are int category
    # codes (the integer backing is the canonical storage dtype).
    "StateSigned": "int64",
    "GroupKey": "int64",
    "FiscalPeriodId": "int64",
    "StatusCode": "int64",
    "CategoryCode": "int64",
    # Timestamp-role outputs.
    "KnowledgeTimestamp": "datetime64[ns]",
    "EffectiveTimestamp": "datetime64[ns]",
    "RevisionTimestamp": "datetime64[ns]",
}


def semantic_kind_to_output_dtype(semantic_kind: str | None) -> str:
    """Derive the q ``value`` output dtype from a canonical semantic kind.

    ``None`` / unknown kinds are numeric-derived and return ``float64``.  The
    call site never hardcodes the dtype; it always goes through this authority.
    """
    if semantic_kind is None:
        return "float64"
    return _SEMANTIC_KIND_TO_DTYPE.get(semantic_kind, "float64")


# output_null_policy values
NULL_POLICY_ANY_ALLOWED = "any_allowed"    # sparse condition/event/mask panel: NaN == absent
NULL_POLICY_WARMUP_OK = "warmup_ok"        # numeric/categorical: rolling-warmup NaN legal
NULL_POLICY_STRICT = "strict"              # timestamp-role: no nulls permitted


def semantic_null_policy(
    semantic_kind: str | None,
) -> tuple[bool, bool, str]:
    """Derive null policy from the canonical semantic kind.

    Returns ``(warmup_nulls_allowed, structural_nulls_allowed,
    output_null_policy)``.  This replaces the blanket ``allow_nulls=True``: a
    sparse event/mask panel is structurally nullable, a numeric series only
    permits rolling-warmup NaN, and a timestamp role forbids nulls entirely.
    """
    if semantic_kind in {"EventBool", "MaskBool", "ConditionBool"}:
        return True, True, NULL_POLICY_ANY_ALLOWED
    if semantic_kind in {
        "KnowledgeTimestamp",
        "EffectiveTimestamp",
        "RevisionTimestamp",
    }:
        return False, False, NULL_POLICY_STRICT
    # numeric / categorical / state outputs: rolling-warmup NaN allowed, but a
    # structural (interior) null is not silently waved through.
    return True, False, NULL_POLICY_WARMUP_OK


def semantic_null_policy_of_contract(
    semantic_kind: str | None,
) -> dict[str, object]:
    """Full derived null-policy dict for QOutputContract construction."""
    warmup, structural, policy = semantic_null_policy(semantic_kind)
    return {
        "warmup_nulls_allowed": warmup,
        "structural_nulls_allowed": structural,
        "output_null_policy": policy,
        "allow_nulls": policy != NULL_POLICY_STRICT,
    }


# ---------------------------------------------------------------------------
# Q2-P0-016: Missing base data must raise typed error
# ---------------------------------------------------------------------------

class QDataUnavailableError(DataQualityError):
    """Q backend 缺失基础数据（Q2-P0-016）。

    Production 要求：
    - 不得返回空 DataFrame
    - 必须抛出 typed error
    - retry=False（需上游修复数据源）
    """
    pass


# ---------------------------------------------------------------------------
# Q2-P0-017: Output schema validation failures
# ---------------------------------------------------------------------------

class QSchemaError(SchemaError):
    """Q backend schema 不匹配（Q2-P0-017）。

    Production 要求：
    - timestamp/instrument/value 列必须存在
    - dtype 必须匹配预期
    - row_count/grain/sort order/nullability 必须验证
    - schema mismatch → hard fail
    """
    pass


class QOutputContractViolation(QSchemaError):
    """Q backend 输出契约违例（Q2-P0-017）。

    包括：
    - 缺失必需列（timestamp, instrument, value）
    - dtype 不匹配
    - 未排序（但要求排序）
    - 非法 null 值
    - row_count 不符合预期
    """
    pass


# ---------------------------------------------------------------------------
# Q2-P0-017: Output contract specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QOutputContract:
    """Q backend 输出契约（Q2-P0-017）。

    Production 验证要求：
    - expected_columns: 必须包含的列
    - expected_dtypes: 每列的预期 dtype
    - require_sorted: 是否要求按 (timestamp, instrument) 排序
    - allow_nulls: 是否允许 null 值
    - grain: 数据粒度（"daily", "minute", etc.）
    - min_rows: 最少行数（None = 不检查）
    - max_rows: 最多行数（None = 不检查）
    """
    expected_columns: tuple[str, ...]
    expected_dtypes: dict[str, str]  # column -> dtype name
    require_sorted: bool = True
    allow_nulls: bool = True
    grain: str | None = None
    min_rows: int | None = None
    max_rows: int | None = None
    # Derived from the canonical semantic kind (not blanket allow_nulls).
    warmup_nulls_allowed: bool = True
    structural_nulls_allowed: bool = True
    output_null_policy: str = NULL_POLICY_WARMUP_OK


# ---------------------------------------------------------------------------
# Q2-P0-018: Runtime failure taxonomy (not planning-time fallback)
# ---------------------------------------------------------------------------

class QRuntimeError(FactorEngineError):
    """Q runtime 执行失败（Q2-P0-018）。

    Production 要求：
    - Runtime 失败必须 hard fail
    - 禁止静默回退到 Pandas
    - 只有 planning-time capability decision 允许选择其他 backend

    retry=False（需修复 q 代码或数据）
    """
    pass


class QSemanticError(SemanticError):
    """Q backend 语义错误（Q2-P0-018）。

    包括：
    - q 代码编译失败
    - q 运行时语义错误（类型不匹配、非法操作）
    - PIT 违例（时间因果性）

    retry=False
    replan=False
    abort=True
    """
    pass


class QCompilerError(QRuntimeError):
    """Q compiler 错误（Q2-P0-018）。

    包括：
    - 算子无法编译为 q
    - Region 依赖图错误
    - 参数验证失败

    retry=False
    replan=True（可能需要选择其他 backend）
    """
    pass


class QExecutionError(QRuntimeError):
    """Q executor 执行错误（Q2-P0-018）。

    包括：
    - q 进程不可用
    - q 代码执行失败
    - 数据传输失败

    retry=True（瞬时错误）
    retry=False（语义错误）
    """
    pass


class QProcessUnavailableError(QExecutionError):
    """Q process 不可用（Q2-P0-018）。

    Production 要求：
    - q 不可用时必须 fail（不得静默回退）
    - 或者 planning-time 就选择其他 backend

    retry=True（可能是瞬时启动延迟）
    """
    pass


class QPlanningFallbackAllowed(FactorEngineError):
    """Planning-time 允许的 fallback（Q2-P0-018）。

    这是唯一合法的 fallback 场景：
    - Planner 在 planning-time 检查 capability
    - 发现 q 不支持某些算子
    - 决策使用其他 certified backend

    Runtime 发现失败后不得使用此路径。
    """
    pass


# ---------------------------------------------------------------------------
# Q2-P0-015: PhysicalBackendRegion gap documentation
# ---------------------------------------------------------------------------

class QPhysicalRegionNotImplemented(FactorEngineError):
    """Q backend 尚未实现真正的 PhysicalBackendRegion（Q2-P0-015）。

    当前状态：
    - QBackend 接收整棵逻辑树（simplified implementation）
    - 应该是：Canonical DAG → PhysicalBackendRegion → QCompiler → QExecutor

    Production blocker: FAIL until properly implemented.

    Required contract:
    1. Planner 生成 PhysicalBackendRegion（不是整棵树）
    2. QBackend 只接收已切分的 region
    3. Region 边界由 Planner 决策（不是 backend）
    """
    pass
