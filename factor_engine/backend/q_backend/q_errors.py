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

from dataclasses import dataclass

from runtime.exceptions import (
    DataQualityError,
    FactorEngineError,
    SchemaError,
    SemanticError,
)


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
    allow_nulls: bool = False
    grain: str = "daily"
    min_rows: int | None = None
    max_rows: int | None = None


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
