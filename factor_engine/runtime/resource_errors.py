# -*- coding: utf-8 -*-
"""严格异常分类：只有 capability 类错误允许跨后端 fallback。

背景（Phase 5 R11）
    历史上 SQL/minute pushdown 用 ``except Exception: fallback pandas`` 吞掉一切
    失败。这在大规模执行时是危险的：DuckDB 因 OOM / deadline / PIT / schema 失败
    后，引擎改试 Pandas，通常更容易把机器炸掉，而且掩盖真实错误。

约定
    - ``CapabilityMiss`` / ``CompilationUnsupported``：**可以** fallback（换后端重试）。
    - 其余 ``ResourceGovernanceError`` 子类：production **必须** fail-closed，
      直接向上抛，禁止跨后端 fallback。

生产策略由 :func:`may_fallback` / :func:`is_fail_closed_error` 判断；
DataAccess 侧异常通过 :func:`is_data_access_governance_failure` 映射进来。
"""
from __future__ import annotations


class ResourceGovernanceError(RuntimeError):
    """资源/执行治理失败基类：默认 fail-closed，不允许跨后端 fallback。"""


class CapabilityMiss(ResourceGovernanceError):
    """算子/后端缺少某项能力（canonical 未注册 / 未认证 / 不支持）。

    这是唯一允许触发跨后端 fallback 的类别之一：换一个后端可能就能算。
    """


class CompilationUnsupported(ResourceGovernanceError):
    """计划无法在后端编译（算子组合不支持 / 计划形态不支持）。

    允许 fallback 的另一类别：编译失败不代表计算本身失败。
    """


class ResourceBudgetExceeded(ResourceGovernanceError):
    """内存 / CPU / 结果字节预算超限。禁止 fallback。"""


class DeadlineExceeded(ResourceGovernanceError):
    """查询超时（deadline）。禁止 fallback。"""


class OutOfMemory(ResourceGovernanceError):
    """检测到 OOM 压力或内存分配失败。禁止 fallback。"""


class SemanticContractError(ResourceGovernanceError):
    """语义契约违约（单位/口径/join fan-out/字段歧义）。禁止 fallback。"""


class PITViolation(ResourceGovernanceError):
    """Point-in-Time 违约（未来数据 / 回滚 / 报告期错误）。禁止 fallback。"""


class SchemaError(ResourceGovernanceError):
    """Schema 不匹配（列缺失 / 类型错误 / 物理布局不符）。禁止 fallback。"""


class DataQualityError(ResourceGovernanceError):
    """数据质量失败（空表 / 全 NaN / 覆盖率不足 / 未来时间戳）。禁止 fallback。"""


class TransientIOError(ResourceGovernanceError):
    """瞬时 IO 失败（网络 / COS 抖动）。可重试，**不**换后端。"""


class BackendExecutionError(ResourceGovernanceError):
    """后端执行失败（非上述已知类别）。禁止 fallback。"""


#: 允许跨后端 fallback 的类别（语义上等价于「这个后端做不到」）。
_FALLBACK_SAFE = (CapabilityMiss, CompilationUnsupported)


def may_fallback(exc: BaseException) -> bool:
    """该异常是否允许跨后端 fallback（仅 CapabilityMiss / CompilationUnsupported）。"""
    return isinstance(exc, _FALLBACK_SAFE)


def is_fail_closed_error(exc: BaseException) -> bool:
    """该异常是否必须 fail-closed（治理类错误且不可 fallback）。"""
    return isinstance(exc, ResourceGovernanceError) and not isinstance(exc, _FALLBACK_SAFE)


#: DataAccess 侧「治理失败」异常类型（禁止 fallback）。
_DA_GOVERNANCE_EXCEPTIONS: tuple[type, ...] = ()
try:
    import data_access.core.exceptions as _da_exc  # type: ignore[import-not-found]
    _DA_GOVERNANCE_EXCEPTIONS = tuple(
        cls
        for name, cls in vars(_da_exc).items()
        if isinstance(cls, type) and issubclass(cls, _da_exc.DataAccessError)
    )
except Exception:  # pragma: no cover - dataaccess 未安装时保持空
    _DA_GOVERNANCE_EXCEPTIONS = ()


def is_data_access_governance_failure(exc: BaseException) -> bool:
    """DataAccess 层抛出的治理错误（ValidationError / DeadlineExceeded / …）一律 fail-closed。

    DataAccess 的契约错误（ValidationError / DeadlineExceeded / DataError / EngineError）
    反映的是「请求本身非法或预算/数据问题」，不是「后端能力缺失」，因此禁止 fallback。
    """
    return isinstance(exc, _DA_GOVERNANCE_EXCEPTIONS)


def classify_exception(exc: BaseException) -> tuple[bool, bool]:
    """返回 ``(may_fallback, fail_closed)`` 决策对。

    DataAccess 治理异常 → fail-closed；本地治理异常 → 按类别；
    其余未知异常 → 默认 fail-closed（不静默 fallback，宁可显式失败）。
    """
    if is_data_access_governance_failure(exc):
        return False, True
    if isinstance(exc, ResourceGovernanceError):
        return may_fallback(exc), is_fail_closed_error(exc)
    return False, True
