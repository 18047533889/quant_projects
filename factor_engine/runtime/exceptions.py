from typing import Any


class FactorEngineError(Exception):
    pass


# ---------------------------------------------------------------------------
# R40 #96：request-scoped CancellationToken（HTTP service → FE 入口）
#
# 一个 job 从 ``_submit_job`` 构造 token 后经 ``_job_wrapper`` 放入 ContextVar，
# ``_execute_inline`` / ``_execute_config_path`` 再传给 ``FactorEngine.run()``。
# engine 关键阶段边界（admission / source scan / root dispatch）查
# ``token.is_cancelled`` —— cancel 事件一旦 set，阶段边界即停止新工作。
# deadline 用 monotonic 时间戳（wall-clock 回拨不影响超时判定）。
# ---------------------------------------------------------------------------
class CancellationToken:
    """Request-scoped cancellation + deadline authority.

    - ``cancel()`` sets an internal ``threading.Event``; ``is_cancelled`` is True
      from then on.
    - ``deadline_monotonic`` is a ``time.monotonic()`` timestamp; ``expired`` is
      True once the monotonic clock passes it.  ``None`` means no deadline.
    - ``raise_if_cancelled()`` raises :class:`Cancellation` at a stage boundary
      so the engine aborts deterministically instead of continuing to run.
    """

    __slots__ = ("_cancel_event", "deadline_monotonic")

    def __init__(self, *, deadline_monotonic: float | None = None) -> None:
        import threading

        self._cancel_event = threading.Event()
        self.deadline_monotonic = deadline_monotonic

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def expired(self) -> bool:
        if self.deadline_monotonic is None:
            return False
        import time

        return time.monotonic() >= self.deadline_monotonic

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise Cancellation("job cancellation requested")
        if self.expired:
            raise DeadlineExceeded("job deadline exceeded")


#: R40 #96: 当前请求的 request-scoped CancellationToken（worker 线程可见）。
#: ``_job_wrapper`` 在 job 执行期间把它 set 进本 ContextVar；engine / scheduler
#: 阶段边界用 :func:`get_active_cancellation_token` 查询并在 cancel/deadline 时
#: 停止新工作（不穿到 DA/backend —— 后端传播 PARTIAL）。
_cancellation_token_ctx_var: Any = None


def get_active_cancellation_token() -> "CancellationToken | None":
    """返回当前请求的 CancellationToken（无则 None）。"""
    from contextvars import ContextVar

    global _cancellation_token_ctx_var
    if _cancellation_token_ctx_var is None:
        _cancellation_token_ctx_var = ContextVar(
            "engine_cancellation_token", default=None
        )
    return _cancellation_token_ctx_var.get()


def set_active_cancellation_token(token: "CancellationToken | None") -> Any:
    """设置当前请求的 CancellationToken（返回 reset token）。"""
    from contextvars import ContextVar

    global _cancellation_token_ctx_var
    if _cancellation_token_ctx_var is None:
        _cancellation_token_ctx_var = ContextVar(
            "engine_cancellation_token", default=None
        )
    return _cancellation_token_ctx_var.set(token)


def reset_active_cancellation_token(token: Any) -> None:
    """恢复之前 ContextVar 的值（与 set 返回的 token 配对）。"""
    if _cancellation_token_ctx_var is not None:
        _cancellation_token_ctx_var.reset(token)


# ---------------------------------------------------------------------------
# R37-P0-072：统一 FailureTaxonomy —— 每类明确 retry/replan/shard/fallback/abort 语义
# ---------------------------------------------------------------------------

class SemanticError(FactorEngineError):
    """公式/算子语义错误。retry=False；replan=False。"""


class PITViolation(FactorEngineError):
    """PIT / 时间因果性违例。retry=False；replan=False；abort=True。"""


class DataQualityError(FactorEngineError):
    """数据质量失败（all-NaN / low coverage / Inf 等）。retry=False；不得 fillna 后继续。"""


class SchemaError(FactorEngineError):
    """schema 不匹配 / schema epoch 冲突。retry=False；需迁移。"""


class ParameterDomainError(FactorEngineError):
    """具体调用参数点不在 certified parameter domain（R37-P0-009 fail closed）。"""


class ResourceAdmissionError(FactorEngineError):
    """资源准入被拒（headroom 不足）。retry=True（延迟）；replan=True。"""


class ResourceUnderpredictionError(FactorEngineError):
    """资源用量超出预测（P50/P90 校准不足）。replan=True；shard=True。"""


class OOMReplanRequired(ResourceUnderpredictionError):
    """OOM 后禁止原尺寸重试 —— 必须 legal shard / 降低并发 / spill。"""


class TransientIOError(FactorEngineError):
    """瞬时 IO 错误。retry=True（有限次）。"""


class PermanentIOError(FactorEngineError):
    """永久 IO 错误。retry=False；abort=True。"""


class WriterFatalError(FactorEngineError):
    """writer fatal（disk full / permission / hang）—— 必须停止新 admission。"""


class Cancellation(FactorEngineError):
    """CancellationToken 传播的取消。retry=False；必须释放全部 lease。"""


class DeadlineExceeded(FactorEngineError):
    """绝对 deadline 超时。retry=False（或按策略）。"""


class BackendBug(FactorEngineError):
    """backend 行为与参考不一致。backend quarantine。"""


class OptimizerMismatch(FactorEngineError):
    """优化前后语义不一致。optimizer quarantine。"""


class CheckpointInvalid(FactorEngineError):
    """checkpoint fingerprint 与输入 identity 不匹配 —— 必须 full replay，不得静默继续。"""


class ProductionEventAutoPublishDisabled(FactorEngineError):
    """production 下 DataEvent 自动发布被禁用 / 缺 event_id（原子两阶段必需）。"""

