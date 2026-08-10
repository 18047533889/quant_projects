class FactorEngineError(Exception):
    pass


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

