"""q/K Backend Capability Registry.

定义哪些算子可以在 q 中原生执行（NATIVE），哪些需要委托（DELEGATE）。

Hard Gate (文档 §85):
- Q_BACKEND_ZERO_SEMANTIC_AUTHORITY: q 不得成为语义权威
- Q_BACKEND_CANONICAL_IR_ONLY: 只能是 Canonical IR 的编译目标
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class QCapabilityLevel(Enum):
    """q 算子能力级别。"""
    NATIVE = "NATIVE"  # q 原生实现
    DELEGATE = "DELEGATE"  # 需要委托到 Python/Pandas
    RESEARCH_ONLY = "RESEARCH_ONLY"  # 仅研究模式
    UNSUPPORTED = "UNSUPPORTED"  # 不支持


@dataclass(frozen=True)
class QOperatorCapability:
    """单个算子在 q 中的能力声明。"""
    op_name: str
    capability: QCapabilityLevel
    streaming_safe: bool = False
    requires_global_sort: bool = False
    requires_full_group: bool = False
    notes: str = ""


# Phase 1 高适配场景算子（文档 §24）
_PHASE1_NATIVE_OPS = {
    # Arithmetic
    "add", "subtract", "multiply", "divide", "negate", "abs",
    "power", "sqrt", "log", "exp", "log1p", "expm1",
    "sign", "floor", "ceil", "round",

    # Comparison
    "greater", "less", "greater_equal", "less_equal", "equal", "not_equal",

    # Lag/Delta
    "lag", "delta", "pct_change", "ts_returns", "ts_diff",

    # Rolling aggregations
    "ts_mean", "ts_sum", "ts_std", "ts_min", "ts_max",
    "ts_count", "ts_median", "ts_product", "ts_var",

    # Cumulative operations
    "ts_cumsum", "ts_cumprod", "ts_cummax", "ts_cummin",

    # Time series statistical moments
    "ts_skew", "ts_kurt", "ts_moment",

    # Time series position/extrema
    "ts_argmax", "ts_argmin", "ts_argmax_age", "ts_argmin_age",
    "ts_days_since_high", "ts_days_since_low",
    "ts_new_high", "ts_new_low",

    # Time series drawdown/distance
    "ts_max_drawdown", "ts_distance_to_high", "ts_distance_to_low",

    # Time series decay
    "ts_decay_linear", "ts_decay_exp", "ts_sum_decay",

    # Time series rank/zscore
    "ts_rank", "ts_zscore", "ts_demean", "ts_normalize",

    # Time series quantile
    "ts_quantile", "ts_percentile",

    # Correlation/Covariance
    "ts_corr", "ts_cov", "ts_beta",

    # Cross-sectional operations
    "rank", "cs_rank", "cs_zscore", "cs_demean", "cs_normalize",
    "cs_quantile", "cs_percentile_rank",
    "cs_winsorize", "cs_clip",
    "cs_mean", "cs_std", "cs_median", "cs_var",

    # Group operations
    "group_mean", "group_sum", "group_std", "group_median",
    "group_min", "group_max", "group_count",

    # Conditional/Fill operations
    "where", "fillna", "ffill", "bfill",
    "clip", "replace",

    # VWAP/Basic aggregation
    "vwap", "mean", "sum", "std", "min", "max", "median",
    "product", "var", "count_nonzero",
    "first", "last",

    # Time operations
    "resample", "time_bucket",

    # Simple indicators
    "true_range", "ema", "wma", "sma",
}

# 暂缓场景（文档 §24）
_PHASE1_DEFERRED_OPS = {
    # 复杂模型训练
    "garch", "har", "kalman_filter", "ar", "var",

    # 复杂 checkpoint state
    "ema_with_deadband", "recursive_filter",

    # Topology/特殊科研
    "network_centrality", "graph_distance", "topological_sort",
    "mutual_information", "transfer_entropy",
}


class QBackendCapability:
    """q/K Backend 能力查询服务。

    遵循文档 §8-9 要求：
    - 必须 backend-specific
    - 不使用 generic "sql"
    - DELEGATE 不得算 NATIVE
    """

    def __init__(self):
        self._capabilities: dict[str, QOperatorCapability] = {}
        self._initialize_phase1()

    def _initialize_phase1(self):
        """初始化 Phase 1 算子能力。"""
        # Phase 1 native ops
        for op in _PHASE1_NATIVE_OPS:
            streaming = op in {"ts_mean", "ts_sum", "lag", "delta"}
            self._capabilities[op] = QOperatorCapability(
                op_name=op,
                capability=QCapabilityLevel.NATIVE,
                streaming_safe=streaming,
                requires_global_sort=False,
                requires_full_group=op.startswith("group_"),
            )

        # Deferred ops
        for op in _PHASE1_DEFERRED_OPS:
            self._capabilities[op] = QOperatorCapability(
                op_name=op,
                capability=QCapabilityLevel.UNSUPPORTED,
                notes="Phase 1 deferred - complex state/model/topology",
            )

    def get_capability(
        self,
        op_name: str,
        *,
        mode: Literal["production", "research"] = "production",
    ) -> QCapabilityLevel:
        """查询算子能力级别。

        参数:
            op_name: 算子名称
            mode: 执行模式

        返回:
            能力级别
        """
        cap = self._capabilities.get(op_name)
        if cap is None:
            return QCapabilityLevel.UNSUPPORTED

        if cap.capability == QCapabilityLevel.RESEARCH_ONLY and mode == "production":
            return QCapabilityLevel.UNSUPPORTED

        return cap.capability

    def supports_native(self, op_name: str) -> bool:
        """算子是否原生支持（不含 DELEGATE）。"""
        return self.get_capability(op_name) == QCapabilityLevel.NATIVE

    def is_streaming_safe(self, op_name: str) -> bool:
        """算子是否可流式执行。"""
        cap = self._capabilities.get(op_name)
        return cap.streaming_safe if cap else False

    def requires_global_sort(self, op_name: str) -> bool:
        """算子是否需要全局排序。"""
        cap = self._capabilities.get(op_name)
        return cap.requires_global_sort if cap else False

    def requires_full_group(self, op_name: str) -> bool:
        """算子是否需要完整分组数据。"""
        cap = self._capabilities.get(op_name)
        return cap.requires_full_group if cap else False

    def compute_native_fraction(
        self,
        op_names: list[str],
        *,
        mode: Literal["production", "research"] = "production",
    ) -> tuple[float, list[str], list[str]]:
        """计算算子列表的原生支持比例。

        遵循文档 §9: DELEGATE 不得算 NATIVE。

        参数:
            op_names: 算子名称列表
            mode: 执行模式

        返回:
            (native_fraction, native_ops, unsupported_ops)
        """
        if not op_names:
            return 0.0, [], []

        native_ops = []
        unsupported_ops = []

        for op in op_names:
            cap = self.get_capability(op, mode=mode)
            if cap == QCapabilityLevel.NATIVE:
                native_ops.append(op)
            elif cap in (QCapabilityLevel.UNSUPPORTED, QCapabilityLevel.DELEGATE):
                unsupported_ops.append(op)

        native_fraction = len(native_ops) / len(op_names)
        return native_fraction, native_ops, unsupported_ops


# Global singleton
_CAPABILITY_REGISTRY: QBackendCapability | None = None


def get_q_capability() -> QBackendCapability:
    """获取全局 q capability registry。"""
    global _CAPABILITY_REGISTRY
    if _CAPABILITY_REGISTRY is None:
        _CAPABILITY_REGISTRY = QBackendCapability()
    return _CAPABILITY_REGISTRY
