# -*- coding: utf-8 -*-
"""MB-P1-011: Polars Lazy DAG fusion optimization.

Polars lazy API 构建 LogicalPlan DAG，在 `.collect()` 时经 optimizer 自动优化：
    - Predicate pushdown（过滤下推到 scan）
    - Projection pushdown（只读需要的列）
    - Common subexpression elimination
    - Filter/predicate coalescence
    - Type coercion optimization

本模块为 FactorEngine 的 Polars backend region 提供：
    1. Lazy DAG 构建器（延迟物化，累积多个算子后统一 collect）
    2. Fusion 边界识别（何时必须 collect：sort/join/group 需要 eager frame）
    3. Streaming collect（out-of-core 大数据集，`collect(streaming=True)`）
    4. 成本评估（lazy plan → optimized plan 预期节省）
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class LazyFusionMetrics:
    """Polars lazy fusion 统计（优化效果）。"""

    plans_built: int = 0
    plans_collected: int = 0
    forced_collect_count: int = 0
    streaming_collect_count: int = 0
    total_operators_fused: int = 0
    estimated_saved_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plans_built": self.plans_built,
            "plans_collected": self.plans_collected,
            "forced_collect_count": self.forced_collect_count,
            "streaming_collect_count": self.streaming_collect_count,
            "total_operators_fused": self.total_operators_fused,
            "estimated_saved_bytes": self.estimated_saved_bytes,
        }


class PolarsLazyFusionOptimizer:
    """Polars LazyFrame DAG 构建与 fusion 控制器。

    用法：
        >>> optimizer = PolarsLazyFusionOptimizer()
        >>> lf = optimizer.begin_lazy(df)  # DataFrame → LazyFrame
        >>> lf = lf.filter(pl.col("x") > 0).select(["a", "b"])
        >>> result = optimizer.collect(lf, streaming=True)  # 统一优化
    """

    def __init__(self, *, streaming_threshold_bytes: int = 2 * 1024**3) -> None:
        """
        Args:
            streaming_threshold_bytes: 超过此阈值自动 streaming collect（out-of-core）
        """
        self.streaming_threshold_bytes = streaming_threshold_bytes
        self._metrics = LazyFusionMetrics()
        self._lock = threading.RLock()

    def begin_lazy(self, df: Any) -> Any:
        """DataFrame → LazyFrame（开始累积 lazy operations）。

        Args:
            df: Polars DataFrame 或已经是 LazyFrame（幂等）

        Returns:
            Polars LazyFrame
        """
        try:
            import polars as pl
        except ImportError:
            _logger.warning("polars not installed; lazy fusion disabled")
            return df

        with self._lock:
            self._metrics.plans_built += 1

        if isinstance(df, pl.LazyFrame):
            return df
        if isinstance(df, pl.DataFrame):
            return df.lazy()
        # 其它类型（pandas / arrow）先转 Polars DataFrame 再 lazy
        try:
            return pl.DataFrame(df).lazy()
        except Exception as exc:
            _logger.debug("begin_lazy failed: %s; returning original", exc)
            return df

    def should_force_collect(self, operation: str) -> bool:
        """判断算子是否需要强制 eager collect（打断 lazy chain）。

        某些算子必须在真实 DataFrame 上执行：
            - sort（Polars lazy sort 需要全局数据）
            - join（多 LazyFrame join 需要至少一个 eager）
            - pivot / melt（全局 reshape）
            - 自定义 UDF apply（lazy 无法 push 进计划）

        Args:
            operation: 算子名称（如 "sort", "join", "pivot"）

        Returns:
            True 表示必须先 collect，False 表示可继续 lazy
        """
        force_eager = {
            "sort",
            "join",
            "pivot",
            "melt",
            "apply",
            "map",
            "explode",  # explode 后续算子难以下推
        }
        return operation.lower() in force_eager

    def collect(
        self,
        lf: Any,
        *,
        streaming: bool | None = None,
        force: bool = False,
    ) -> Any:
        """统一 collect：触发 Polars optimizer，返回 eager DataFrame。

        Args:
            lf: Polars LazyFrame
            streaming: True 强制 streaming；None 自动判断；False 禁用
            force: True 表示强制 collect（边界算子），记入 forced_collect_count

        Returns:
            Polars DataFrame
        """
        try:
            import polars as pl
        except ImportError:
            return lf

        if not isinstance(lf, pl.LazyFrame):
            return lf

        with self._lock:
            self._metrics.plans_collected += 1
            if force:
                self._metrics.forced_collect_count += 1

        # 自动判断是否 streaming：估算数据量 > threshold
        use_streaming = streaming
        if streaming is None:
            try:
                estimated_bytes = self._estimate_plan_bytes(lf)
                use_streaming = estimated_bytes >= self.streaming_threshold_bytes
            except Exception:
                use_streaming = False

        try:
            if use_streaming:
                with self._lock:
                    self._metrics.streaming_collect_count += 1
                return lf.collect(streaming=True)
            return lf.collect()
        except Exception as exc:
            _logger.warning(
                "lazy collect failed (%s); fallback to non-streaming", exc
            )
            try:
                return lf.collect(streaming=False)
            except Exception:
                # 最后手段：如果已是 DataFrame，直接返回
                return lf

    def _estimate_plan_bytes(self, lf: Any) -> int:
        """粗略估算 LazyFrame collect 后的内存占用（行数 × 列数 × 8 bytes）。

        实际 Polars 内部有更精确的 cost model；这里只作 streaming 触发判断。
        """
        try:
            # Polars LazyFrame.schema 返回 {col: dtype}，不触发 collect
            schema = lf.schema
            n_cols = len(schema)
            # 行数无法精确获取（需要 collect），保守估算：假设 scan 返回 1M 行
            # 实际行数可从 DataFrameContext.rows 传递（调用方提供）
            estimated_rows = 1_000_000
            return estimated_rows * n_cols * 8
        except Exception:
            return 0

    def explain_plan(self, lf: Any) -> str:
        """返回 Polars optimized logical plan（诊断用，不触发 collect）。

        Returns:
            优化后的 LogicalPlan 文本表示
        """
        try:
            import polars as pl
        except ImportError:
            return "polars not installed"

        if not isinstance(lf, pl.LazyFrame):
            return "not a LazyFrame"

        try:
            # Polars >= 0.18.0 用 explain(optimized=True)
            return lf.explain(optimized=True)
        except Exception:
            try:
                # 旧版本 fallback
                return str(lf.describe_optimized_plan())
            except Exception:
                return "explain not available"

    def record_fusion(self, operator_count: int, saved_bytes: int = 0) -> None:
        """记录一次 fusion 优化（供调用方显式标记）。

        Args:
            operator_count: 合并的算子数
            saved_bytes: 估算节省的中间结果字节数
        """
        with self._lock:
            self._metrics.total_operators_fused += operator_count
            self._metrics.estimated_saved_bytes += saved_bytes

    def metrics(self) -> LazyFusionMetrics:
        """返回当前累计统计。"""
        with self._lock:
            return LazyFusionMetrics(
                plans_built=self._metrics.plans_built,
                plans_collected=self._metrics.plans_collected,
                forced_collect_count=self._metrics.forced_collect_count,
                streaming_collect_count=self._metrics.streaming_collect_count,
                total_operators_fused=self._metrics.total_operators_fused,
                estimated_saved_bytes=self._metrics.estimated_saved_bytes,
            )

    def reset_metrics(self) -> None:
        """重置统计（测试或分段统计用）。"""
        with self._lock:
            self._metrics = LazyFusionMetrics()


def build_polars_lazy_chain(
    operations: list[dict[str, Any]],
    source_df: Any,
    optimizer: PolarsLazyFusionOptimizer | None = None,
) -> Any:
    """批量构建 Polars lazy chain（多个算子累积后统一 collect）。

    Args:
        operations: 算子列表，每项 {"op": "filter|select|...", "args": {...}}
        source_df: 起始 DataFrame
        optimizer: PolarsLazyFusionOptimizer 实例（None 则新建）

    Returns:
        优化后的 Polars DataFrame

    Example:
        >>> ops = [
        ...     {"op": "filter", "args": {"predicate": pl.col("x") > 0}},
        ...     {"op": "select", "args": {"exprs": ["a", "b"]}},
        ...     {"op": "with_columns", "args": {"exprs": [pl.col("a") * 2]}},
        ... ]
        >>> result = build_polars_lazy_chain(ops, df)
    """
    if optimizer is None:
        optimizer = PolarsLazyFusionOptimizer()

    lf = optimizer.begin_lazy(source_df)
    force_collect_ops: set[str] = set()

    for i, step in enumerate(operations):
        op_name = step.get("op", "")
        args = step.get("args", {})

        # 边界算子必须先 collect
        if optimizer.should_force_collect(op_name):
            lf = optimizer.collect(lf, force=True)
            force_collect_ops.add(op_name)

        # 应用算子（lazy API）
        try:
            if op_name == "filter":
                lf = lf.filter(args.get("predicate"))
            elif op_name == "select":
                lf = lf.select(args.get("exprs", []))
            elif op_name == "with_columns":
                lf = lf.with_columns(args.get("exprs", []))
            elif op_name == "group_by":
                lf = lf.group_by(args.get("by", [])).agg(args.get("agg", []))
            elif op_name == "sort":
                # sort 需要 eager frame
                df = optimizer.collect(lf, force=True)
                df = df.sort(args.get("by", []))
                lf = optimizer.begin_lazy(df)
            elif op_name == "join":
                # join 右表必须 eager
                other = args.get("other")
                if other is not None:
                    df = optimizer.collect(lf, force=True)
                    df = df.join(other, **args.get("join_kwargs", {}))
                    lf = optimizer.begin_lazy(df)
            else:
                _logger.debug("unsupported lazy op: %s; skipping", op_name)
        except Exception as exc:
            _logger.warning("lazy op %s failed: %s; stopping chain", op_name, exc)
            break

    # 最终 collect（自动判断 streaming）
    result = optimizer.collect(lf)
    optimizer.record_fusion(len(operations), 0)
    return result


# 便捷函数：全局单例 optimizer
_global_optimizer: PolarsLazyFusionOptimizer | None = None
_global_lock = threading.Lock()


def get_global_polars_optimizer() -> PolarsLazyFusionOptimizer:
    """返回全局 Polars lazy fusion optimizer（进程级单例）。"""
    global _global_optimizer
    if _global_optimizer is None:
        with _global_lock:
            if _global_optimizer is None:
                _global_optimizer = PolarsLazyFusionOptimizer()
    return _global_optimizer
