# -*- coding: utf-8 -*-
"""MB-P1-020: Batch Polars expression compilation.

Polars expression 批量编译（避免逐表达式解析）：
    - Polars lazy API 构建 LogicalPlan 时需要解析每个 expression
    - 批量编译：一次性解析多个 expression（共享 schema context）
    - Expression cache：相同 expression 复用已编译结果
    - Optimized expression tree：预优化（constant folding / 类型推断）

集成点：
    - PolarsLazyFusionOptimizer 构建 lazy chain 时批量编译 expressions
    - Backend executor 执行 Polars region 前预编译所有 expressions
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class CompiledExpression:
    """已编译的 Polars expression（可复用）。"""

    expression_str: str
    expression_hash: str
    compiled_expr: Any  # Polars Expr object
    schema_context: dict[str, str]  # {col: dtype}
    compilation_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression_str": self.expression_str,
            "expression_hash": self.expression_hash,
            "schema_context": self.schema_context,
            "compilation_time_ms": round(self.compilation_time_ms, 3),
        }


@dataclass
class ExpressionCompilerMetrics:
    """表达式编译统计。"""

    total_compilations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    batch_compilations: int = 0
    total_compilation_time_ms: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_compilations": self.total_compilations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "batch_compilations": self.batch_compilations,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "total_compilation_time_ms": round(self.total_compilation_time_ms, 2),
        }


class PolarsExpressionCompiler:
    """Polars expression 批量编译器。

    用法：
        >>> compiler = PolarsExpressionCompiler()
        >>> exprs = [
        ...     "pl.col('x') > 0",
        ...     "pl.col('y').mean()",
        ...     "(pl.col('a') + pl.col('b')) / 2",
        ... ]
        >>> compiled = compiler.batch_compile(exprs, schema)
    """

    def __init__(self, *, cache_size: int = 1000) -> None:
        """
        Args:
            cache_size: Expression cache 最大容量（LRU eviction）
        """
        self.cache_size = cache_size
        self._cache: dict[str, CompiledExpression] = {}
        self._cache_order: list[str] = []  # LRU tracking
        self._metrics = ExpressionCompilerMetrics()
        self._lock = threading.RLock()

    def batch_compile(
        self,
        expressions: list[str],
        schema: dict[str, str] | None = None,
    ) -> list[Any]:
        """批量编译 Polars expressions（共享 schema context）。

        Args:
            expressions: 表达式字符串列表（如 "pl.col('x') > 0"）
            schema: Schema context（{col: dtype}）

        Returns:
            已编译的 Polars Expr 对象列表
        """
        try:
            import polars as pl
        except ImportError:
            _logger.warning("polars not installed; expression compilation disabled")
            return []

        import time

        start_ms = time.monotonic() * 1000.0
        schema = schema or {}
        compiled: list[Any] = []

        with self._lock:
            self._metrics.batch_compilations += 1

            for expr_str in expressions:
                # 计算 expression hash（包含 schema context）
                expr_hash = self._compute_hash(expr_str, schema)

                # 缓存命中
                if expr_hash in self._cache:
                    cached = self._cache[expr_hash]
                    compiled.append(cached.compiled_expr)
                    self._metrics.cache_hits += 1
                    # LRU: 移到队列末尾
                    self._cache_order.remove(expr_hash)
                    self._cache_order.append(expr_hash)
                    continue

                # 缓存未命中：编译 expression
                self._metrics.cache_misses += 1
                try:
                    # eval() expression string → Polars Expr
                    # 安全性：只允许 pl.* 表达式（生产环境应使用 AST parser）
                    if not expr_str.strip().startswith("pl."):
                        _logger.warning("unsafe expression: %s; skipping", expr_str)
                        continue

                    compiled_expr = eval(expr_str, {"pl": pl})
                    compiled.append(compiled_expr)

                    # 缓存已编译 expression
                    compilation_time = (time.monotonic() * 1000.0) - start_ms
                    self._cache[expr_hash] = CompiledExpression(
                        expression_str=expr_str,
                        expression_hash=expr_hash,
                        compiled_expr=compiled_expr,
                        schema_context=schema.copy(),
                        compilation_time_ms=compilation_time,
                    )
                    self._cache_order.append(expr_hash)

                    # LRU eviction
                    if len(self._cache) > self.cache_size:
                        oldest = self._cache_order.pop(0)
                        self._cache.pop(oldest, None)

                except Exception as exc:
                    _logger.error("expression compilation failed: %s (%s)", expr_str, exc)
                    continue

            elapsed_ms = (time.monotonic() * 1000.0) - start_ms
            self._metrics.total_compilations += len(expressions)
            self._metrics.total_compilation_time_ms += elapsed_ms

        _logger.debug(
            "batch compiled %d expressions in %.2f ms (cache hit rate: %.1f%%)",
            len(expressions), elapsed_ms, self._metrics.cache_hit_rate * 100,
        )

        return compiled

    def compile_single(
        self,
        expression: str,
        schema: dict[str, str] | None = None,
    ) -> Any | None:
        """编译单个 expression（优先从 cache 读取）。

        Args:
            expression: 表达式字符串
            schema: Schema context

        Returns:
            已编译的 Polars Expr 或 None（失败）
        """
        compiled = self.batch_compile([expression], schema)
        return compiled[0] if compiled else None

    def _compute_hash(self, expression: str, schema: dict[str, str]) -> str:
        """计算 expression 的唯一 hash（包含 schema context）。

        相同表达式在不同 schema 下可能有不同语义（类型推断），因此 hash 需包含 schema。
        """
        # schema 排序后序列化（保证确定性）
        schema_str = ",".join(f"{k}:{v}" for k, v in sorted(schema.items()))
        combined = f"{expression}|{schema_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def precompile_common_patterns(self) -> None:
        """预编译常见 expression 模式（warmup cache）。

        常见模式：
            - pl.col('x') > 0
            - pl.col('x').mean()
            - pl.col('x').shift(1)
            - pl.col('x').rolling_mean(window_size=20)
        """
        common_patterns = [
            "pl.col('x') > 0",
            "pl.col('x') < 0",
            "pl.col('x').is_null()",
            "pl.col('x').is_not_null()",
            "pl.col('x').mean()",
            "pl.col('x').sum()",
            "pl.col('x').std()",
            "pl.col('x').min()",
            "pl.col('x').max()",
            "pl.col('x').shift(1)",
            "pl.col('x').diff()",
            "pl.col('x').pct_change()",
        ]

        # 泛型 schema（支持任意列）
        generic_schema = {"x": "float64"}
        self.batch_compile(common_patterns, generic_schema)
        _logger.info("precompiled %d common expression patterns", len(common_patterns))

    def optimize_expression(self, expression: str) -> str:
        """优化 expression（常量折叠 / 类型推断）。

        示例：
            "(pl.col('x') + 0) * 1" → "pl.col('x')"
            "pl.col('x').cast(pl.Float64).cast(pl.Float64)" → "pl.col('x').cast(pl.Float64)"
        """
        # 简化实现：实际需要完整 expression rewriter
        optimized = expression

        # 常量折叠
        optimized = optimized.replace(" + 0", "").replace(" - 0", "")
        optimized = optimized.replace(" * 1", "").replace(" / 1", "")

        # 冗余 cast 消除
        # （实际需要 AST parser）

        return optimized

    def clear_cache(self) -> None:
        """清空 expression cache（测试或内存压力时使用）。"""
        with self._lock:
            self._cache.clear()
            self._cache_order.clear()
            _logger.debug("expression cache cleared")

    def metrics(self) -> ExpressionCompilerMetrics:
        """返回累计统计。"""
        with self._lock:
            return ExpressionCompilerMetrics(
                total_compilations=self._metrics.total_compilations,
                cache_hits=self._metrics.cache_hits,
                cache_misses=self._metrics.cache_misses,
                batch_compilations=self._metrics.batch_compilations,
                total_compilation_time_ms=self._metrics.total_compilation_time_ms,
            )

    def summary(self) -> dict[str, Any]:
        """返回编译器状态摘要。"""
        with self._lock:
            metrics = self.metrics().to_dict()
            return {
                "cache_size": len(self._cache),
                "cache_capacity": self.cache_size,
                "metrics": metrics,
            }


# 全局单例
_global_compiler: PolarsExpressionCompiler | None = None
_global_lock = threading.Lock()


def get_global_polars_compiler() -> PolarsExpressionCompiler:
    """返回全局 PolarsExpressionCompiler（进程级单例）。"""
    global _global_compiler
    if _global_compiler is None:
        with _global_lock:
            if _global_compiler is None:
                _global_compiler = PolarsExpressionCompiler()
                # Warmup: 预编译常见模式
                try:
                    _global_compiler.precompile_common_patterns()
                except Exception as exc:
                    _logger.debug("precompile warmup failed: %s", exc)
    return _global_compiler
