# -*- coding: utf-8 -*-
"""智能后端选择器：数据规模 × 算子类型 × 内存约束 → 最优 Backend。

实现智能路由策略：
    1. 数据规模评估（< 10k → Pandas, 10k-1M → Polars, > 1M → DuckDB）
    2. 算子类型匹配（窗口密集 → DuckDB, 截面密集 → Polars, 复杂逻辑 → Pandas）
    3. 内存约束（充足 → Polars, 受限 → DuckDB Streaming, 极限 → Pandas 分块）
    4. 成本模型动态采样（实际执行时间 → 自适应调整）
    5. 混合执行（不同 factor 不同 backend, Transfer 成本最小化）

Section references: §43-§45 (multi-backend optimization)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from factor_engine.planning.backend_region import ExecutionAxis, PhysicalBackend


class DataScale(str, Enum):
    """数据规模分类（决定启动成本 vs 批量吞吐权衡）。"""

    TINY = "tiny"  # < 1k rows: Pandas 启动最快
    SMALL = "small"  # 1k-10k rows: Pandas 仍有优势
    MEDIUM = "medium"  # 10k-100k rows: Polars 平衡点
    LARGE = "large"  # 100k-1M rows: Polars/DuckDB 均可
    HUGE = "huge"  # > 1M rows: DuckDB 批量优势
    MASSIVE = "massive"  # > 10M rows: DuckDB Streaming 必选


class OperatorProfile(str, Enum):
    """算子密集度分类（决定 backend 专长匹配）。"""

    WINDOW_HEAVY = "window_heavy"  # 窗口函数密集 → DuckDB SQL 优化
    CROSS_SECTION_HEAVY = "cross_section_heavy"  # 截面操作密集 → Polars 并行
    ELEMENTWISE_HEAVY = "elementwise_heavy"  # 逐元素密集 → Pandas/NumPy
    REGRESSION_HEAVY = "regression_heavy"  # 回归/中和密集 → Pandas statsmodels
    MIXED = "mixed"  # 混合负载 → 需成本模型决策


class MemoryPressure(str, Enum):
    """内存压力分类（决定 streaming vs in-memory）。"""

    LOW = "low"  # < 30% 内存占用：性能优先
    MODERATE = "moderate"  # 30%-60%：平衡
    HIGH = "high"  # 60%-80%：内存优先
    CRITICAL = "critical"  # > 80%：强制 streaming/分块


@dataclass(frozen=True)
class BackendCharacteristics:
    """单个 Backend 的性能特征画像（用于决策匹配）。"""

    backend: PhysicalBackend
    startup_cost_ms: float  # 启动成本（import, 连接池初始化等）
    per_row_throughput_mrows_per_sec: float  # 吞吐率（百万行/秒）
    memory_overhead_factor: float  # 内存开销系数（相对原始数据）
    supports_streaming: bool  # 是否支持 streaming
    preferred_scale: tuple[DataScale, ...]  # 擅长的数据规模
    preferred_profiles: tuple[OperatorProfile, ...]  # 擅长的算子类型
    parallel_efficiency: float  # 并行效率（0.0-1.0）
    conversion_penalty_ms: float  # 转换惩罚（从其他 backend 转入）
    # R21-NUMBA-COST-MODEL: Numba JIT specific fields for TTDC calculation
    cold_jit_ms: float = 0.0  # Cold JIT compilation cost (one-time)
    warm_ms: float = 0.0  # Warm execution cost after JIT cache hit
    cache_hit_probability: float = 0.0  # Probability of JIT cache hit (0.0-1.0)


# Backend 性能特征库（基于实测 + 架构特性）
_BACKEND_CHARACTERISTICS: dict[PhysicalBackend, BackendCharacteristics] = {
    PhysicalBackend.PANDAS_NUMPY: BackendCharacteristics(
        backend=PhysicalBackend.PANDAS_NUMPY,
        startup_cost_ms=2.0,  # 极快启动
        per_row_throughput_mrows_per_sec=0.5,  # 中等吞吐
        memory_overhead_factor=1.2,  # 较低内存开销
        supports_streaming=False,
        preferred_scale=(DataScale.TINY, DataScale.SMALL, DataScale.MEDIUM),
        preferred_profiles=(
            OperatorProfile.ELEMENTWISE_HEAVY,
            OperatorProfile.REGRESSION_HEAVY,
            OperatorProfile.MIXED,
        ),
        parallel_efficiency=0.3,  # GIL 限制
        conversion_penalty_ms=5.0,
    ),
    PhysicalBackend.POLARS_PANEL: BackendCharacteristics(
        backend=PhysicalBackend.POLARS_PANEL,
        startup_cost_ms=8.0,  # 中等启动
        per_row_throughput_mrows_per_sec=2.5,  # 高吞吐
        memory_overhead_factor=1.5,  # 中等内存开销
        supports_streaming=False,
        preferred_scale=(DataScale.MEDIUM, DataScale.LARGE),
        preferred_profiles=(
            OperatorProfile.CROSS_SECTION_HEAVY,
            OperatorProfile.WINDOW_HEAVY,
            OperatorProfile.MIXED,
        ),
        parallel_efficiency=0.85,  # 高并行效率
        conversion_penalty_ms=8.0,
    ),
    PhysicalBackend.POLARS_LONG: BackendCharacteristics(
        backend=PhysicalBackend.POLARS_LONG,
        startup_cost_ms=10.0,  # 稍慢启动（优化器开销）
        per_row_throughput_mrows_per_sec=3.0,  # 最高吞吐（优化后）
        memory_overhead_factor=1.3,  # 低内存开销（lazy）
        supports_streaming=True,
        preferred_scale=(DataScale.LARGE, DataScale.HUGE, DataScale.MASSIVE),
        preferred_profiles=(
            OperatorProfile.CROSS_SECTION_HEAVY,
            OperatorProfile.WINDOW_HEAVY,
            OperatorProfile.MIXED,
        ),
        parallel_efficiency=0.9,  # 最高并行效率
        conversion_penalty_ms=10.0,
    ),
    PhysicalBackend.DUCKDB_SQL: BackendCharacteristics(
        backend=PhysicalBackend.DUCKDB_SQL,
        startup_cost_ms=15.0,  # 最慢启动（SQL 解析 + 优化）
        per_row_throughput_mrows_per_sec=4.0,  # 最高批量吞吐
        memory_overhead_factor=1.1,  # 最低内存开销（列存 + 压缩）
        supports_streaming=True,
        preferred_scale=(DataScale.LARGE, DataScale.HUGE, DataScale.MASSIVE),
        preferred_profiles=(
            OperatorProfile.WINDOW_HEAVY,
            OperatorProfile.CROSS_SECTION_HEAVY,
        ),
        parallel_efficiency=0.95,  # 最高并行效率（vectorized）
        conversion_penalty_ms=20.0,  # 最高转换成本（SQL 边界）
    ),
}


@dataclass(frozen=True)
class RoutingContext:
    """路由决策上下文（输入）。"""

    estimated_rows: int
    estimated_columns: int
    estimated_bytes: int
    available_memory_bytes: int
    operator_profile: OperatorProfile
    execution_axis: ExecutionAxis
    window_params: list[int]  # 窗口参数列表（用于判断 window_heavy）
    has_regression: bool  # 是否包含回归/中和算子
    parent_backend: PhysicalBackend | None  # 父节点 backend（用于 transfer affinity）
    allows_streaming: bool  # 是否允许 streaming
    performance_priority: str  # "speed" | "memory" | "balanced"


@dataclass(frozen=True)
class RoutingDecision:
    """路由决策输出（带详细理由）。"""

    chosen_backend: PhysicalBackend
    estimated_cost_ms: float
    memory_footprint_bytes: int
    confidence_score: float  # 0.0-1.0（决策置信度）
    reasoning: list[str]  # 决策理由（可解释性）
    alternatives: list[tuple[PhysicalBackend, float, str]]  # 备选方案（backend, cost, reason）


class IntelligentBackendSelector:
    """智能后端选择器：多维度决策树 + 成本模型。

    决策流程：
        1. 数据规模分类 → 候选 backend 集合
        2. 算子类型匹配 → 候选集过滤/权重调整
        3. 内存约束检查 → 强制 streaming/分块
        4. 成本模型评估 → 选择最优（compute + transfer + memory_risk）
        5. Transfer affinity → 父子节点 backend 亲和性调整

    R21-ROUTING-AUTHORITY: this selector is a capability/cost CANDIDATE
    PROVIDER only.  It may propose a backend but never finalizes the production
    route.  The Global Physical Planner
    (``runtime.multibackend.batch_global_optimizer.PhysicalBatchGlobalOptimizer``)
    is the sole production routing authority.
    """

    def __init__(
        self,
        *,
        enable_adaptive_learning: bool = True,
        cost_history_window: int = 100,
    ) -> None:
        """初始化选择器。

        Args:
            enable_adaptive_learning: 是否启用自适应学习（实测成本反馈）
            cost_history_window: 成本历史窗口大小（用于动态调整）
        """
        self.enable_adaptive_learning = enable_adaptive_learning
        self.cost_history_window = cost_history_window
        # 成本历史: {(backend, scale, profile): [actual_cost_ms, ...]}
        self._cost_history: dict[tuple[str, str, str], list[float]] = {}
        # 动态调整系数: {backend: adjustment_factor}
        self._adjustment_factors: dict[PhysicalBackend, float] = {
            b: 1.0 for b in PhysicalBackend
        }

    def select_backend(self, ctx: RoutingContext) -> RoutingDecision:
        """智能选择最优 backend（主入口）。

        Args:
            ctx: 路由决策上下文

        Returns:
            RoutingDecision 包含选择的 backend 及详细理由
        """
        reasoning: list[str] = []

        # Phase 1: 数据规模分类
        scale = self._classify_data_scale(ctx.estimated_rows)
        reasoning.append(f"Data scale: {scale.value} ({ctx.estimated_rows:,} rows)")

        # Phase 2: 内存压力评估
        memory_pressure = self._assess_memory_pressure(
            ctx.estimated_bytes, ctx.available_memory_bytes
        )
        reasoning.append(
            f"Memory pressure: {memory_pressure.value} "
            f"({ctx.estimated_bytes / 1024**2:.1f}MB / "
            f"{ctx.available_memory_bytes / 1024**2:.1f}MB)"
        )

        # Phase 3: 算子 profile 确认
        reasoning.append(f"Operator profile: {ctx.operator_profile.value}")
        if ctx.has_regression:
            reasoning.append("Contains regression operators")
        if ctx.window_params:
            avg_window = sum(ctx.window_params) / len(ctx.window_params)
            reasoning.append(f"Avg window size: {avg_window:.0f}")

        # Phase 4: 候选 backend 集合（基于规模 + profile）
        candidates = self._filter_candidates(scale, ctx.operator_profile, memory_pressure)
        reasoning.append(f"Candidates: {[c.value for c in candidates]}")

        # Phase 5: 成本模型评估（含 transfer affinity）
        costs = self._evaluate_costs(ctx, candidates, reasoning)

        # Phase 6: 选择最优
        best_backend, best_cost, best_memory = min(
            costs, key=lambda x: x[1]
        )

        # Phase 7: 置信度评分
        confidence = self._compute_confidence(costs, scale, memory_pressure)

        # Phase 8: 备选方案
        alternatives = [
            (backend, cost, f"cost={cost:.1f}ms, memory={mem / 1024**2:.1f}MB")
            for backend, cost, mem in costs
            if backend != best_backend
        ][:3]

        return RoutingDecision(
            chosen_backend=best_backend,
            estimated_cost_ms=best_cost,
            memory_footprint_bytes=best_memory,
            confidence_score=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
        )

    def _classify_data_scale(self, rows: int) -> DataScale:
        """数据规模分类。"""
        if rows < 1000:
            return DataScale.TINY
        elif rows < 10_000:
            return DataScale.SMALL
        elif rows < 100_000:
            return DataScale.MEDIUM
        elif rows < 1_000_000:
            return DataScale.LARGE
        elif rows < 10_000_000:
            return DataScale.HUGE
        else:
            return DataScale.MASSIVE

    def _assess_memory_pressure(self, required: int, available: int) -> MemoryPressure:
        """内存压力评估。"""
        ratio = required / max(available, 1)
        if ratio < 0.3:
            return MemoryPressure.LOW
        elif ratio < 0.6:
            return MemoryPressure.MODERATE
        elif ratio < 0.8:
            return MemoryPressure.HIGH
        else:
            return MemoryPressure.CRITICAL

    def _filter_candidates(
        self,
        scale: DataScale,
        profile: OperatorProfile,
        memory: MemoryPressure,
    ) -> list[PhysicalBackend]:
        """候选 backend 过滤（基于规模 + profile + 内存）。"""
        candidates: list[PhysicalBackend] = []

        for backend, chars in _BACKEND_CHARACTERISTICS.items():
            # 规模匹配
            if scale not in chars.preferred_scale:
                # 允许跨规模使用，但有惩罚
                if scale in (DataScale.TINY, DataScale.SMALL) and backend in (
                    PhysicalBackend.DUCKDB_SQL,
                    PhysicalBackend.POLARS_LONG,
                    PhysicalBackend.POLARS_PANEL,
                ):
                    # 小数据不推荐重型 backend
                    continue

            # 内存约束
            if memory == MemoryPressure.CRITICAL:
                if not chars.supports_streaming:
                    continue  # Critical 内存压力必须支持 streaming

            # Profile 匹配（宽松）
            if profile != OperatorProfile.MIXED:
                if profile not in chars.preferred_profiles:
                    # 允许非最优匹配，但会在成本模型中惩罚
                    pass

            candidates.append(backend)

        # 至少返回一个候选（fallback）
        if not candidates:
            candidates.append(PhysicalBackend.PANDAS_NUMPY)

        return candidates

    def _evaluate_costs(
        self,
        ctx: RoutingContext,
        candidates: list[PhysicalBackend],
        reasoning: list[str],
    ) -> list[tuple[PhysicalBackend, float, int]]:
        """成本模型评估（返回 [(backend, cost_ms, memory_bytes), ...]）。

        OPTIMIZED: Added data skew awareness, non-linear scaling, query complexity scoring.
        """
        results: list[tuple[PhysicalBackend, float, int]] = []

        for backend in candidates:
            chars = _BACKEND_CHARACTERISTICS[backend]

            # 1. Startup cost (scale down for small data)
            startup = chars.startup_cost_ms
            if ctx.estimated_rows < 10_000:
                startup *= 0.5  # Reduced startup overhead for small data

            # 2. Compute cost (throughput-based with non-linear scaling)
            rows_millions = ctx.estimated_rows / 1_000_000.0
            # Non-linear scaling: larger data gets better amortization
            scaling_factor = 1.0 + (rows_millions ** 0.15) * 0.1 if rows_millions > 1.0 else 1.0
            compute_ms = (rows_millions / chars.per_row_throughput_mrows_per_sec * 1000.0) * scaling_factor

            # 3. Memory footprint
            memory_bytes = int(ctx.estimated_bytes * chars.memory_overhead_factor)

            # 4. Transfer cost with conversion cost matrix (if parent backend differs)
            transfer_ms = 0.0
            if ctx.parent_backend and ctx.parent_backend != backend:
                transfer_ms = self._get_conversion_cost(ctx.parent_backend, backend, ctx.estimated_bytes)

            # 5. Profile mismatch penalty (refined)
            profile_penalty = 0.0
            if ctx.operator_profile not in chars.preferred_profiles:
                # Stronger penalty for regression on non-Pandas
                if ctx.has_regression and backend != PhysicalBackend.PANDAS_NUMPY:
                    profile_penalty = compute_ms * 0.5
                else:
                    profile_penalty = compute_ms * 0.3

            # 6. Window size penalty (refined for large windows)
            window_penalty = 0.0
            if ctx.window_params and ctx.operator_profile == OperatorProfile.WINDOW_HEAVY:
                avg_window = sum(ctx.window_params) / len(ctx.window_params)
                max_window = max(ctx.window_params) if ctx.window_params else 0
                # Non-linear penalty for large windows
                if max_window > 500:
                    if backend == PhysicalBackend.PANDAS_NUMPY:
                        window_penalty = compute_ms * 0.8  # Heavy penalty
                    elif backend == PhysicalBackend.DUCKDB_SQL:
                        window_penalty = 0.0  # DuckDB optimized for large windows
                elif avg_window > 100 and backend == PhysicalBackend.PANDAS_NUMPY:
                    window_penalty = compute_ms * 0.5

            # 7. Memory pressure penalty (refined)
            memory_penalty = 0.0
            memory_ratio = memory_bytes / max(ctx.available_memory_bytes, 1)
            if memory_ratio > 0.9:
                if not chars.supports_streaming:
                    memory_penalty = 5000.0  # Extreme penalty for OOM risk
                else:
                    memory_penalty = compute_ms * 0.4  # Significant streaming overhead
            elif memory_ratio > 0.8:
                if not chars.supports_streaming:
                    memory_penalty = 1000.0
                else:
                    memory_penalty = compute_ms * 0.2

            # 8. Query complexity scoring (NEW)
            complexity_penalty = self._compute_complexity_penalty(ctx, backend, compute_ms)

            # 9. Data distribution awareness (NEW): sparsity and skew
            distribution_penalty = self._compute_distribution_penalty(ctx, backend, compute_ms)

            # 10. Adaptive adjustment (from historical measurements)
            adjustment = self._adjustment_factors.get(backend, 1.0)

            # Total cost
            total_cost = (
                startup
                + compute_ms
                + transfer_ms
                + profile_penalty
                + window_penalty
                + memory_penalty
                + complexity_penalty
                + distribution_penalty
            ) * adjustment

            results.append((backend, total_cost, memory_bytes))

        return results

    def _get_conversion_cost(
        self, source: PhysicalBackend, target: PhysicalBackend, bytes_size: int
    ) -> float:
        """Get conversion cost from conversion cost matrix.

        OPTIMIZED: Precise conversion costs based on backend pairs.
        """
        # Conversion cost matrix (ms baseline + per-GB scaling)
        # Key: (source, target) -> (baseline_ms, ms_per_gb)
        conversion_matrix = {
            (PhysicalBackend.PANDAS_NUMPY, PhysicalBackend.POLARS_PANEL): (5.0, 30.0),
            (PhysicalBackend.PANDAS_NUMPY, PhysicalBackend.POLARS_LONG): (8.0, 40.0),
            (PhysicalBackend.PANDAS_NUMPY, PhysicalBackend.DUCKDB_SQL): (15.0, 60.0),
            (PhysicalBackend.POLARS_PANEL, PhysicalBackend.PANDAS_NUMPY): (4.0, 25.0),
            (PhysicalBackend.POLARS_PANEL, PhysicalBackend.POLARS_LONG): (2.0, 10.0),
            (PhysicalBackend.POLARS_PANEL, PhysicalBackend.DUCKDB_SQL): (10.0, 45.0),
            (PhysicalBackend.POLARS_LONG, PhysicalBackend.PANDAS_NUMPY): (6.0, 35.0),
            (PhysicalBackend.POLARS_LONG, PhysicalBackend.POLARS_PANEL): (3.0, 15.0),
            (PhysicalBackend.POLARS_LONG, PhysicalBackend.DUCKDB_SQL): (8.0, 40.0),
            (PhysicalBackend.DUCKDB_SQL, PhysicalBackend.PANDAS_NUMPY): (12.0, 50.0),
            (PhysicalBackend.DUCKDB_SQL, PhysicalBackend.POLARS_PANEL): (10.0, 45.0),
            (PhysicalBackend.DUCKDB_SQL, PhysicalBackend.POLARS_LONG): (8.0, 40.0),
        }

        key = (source, target)
        if key in conversion_matrix:
            baseline, ms_per_gb = conversion_matrix[key]
            gb_size = bytes_size / (1024**3)
            return baseline + gb_size * ms_per_gb
        else:
            # Fallback: use target characteristics
            chars = _BACKEND_CHARACTERISTICS[target]
            return chars.conversion_penalty_ms + (bytes_size / 1024**3) * 50.0

    def _compute_complexity_penalty(
        self, ctx: RoutingContext, backend: PhysicalBackend, base_compute_ms: float
    ) -> float:
        """Compute query complexity penalty.

        OPTIMIZED: Score complexity based on window params, columns, and operator profile.
        """
        complexity_score = 0.0

        # Window complexity
        if ctx.window_params:
            max_window = max(ctx.window_params)
            avg_window = sum(ctx.window_params) / len(ctx.window_params)
            window_complexity = (max_window / 1000.0) + (avg_window / 500.0)
            complexity_score += window_complexity

        # Column complexity
        column_complexity = ctx.estimated_columns / 20.0  # Normalize by 20 columns
        complexity_score += column_complexity

        # Profile complexity
        if ctx.operator_profile == OperatorProfile.REGRESSION_HEAVY:
            complexity_score += 2.0
        elif ctx.operator_profile == OperatorProfile.WINDOW_HEAVY:
            complexity_score += 1.5
        elif ctx.operator_profile == OperatorProfile.CROSS_SECTION_HEAVY:
            complexity_score += 1.0

        # Apply complexity penalty based on backend capability
        if backend == PhysicalBackend.PANDAS_NUMPY and complexity_score > 3.0:
            return base_compute_ms * 0.2 * complexity_score
        elif backend in (PhysicalBackend.POLARS_LONG, PhysicalBackend.DUCKDB_SQL):
            # These backends handle complexity better
            return base_compute_ms * 0.05 * complexity_score
        else:
            return base_compute_ms * 0.1 * complexity_score

    def _compute_distribution_penalty(
        self, ctx: RoutingContext, backend: PhysicalBackend, base_compute_ms: float
    ) -> float:
        """Compute data distribution penalty (sparsity, skew).

        OPTIMIZED: Account for data distribution characteristics.
        """
        penalty = 0.0

        # Estimate sparsity from rows vs bytes ratio
        expected_bytes = ctx.estimated_rows * ctx.estimated_columns * 8
        if expected_bytes > 0:
            density = ctx.estimated_bytes / expected_bytes
            sparsity = 1.0 - density

            # High sparsity benefits columnar formats
            if sparsity > 0.5:
                if backend == PhysicalBackend.DUCKDB_SQL:
                    penalty -= base_compute_ms * 0.15  # Benefit from compression
                elif backend == PhysicalBackend.PANDAS_NUMPY:
                    penalty += base_compute_ms * 0.1  # Less efficient

        # Estimate skew from estimated rows per instrument
        if ctx.estimated_rows > 0 and ctx.estimated_columns > 0:
            # High row count suggests potential skew
            rows_per_group = ctx.estimated_rows / max(ctx.estimated_columns, 1)
            if rows_per_group > 10000:
                # Large groups benefit from vectorized backends
                if backend in (PhysicalBackend.POLARS_PANEL, PhysicalBackend.DUCKDB_SQL):
                    penalty -= base_compute_ms * 0.1
                else:
                    penalty += base_compute_ms * 0.05

        return penalty

    def _compute_confidence(
        self,
        costs: list[tuple[PhysicalBackend, float, int]],
        scale: DataScale,
        memory: MemoryPressure,
    ) -> float:
        """计算决策置信度（基于成本差异 + 规模确定性）。"""
        if len(costs) < 2:
            return 1.0

        sorted_costs = sorted(costs, key=lambda x: x[1])
        best_cost = sorted_costs[0][1]
        second_cost = sorted_costs[1][1]

        # Cost gap confidence
        if best_cost < 1.0:
            gap_confidence = 0.5
        else:
            gap = (second_cost - best_cost) / best_cost
            gap_confidence = min(gap / 0.5, 1.0)  # 50%+ gap → full confidence

        # Scale certainty (larger data → higher confidence in throughput model)
        if scale in (DataScale.HUGE, DataScale.MASSIVE):
            scale_confidence = 0.95
        elif scale in (DataScale.LARGE,):
            scale_confidence = 0.85
        elif scale == DataScale.MEDIUM:
            scale_confidence = 0.7
        else:
            scale_confidence = 0.6

        # Memory certainty
        if memory == MemoryPressure.CRITICAL:
            memory_confidence = 0.95  # High certainty: must use streaming
        elif memory == MemoryPressure.LOW:
            memory_confidence = 0.9
        else:
            memory_confidence = 0.8

        # Combined confidence
        return (gap_confidence * 0.5 + scale_confidence * 0.3 + memory_confidence * 0.2)

    def record_actual_cost(
        self,
        backend: PhysicalBackend,
        scale: DataScale,
        profile: OperatorProfile,
        actual_cost_ms: float,
    ) -> None:
        """记录实际执行成本（用于自适应学习）。

        OPTIMIZED: Enhanced feedback loop with more granular learning.
        """
        if not self.enable_adaptive_learning:
            return

        key = (backend.value, scale.value, profile.value)
        history = self._cost_history.setdefault(key, [])
        history.append(actual_cost_ms)

        # Keep only recent window
        if len(history) > self.cost_history_window:
            history.pop(0)

        # Update adjustment factor (EWMA) - more responsive
        if len(history) >= 5:
            # Compare actual vs predicted with recent bias
            recent_avg = sum(history[-5:]) / 5
            all_avg = sum(history) / len(history)
            # Weight recent more heavily
            weighted_avg = recent_avg * 0.7 + all_avg * 0.3

            chars = _BACKEND_CHARACTERISTICS[backend]
            predicted = 10.0 / chars.per_row_throughput_mrows_per_sec * 1000.0

            if predicted > 0:
                ratio = weighted_avg / predicted
                # More aggressive EWMA for faster adaptation
                alpha = 0.2 if len(history) < 20 else 0.1
                old_factor = self._adjustment_factors.get(backend, 1.0)
                # Clamp adjustment factor to reasonable range
                new_factor = old_factor * (1 - alpha) + ratio * alpha
                self._adjustment_factors[backend] = max(0.5, min(2.0, new_factor))

    def get_calibration_stats(self) -> dict[str, Any]:
        """Get calibration statistics for monitoring.

        OPTIMIZED: Added monitoring capabilities.
        """
        stats = {
            "adjustment_factors": {
                backend.value: round(factor, 3)
                for backend, factor in self._adjustment_factors.items()
            },
            "history_size": {
                f"{backend}_{scale}_{profile}": len(history)
                for (backend, scale, profile), history in self._cost_history.items()
            },
            "total_samples": sum(len(h) for h in self._cost_history.values()),
        }
        return stats


def select_optimal_backend_for_node(
    *,
    estimated_rows: int,
    estimated_columns: int,
    estimated_bytes: int,
    available_memory_bytes: int,
    operator_names: list[str],
    window_params: list[int] | None = None,
    parent_backend: PhysicalBackend | None = None,
    performance_priority: str = "balanced",
) -> RoutingDecision:
    """便捷函数：为单个节点选择最优 backend。

    R21-ROUTING-AUTHORITY: this is a capability/cost CANDIDATE PROVIDER only.
    It may propose a backend but never finalizes the production route.  The
    Global Physical Planner
    (``runtime.multibackend.batch_global_optimizer.PhysicalBatchGlobalOptimizer``)
    is the sole production routing authority.

    Args:
        estimated_rows: 估计行数
        estimated_columns: 估计列数
        estimated_bytes: 估计字节数
        available_memory_bytes: 可用内存
        operator_names: 算子名称列表
        window_params: 窗口参数列表
        parent_backend: 父节点 backend
        performance_priority: 性能优先级

    Returns:
        RoutingDecision
    """
    # Infer operator profile
    profile = _infer_operator_profile(operator_names, window_params or [])

    # Infer execution axis
    axis = _infer_execution_axis(operator_names)

    # Check for regression
    has_regression = any(
        "regression" in op or "neutralize" in op or "resid" in op
        for op in operator_names
    )

    ctx = RoutingContext(
        estimated_rows=estimated_rows,
        estimated_columns=estimated_columns,
        estimated_bytes=estimated_bytes,
        available_memory_bytes=available_memory_bytes,
        operator_profile=profile,
        execution_axis=axis,
        window_params=window_params or [],
        has_regression=has_regression,
        parent_backend=parent_backend,
        allows_streaming=True,
        performance_priority=performance_priority,
    )

    selector = IntelligentBackendSelector()
    return selector.select_backend(ctx)


def _infer_operator_profile(operator_names: list[str], window_params: list[int]) -> OperatorProfile:
    """从算子列表推断 profile。"""
    window_count = sum(1 for op in operator_names if "ts_" in op or "rolling" in op)
    cs_count = sum(1 for op in operator_names if "cs_" in op or "rank" in op or "zscore" in op)
    regress_count = sum(1 for op in operator_names if "regress" in op or "neutralize" in op)

    total = len(operator_names)
    if total == 0:
        return OperatorProfile.MIXED

    if regress_count > 0:
        return OperatorProfile.REGRESSION_HEAVY

    window_ratio = window_count / total
    cs_ratio = cs_count / total

    # Large windows → window_heavy
    if window_params and sum(window_params) / len(window_params) > 100:
        return OperatorProfile.WINDOW_HEAVY

    if window_ratio > 0.6:
        return OperatorProfile.WINDOW_HEAVY
    elif cs_ratio > 0.6:
        return OperatorProfile.CROSS_SECTION_HEAVY
    elif window_ratio < 0.2 and cs_ratio < 0.2:
        return OperatorProfile.ELEMENTWISE_HEAVY
    else:
        return OperatorProfile.MIXED


def _infer_execution_axis(operator_names: list[str]) -> ExecutionAxis:
    """从算子列表推断执行轴。"""
    if any("ts_" in op or "rolling" in op for op in operator_names):
        return ExecutionAxis.TIME_PER_INSTRUMENT
    elif any("cs_" in op or "rank" in op for op in operator_names):
        return ExecutionAxis.CROSS_SECTION_PER_DATE
    else:
        return ExecutionAxis.GLOBAL_PANEL
