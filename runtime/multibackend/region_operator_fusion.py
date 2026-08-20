# -*- coding: utf-8 -*-
"""MB-P1-015: Region internal operator fusion.

Backend execution region 内算子融合（region = 同一 backend 的连续算子序列）：
    - Pandas: vectorized expression fusion（多个 assign/filter/select 合并）
    - Polars: lazy chain（已由 polars_lazy_fusion.py 覆盖）
    - DuckDB: SQL subquery fusion（多层 SELECT 合并成单 CTE）
    - 避免中间 DataFrame 物化（内存节省 + cache miss 减少）

Region fusion 与 native fusion 区别：
    - Native fusion: 跨 root 的 batch-global 优化（多个因子共享 scan）
    - Region fusion: 单个因子内部的算子链优化（减少中间结果）
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class RegionFusionMetrics:
    """Region 内部融合统计。"""

    regions_analyzed: int = 0
    operators_fused: int = 0
    fusion_opportunities: int = 0
    fusion_applied: int = 0
    estimated_memory_saved_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions_analyzed": self.regions_analyzed,
            "operators_fused": self.operators_fused,
            "fusion_opportunities": self.fusion_opportunities,
            "fusion_applied": self.fusion_applied,
            "estimated_memory_saved_bytes": self.estimated_memory_saved_bytes,
        }


@dataclass
class OperatorRegion:
    """单个 execution region（连续同 backend 算子）。"""

    region_id: str
    backend: str
    operators: list[dict[str, Any]]
    fusible: bool = True
    estimated_memory_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "backend": self.backend,
            "operator_count": len(self.operators),
            "fusible": self.fusible,
            "estimated_memory_bytes": self.estimated_memory_bytes,
        }


class RegionOperatorFusion:
    """Region 内部算子融合优化器。

    集成点：
        - PhysicalLowerer 构建 region 时标记 fusible operators
        - Backend executor 执行 region 前调用 fuse_region()
        - AdaptiveBatchScheduler 根据 fusion 后的成本调度
    """

    def __init__(self) -> None:
        self._metrics = RegionFusionMetrics()
        self._lock = threading.RLock()

    def identify_regions(
        self, operators: list[dict[str, Any]]
    ) -> list[OperatorRegion]:
        """识别连续同 backend 的 operator region。

        Args:
            operators: 算子列表，每项 {"op": "filter", "backend": "pandas", ...}

        Returns:
            Region 列表（每个 region 是连续同 backend 的算子序列）
        """
        if not operators:
            return []

        regions: list[OperatorRegion] = []
        current_backend = operators[0].get("backend", "pandas")
        current_ops: list[dict[str, Any]] = []
        region_counter = 0

        for op in operators:
            backend = op.get("backend", "pandas")
            if backend != current_backend:
                # Backend 切换：封闭当前 region
                if current_ops:
                    region_counter += 1
                    regions.append(
                        OperatorRegion(
                            region_id=f"region_{region_counter}",
                            backend=current_backend,
                            operators=current_ops,
                        )
                    )
                current_backend = backend
                current_ops = [op]
            else:
                current_ops.append(op)

        # 最后一个 region
        if current_ops:
            region_counter += 1
            regions.append(
                OperatorRegion(
                    region_id=f"region_{region_counter}",
                    backend=current_backend,
                    operators=current_ops,
                )
            )

        with self._lock:
            self._metrics.regions_analyzed += len(regions)

        return regions

    def fuse_region(self, region: OperatorRegion) -> dict[str, Any]:
        """融合 region 内的算子（返回优化后的执行计划）。

        Args:
            region: OperatorRegion

        Returns:
            融合后的执行计划（backend-specific）
        """
        with self._lock:
            self._metrics.fusion_opportunities += 1

        if len(region.operators) <= 1:
            # 单算子：无需融合
            return {"fused": False, "plan": region.operators}

        backend = region.backend.lower()

        if backend == "pandas":
            return self._fuse_pandas_region(region)
        elif backend == "polars":
            return self._fuse_polars_region(region)
        elif backend == "duckdb":
            return self._fuse_duckdb_region(region)
        else:
            # 未知 backend：不融合
            return {"fused": False, "plan": region.operators}

    def _fuse_pandas_region(self, region: OperatorRegion) -> dict[str, Any]:
        """融合 Pandas 算子（vectorized expression fusion）。

        示例：
            filter(x > 0) + assign(y=x*2) + select(['a','y'])
            → 单次 eval 合并：df.query("x > 0").assign(y=lambda d: d.x*2)[['a','y']]
        """
        ops = region.operators
        fusible_types = {"filter", "assign", "select", "with_columns"}

        # 检查是否全部可融合
        if not all(op.get("op", "") in fusible_types for op in ops):
            return {"fused": False, "plan": ops}

        # 构造融合后的 Pandas chain
        fused_plan = {
            "fused": True,
            "backend": "pandas",
            "chain": [],
        }

        for op in ops:
            op_type = op.get("op", "")
            if op_type == "filter":
                fused_plan["chain"].append(
                    {"method": "query", "args": [op.get("predicate", "")]}
                )
            elif op_type == "assign":
                fused_plan["chain"].append(
                    {"method": "assign", "kwargs": op.get("columns", {})}
                )
            elif op_type == "select":
                fused_plan["chain"].append(
                    {"method": "getitem", "args": [op.get("columns", [])]}
                )

        with self._lock:
            self._metrics.fusion_applied += 1
            self._metrics.operators_fused += len(ops)
            # 估算节省内存：每个中间结果 ~10MB
            self._metrics.estimated_memory_saved_bytes += (len(ops) - 1) * 10 * 1024**2

        return fused_plan

    def _fuse_polars_region(self, region: OperatorRegion) -> dict[str, Any]:
        """融合 Polars 算子（lazy chain，由 polars_lazy_fusion 模块完成）。

        此处只标记可融合，实际融合在 PolarsLazyFusionOptimizer 中。
        """
        ops = region.operators

        with self._lock:
            self._metrics.fusion_applied += 1
            self._metrics.operators_fused += len(ops)

        return {
            "fused": True,
            "backend": "polars",
            "lazy_chain": ops,
            "note": "actual fusion delegated to PolarsLazyFusionOptimizer",
        }

    def _fuse_duckdb_region(self, region: OperatorRegion) -> dict[str, Any]:
        """融合 DuckDB 算子（SQL subquery fusion）。

        示例：
            SELECT a FROM (SELECT a, b FROM t WHERE x > 0) WHERE a < 10
            → 融合成：SELECT a FROM t WHERE x > 0 AND a < 10
        """
        ops = region.operators
        fusible_types = {"select", "filter", "aggregate"}

        if not all(op.get("op", "") in fusible_types for op in ops):
            return {"fused": False, "plan": ops}

        # 构造融合后的 SQL（简化版本：实际需要完整 SQL rewriter）
        fused_sql_parts = {
            "select": [],
            "from": "input_table",
            "where": [],
            "group_by": [],
            "having": [],
        }

        for op in ops:
            op_type = op.get("op", "")
            if op_type == "select":
                fused_sql_parts["select"].extend(op.get("columns", []))
            elif op_type == "filter":
                fused_sql_parts["where"].append(op.get("predicate", ""))
            elif op_type == "aggregate":
                fused_sql_parts["group_by"].extend(op.get("group_by", []))

        # 构造最终 SQL
        select_clause = ", ".join(fused_sql_parts["select"]) or "*"
        where_clause = (
            " AND ".join(fused_sql_parts["where"]) if fused_sql_parts["where"] else ""
        )
        group_clause = (
            ", ".join(fused_sql_parts["group_by"]) if fused_sql_parts["group_by"] else ""
        )

        fused_sql = f"SELECT {select_clause} FROM {fused_sql_parts['from']}"
        if where_clause:
            fused_sql += f" WHERE {where_clause}"
        if group_clause:
            fused_sql += f" GROUP BY {group_clause}"

        with self._lock:
            self._metrics.fusion_applied += 1
            self._metrics.operators_fused += len(ops)
            self._metrics.estimated_memory_saved_bytes += (len(ops) - 1) * 10 * 1024**2

        return {
            "fused": True,
            "backend": "duckdb",
            "sql": fused_sql,
            "original_operators": len(ops),
        }

    def can_fuse(self, op1: dict[str, Any], op2: dict[str, Any]) -> bool:
        """判断两个相邻算子是否可融合。

        Args:
            op1: 第一个算子
            op2: 第二个算子

        Returns:
            True 表示可融合，False 表示必须独立执行
        """
        # Backend 不同：不可融合
        if op1.get("backend") != op2.get("backend"):
            return False

        # 某些算子打断融合边界（如 sort / join / 自定义 UDF）
        boundary_ops = {"sort", "join", "pivot", "melt", "apply"}
        if op1.get("op") in boundary_ops or op2.get("op") in boundary_ops:
            return False

        # 副作用算子（写文件、发请求）不可融合
        if op1.get("side_effect", False) or op2.get("side_effect", False):
            return False

        return True

    def metrics(self) -> RegionFusionMetrics:
        """返回累计统计。"""
        with self._lock:
            return RegionFusionMetrics(
                regions_analyzed=self._metrics.regions_analyzed,
                operators_fused=self._metrics.operators_fused,
                fusion_opportunities=self._metrics.fusion_opportunities,
                fusion_applied=self._metrics.fusion_applied,
                estimated_memory_saved_bytes=self._metrics.estimated_memory_saved_bytes,
            )

    def reset_metrics(self) -> None:
        """重置统计。"""
        with self._lock:
            self._metrics = RegionFusionMetrics()


# 全局单例
_global_region_fusion: RegionOperatorFusion | None = None
_global_lock = threading.Lock()


def get_global_region_fusion() -> RegionOperatorFusion:
    """返回全局 RegionOperatorFusion（进程级单例）。"""
    global _global_region_fusion
    if _global_region_fusion is None:
        with _global_lock:
            if _global_region_fusion is None:
                _global_region_fusion = RegionOperatorFusion()
    return _global_region_fusion
