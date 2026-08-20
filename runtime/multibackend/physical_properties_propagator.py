# -*- coding: utf-8 -*-
"""MB-P1-018: Physical properties propagation.

执行计划物理属性传播（优化数据流）：
    - Partitioning（数据分区：hash/range/random）
    - Ordering（排序状态：sorted by x ASC）
    - Distribution（分布方式：broadcast/shuffle/replicated）
    - Equivalence classes（等价列集合：join 后 a=b）

物理属性传播 → 避免冗余 shuffle/sort → 减少数据传输 + 计算开销。

示例：
    SELECT * FROM t1 JOIN t2 ON t1.id = t2.id ORDER BY id
    → t1 已按 id hash 分区 + t2 broadcast → join 无需 shuffle
    → join 后保留 id 排序 → ORDER BY 无需重排
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class PhysicalProperties:
    """执行节点的物理属性（数据布局特征）。"""

    # 分区属性
    partitioning: str | None = None  # "hash:col" | "range:col" | "random" | None
    partition_count: int = 1
    # 排序属性
    ordering: list[tuple[str, str]] = field(default_factory=list)  # [(col, "ASC"|"DESC")]
    # 分布属性（分布式执行）
    distribution: str = "single"  # "single" | "broadcast" | "partitioned"
    # 等价类（join 后推导）
    equivalence_classes: list[set[str]] = field(default_factory=list)
    # 估算行数（物理属性依赖 cardinality）
    estimated_rows: int = 0

    def is_partitioned_by(self, columns: list[str]) -> bool:
        """判断是否按指定列分区。"""
        if not self.partitioning:
            return False
        if not self.partitioning.startswith("hash:") and not self.partitioning.startswith("range:"):
            return False
        part_cols = self.partitioning.split(":", 1)[1].split(",")
        return set(part_cols) == set(columns)

    def is_sorted_by(self, columns: list[str]) -> bool:
        """判断是否按指定列排序（前缀匹配）。"""
        if len(columns) > len(self.ordering):
            return False
        for i, col in enumerate(columns):
            if self.ordering[i][0] != col:
                return False
        return True

    def get_equivalent_columns(self, col: str) -> set[str]:
        """返回与 col 等价的所有列（等价类传播）。"""
        for eq_class in self.equivalence_classes:
            if col in eq_class:
                return eq_class
        return {col}

    def to_dict(self) -> dict[str, Any]:
        return {
            "partitioning": self.partitioning,
            "partition_count": self.partition_count,
            "ordering": self.ordering,
            "distribution": self.distribution,
            "equivalence_classes": [list(s) for s in self.equivalence_classes],
            "estimated_rows": self.estimated_rows,
        }


class PhysicalPropertiesPropagator:
    """物理属性传播器（计划树自下而上推导）。

    集成点：
        - PhysicalLowerer 构建 PhysicalFactorDAG 时调用 propagate()
        - Optimizer 根据物理属性选择最优 join/sort 算法
        - Scheduler 根据 distribution 属性决定是否并行
    """

    def propagate_scan(
        self,
        source_id: str,
        columns: list[str],
        predicate: str | None = None,
    ) -> PhysicalProperties:
        """Scan 算子的物理属性（输入边界）。

        Args:
            source_id: 数据源标识
            columns: 输出列
            predicate: 过滤条件

        Returns:
            PhysicalProperties（Scan 输出无特定分区/排序）
        """
        # Scan 默认无分区、无排序（除非数据源本身有序，如时间序列）
        return PhysicalProperties(
            partitioning=None,
            ordering=[],
            distribution="single",
            estimated_rows=1_000_000,  # 默认估算
        )

    def propagate_filter(
        self,
        input_props: PhysicalProperties,
        predicate: str,
        selectivity: float = 0.5,
    ) -> PhysicalProperties:
        """Filter 算子的物理属性传播。

        Filter 保留输入的分区/排序（不打乱数据布局）。
        """
        return PhysicalProperties(
            partitioning=input_props.partitioning,
            partition_count=input_props.partition_count,
            ordering=input_props.ordering.copy(),
            distribution=input_props.distribution,
            equivalence_classes=[s.copy() for s in input_props.equivalence_classes],
            estimated_rows=int(input_props.estimated_rows * selectivity),
        )

    def propagate_project(
        self,
        input_props: PhysicalProperties,
        output_columns: list[str],
    ) -> PhysicalProperties:
        """Project 算子的物理属性传播。

        Project 可能改变排序（如果 ORDER BY 的列被移除）。
        """
        # 检查排序列是否仍在输出中
        preserved_ordering = [
            (col, direction)
            for col, direction in input_props.ordering
            if col in output_columns
        ]

        # 检查分区列是否仍在输出中
        preserved_partitioning = input_props.partitioning
        if input_props.partitioning:
            part_type, part_cols_str = input_props.partitioning.split(":", 1)
            part_cols = part_cols_str.split(",")
            if not all(c in output_columns for c in part_cols):
                preserved_partitioning = None

        return PhysicalProperties(
            partitioning=preserved_partitioning,
            partition_count=input_props.partition_count,
            ordering=preserved_ordering,
            distribution=input_props.distribution,
            equivalence_classes=[
                s & set(output_columns) for s in input_props.equivalence_classes
            ],
            estimated_rows=input_props.estimated_rows,
        )

    def propagate_sort(
        self,
        input_props: PhysicalProperties,
        sort_columns: list[tuple[str, str]],
    ) -> PhysicalProperties:
        """Sort 算子的物理属性传播。

        Sort 建立新的排序属性（覆盖输入排序）。
        """
        return PhysicalProperties(
            partitioning=input_props.partitioning,
            partition_count=input_props.partition_count,
            ordering=sort_columns.copy(),
            distribution=input_props.distribution,
            equivalence_classes=[s.copy() for s in input_props.equivalence_classes],
            estimated_rows=input_props.estimated_rows,
        )

    def propagate_hash_join(
        self,
        left_props: PhysicalProperties,
        right_props: PhysicalProperties,
        join_keys: list[tuple[str, str]],  # [(left_col, right_col)]
        join_type: str = "inner",
    ) -> PhysicalProperties:
        """Hash Join 算子的物理属性传播。

        Args:
            left_props: 左表物理属性
            right_props: 右表物理属性
            join_keys: join 键列表
            join_type: "inner" | "left" | "right" | "outer"

        Returns:
            Join 输出的物理属性
        """
        # Join 后输出分区：如果左表按 join key 分区，则输出也按此分区
        output_partitioning = None
        left_join_cols = [k[0] for k in join_keys]
        if left_props.is_partitioned_by(left_join_cols):
            output_partitioning = left_props.partitioning

        # Join 后排序丢失（hash join 无序输出）
        output_ordering: list[tuple[str, str]] = []

        # 等价类合并（join key 建立等价关系）
        new_equivalence_classes = [
            s.copy() for s in left_props.equivalence_classes
        ] + [s.copy() for s in right_props.equivalence_classes]

        for left_col, right_col in join_keys:
            # 合并两列所在的等价类
            left_class = left_props.get_equivalent_columns(left_col)
            right_class = right_props.get_equivalent_columns(right_col)
            merged_class = left_class | right_class
            new_equivalence_classes = [
                s for s in new_equivalence_classes if not (s & merged_class)
            ]
            new_equivalence_classes.append(merged_class)

        # 估算输出行数（join selectivity）
        join_selectivity = 1.0 if join_type == "inner" else 1.5
        output_rows = int(
            left_props.estimated_rows * right_props.estimated_rows * join_selectivity / 1000
        )

        return PhysicalProperties(
            partitioning=output_partitioning,
            partition_count=left_props.partition_count,
            ordering=output_ordering,
            distribution="partitioned" if output_partitioning else "single",
            equivalence_classes=new_equivalence_classes,
            estimated_rows=output_rows,
        )

    def propagate_aggregate(
        self,
        input_props: PhysicalProperties,
        group_by_columns: list[str],
    ) -> PhysicalProperties:
        """Aggregate 算子的物理属性传播。

        Aggregate 输出按 GROUP BY 列分区（hash aggregate）。
        """
        output_partitioning = None
        if group_by_columns:
            output_partitioning = f"hash:{','.join(group_by_columns)}"

        # Aggregate 输出行数 = 分组数（通常远小于输入）
        output_rows = max(1, input_props.estimated_rows // 100)

        return PhysicalProperties(
            partitioning=output_partitioning,
            partition_count=input_props.partition_count,
            ordering=[],  # aggregate 输出无序
            distribution="partitioned" if group_by_columns else "single",
            equivalence_classes=[],  # aggregate 不保留等价类
            estimated_rows=output_rows,
        )

    def requires_shuffle(
        self,
        operator: str,
        input_props: PhysicalProperties,
        required_partitioning: str | None,
    ) -> bool:
        """判断算子是否需要 shuffle（数据重分区）。

        Args:
            operator: 算子类型
            input_props: 输入物理属性
            required_partitioning: 算子要求的分区（如 "hash:id"）

        Returns:
            True 表示需要 shuffle，False 表示输入已满足要求
        """
        if required_partitioning is None:
            return False

        if input_props.partitioning == required_partitioning:
            return False

        # 等价列检查：如果要求按 a 分区，但输入按 b 分区且 a=b，则无需 shuffle
        if required_partitioning.startswith("hash:"):
            required_cols = set(required_partitioning.split(":", 1)[1].split(","))
            for col in required_cols:
                equiv = input_props.get_equivalent_columns(col)
                if input_props.is_partitioned_by(list(equiv)):
                    return False

        return True

    def requires_sort(
        self,
        input_props: PhysicalProperties,
        required_ordering: list[tuple[str, str]],
    ) -> bool:
        """判断是否需要显式 Sort。

        Args:
            input_props: 输入物理属性
            required_ordering: 要求的排序

        Returns:
            True 表示需要 Sort，False 表示输入已排序
        """
        if not required_ordering:
            return False

        # 检查输入是否已按要求排序（前缀匹配）
        required_cols = [col for col, _ in required_ordering]
        return not input_props.is_sorted_by(required_cols)


# 全局单例
_global_propagator: PhysicalPropertiesPropagator | None = None


def get_global_properties_propagator() -> PhysicalPropertiesPropagator:
    """返回全局 PhysicalPropertiesPropagator（进程级单例）。"""
    global _global_propagator
    if _global_propagator is None:
        _global_propagator = PhysicalPropertiesPropagator()
    return _global_propagator
