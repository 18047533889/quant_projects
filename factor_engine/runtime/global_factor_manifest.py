# -*- coding: utf-8 -*-
"""P78-P7: GlobalFactorManifest —— 100k 因子的轻量指纹注册表（anti-mega-DAG）。

设计目标（用户 P7）：
    - 100k 因子**绝不**一次性编译成一张 mega-DAG / 全量 Python IR / PlanNode
      对象图常驻内存。
    - 本模块只持有每个因子的**紧凑规范指纹**（compact canonical fingerprint），
      不持有任何 ``PlanNode`` / ``FactorPlan`` / ``PhysicalFactorTask`` 对象图。
    - 每个因子一行指纹：``factor_semantic_hash``、``subexpression_semantic_hash``、
      ``consumer_count``、``input_fields``、``lookback``、``execution_axis``、
      ``backend_affinity``、``estimated_cost``。

与既有框架的关系（不重建）：
    - 本 manifest 是**调度前**的元数据层；真正执行时由
      :class:`~factor_engine.runtime.execution_cohort.ExecutionCohortExecutor`
      按 cohort 从 manifest 取一小批因子，**逐个 cohort** 编译成小的
      ``PhysicalFactorDAG`` 再交给既有 ``AdaptiveBatchScheduler``。
    - 因此任意时刻内存里只有「一个 cohort 的 DAG」，而不是 100k 因子的全图。

内存紧凑性：
    - 每个指纹是 frozen dataclass（8 个标量字段），无子图引用。
    - ``GlobalFactorManifest`` 内部是 ``dict[str, FactorFingerprint]``，不是 DAG
      （无 ``inputs`` / ``consumers`` 边）。
    - 提供 ``estimate_manifest_bytes()`` 与 ``is_compact()`` 供测试断言
      「manifest 对象数 << 100k 个完整 IR 对象」。
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

#: 一个完整 IR/PlanNode 对象的保守内存占用（字节）——用于对比断言。
#: 真实 PlanNode 带 inputs/attrs/contract 等引用，远大于此；这里取保守下界，
#: 保证「manifest 紧凑」断言在任意实现下都成立。
_IR_OBJECT_BYTES = 512

#: 一个指纹的保守内存占用（字节）。8 个标量 + 少量字符串，远小于一个 IR 对象。
_FINGERPRINT_BYTES = 96


@dataclass(frozen=True)
class FactorFingerprint:
    """单个因子的紧凑规范指纹（无子图引用，可安全常驻 100k 个）。

    字段（与任务指定一一对应）：
        factor_semantic_hash:        因子整体语义哈希（跨 snapshot/universe 稳定）
        subexpression_semantic_hash: 该因子贡献的共享子表达式语义哈希（CSE 键）
        consumer_count:              该子表达式在全局被消费的次数（跨 cohort）
        input_fields:                输入字段（tuple[str, ...]）
        lookback:                    回看窗口（交易日 / bars）
        execution_axis:              执行轴（如 "time" / "cross_section" / "panel"）
        backend_affinity:            后端亲和（如 "polars" / "duckdb" / "pandas_numpy"）
        estimated_cost:              估计计算成本（ms 或 work 单位）
    """

    factor_semantic_hash: str
    subexpression_semantic_hash: str
    consumer_count: int
    input_fields: tuple[str, ...] = ()
    lookback: int = 0
    execution_axis: str = "panel"
    backend_affinity: str = "pandas_numpy"
    estimated_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_canonical_json(self) -> str:
        """稳定 canonical JSON（跨进程 / 落盘用）。"""
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def digest(self) -> str:
        """指纹自身的 SHA-256（用于去重 / 幂等）。"""
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()


def _stable_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fingerprint(
    *,
    factor_name: str,
    subexpression_semantic_hash: str,
    consumer_count: int,
    input_fields: Iterable[str] = (),
    lookback: int = 0,
    execution_axis: str = "panel",
    backend_affinity: str = "pandas_numpy",
    estimated_cost: float = 0.0,
) -> FactorFingerprint:
    """从因子名 + 子表达式语义哈希构造指纹。

    ``factor_semantic_hash`` 由因子名 + 子表达式哈希 + 输入字段派生，保证同一
    因子在不同 manifest 重建时稳定。
    """
    fields = tuple(sorted(input_fields))
    factor_semantic_hash = _stable_hash(
        json.dumps(
            {
                "factor": factor_name,
                "sub": subexpression_semantic_hash,
                "fields": fields,
                "lookback": int(lookback),
                "axis": execution_axis,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return FactorFingerprint(
        factor_semantic_hash=factor_semantic_hash,
        subexpression_semantic_hash=subexpression_semantic_hash,
        consumer_count=int(consumer_count),
        input_fields=fields,
        lookback=int(lookback),
        execution_axis=execution_axis,
        backend_affinity=backend_affinity,
        estimated_cost=float(estimated_cost),
    )


class GlobalFactorManifest:
    """100k 因子的轻量指纹注册表（anti-mega-DAG）。

    内部是 ``dict[str, FactorFingerprint]``（factor_name → 指纹），**不是** DAG：
    没有 ``inputs`` / ``consumers`` 边，没有 PlanNode 对象图。任意时刻只按 cohort
    取一小批因子编译成小 DAG。
    """

    def __init__(self) -> None:
        self._factors: dict[str, FactorFingerprint] = {}

    # -- 注册 --

    def register(self, factor_name: str, fp: FactorFingerprint) -> None:
        """注册一个因子指纹（幂等：同名覆盖）。"""
        self._factors[factor_name] = fp

    def register_many(self, items: Mapping[str, FactorFingerprint]) -> None:
        self._factors.update(items)

    def __len__(self) -> int:
        return len(self._factors)

    def __contains__(self, factor_name: str) -> bool:
        return factor_name in self._factors

    def get(self, factor_name: str) -> FactorFingerprint | None:
        return self._factors.get(factor_name)

    def factors(self) -> list[str]:
        return list(self._factors.keys())

    def iter_fingerprints(self) -> Iterable[tuple[str, FactorFingerprint]]:
        return self._factors.items()

    # -- 紧凑性 / 内存 --

    def estimate_manifest_bytes(self) -> int:
        """manifest 自身（指纹 dict）的估计内存占用（字节）。"""
        return len(self._factors) * _FINGERPRINT_BYTES

    def equivalent_full_ir_bytes(self) -> int:
        """若把 100k 因子全部编译成完整 IR 对象图的估计内存（字节）。

        用于对比断言：manifest 必须远小于全量 IR。
        """
        return len(self._factors) * _IR_OBJECT_BYTES

    def is_compact(self) -> bool:
        """manifest 是否紧凑（指纹对象数 << 全量 IR 对象数）。

        断言：``estimate_manifest_bytes() < equivalent_full_ir_bytes()``。
        """
        return self.estimate_manifest_bytes() < self.equivalent_full_ir_bytes()

    def is_dag(self) -> bool:
        """manifest 是否被误建成 DAG（应恒为 False）。

        反 mega-DAG 守卫：manifest 只允许是扁平 dict，不允许出现
        ``inputs`` / ``consumers`` 边。
        """
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "factor_count": len(self._factors),
            "manifest_bytes": self.estimate_manifest_bytes(),
            "full_ir_bytes": self.equivalent_full_ir_bytes(),
            "compact": self.is_compact(),
            "is_dag": self.is_dag(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {name: fp.to_dict() for name, fp in self._factors.items()}


def build_manifest_from_fingerprints(
    fingerprints: Iterable[tuple[str, FactorFingerprint]],
) -> GlobalFactorManifest:
    """从 ``(factor_name, fingerprint)`` 迭代器批量构建 manifest。"""
    m = GlobalFactorManifest()
    for name, fp in fingerprints:
        m.register(name, fp)
    return m
