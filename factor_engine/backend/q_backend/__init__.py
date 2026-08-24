"""q/K Backend - Phase 1 High-Compatibility Execution Backend.

q/K 作为 FactorEngine 的可选物理执行后端，而非语义权威。

架构原则（文档 §24-26）：
1. q/K 只作为 execution backend
2. 不改变 DSL/IR/Semantic authority
3. Region 级边界（禁止 Python→q op1→Python→q op2）
4. 类型 adapter 版本化
5. PIT 语义保持（available_at/revision/snapshot 由 DataAccess 给定）
6. PyKX zero-copy 只是优化，非 correctness 前提
7. q process availability 治理

Phase 1 优先场景（文档 §24）：
- arithmetic/comparison
- lag/delta
- rolling mean/sum/std/min/max
- corr/cov
- rank/group basic
- VWAP/basic aggregation
- time/as-of preparation
- minute→daily aggregation

暂缓场景：
- 复杂 model training
- 复杂 checkpoint state
- topology/特殊科研算法
"""

from __future__ import annotations

__all__ = [
    "QBackend",
    "get_q_backend",
    "QBackendCapability",
    "QProcessManager",
    "QTypeAdapter",
    "QExecutor",
    "QCompiler",
    "QRegionPlan",
    "get_q_compiler",
    "check_q_availability",
    "is_q_available",
]

from factor_engine.backend.q_backend.q_backend import QBackend, get_q_backend
from factor_engine.backend.q_backend.q_capability import QBackendCapability
from factor_engine.backend.q_backend.q_process_manager import (
    QProcessManager,
    check_q_availability,
    is_q_available,
)
from factor_engine.backend.q_backend.q_adapter import QTypeAdapter
from factor_engine.backend.q_backend.q_executor import QExecutor
from factor_engine.backend.q_backend.q_compiler import QCompiler, QRegionPlan, get_q_compiler
