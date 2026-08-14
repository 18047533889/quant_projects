"""Q backend capability queries backed by the physical implementation registry."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from backend.q_backend.q_physical_implementation_registry import (
    QPhysicalImplementationRegistry,
    get_q_physical_implementation_registry,
)

class QCapabilityLevel(Enum):
    NATIVE = "NATIVE"
    DELEGATE = "DELEGATE"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    UNSUPPORTED = "UNSUPPORTED"

@dataclass(frozen=True)
class QOperatorCapability:
    op_name: str
    capability: QCapabilityLevel
    streaming_safe: bool = False
    requires_global_sort: bool = False
    requires_full_group: bool = False
    notes: str = ""

# Compatibility-only declaration surface. It is never consulted for admission.
_PHASE1_NATIVE_OPS = frozenset()
_PHASE1_DEFERRED_OPS = frozenset({"garch", "har", "kalman_filter", "ar", "var", "network_centrality", "graph_distance", "topological_sort", "mutual_information", "transfer_entropy"})

class QBackendCapability:
    def __init__(self, registry=None):
        self.registry = registry or get_q_physical_implementation_registry()

    def get_capability(self, op_name: str, *, mode: Literal["production", "research"] = "production") -> QCapabilityLevel:
        if mode == "production":
            return QCapabilityLevel.NATIVE if self.registry.is_production_certified(op_name) else QCapabilityLevel.UNSUPPORTED
        return QCapabilityLevel.NATIVE if self.registry.has_lowering(op_name) else QCapabilityLevel.UNSUPPORTED

    def supports_native(self, op_name: str) -> bool:
        return self.get_capability(op_name) == QCapabilityLevel.NATIVE

    def is_streaming_safe(self, op_name: str) -> bool:
        return op_name in {"ts_mean", "ts_sum", "lag", "delta"} and self.registry.has_lowering(op_name)

    def requires_global_sort(self, op_name: str) -> bool:
        return False

    def requires_full_group(self, op_name: str) -> bool:
        return op_name.startswith("group_") and self.registry.has_lowering(op_name)

    def compute_native_fraction(self, op_names: list[str], *, mode: Literal["production", "research"] = "production"):
        if not op_names:
            return 0.0, [], []
        native = [op for op in op_names if self.get_capability(op, mode=mode) == QCapabilityLevel.NATIVE]
        unsupported = [op for op in op_names if op not in native]
        return len(native) / len(op_names), native, unsupported

_CAPABILITY_REGISTRY: QBackendCapability | None = None
def get_q_capability() -> QBackendCapability:
    global _CAPABILITY_REGISTRY
    if _CAPABILITY_REGISTRY is None:
        _CAPABILITY_REGISTRY = QBackendCapability()
    return _CAPABILITY_REGISTRY
