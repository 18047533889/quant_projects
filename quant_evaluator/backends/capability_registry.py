"""Backend capability registry (spec §11).

Tracks, per primitive operation, which backends have a real implementation,
their parity status, supported dtypes/layouts, and scratch estimator.  A
metric being STABLE in the MetricRegistry does NOT imply GPU readiness — that
is decided here.

Only the coordinator modifies the canonical integration surface; subagents
register their own primitive implementations through the public API below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from quant_evaluator.contracts.backend_policy import BackendPolicy


class ParityStatus(Enum):
    """Parity status of a backend implementation vs CPU reference."""

    UNVERIFIED = "unverified"
    PARITY_PASS = "parity_pass"
    PARITY_FAIL = "parity_fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class BackendImplementation:
    """A concrete implementation of a primitive on a backend (spec §11)."""

    operation_id: str
    backend: str  # "cpu_reference" | "cpu_fast" | "cuda"
    implementation_version: str
    implementation_hash: str
    supported_dtypes: Tuple[str, ...] = ("float64",)
    supported_layouts: Tuple[str, ...] = ("T,N,F",)
    deterministic: bool = True
    exact_semantics: bool = True
    scratch_estimator: Optional[Callable[..., int]] = None
    parity_status: ParityStatus = ParityStatus.UNVERIFIED
    fn: Optional[Callable[..., Any]] = None


class BackendCapabilityRegistry:
    """Registry of primitive → backend implementations.

    Thread-safe enough for concurrent subagent registration (single writer
    during build, read-only after seal).
    """

    def __init__(self) -> None:
        self._impls: Dict[str, List[BackendImplementation]] = {}
        self._sealed = False

    def register(self, impl: BackendImplementation) -> None:
        if self._sealed:
            raise RuntimeError("BackendCapabilityRegistry is sealed")
        self._impls.setdefault(impl.operation_id, []).append(impl)

    def implementations(self, operation_id: str) -> List[BackendImplementation]:
        return list(self._impls.get(operation_id, []))

    def has_backend(self, operation_id: str, backend: str) -> bool:
        return any(i.backend == backend for i in self._impls.get(operation_id, []))

    def cuda_ready(self, operation_id: str) -> bool:
        return self.has_backend(operation_id, "cuda")

    def seal(self) -> None:
        self._sealed = True

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for op, impls in self._impls.items():
            out[op] = [
                {
                    "backend": i.backend,
                    "version": i.implementation_version,
                    "hash": i.implementation_hash,
                    "dtypes": list(i.supported_dtypes),
                    "layouts": list(i.supported_layouts),
                    "deterministic": i.deterministic,
                    "exact_semantics": i.exact_semantics,
                    "parity_status": i.parity_status.value,
                }
                for i in impls
            ]
        return out


_REGISTRY = BackendCapabilityRegistry()


def get_backend_capability_registry() -> BackendCapabilityRegistry:
    return _REGISTRY


def register_backend_implementation(impl: BackendImplementation) -> None:
    _REGISTRY.register(impl)


def seal_backend_capability_registry() -> None:
    _REGISTRY.seal()
