"""Backend / precision / GPU execution contracts for quant_evaluator.

These are the canonical runtime contracts introduced by the GPU batch
evaluation refactor (spec §4).  They are CPU/GPU-agnostic and must not import
CuPy at module load so that a CPU-only environment imports cleanly.

Only the coordinator modifies this module (canonical authority surface).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class BackendPolicy(Enum):
    """Execution backend policy (spec §4.1)."""

    AUTO = "auto"
    CPU_REFERENCE = "cpu_reference"
    CPU_FAST = "cpu_fast"
    CUDA = "cuda"
    CUDA_STRICT = "cuda_strict"

    @property
    def is_cuda(self) -> bool:
        return self in (BackendPolicy.CUDA, BackendPolicy.CUDA_STRICT)

    @property
    def strict(self) -> bool:
        return self is BackendPolicy.CUDA_STRICT


class PrecisionPolicy(Enum):
    """Precision policy (spec §4.2)."""

    REFERENCE_FP64 = "reference_fp64"
    GPU_FP64 = "gpu_fp64"
    GPU_MIXED = "gpu_mixed"


@dataclass(frozen=True)
class GPUExecutionPolicy:
    """GPU execution policy (spec §4.3)."""

    device_ids: Tuple[int, ...] = (0,)
    max_vram_fraction: float = 0.75
    pinned_host_memory: bool = True
    async_transfer: bool = True
    double_buffer: bool = True
    precision_policy: PrecisionPolicy = PrecisionPolicy.GPU_MIXED
    oom_retile: bool = True
    strict_backend: bool = False
    required_capabilities: Tuple[str, ...] = ()
    # One worker's materialized results, not the entire queued factor universe.
    max_host_result_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if (isinstance(self.max_host_result_bytes, bool)
                or not isinstance(self.max_host_result_bytes, int)
                or self.max_host_result_bytes <= 0):
            raise ValueError("max_host_result_bytes must be a positive integer")
        if not 0.0 < self.max_vram_fraction <= 1.0:
            raise ValueError(
                f"max_vram_fraction must be in (0,1], got {self.max_vram_fraction}"
            )
        if not self.device_ids:
            raise ValueError("device_ids must be non-empty")
        object.__setattr__(self, 'required_capabilities', tuple(self.required_capabilities))
        if any(x not in ('async_transfer', 'pinned_host_memory', 'double_buffer') for x in self.required_capabilities):
            raise ValueError('unknown mandatory GPU capability')


class DeviceFactorBatch:
    """Runtime-internal device-resident factor batch.

    Public :class:`~quant_evaluator.contracts.factor_batch.FactorBatch` stays
    ``(T, N, F)``; this internal type uses the GPU-friendly ``(T, F_tile, N)``
    layout (spec §4.4).  It is a thin wrapper over a device array plus the
    factor ids it carries.
    """

    __slots__ = ("values", "factor_ids", "layout", "dtype")

    def __init__(self, values, factor_ids, layout: str = "T,F,N", dtype=None):
        self.values = values
        self.factor_ids = tuple(factor_ids)
        self.layout = layout
        self.dtype = dtype

    @property
    def num_factors(self) -> int:
        return len(self.factor_ids)

    def __len__(self) -> int:
        return self.num_factors


class DeviceLabelPanel:
    """Runtime-internal device-resident label panel ``(T, N)`` (spec §4.5).

    Uploaded once and resident for the whole session.
    """

    __slots__ = ("values", "target_id", "dtype")

    def __init__(self, values, target_id: str = "next_ret", dtype=None):
        self.values = values
        self.target_id = target_id
        self.dtype = dtype


@dataclass
class DeviceEvaluationContext:
    """Optional device-resident evaluation context (spec §4.6).

    Only fields actually needed by the requested metrics are uploaded.
    """

    validity: Optional[object] = None
    universe: Optional[object] = None
    tradability: Optional[object] = None
    industry: Optional[object] = None
    size: Optional[object] = None
    beta: Optional[object] = None
    market_cap: Optional[object] = None
    adv: Optional[object] = None
    cost: Optional[object] = None
    calendar: Optional[object] = None

    def required_fields(self) -> Tuple[str, ...]:
        return tuple(
            k
            for k, v in self.__dict__.items()
            if v is not None
        )
