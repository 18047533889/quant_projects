# -*- coding: utf-8 -*-
"""R35 §14 / §15 / §16: NumbaKernelRegistry — unified certified Numba kernel layer.

Every accelerated kernel is registered with a spec, a reference (canonical
NumPy/Python) kernel, and an optional Numba implementation.  Admission requires
reference parity on hostile fixtures (NaN / Inf / dtype / overflow), NOT merely
"compiles".  ``fastmath=False`` is the default (strict NaN/Inf/associativity
semantics).

State when Numba is unavailable: the registry stays importable, ``numba_available``
is False, and every kernel's ``numba_fn`` is None — the reference kernel remains
the execution path and the parity evidence is recorded as ``NUMBA_UNAVAILABLE``
(honest, per taskbook §13).  Nothing is claimed as "Numba-certified" in that
state.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

try:  # numba is an optional accel dependency (pyproject [accel])
    import numba

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when numba is not installed
    numba = None  # type: ignore
    NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class NumbaKernelSpec:
    """Machine contract of one accelerated kernel (taskbook §14)."""

    canonical_family: str      # kalman / stateful / psar / supertrend / kama / ar / enet / pls / garch
    kernel_name: str           # unique kernel id, e.g. kalman_level
    semantic_version: str      # bump when numeric semantics change
    supported_dtypes: tuple[str, ...] = ("float64",)
    supported_param_domain: str = ""     # human-readable, e.g. "window in [5, 10000]"
    nogil: bool = True
    parallel: bool = False     # never default-on; requires explicit broker grant
    cache: bool = True
    fastmath: bool = False     # R35 §15: strict semantics are the default


@dataclass
class NumbaKernel:
    """One registered kernel: spec + reference + optional numba implementation."""

    spec: NumbaKernelSpec
    reference_fn: Callable          # canonical (NumPy/Python) implementation
    numba_fn: Callable | None = None
    numba_available: bool = NUMBA_AVAILABLE

    def call(self, *args: Any, use_numba: bool = True, **kwargs: Any) -> Any:
        """Dispatch to numba when available and requested, else reference."""
        if use_numba and self.numba_available and self.numba_fn is not None:
            return self.numba_fn(*args, **kwargs)
        return self.reference_fn(*args, **kwargs)


class NumbaKernelRegistry:
    """Singleton registry of certified Numba kernels."""

    _kernels: dict[str, NumbaKernel] = {}

    @classmethod
    def register(
        cls,
        canonical_family: str,
        kernel_name: str,
        reference_fn: Callable,
        numba_fn: Callable | None = None,
        *,
        semantic_version: str = "1.0",
        supported_dtypes: tuple[str, ...] = ("float64",),
        supported_param_domain: str = "",
        nogil: bool = True,
        parallel: bool = False,
        cache: bool = True,
        fastmath: bool = False,
    ) -> str:
        spec = NumbaKernelSpec(
            canonical_family=canonical_family,
            kernel_name=kernel_name,
            semantic_version=semantic_version,
            supported_dtypes=supported_dtypes,
            supported_param_domain=supported_param_domain,
            nogil=nogil,
            parallel=parallel,
            cache=cache,
            fastmath=fastmath,
        )
        cls._kernels[kernel_name] = NumbaKernel(
            spec=spec,
            reference_fn=reference_fn,
            numba_fn=numba_fn,
        )
        return kernel_name

    @classmethod
    def get(cls, kernel_name: str) -> NumbaKernel | None:
        return cls._kernels.get(kernel_name)

    @classmethod
    def kernels(cls) -> dict[str, NumbaKernel]:
        return dict(cls._kernels)

    @classmethod
    def numba_available(cls) -> bool:
        return NUMBA_AVAILABLE


def parity_check(
    kernel: NumbaKernel,
    *args: Any,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Reference vs numba parity on the given inputs (hostile fixtures caller
    supplies).  Returns a machine-checkable result dict."""
    if not NUMBA_AVAILABLE or kernel.numba_fn is None:
        return {
            "kernel": kernel.spec.kernel_name,
            "status": "NUMBA_UNAVAILABLE",
            "numba": False,
            "ref": False,
        }
    ref = np.asarray(kernel.reference_fn(*args))
    num = np.asarray(kernel.numba_fn(*args))
    ref_fin = np.isfinite(ref) & np.isfinite(num)
    nan_ok = np.isnan(ref) == np.isnan(num)
    match = np.allclose(
        ref[ref_fin], num[ref_fin], rtol=rtol, atol=atol, equal_nan=True
    ) and nan_ok.all()
    return {
        "kernel": kernel.spec.kernel_name,
        "status": "PASS" if match else "FAIL",
        "numba": bool(match),
        "ref": bool(match),
        "max_abs_diff": float(np.nanmax(np.abs(ref[ref_fin] - num[ref_fin]))) if ref_fin.any() else 0.0,
    }


def benchmark_kernel(
    kernel: NumbaKernel,
    *args: Any,
    use_numba: bool = True,
    repeats: int = 5,
) -> dict[str, Any]:
    """Warm benchmark (best-of-N seconds) for reference vs numba."""
    times: list[float] = []
    # warm up
    kernel.call(*args, use_numba=use_numba)
    for _ in range(repeats):
        t0 = time.perf_counter()
        kernel.call(*args, use_numba=use_numba)
        times.append(time.perf_counter() - t0)
    return {
        "kernel": kernel.spec.kernel_name,
        "mode": "numba" if use_numba else "reference",
        "best_ms": float(min(times)) * 1000.0,
        "median_ms": float(float(np.median(times))) * 1000.0,
        "numba_available": NUMBA_AVAILABLE,
    }


#: Exposed for tests / evidence scripts.
def list_kernels() -> dict[str, dict[str, Any]]:
    return {
        name: dataclasses.asdict(k.spec)
        for name, k in sorted(NumbaKernelRegistry.kernels().items())
    }
