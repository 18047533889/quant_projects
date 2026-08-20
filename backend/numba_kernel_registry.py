# -*- coding: utf-8 -*-
"""R35 §14 / §15 / §16: NumbaKernelRegistry — unified certified Numba kernel layer.

Every accelerated kernel is registered with a spec, a reference (canonical
NumPy/Python) kernel, and an optional Numba implementation.  Admission requires
reference parity on hostile fixtures (NaN / Inf / dtype / overflow), NOT merely
"compiles".  ``fastmath=False`` is the default (strict NaN/Inf/associativity
semantics).

Dedup policy (R21-NUMBA-IMPL-HASH §14.1):
    NumbaKernelImplementationID = SHA256(reference_fn source + numba_fn source + transitive helpers).
    Same spec but different implementation_id → DuplicateKernelRegistrationError.
    Same spec + same implementation_id (idempotent re-registration) → allowed.

State when Numba is unavailable: the registry stays importable, ``numba_available``
is False, and every kernel's ``numba_fn`` is None — the reference kernel remains
the execution path and the parity evidence is recorded as ``NUMBA_UNAVAILABLE``
(honest, per taskbook §13).  Nothing is claimed as "Numba-certified" in that
state.

Production vs Research mode (R21-NUMBA-FAIL-CLOSED):
- Production mode (FACTOR_ENGINE_RUN_MODE=production): if numba unavailable,
  NumbaKernel.call() raises KernelUnavailableError.
- Research mode (default): if numba unavailable, falls back to reference with
  a UserWarning.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class KernelUnavailableError(RuntimeError):
    """Raised when a requested kernel is unavailable (e.g. Numba missing in production)."""

class DuplicateKernelRegistrationError(RuntimeError):
    """Raised when the same kernel_name is registered with a different NumbaKernelImplementationID."""


class DuplicateSpecMismatchError(RuntimeError):
    """Raised when the same kernel_name is registered with a different NumbaKernelSpec."""


try:  # numba is an optional accel dependency (pyproject [accel])
    import numba

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when numba is not installed
    numba = None  # type: ignore
    NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Source hashing helpers
# ---------------------------------------------------------------------------

def _get_source_safely(fn: Callable | None) -> str:
    """Return the source text of fn, or empty string if unavailable (C/builtin/numba-JIT)."""
    if fn is None:
        return ""
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        # numba @njit compiled functions, C extensions, builtins
        return ""


def _resolve_helper_sources(fn: Callable | None) -> list[str]:
    """Collect transitive helper sources referenced in fn's source text.

    Heuristics:
        1. Walk the module's __dict__ for callables whose qualified name
           contains the kernel's module basename (e.g. '_rolling_sum_reference'
           helpers like '_sum_nan_window').
        2. Scan fn's source text for identifiers that are callable and private.
    """
    if fn is None:
        return []
    try:
        mod = inspect.getmodule(fn)
    except Exception:
        return []
    if mod is None:
        return []

    src = _get_source_safely(fn)
    if not src:
        return []

    helpers: list[str] = []
    # Collect callables from the same module that are referenced in fn's source
    for name, obj in vars(mod).items():
        if not callable(obj) or obj is fn:
            continue
        # Heuristic: private helpers in the same module
        if name.startswith("_") and hasattr(obj, "__module__") and obj.__module__ == mod.__name__:
            # Check if name appears in the source text (substring match)
            if name in src:
                helpers.append(_get_source_safely(obj))
    return [h for h in helpers if h]


def _compute_numba_kernel_implementation_id(
    reference_fn: Callable,
    numba_fn: Callable | None,
) -> str:
    """Compute NumbaKernelImplementationID = SHA256(ref src | numba src | helpers).

    Uses only the function body text (ignoring qualname/module) so that two
    functions with identical source produce the same implementation_id.
    Falls back to SHA256 of qualified names when source is unavailable
    (e.g. numba JIT compiled from bytecode).
    """
    ref_src = _get_source_safely(reference_fn)
    numba_src = _get_source_safely(numba_fn)

    # Collect helpers from both modules
    ref_helpers = _resolve_helper_sources(reference_fn)
    numba_helpers = _resolve_helper_sources(numba_fn) if numba_fn is not None else []

    # Build the hash payload (body text only, no qualname)
    parts: list[str] = []

    if ref_src:
        parts.append(f"REF:{ref_src}")
    else:
        parts.append(f"REF_NAME:{reference_fn.__qualname__}")

    if numba_src:
        parts.append(f"NUMBA:{numba_src}")
    elif numba_fn is not None:
        parts.append(f"NUMBA_NAME:{numba_fn.__qualname__}")
    else:
        parts.append("NUMBA:none")

    for i, h in enumerate(ref_helpers):
        parts.append(f"REF_HELPER[{i}]:{h}")
    for i, h in enumerate(numba_helpers):
        parts.append(f"NUMBA_HELPER[{i}]:{h}")

    blob = "\n---\n".join(parts)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"nki:v1:{digest}"


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
    implementation_id: str = ""     # NumbaKernelImplementationID, set at registration

    def call(self, *args: Any, use_numba: bool = True, **kwargs: Any) -> Any:
        """Dispatch to numba when available and requested, else reference.

        In production mode, if numba is requested but unavailable, raise KernelUnavailableError.
        In research mode, fall back to reference with a warning.
        """
        if use_numba and self.numba_available and self.numba_fn is not None:
            return self.numba_fn(*args, **kwargs)

        # Numba requested but unavailable
        if use_numba and not self.numba_available:
            from runtime.production_policy import is_production_mode
            if is_production_mode():
                raise KernelUnavailableError(
                    f"Numba kernel '{self.spec.kernel_name}' unavailable in production mode. "
                    f"Install numba or set FACTOR_ENGINE_RUN_MODE=research."
                )
            else:
                import warnings
                warnings.warn(
                    f"Numba kernel '{self.spec.kernel_name}' unavailable, falling back to reference.",
                    UserWarning,
                    stacklevel=2,
                )
        return self.reference_fn(*args, **kwargs)


class NumbaKernelRegistry:
    """Singleton registry of certified Numba kernels.

    Dedup policy (§14.1):
        - Same kernel_name + same spec + same implementation_id → idempotent, allowed.
        - Same kernel_name + same spec + different implementation_id → DuplicateKernelRegistrationError
          (spec matches but the actual function bodies differ — a silent swap).
        - Same kernel_name + different spec → DuplicateSpecMismatchError.
    """

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

        impl_id = _compute_numba_kernel_implementation_id(reference_fn, numba_fn)

        new_kernel = NumbaKernel(
            spec=spec,
            reference_fn=reference_fn,
            numba_fn=numba_fn,
            implementation_id=impl_id,
        )

        # --- Registry dedup ---
        existing = cls._kernels.get(kernel_name)
        if existing is not None:
            if dataclasses.asdict(existing.spec) != dataclasses.asdict(spec):
                raise DuplicateSpecMismatchError(
                    f"kernel_name={kernel_name!r} already registered with "
                    f"spec={dataclasses.asdict(existing.spec)}; "
                    f"new spec={dataclasses.asdict(spec)} differs"
                )
            if existing.implementation_id != impl_id:
                raise DuplicateKernelRegistrationError(
                    f"kernel_name={kernel_name!r} already registered with "
                    f"implementation_id={existing.implementation_id!r}; "
                    f"new implementation_id={impl_id!r} differs — "
                    f"same spec but different function implementations"
                )

        cls._kernels[kernel_name] = new_kernel
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

    # --- R21-NUMBA-PARITY-INF-FIX: explicit +Inf / -Inf / finite masks ---
    ref_posinf = np.isposinf(ref)
    num_posinf = np.isposinf(num)
    ref_neginf = np.isneginf(ref)
    num_neginf = np.isneginf(num)
    ref_fin = np.isfinite(ref)
    num_fin = np.isfinite(num)

    posinf_ok = bool(np.array_equal(ref_posinf, num_posinf))
    neginf_ok = bool(np.array_equal(ref_neginf, num_neginf))
    finite_mask_ok = bool(np.array_equal(ref_fin, num_fin))
    nan_ok = bool(np.array_equal(np.isnan(ref), np.isnan(num)))
    dtype_ok = ref.dtype == num.dtype
    shape_ok = ref.shape == num.shape

    # When shapes differ, skip the finite-value comparison (it would index-fail).
    if shape_ok and ref_fin.any():
        finite_match = bool(
            np.allclose(ref[ref_fin], num[ref_fin], rtol=rtol, atol=atol)
        )
    elif shape_ok:
        finite_match = True
    else:
        finite_match = False

    match = (
        posinf_ok
        and neginf_ok
        and finite_mask_ok
        and nan_ok
        and dtype_ok
        and shape_ok
        and finite_match
    )

    max_abs_diff = 0.0
    if shape_ok and ref_fin.any():
        max_abs_diff = float(np.nanmax(np.abs(ref[ref_fin] - num[ref_fin])))

    return {
        "kernel": kernel.spec.kernel_name,
        "status": "PASS" if match else "FAIL",
        "numba": bool(match),
        "ref": bool(match),
        "posinf_ok": posinf_ok,
        "neginf_ok": neginf_ok,
        "finite_mask_ok": finite_mask_ok,
        "nan_ok": nan_ok,
        "dtype_ok": dtype_ok,
        "shape_ok": shape_ok,
        "finite_match": finite_match,
        "max_abs_diff": max_abs_diff,
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


def get_implementation_registry() -> dict[str, dict[str, Any]]:
    """Return implementation IDs for all registered kernels (evidence/audit)."""
    return {
        name: {
            "kernel_name": k.spec.kernel_name,
            "implementation_id": k.implementation_id,
            "semantic_version": k.spec.semantic_version,
        }
        for name, k in sorted(NumbaKernelRegistry.kernels().items())
    }
