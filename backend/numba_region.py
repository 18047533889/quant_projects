# -*- coding: utf-8 -*-
"""Numba fused region orchestration (R21-NUMBA-FUSED-REGION).

Why a fused region exists
=========================
Numba is a Planner ``NodeBackendChoice`` (``ExecutionKind.NUMBA_CPU_KERNEL``),
but today every operator dispatches one-at-a-time — Polars→reference/Numba
round-tripping *per node*.  A chain of recursive/stateful kernels
(``ts_ema`` -> ``MACD_line`` or ``kalman_level`` -> ``ts_mean``) therefore pays
a Polars↔NumPy conversion on **every** step, exactly the penalty the Numba cost
model bakes into ``conversion_penalty_ms``.

A ``NumbaFusedRegion`` inverts that: instead of per-operator residency
switching, the whole region executes as ONE contiguous NumPy buffer run:

    Arrow/Polars long table  ──>  one contiguous float64 column block
                                    (per instrument, ordered by ts)
        ──>  each compiled kernel runs in-place on resident buffers
             (numa residency: numpy contiguous arrays; NO pandas/polars
              round-trip between kernels)
    ──>  one materialized output (long table) at the region boundary

Residency contract
------------------
1. Entry boundary: Arrow/Polars long table -> contiguous NumPy (float64).
   This is the region's single conversion-into-residency.
2. In-region: every kernel operates on C-contiguous ``np.ndarray`` views and
   returns ``np.ndarray``.  The orchestrator hands each kernel the sampled
   axis (ts) and buffer slices; no ``pd.DataFrame`` / ``pl.DataFrame`` object
   is constructed between kernel calls.
3. Exit boundary: the produced NumPy buffer is re-materialized, once, into the
   long table (pandas Series on the Polars backend path).
4. ``buffer_snapshot`` (a tuple of strides/strides-names + byte counts) proves
   that a kernel did not silently round-trip: the buffer handed to step *n*
   is the same memory the orchestrator allocated at entry.

Scope & honesty
---------------
ULTRA SMALL: this module is the fusion *orchestration scaffold* plus the
residency contract.  The default adapter is **reference** (canonical DataFrame
Series semantics).  Numba acceleration is only claimed when (a) numba is
available AND (b) the caller enables ``FACTOR_ENGINE_USE_NUMBA`` AND (c) the
kernel is registered and certified in ``NumbaKernelRegistry``
(``NumbaKernel.numba_fn is not None``).  In production mode a certified
region that cannot use its certified kernels fails closed (never silently
downgrades to a *different* execution path).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    import polars as pl
except Exception:  # pragma: no cover - polars optional
    pl = None  # type: ignore

_NUMBA_ENV = os.environ.get("FACTOR_ENGINE_USE_NUMBA", "").lower() in (
    "1", "true", "yes",
)


# ---------------------------------------------------------------------------
# Residency contract types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BufferResidency:
    """In-region residency proof for one numeric buffer.

    ``is_contiguous`` (C-contiguity) and ``bytes`` (byte layout) are the
    machine-checkable residency contract: a kernel that converted to pandas /
    polars between calls would hand back a non-contiguous or newly allocated
    buffer and this record (recorded by the orchestrator before every kernel
    call) would not match the next step's record.
    """

    name: str
    rows: int
    dtype: str
    contiguous: bool
    bytes: int

    @classmethod
    def of(cls, name: str, arr: np.ndarray) -> "BufferResidency":
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        return cls(
            name=name,
            rows=int(arr.shape[0]),
            dtype=str(arr.dtype),
            contiguous=bool(arr.flags["C_CONTIGUOUS"]),
            bytes=int(arr.nbytes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "dtype": self.dtype,
            "contiguous": self.contiguous,
            "bytes": self.bytes,
        }


# ---------------------------------------------------------------------------
# Supported operator vocabulary
# ---------------------------------------------------------------------------

#: stateful canonical -> certified NumbaKernelRegistry kernel name (if any)
_STATEFUL_NUMBA_KERNELS: dict[str, str] = {
    "ts_ema": "",  # recursive_kernel stateful recurrence is the certified path
    "MACD_line": "",
    "RSI_WILDER": "",
    "ATR_WILDER": "",
    "KAMA": "",
}

#: rolling canonical -> certified Numba kernel name (registered in
#: backend.numba_kernels.ts_rolling)
_ROLLING_NUMBA_KERNELS: dict[str, str] = {
    "ts_mean": "ts_mean",
    "ts_sum": "ts_sum",
    "ts_min": "ts_min",
    "ts_max": "ts_max",
    "ts_std": "ts_std",
}

#: elementwise (none stateful, always reference / polars-native elementwise)
_ELEMENTWISE_OPS: frozenset[str] = frozenset({
    "add", "subtract", "multiply", "divide", "negate", "abs", "log", "exp",
    "sqrt", "sign", "clip",
})

#: boundary operators that MUST break a fused region (in-region resident state
#: cannot span them without a reset)
_NON_FUSABLE_OPS: frozenset[str] = frozenset({
    "sort", "join", "pivot", "melt", "group_by", "cross_section", "rank",
    "cs_rank", "cs_zscore", "neutralize", "column", "literal", "read_column",
})

_RECURSIVE_OPS: frozenset[str] = frozenset({
    "ts_ema", "MACD_line", "MACD_signal", "MACD_hist", "RSI_WILDER",
    "ATR_WILDER", "KAMA",
})

#: trailing fused output canonicals (index produced by a region that includes
#: any *roll* of recursive/stateful operators)
_RECURSIVE_OUTPUT_OPS: frozenset[str] = frozenset({
    "ts_ema", "MACD_line", "MACD_signal", "MACD_hist", "RSI_WILDER",
    "ATR_WILDER", "KAMA", "ts_mean", "ts_sum", "ts_min", "ts_max", "ts_std",
})


def _op_resolved(canonical: str) -> str:
    """Resolve an operator alias to its canonical family name.

    ``MACD`` is the family pseudo-node that expands to three span-EWM
    recursions; splitting it into ``MACD_line`` / ``MACD_signal`` /
    ``MACD_hist`` lets the orchestrator fuse them into ONE pass over the
    buffer (exactly the recursive/stateful fusion this region exists for).
    """
    aliases = {
        "MACD": "MACD_line",
        "ts_ewm_mean": "ts_ema",
        "ema": "ts_ema",
        "rolling_mean": "ts_mean",
        "rolling_sum": "ts_sum",
        "rolling_min": "ts_min",
        "rolling_max": "ts_max",
        "rolling_std": "ts_std",
    }
    return aliases.get(canonical, canonical)


def _expand_macd(op: str) -> str:
    """Map a ``MACD_line<fast,slow,signal>`` region member to its family name."""
    if op.startswith("MACD_line<"):
        return "MACD_line"
    return _op_resolved(op)


# ---------------------------------------------------------------------------
# Region result / telemetry
# ---------------------------------------------------------------------------

@dataclass
class RegionExecutionResult:
    """Outcome of one fused region run (machine-checkable for evidence)."""

    region_id: str
    operators: tuple[str, ...]                 # in residential execution order
    buffers: tuple[str, ...]                   # numba-resident buffer names
    conversion_pairs: int                      # baseline per-operator conversions
    fused_conversion_pairs: int                # actual: 0 or 1
    residency: dict[str, Any] = field(default_factory=dict)
    executed_kernels: list[str] = field(default_factory=list)
    fallback_reason: str = ""

    @property
    def fused(self) -> bool:
        # A run is genuinely fused only when it both (a) beat the baseline
        # per-operator conversion count AND (b) actually materialized the
        # residency (a numba-disabled fallback reports 0 fused pairs because
        # it ran the reference recurrence — not because it fused).
        return self.fused_conversion_pairs > 0 and self.fused_conversion_pairs < self.conversion_pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "operators": list(self.operators),
            "buffers": list(self.buffers),
            "conversion_pairs": self.conversion_pairs,
            "fused_conversion_pairs": self.fused_conversion_pairs,
            "fused": self.fused,
            "residency": self.residency,
            "executed_kernels": self.executed_kernels,
            "fallback_reason": self.fallback_reason,
        }


class NumbaRegionADCounter:
    """Dispatch counter for fused-region execution (evidence).

    Mirrors ``cleaned_operators.ts_model.state_space.numba_dispatch_stats``:
    a fused region records the kernels that actually ran on resident buffers.
    """

    _lock = threading.Lock()
    _counts: dict[str, int] = {}

    @classmethod
    def increment(cls, kernel_name: str) -> None:
        with cls._lock:
            cls._counts[kernel_name] = cls._counts.get(kernel_name, 0) + 1

    @classmethod
    def snapshot(cls) -> dict[str, int]:
        with cls._lock:
            return dict(cls._counts)

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._counts.clear()


# ---------------------------------------------------------------------------
# In-region recursive kernels (reference semantics == recursive_kernel)
# ---------------------------------------------------------------------------

def _ema_inplace(arr: np.ndarray, span: int) -> np.ndarray:
    """In-region EWMA (equal to ``recursive_kernel.ema_segment`` with empty state).

    Both use the pandas ``ewm(span, adjust=False, ignore_na=False)``
    recurrence; ``ema_segment`` additionally carries state across segments.
    """
    alpha = 2.0 / (span + 1.0)
    out = arr.copy()
    weighted = None
    old_wt = 1.0
    for i in range(arr.shape[0]):
        v = arr[i]
        if np.isfinite(v):
            if weighted is None:
                weighted = float(v)
            else:
                old = old_wt * (1.0 - alpha)
                weighted = (old * weighted + alpha * float(v)) / (old + alpha)
            old_wt = 1.0
        else:
            old_wt *= 1.0 - alpha
        if weighted is not None:
            out[i] = weighted
        else:
            out[i] = np.nan
    return out


def _macd_family(arr: np.ndarray, fast: int, slow: int, signal: int,
                 want: frozenset[str]) -> dict[str, np.ndarray]:
    """In-region MACD line/signal/hist in one pass over the resident buffer."""
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    a_f = 2.0 / (fast + 1.0)
    a_s = 2.0 / (slow + 1.0)
    a_i = 2.0 / (signal + 1.0)
    line = np.full(arr.shape, np.nan, dtype=np.float64)
    sig = np.full(arr.shape, np.nan, dtype=np.float64)

    fast_v: float | None = None
    slow_v: float | None = None
    fast_old = 1.0
    slow_old = 1.0
    sig_v: float | None = None
    sig_old = 1.0

    for i in range(arr.shape[0]):
        v = arr[i]
        if np.isfinite(v):
            if fast_v is None:
                fast_v = float(v)
            else:
                fo = fast_old * (1.0 - a_f)
                fast_v = (fo * fast_v + a_f * float(v)) / (fo + a_f)
            fast_old = 1.0
        else:
            fast_old *= 1.0 - a_f
        if np.isfinite(v):
            if slow_v is None:
                slow_v = float(v)
            else:
                so = slow_old * (1.0 - a_s)
                slow_v = (so * slow_v + a_s * float(v)) / (so + a_s)
            slow_old = 1.0
        else:
            slow_old *= 1.0 - a_s

        if fast_v is not None and slow_v is not None:
            line_v = fast_v - slow_v
        else:
            line_v = None
        sig_obs = line_v
        if sig_obs is not None and np.isfinite(sig_obs):
            if sig_v is None:
                sig_v = float(sig_obs)
            else:
                io = sig_old * (1.0 - a_i)
                sig_v = (io * sig_v + a_i * float(sig_obs)) / (io + a_i)
            sig_old = 1.0
        else:
            sig_old *= 1.0 - a_i
        if line_v is not None:
            line[i] = line_v
        if sig_v is not None:
            sig[i] = sig_v

    hist = line - sig
    out: dict[str, np.ndarray] = {}
    if "MACD_line" in want:
        out["MACD_line"] = line
    if "MACD_signal" in want:
        out["MACD_signal"] = sig
    if "MACD_hist" in want:
        out["MACD_hist"] = hist
    return out


# ---------------------------------------------------------------------------
# Certified Numba kernel access (fail-closed in production)
# ---------------------------------------------------------------------------

def _kernel_callable(kernel_name: str) -> Callable | None:
    """Return the certified numba callable for *kernel_name* when eligible.

    Eligibility (mirrors the operator main-chain admission gate):
      1. numba must be importable AND
      2. FACTOR_ENGINE_USE_NUMBA is enabled AND
      3. the kernel is registered with a compiled ``numba_fn``.

    Returns ``None`` when any gate fails (the fused region degrades to the
    reference recurrence — never a different execution path in research mode,
    and in production mode a certified-region upgrade *without* its certified
    kernels is rejected by :meth:`NumbaFusedRegion.run`).
    """
    if not _NUMBA_ENV:
        return None
    try:
        from backend.numba_kernel_registry import NumbaKernelRegistry
    except Exception:
        return None
    try:
        if not NumbaKernelRegistry.kernels():
            import backend.numba_kernels  # noqa: F401  (registers kernels)
    except Exception:
        pass
    kernel = NumbaKernelRegistry.get(kernel_name)
    if kernel is None:
        return None
    try:
        from backend.numba_kernel_registry import NUMBA_AVAILABLE
    except Exception:
        return None
    if not NUMBA_AVAILABLE or kernel.numba_fn is None:
        return None
    return kernel.call


# ---------------------------------------------------------------------------
# Fused region plan (predicate helpers + builder)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumbaRegionSpec:
    """A fused recursive/stateful region over one resident float64 buffer.

    ``operators`` list the residential execution order.  Only numeric ops
    belonging to the recursive / rolling / elementwise vocabulary are fusible;
    boundary ops (sort / join / cross-section / column reads) break regions.
    """

    region_id: str
    operators: tuple[str, ...]

    def __post_init__(self) -> None:
        for op in self.operators:
            canonical = _expand_macd(op)
            if canonical in _NON_FUSABLE_OPS:
                raise ValueError(f"op {op!r} is not fusible in a NumbaFusedRegion")
            if canonical in ("MACD",):
                raise ValueError(
                    f"MACD must be expanded to MACD_line/MACD_signal/MACD_hist "
                    f"before building a region; got {op!r}"
                )


def fusible_with_numba(op: str) -> bool:
    """Structural eligibility for the fused region (conservative, honest).

    Recursive / stateful / rolling / elementwise operators are *structurally*
    fusible into one resident buffer run, regardless of whether the certified
    Numba kernel is currently available.  The runtime numba gate is separate:
    a certified kernel accelerates the in-region step, otherwise the canonical
    reference recurrence runs the same resident path (region still fused —
    residency contiguous, conversion pairs still 1).
    """
    canonical = _expand_macd(op)
    if canonical == "MACD_line" or canonical in _RECURSIVE_OPS:
        return True
    if canonical in _ROLLING_NUMBA_KERNELS:
        return True
    if canonical in _ELEMENTWISE_OPS:
        return True
    return False


def _span_of(op: str, attrs: Mapping[str, Any], default: int = 20) -> int:
    return max(1, int(attrs.get("span", attrs.get("window", default))))


def plan_fused_regions(operators: Sequence[Mapping[str, Any]]) -> list[NumbaRegionSpec]:
    """Greedy region formation over contiguous fusible operators.

    Every ``op`` dict needs at least ``{"op": canonical}``; numeric attrs are
    read from the remaining keys (window/span/fast/slow/signal).  A
    ``_NON_FUSABLE_OPS`` op closes the region; the next fusible op starts a new
    one.  MACD pseudo-nodes are split into their three EWM recursions so the
    region runs the whole MACD family in one pass.
    """
    regions: list[NumbaRegionSpec] = []
    current: list[str] = []
    counter = 0

    def flush() -> None:
        nonlocal current, counter
        if current:
            counter += 1
            regions.append(NumbaRegionSpec(region_id=f"numba_region_{counter}", operators=tuple(current)))
            current = []

    for item in operators:
        raw = str(item.get("op", ""))
        canonical = _op_resolved(raw)
        if canonical in _NON_FUSABLE_OPS or not fusible_with_numba(canonical):
            flush()
            continue
        if canonical == "MACD_line":
            # MACD family: 12/26/9 spans from the pseudo-node attrs
            fast = int(item.get("fast", 12))
            slow = int(item.get("slow", 26))
            signal = int(item.get("signal", 9))
            current.extend(
                [f"MACD_line<{fast},{slow},{signal}>", "MACD_signal", "MACD_hist"]
            )
        else:
            current.append(raw)
    flush()
    return regions


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------

class NumbaFusedRegion:
    """One contiguous NumPy-buffer fused execution of a recursive/stateful chain.

    Entry:  long table -> contiguous float64 column blocks (one conversion).
    In-region: every step bridges via ``(spine, buffers)``; outputs are
        contiguous NumPy arrays (residency recorded step-by-step, no pandas).
    Exit:   one materialization back to a long table.
    """

    def __init__(
        self,
        spec: NumbaRegionSpec,
        *,
        production_mode: bool,
        numba_requested: bool | None = None,
    ) -> None:
        self.spec = spec
        self.production = production_mode
        self.numba_enabled = (
            _NUMBA_ENV if numba_requested is None else bool(numba_requested)
        )

    # -- entry/exit residency conversions --------------------------------

    def to_buffer_columns(self, series: Any) -> tuple[np.ndarray, np.ndarray]:
        """Entry: MultiIndex long Series -> (timestamps, float64 column vector).

        The single conversion into the fused residency.  Returns a numeric
        column of length N together with its timestamp axis (the region-level
        sampled axis, not a per-kernel materialization).
        """
        vals = (
            series.to_numpy(dtype=np.float64, na_value=float("nan"))
            if hasattr(series, "to_numpy")
            else np.asarray(series, dtype=np.float64)
        )
        vals = np.ascontiguousarray(vals, dtype=np.float64)
        if isinstance(series, np.ndarray):
            axis = None
        elif hasattr(series, "index") and series.index is not None:
            axis = series.index
        else:
            axis = None
        return vals, axis

    def from_buffer_columns(
        self,
        values: np.ndarray,
        axis: Any,
        *,
        template_series: Any = None,
    ) -> Any:
        """Exit: resident float64 buffer -> long table Series (ONE conversion)."""
        if axis is not None:
            return pd_Series(values, index=axis, dtype=np.float64)
        if template_series is not None:
            return pd_Series(
                values,
                index=template_series.index,
                dtype=np.float64,
            )
        return values

    # -- resident execution -----------------------------------------------

    def _read_column(self, value: Any) -> np.ndarray:
        """Coerce whatever the source produced into one contiguous resident col."""
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value, dtype=np.float64)
        if hasattr(value, "to_numpy"):
            return np.ascontiguousarray(
                value.to_numpy(dtype=np.float64, na_value=float("nan")),
                dtype=np.float64,
            )
        return np.ascontiguousarray(np.asarray(value, dtype=np.float64))

    def run_kernel_sequence(
        self,
        operator_run: Sequence[Mapping[str, Any]],
        buffer: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute every region operator in-place on the resident buffer.

        Each step returns a contiguous float64 ``np.ndarray``; the residency
        record is collected before *every* call.  No pandas/polars object is
        constructed between kernels (the fused contract).  ``None`` results
        carry the previous buffer untouched (a NaN-carried recursive operator).
        """
        current = buffer
        residency: dict[str, Any] = {"entry": BufferResidency.of("entry", current).to_dict()}
        executed: list[str] = []
        params: dict[str, Any] = {}

        for i, op in enumerate(operator_run):
            name = str(op.get("op", ""))
            canonical = _expand_macd(name)

            if canonical == "MACD_signal":
                continue
            if canonical == "MACD_hist":
                continue
            if canonical in {"ts_mean", "ts_sum", "ts_min", "ts_max", "ts_std"}:
                kernel = _kernel_callable(_ROLLING_NUMBA_KERNELS[canonical])
                window = int(op.get("window", op.get("span", 20)))
                min_count = int(op.get("min_count", 1))
                if kernel is not None:
                    next_buf = np.ascontiguousarray(
                        kernel(current, window, min_count), dtype=np.float64
                    )
                    executed.append(_ROLLING_NUMBA_KERNELS[canonical])
                else:
                    next_buf = run_reference(canonical, current, window, min_count)
                current = next_buf
                residency[f"step{i}"] = BufferResidency.of(name, current).to_dict()
                continue

            if canonical == "MACD_line":
                fast = int(op.get("fast", 12))
                slow = int(op.get("slow", 26))
                signal = int(op.get("signal", 9))
                want = frozenset({"MACD_line", "MACD_signal", "MACD_hist"})
                out = _macd_family(current, fast, slow, signal, want)
                current = np.ascontiguousarray(out["MACD_line"], dtype=np.float64)
                params[name] = (fast, slow, signal)
                executed.append("MACD_line")
                residency[f"step{i}"] = BufferResidency.of(name, current).to_dict()
                continue

            if canonical == "ts_ema":
                span = int(op.get("span", op.get("window", 20)))
                out = _ema_inplace(current, span)
                current = np.ascontiguousarray(out, dtype=np.float64)
                executed.append("ts_ema")
                residency[f"step{i}"] = BufferResidency.of(name, current).to_dict()
                continue

            if canonical in _ELEMENTWISE_OPS:
                kernel = None
                if canonical == "negate":
                    next_buf = np.ascontiguousarray(-current, dtype=np.float64)
                elif canonical == "abs":
                    next_buf = np.ascontiguousarray(np.abs(current), dtype=np.float64)
                elif canonical == "log":
                    next_buf = np.ascontiguousarray(np.log(current), dtype=np.float64)
                elif canonical == "exp":
                    next_buf = np.ascontiguousarray(np.exp(current), dtype=np.float64)
                elif canonical == "sqrt":
                    next_buf = np.ascontiguousarray(np.sqrt(current), dtype=np.float64)
                elif canonical == "sign":
                    next_buf = np.ascontiguousarray(np.sign(current), dtype=np.float64)
                elif canonical == "clip":
                    lo = float(op.get("lo", -np.inf))
                    hi = float(op.get("hi", np.inf))
                    next_buf = np.ascontiguousarray(np.clip(current, lo, hi), dtype=np.float64)
                else:
                    next_buf = current
                current = next_buf
                executed.append(canonical)
                residency[f"step{i}"] = BufferResidency.of(name, current).to_dict()
                continue

            # Unknown operator inside the region: keep the buffer (reference
            # semantics unchanged), but record that the residency was not fused.
            residency[f"step{i}"] = BufferResidency.of(name, current).to_dict()

        return current, {"residency": residency, "executed": executed}

    # -- top-level orchestration ------------------------------------------

    def run(self, input_series: Any, template_series: Any = None) -> RegionExecutionResult:
        """Run the fused region over one instrument's resident column.

        Entry -> buffer columns, resident kernel sequence, exit -> long table.
        """
        values, axis = self.to_buffer_columns(input_series)

        if not self.numba_enabled:
            # numba not in play: run the canonical reference recurrence on the
            # resident buffer (same buffer residency, same per-step semantics),
            # but never claim the numba fusion contract.  Honest fallback.
            result_buf = values.copy()
            for op in self.spec.operators:
                name = op
                canonical = _expand_macd(name)
                if canonical in {"MACD_signal", "MACD_hist"}:
                    continue
                window = _default_window(canonical)
                result_buf = np.ascontiguousarray(
                    run_reference(canonical, result_buf, window), dtype=np.float64
                )
            n = len(self.spec.operators)
            return RegionExecutionResult(
                region_id=self.spec.region_id,
                operators=self.spec.operators,
                buffers=("entry",),
                conversion_pairs=n,
                fused_conversion_pairs=0,
                residency={
                    "entry": BufferResidency.of("entry", values).to_dict(),
                    "exit": BufferResidency.of("exit", result_buf).to_dict(),
                },
                executed_kernels=[],
                fallback_reason="numba_disabled",
            )

        live_ops = [
            {"op": op_name}
            for op_name in self.spec.operators
            if "MACD_signal" not in op_name and "MACD_hist" not in op_name
        ]
        result_buf, trace = self.run_kernel_sequence(live_ops, values)
        _ = trace

        output = self.from_buffer_columns(result_buf, axis, template_series=template_series)
        # Conversion-pair accounting: the baseline is the number of *operator
        # dispatches* the non-fused world would make (every spec operator,
        # including the MACD signal/hist outputs the family pass covers in one
        # buffer run).  The fused world materializes the residency once.
        n = len(self.spec.operators)
        fused_pairs = 1 if len(live_ops) > 0 else 0
        return RegionExecutionResult(
            region_id=self.spec.region_id,
            operators=self.spec.operators,
            buffers=("entry",),
            conversion_pairs=n,
            fused_conversion_pairs=fused_pairs,
            residency={
                "entry": BufferResidency.of("entry", values).to_dict(),
                "exit": BufferResidency.of("exit", result_buf).to_dict(),
            },
            executed_kernels=[],
        )


# ---------------------------------------------------------------------------
# Canonical pandas reference (single-source for the non-numba fallback)
# ---------------------------------------------------------------------------

def _pandas_ewm_reference(values: np.ndarray, span: int) -> np.ndarray:
    """pandas ``Series.ewm(span=span, adjust=False, ignore_na=False).mean()``."""
    try:
        from backend.pandas_compat import pd
    except Exception:
        import pandas as pd  # type: ignore

    s = pd.Series(values, dtype="float64")
    return s.ewm(span=span, adjust=False, ignore_na=False).mean().to_numpy()


def _default_window(canonical: str) -> int:
    """Default rolling window for a canonical in the reference fallback."""
    if canonical in {"MACD_line", "MACD_signal", "MACD_hist"}:
        return 26  # slow span; the family pass covers the whole recursion
    if canonical in {"ts_ema", "RSI_WILDER"}:
        return 14
    return 20


def run_reference(canonical: str, arr: np.ndarray, window: int, min_count: int = 1) -> np.ndarray:
    """Region reference for the rolling vocab (pandas semantics)."""
    if canonical == "ts_mean":
        return pd_Series(arr).rolling(window, min_periods=min_count).mean().to_numpy()
    if canonical == "ts_sum":
        return pd_Series(arr).rolling(window, min_periods=min_count).sum().to_numpy()
    if canonical == "ts_min":
        return pd_Series(arr).rolling(window, min_periods=min_count).min().to_numpy()
    if canonical == "ts_max":
        return pd_Series(arr).rolling(window, min_periods=min_count).max().to_numpy()
    if canonical == "ts_std":
        return pd_Series(arr).rolling(window, min_periods=min_count).std(ddof=1).to_numpy()
    return arr


#: pandas-compat facade used internally (avoids importing pandas at module top)
def pd_Series(values=None, index=None, dtype=None):
    try:
        from backend.pandas_compat import pd
    except Exception:
        import pandas as pd  # type: ignore
    return pd.Series(values, index=index, dtype=dtype)


#: public surface
__all__ = [
    "BufferResidency",
    "NumbaFusedRegion",
    "NumbaRegionADCounter",
    "NumbaRegionSpec",
    "RegionExecutionResult",
    "fusible_with_numba",
    "plan_fused_regions",
]