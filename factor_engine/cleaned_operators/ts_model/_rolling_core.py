# -*- coding: utf-8 -*-
"""Shared rolling regression kernels for time-series model operators."""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from threading import Lock, local
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata

# Thread-parallel execution of the batched kernels' per-window loops.  The
# per-window computations (LAPACK solves, HiGHS LPs, dgemv certificates) are
# mutually independent and free of shared mutable state (the HiGHS instance /
# LP-skeleton cache are thread-local), so the results are bit-identical to the
# sequential order regardless of scheduling.
_MAX_WORKER_THREADS = min(8, __import__("os").cpu_count() or 1)
_PARALLEL_MIN_ITEMS = 64
_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR is None:
                _EXECUTOR = ThreadPoolExecutor(
                    max_workers=_MAX_WORKER_THREADS,
                    thread_name_prefix="rolling_core")
    return _EXECUTOR


def _parallel_for(items: np.ndarray, fn: Callable[[int], bool]) -> bool:
    """Run ``fn(item)`` for every item, possibly across worker threads.

    ``fn`` returns False to flag a failure (e.g. LAPACK raised); the caller
    then falls back exactly like the sequential loop would.  Sequential for
    small item counts (thread hand-off would dominate) and inside process
    workers (the GIL-bound numpy call wrappers do not overlap there).
    """
    n = len(items)
    if (_IN_PROC_WORKER or n < _PARALLEL_MIN_ITEMS
            or _MAX_WORKER_THREADS <= 1):
        for i in items:
            if not fn(int(i)):
                return False
        return True
    ex = _get_executor()
    bounds = np.linspace(0, n, _MAX_WORKER_THREADS + 1).astype(int)

    def _run(lo: int, hi: int) -> bool:
        for j in range(lo, hi):
            if not fn(int(items[j])):
                return False
        return True

    futures = [ex.submit(_run, lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
    return all(f.result() for f in futures)


# Process-level parallelism for the batched kernels.  The numpy call wrappers
# (np.linalg.lstsq / highspy) hold the GIL for a large share of their runtime,
# which caps thread-level speedup at ~2x; worker processes remove that cap.
# Every step inside the batched kernels is per-window (all reductions run
# along the window axis), so a contiguous axis-0 chunk returns bit-identical
# results to the full-batch call — chunking cannot change any coefficient.
_IN_PROC_WORKER = False
_PROC_MIN_ITEMS = 128
# ``RC_PROC_WORKERS`` overrides the worker count; ``RC_PROC_WORKERS=0``
# disables the process path entirely (pure sequential/threads).
_PROC_MAX_WORKERS = (int(__import__("os").environ["RC_PROC_WORKERS"])
                     if __import__("os").environ.get("RC_PROC_WORKERS")
                     else min(12, __import__("os").cpu_count() or 1))
_PROC_POOL = None
_PROC_POOL_LOCK = Lock()
_PROC_POOL_BROKEN = False


def _get_proc_pool():
    global _PROC_POOL
    if _PROC_POOL is None:
        with _PROC_POOL_LOCK:
            if _PROC_POOL is None:
                import multiprocessing as _mp
                from concurrent.futures import ProcessPoolExecutor as _PPE
                # ``forkserver``: worker processes are forked from a clean,
                # single-threaded server — never from the engine's worker
                # threads (fork there risks inheriting held locks).  The
                # kernel module is preloaded into the server so every forked
                # worker starts with it already imported (COW-shared).
                _mp.set_forkserver_preload(
                    ["factor_engine.cleaned_operators.ts_model._rolling_core"])
                # Workers must not re-execute the caller's main module: the
                # server already imported it, and forked workers inherit that
                # state.  Stripping the fixup keys avoids re-running main().
                _orig_gpd = _mp.spawn.get_preparation_data

                def _gpd_no_main(*a, **k):
                    d = _orig_gpd(*a, **k)
                    d.pop("init_main_from_path", None)
                    d.pop("init_main_from_name", None)
                    return d

                _mp.spawn.get_preparation_data = _gpd_no_main
                ctx = _mp.get_context("forkserver")
                _PROC_POOL = _PPE(max_workers=_PROC_MAX_WORKERS, mp_context=ctx)
    return _PROC_POOL


def _batch_kernel_chunk(kernel: Callable[..., tuple], args: tuple,
                        kwargs: dict) -> tuple:
    """Worker-side entry: run ``kernel`` on one axis-0 chunk of the stack."""
    global _IN_PROC_WORKER
    _IN_PROC_WORKER = True
    try:
        return kernel(*args, **kwargs)
    finally:
        _IN_PROC_WORKER = False


def _proc_fanout(kernel: Callable[..., tuple], args: tuple,
                 kwargs: dict) -> tuple | None:
    """Split the stacked inputs' axis 0 across worker processes.

    Returns ``(betas, ok)`` reassembled in order, or ``None`` when process
    execution is unavailable — the caller then runs its normal path, so any
    failure here degrades to the sequential result (never a wrong one).
    """
    import math
    global _PROC_POOL_BROKEN
    if _PROC_POOL_BROKEN:
        return None
    try:
        K = int(args[0].shape[0])
        nchunk = min(_PROC_MAX_WORKERS, math.ceil(K / _PROC_MIN_ITEMS))
        if nchunk < 2:
            return None
        pool = _get_proc_pool()
        bounds = np.linspace(0, K, nchunk + 1).astype(int)
        futures = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            chunk = tuple(
                a[lo:hi] if isinstance(a, np.ndarray) and a.ndim >= 1
                and a.shape[0] == K else a       # scalars pass through
                for a in args)
            futures.append(pool.submit(_batch_kernel_chunk, kernel, chunk,
                                       dict(kwargs)))
        betas_parts, ok_parts = [], []
        for f in futures:
            b, o = f.result()
            betas_parts.append(np.asarray(b, dtype=float))
            ok_parts.append(np.asarray(o, dtype=bool))
    except Exception:
        _PROC_POOL_BROKEN = True
        return None
    return (np.concatenate(betas_parts, axis=0),
            np.concatenate(ok_parts, axis=0))

try:  # HiGHS LP for true pinball-loss quantile regression.
    from scipy.optimize import linprog as _linprog
except Exception:  # pragma: no cover
    _linprog = None

# Thread-local direct-HiGHS state (see :func:`_highs_solve_eq`).  Each thread
# reuses one configured ``_Highs`` instance and one CSC skeleton per
# ``(n, p, q)`` model shape; the per-window numerical payload is refreshed in
# place.  Model and options are byte-for-byte what
# ``linprog(..., method="highs", options={"maxiter": 10000, "time_limit": 5.0})``
# hands to HiGHS via ``_linprog_highs``/``_highs_wrapper``, so the returned
# primal / dual / objective match the legacy path bit for bit.
_HIGHS_TLS = local()


def _highs_solve_eq(z: np.ndarray, yn: np.ndarray, q: float):
    """Solve ``min q·Σu + (1-q)·Σv  s.t.  z·γ + u − v = yn, u,v ≥ 0``.

    Returns ``(x, row_dual, fun)`` for the optimal solve (HiGHS ``kOptimal``)
    or ``None`` on any failure — the caller maps failure to the legacy
    fail-closed ``non_converged`` path exactly as a non-success ``linprog``
    result would be.
    """
    core = getattr(_HIGHS_TLS, "core", None)
    if core is None:
        try:
            from scipy.optimize._highspy import _core as _hc
        except Exception:  # pragma: no cover
            return None
        core = _hc
        _HIGHS_TLS.core = core
        _HIGHS_TLS.solver = None
        _HIGHS_TLS.lp_cache = {}
    try:
        n, p = z.shape
        cache = _HIGHS_TLS.lp_cache.get((n, p, q))
        if cache is None:
            nnz = n * p + 2 * n
            indptr = np.empty(p + 2 * n + 1, dtype=np.int32)
            indices = np.empty(nnz, dtype=np.int32)
            data_const = np.empty(nnz, dtype=float)
            indptr[0] = 0
            for j in range(p):
                indices[j * n:(j + 1) * n] = np.arange(n)
                indptr[j + 1] = (j + 1) * n
            for k in range(n):
                col = p + k
                indptr[col + 1] = indptr[col] + 1
                indices[indptr[col]] = k
                data_const[indptr[col]] = 1.0
            for k in range(n):
                col = p + n + k
                indptr[col + 1] = indptr[col] + 1
                indices[indptr[col]] = k
                data_const[indptr[col]] = -1.0
            inf = float(core.kHighsInf)
            lb = np.concatenate([np.full(p, -inf), np.zeros(2 * n)])
            ub = np.full(p + 2 * n, inf)
            cost = np.concatenate(
                [np.zeros(p), np.full(n, q), np.full(n, 1.0 - q)])
            cache = (indptr, indices, data_const, lb, ub, cost)
            _HIGHS_TLS.lp_cache[(n, p, q)] = cache
        indptr, indices, data_const, lb, ub, cost = cache
        data = data_const.copy()
        for j in range(p):
            data[j * n:(j + 1) * n] = z[:, j]
        # A fresh ``_Highs`` + ``HighsOptions`` per call replicates
        # ``linprog(..., method="highs")`` (which constructs both per call)
        # bit for bit.  Reusing an instance across calls leaks solver state
        # that ``clearSolver()`` does not fully reset: degenerate LPs can
        # return order-dependent duals, flipping the caller's strict-rank
        # certificate.  The CSC skeleton cache below keeps the remaining
        # per-call work at numpy-copy speed.
        solver = core._Highs()
        opts = core.HighsOptions()
        # Mirror the effective (non-None) options of _linprog_highs.
        opts.presolve = "on"
        opts.time_limit = 5.0
        opts.highs_debug_level = int(core.kHighsDebugLevelNone)
        opts.log_to_console = False
        opts.output_flag = False
        opts.simplex_strategy = int(
            core.simplex_constants.SimplexStrategy.kSimplexStrategyDual)
        opts.ipm_iteration_limit = 10_000
        opts.simplex_iteration_limit = 10_000
        if solver.passOptions(opts) == core.HighsStatus.kError:
            return None
        lp = core.HighsLp()
        lp.num_col_ = p + 2 * n
        lp.num_row_ = n
        lp.a_matrix_.num_col_ = p + 2 * n
        lp.a_matrix_.num_row_ = n
        lp.a_matrix_.format_ = core.MatrixFormat.kColwise
        lp.col_cost_ = cost
        lp.col_lower_ = lb
        lp.col_upper_ = ub
        lp.row_lower_ = yn
        lp.row_upper_ = yn
        lp.a_matrix_.start_ = indptr
        lp.a_matrix_.index_ = indices
        lp.a_matrix_.value_ = data
        if solver.passModel(lp) == core.HighsStatus.kError:
            return None
        if solver.run() == core.HighsStatus.kError:
            return None
        if solver.getModelStatus() != core.HighsModelStatus.kOptimal:
            return None
        solution = solver.getSolution()
        info = solver.getInfo()
        return (np.asarray(solution.col_value, dtype=float),
                np.asarray(solution.row_dual, dtype=float),
                float(info.objective_function_value))
    except (MemoryError, RuntimeError, ValueError, OverflowError):
        return None

_INTEGER_PARAMS = frozenset(
    {"window", "min_periods", "order", "lag", "coefficient_index", "max_q", "q"}
)

# Fixed numerical policy — VERSIONED (audit M-055).  The Huber outlier cutoff is a
# fixed, documented constant; do NOT free-search / tune it per dataset.  Any change
# to this value must ship as a NEW semantic version (the operator output changes
# for every window), never silently edited in place.
_HUBER_DELTA = 1.345

#: Failure-reason telemetry of the most recent iterative regression fit (M-060).
#: ``last_fit_status()`` lets callers distinguish a genuinely converged fit from a
#: fail-closed one and *why*, mirroring the telemetry pattern used by
#: ``group_ext`` / ``advanced_expectile``.
# Context-local immutable snapshots prevent concurrent tasks from overwriting
# each other's diagnostics while preserving the legacy accessor API.
_LAST_FIT_STATUS: ContextVar[MappingProxyType] = ContextVar(
    "rolling_core_last_fit_status",
    default=MappingProxyType({"converged": False, "reason": "not_run"}),
)

_DETAIL_MAX_DEPTH = 4
_DETAIL_MAX_ITEMS = 64
_DETAIL_MAX_STRING = 256


def _bounded_text(value: str) -> str | tuple[str, str]:
    if len(value) <= _DETAIL_MAX_STRING:
        return value
    return value[:_DETAIL_MAX_STRING], "<truncated>"


def _detail_key(value: Any) -> str:
    if isinstance(value, str):
        bounded = _bounded_text(value)
        return bounded if isinstance(bounded, str) else bounded[0] + "<truncated>"
    if isinstance(value, int) and value.bit_length() > 256:
        return "<oversized_integer>"
    if value is None or isinstance(value, (bool, int, float, complex)):
        return str(value)
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    return f"<key_type:{kind[:_DETAIL_MAX_STRING]}>"


def _status_reason(value: Any) -> str:
    if isinstance(value, str):
        bounded = _bounded_text(value)
        return bounded if isinstance(bounded, str) else bounded[0] + "<truncated>"
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    return f"<reason_type:{kind[:_DETAIL_MAX_STRING]}>"


def _freeze_detail(
    value: Any, *, depth: int = 0, seen: set[int] | None = None,
    budget: list[int] | None = None,
) -> Any:
    """Create a bounded, detached diagnostic summary without arbitrary repr."""
    if seen is None:
        seen = set()
    if budget is None:
        budget = [_DETAIL_MAX_ITEMS]
    if depth > _DETAIL_MAX_DEPTH:
        return "<max_depth>"
    if isinstance(value, int) and value.bit_length() > 256:
        return "<oversized_integer>"
    if value is None or isinstance(value, (bool, int, float, complex)):
        return value
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, bytes):
        prefix = value[:_DETAIL_MAX_STRING]
        return prefix if len(value) <= _DETAIL_MAX_STRING else (prefix, b"<truncated>")
    if isinstance(value, np.generic):
        return _freeze_detail(value.item(), depth=depth, seen=seen, budget=budget)

    identity = id(value)
    if identity in seen:
        return "<cycle>"
    if budget[0] <= 0:
        return "<truncated>"
    seen.add(identity)
    try:
        if isinstance(value, np.ndarray):
            values = []
            iterator = np.nditer(value, flags=["refs_ok", "zerosize_ok"], order="C")
            for item in iterator:
                if budget[0] <= 0:
                    values.append("<truncated>")
                    break
                budget[0] -= 1
                values.append(_freeze_detail(
                    item.item(), depth=depth + 1, seen=seen, budget=budget))
            return ("<ndarray>", tuple(value.shape), str(value.dtype), tuple(values))
        if isinstance(value, Mapping):
            items = []
            for key, item in value.items():
                if budget[0] <= 0:
                    items.append(("<truncated>", "<truncated>"))
                    break
                budget[0] -= 1
                frozen_key = _freeze_detail(
                    key, depth=depth + 1, seen=seen, budget=budget)
                frozen_item = _freeze_detail(
                    item, depth=depth + 1, seen=seen, budget=budget)
                items.append((frozen_key, frozen_item))
            return tuple(items)
        if isinstance(value, (list, tuple, set, frozenset)):
            items = []
            for item in value:
                if budget[0] <= 0:
                    items.append("<truncated>")
                    break
                budget[0] -= 1
                items.append(_freeze_detail(
                    item, depth=depth + 1, seen=seen, budget=budget))
            return tuple(items)
        kind = f"{type(value).__module__}.{type(value).__qualname__}"
        return "<object_type>", _bounded_text(kind)
    finally:
        seen.remove(identity)


def _freeze_details(details: Any) -> tuple[tuple[str, Any], ...]:
    try:
        pairs = details.items() if isinstance(details, Mapping) else iter(details)
        result = []
        budget = [_DETAIL_MAX_ITEMS]
        for pair in pairs:
            if budget[0] <= 0:
                result.append(("<truncated>", "<truncated>"))
                break
            key, value = pair
            budget[0] -= 1
            result.append((_detail_key(key), _freeze_detail(value, budget=budget)))
        return tuple(result)
    except (TypeError, ValueError):
        return (("<invalid_details>", "<object_type>"),)


@dataclass(frozen=True)
class FitStatus:
    """Immutable status captured at the same call boundary as a fit result."""

    converged: bool
    reason: str
    details: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _status_reason(self.reason))
        object.__setattr__(self, "details", _freeze_details(self.details))

    def as_dict(self) -> dict[str, Any]:
        return {"converged": self.converged, "reason": self.reason, **dict(self.details)}


@dataclass(frozen=True)
class FitResult:
    """A legacy fit value and immutable diagnostic captured together.

    The frozen dataclass prevents rebinding, but ``value`` remains the legacy
    NumPy array and is intentionally not claimed to be deeply immutable.
    """

    value: np.ndarray | None
    status: FitStatus


@dataclass(frozen=True)
class FitScope:
    """Caller-supplied ownership and window coordinates for a fit receipt.

    Runtime identities are deliberately never invented here. If any owning
    identity is absent, ``scope_kind`` is ``kernel_only`` and consumers must not
    attribute the receipt to a factor.
    """

    canonical: str | None = None
    backend: str | None = None
    profile: str | None = None
    instrument: str | None = None
    window_start: Any = None
    window_end: Any = None
    output_row: Any = None
    fit_cutoff: Any = None
    maturity_cutoff: Any = None
    execution_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    factor_id: str | None = None

    @property
    def scope_kind(self) -> str:
        owned = (self.canonical, self.backend, self.profile, self.instrument,
                 self.execution_id, self.run_id, self.task_id, self.factor_id,
                 self.window_start, self.window_end, self.output_row,
                 self.fit_cutoff, self.maturity_cutoff)
        if not all(_is_scope_scalar(v) for v in owned):
            return "kernel_only"
        try:
            valid_window = (
                self.window_start <= self.window_end <= self.output_row
                and self.fit_cutoff <= self.output_row
                and self.maturity_cutoff <= self.output_row
            )
        except (TypeError, ValueError):
            valid_window = False
        return "factor_window" if valid_window else "kernel_only"


@dataclass(frozen=True)
class FitFailureReceipt:
    sequence: int
    status: FitStatus
    scope: FitScope
    scope_kind: str


def _is_scope_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (bool, list, tuple, dict, set, np.ndarray)):
        return False
    if isinstance(value, str):
        return bool(value) and len(value) <= _DETAIL_MAX_STRING
    if isinstance(value, (int, np.integer)):
        return int(value).bit_length() <= 64
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        return not bool(pd.isna(value))
    return False


def _sanitize_scope(scope: FitScope) -> FitScope:
    return FitScope(**{
        name: value if _is_scope_scalar(value) else None
        for name, value in scope.__dict__.items()
    })


@dataclass
class BoundedFitFailureSink:
    """Execution-owned bounded failure aggregation and sampled detail storage."""

    detail_capacity: int = 64
    group_capacity: int = 128
    _counts: Counter[tuple[str, str]] = field(default_factory=Counter, init=False)
    _details: deque[FitFailureReceipt] = field(init=False)
    _sequence: int = field(default=0, init=False)
    _dropped_details: int = field(default=0, init=False)
    _overflow_group_count: int = field(default=0, init=False)
    _unknown_status_count: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if (type(self.detail_capacity) is not int or type(self.group_capacity) is not int
                or self.detail_capacity < 0 or self.group_capacity < 1):
            raise ValueError("detail_capacity must be >= 0 and group_capacity must be >= 1")
        self._details = deque(maxlen=self.detail_capacity or None)

    def record(self, result: FitResult, scope: FitScope | None = None) -> None:
        if result.status.converged:
            return
        if result.status.reason in {"unknown_status", "internal_status_mismatch"}:
            with self._lock:
                self._unknown_status_count += 1
            return
        actual_scope = _sanitize_scope(scope) if isinstance(scope, FitScope) else FitScope()
        canonical = (actual_scope.canonical
                     if isinstance(actual_scope.canonical, str)
                     and 0 < len(actual_scope.canonical) <= _DETAIL_MAX_STRING
                     else "<kernel_only>")
        key = (canonical, result.status.reason)
        with self._lock:
            if key in self._counts or len(self._counts) < self.group_capacity:
                self._counts[key] += 1
            else:
                self._overflow_group_count += 1
            self._sequence += 1
            if self.detail_capacity == 0:
                self._dropped_details += 1
                return
            if len(self._details) == self.detail_capacity:
                self._dropped_details += 1
            self._details.append(FitFailureReceipt(
                self._sequence, result.status, actual_scope, actual_scope.scope_kind))

    @property
    def dropped_details(self) -> int:
        with self._lock:
            return self._dropped_details

    @property
    def overflow_group_count(self) -> int:
        with self._lock:
            return self._overflow_group_count

    @property
    def unknown_status_count(self) -> int:
        with self._lock:
            return self._unknown_status_count

    def counts(self) -> Mapping[tuple[str, str], int]:
        with self._lock:
            return MappingProxyType(dict(self._counts))

    def page(self, *, after_sequence: int = 0, limit: int = 50) -> tuple[FitFailureReceipt, ...]:
        if type(after_sequence) is not int or type(limit) is not int or after_sequence < 0 or limit < 1:
            raise ValueError("limit must be >= 1")
        with self._lock:
            return tuple(r for r in self._details if r.sequence > after_sequence)[:limit]


_FIT_FAILURE_SINK: ContextVar[BoundedFitFailureSink | None] = ContextVar(
    "rolling_core_fit_failure_sink", default=None)
_FIT_RECEIPT_SCOPE: ContextVar[FitScope | None] = ContextVar(
    "rolling_core_fit_receipt_scope", default=None)


@contextmanager
def fit_failure_receipts(sink: BoundedFitFailureSink) -> Iterator[BoundedFitFailureSink]:
    """Bind a caller-owned sink to the current thread/async execution context."""
    token = _FIT_FAILURE_SINK.set(sink)
    try:
        yield sink
    finally:
        _FIT_FAILURE_SINK.reset(token)


@contextmanager
def fit_receipt_scope(scope: FitScope) -> Iterator[FitScope]:
    """Bind caller-provided owner fields; producers add window coordinates."""
    sanitized = _sanitize_scope(scope)
    token = _FIT_RECEIPT_SCOPE.set(sanitized)
    try:
        yield sanitized
    finally:
        _FIT_RECEIPT_SCOPE.reset(token)


def current_fit_scope() -> FitScope | None:
    """Return the current immutable owner scope, without inventing identities."""
    return _FIT_RECEIPT_SCOPE.get()


def record_current_fit(
    result: FitResult, *, instrument: Any, window_start: Any, window_end: Any,
    output_row: Any, fit_cutoff: Any, maturity_cutoff: Any,
    failure_sink: BoundedFitFailureSink | None = None,
) -> None:
    """Merge producer coordinates into the current owner scope and record."""
    sink = failure_sink if failure_sink is not None else _FIT_FAILURE_SINK.get()
    if sink is None:
        return
    base = current_fit_scope() or FitScope()
    sink.record(result, replace(
        base, instrument=instrument, window_start=window_start,
        window_end=window_end, output_row=output_row, fit_cutoff=fit_cutoff,
        maturity_cutoff=maturity_cutoff))


def last_fit_status() -> dict[str, Any]:
    """Return a copy of this thread/async-context's latest fit diagnostic."""
    return dict(_LAST_FIT_STATUS.get())


def _set_fit_status(converged: bool, reason: str, **details: Any) -> None:
    snapshot: dict[str, Any] = {
        "converged": bool(converged), "reason": _status_reason(reason)}
    snapshot.update(dict(_freeze_details(details)))
    _LAST_FIT_STATUS.set(MappingProxyType(snapshot))


def fit_result(fit_fn: Callable[..., np.ndarray | None], *args: Any, **kwargs: Any) -> FitResult:
    """Run a legacy fit callable and capture its context-local status immediately."""
    # A third-party/legacy callable that forgets to publish status must not
    # inherit the previous fit's success or failure.
    _set_fit_status(False, "not_run")
    value = fit_fn(*args, **kwargs)
    snapshot = last_fit_status()
    if snapshot.get("reason") == "not_run":
        snapshot = {"converged": False, "reason": "unknown_status"}
    elif bool(snapshot.get("converged")) != (value is not None):
        snapshot = {"converged": False, "reason": "internal_status_mismatch"}
    status = FitStatus(
        bool(snapshot.pop("converged", False)),
        str(snapshot.pop("reason", "unknown")),
        tuple(sorted(snapshot.items())),
    )
    return FitResult(value, status)


def metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int = 5,
    domain: str = "price_volume",
    input_units: dict[str, str] | None = None,
    output_unit: str | None = None,
    diagnostic_only: bool = False,
    param_specs: "dict[str, Any] | None" = None,
) -> OperatorMetadata:
    # P1-89: the typed-v2 surface carries explicit input_units / output_unit
    # field semantics.  Regression kernels that previously wrote a generic
    # ``unit`` tag now also declare the concrete unit relationships (e.g. beta
    # -> unit(y)/unit(x), residual -> unit(y)); ``unit`` remains in the tags for
    # backward compatibility with legacy consumers.
    #
    # Round-6 §20: ``diagnostic_only`` marks in-sample self-fit operators
    # (``fit_lag=0`` residuals / R² / AR fitted values) whose current-row output
    # is influenced by the current sample itself.  That is not future leakage,
    # but default factor mining should prefer the out-of-sample
    # ``*_prior`` / ``*_forecast_error`` / ``*_prior_innovation`` variants; the
    # tag lets the mining surface hide the self-fit diagnostics.
    tags = [
        "time_series_regression", "daily", "pit_safe", "causal", "typed_v2",
        f"signature:{','.join(params)}->series", f"domain:{domain}",
        f"unit:{unit}", f"cost:{cost}",
    ]
    if diagnostic_only:
        tags.append("diagnostic_only")
    return OperatorMetadata(
        name=name,
        category="time_series_regression",
        description=description,
        param_names=params,
        return_type="series",
        tags=tags,
        input_units=dict(input_units) if input_units else {},
        output_unit=output_unit,
        param_specs=dict(param_specs) if param_specs else {},
    )


def frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def trailing_contiguous_finite(vals: np.ndarray) -> np.ndarray:
    """Most recent suffix of consecutive finite values ending at the last row.

    A missing observation *breaks* the segment: rows on either side of a NaN
    are never treated as adjacent observations (no time-axis compression for
    suspensions / provider gaps).  A NaN at the last row yields an empty block
    so the caller emits NaN instead of re-using the last valid value.
    """
    n = len(vals)
    if n == 0 or not np.isfinite(vals[-1]):
        return np.empty(0, dtype=float)
    i = n - 1
    while i >= 0 and np.isfinite(vals[i]):
        i -= 1
    return vals[i + 1 :]


def aligned(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Strict multi-panel alignment (round-6 P0-25): fail closed, never reindex.

    Two panels with different instrument columns or a shifted index must never
    be paired positionally — silently reindexing hides a missing symbol or a
    one-day date shift as NaN.  The central ``validate_operator_call`` gate
    already rejects misaligned panels before this helper runs; strictness here
    is defense-in-depth for direct kernel calls.
    """
    if not frames:
        return ()
    from factor_engine.cleaned_operators.alignment import align_panel_inputs

    return align_panel_inputs(*frames, strict_axes=True)


def design_is_well_conditioned(design: np.ndarray) -> bool:
    """ModelDesignGate: reject rank-deficient / ill-conditioned linear designs.

    A design with a very large condition number (near-collinear columns) or
    deficient rank makes the least-squares coefficients numerically meaningless,
    so the fit is rejected *before* the solve.  The condition threshold (1e12)
    matches numpy's default ``rcond`` behaviour for double precision; degenerate
    (empty / zero-singular-value) designs are guarded so a 0/0 never propagates.
    """
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] == 0:
        return False
    n, p = design.shape
    if n < p:
        return False
    try:
        s = np.linalg.svd(design, compute_uv=False)
    except (np.linalg.LinAlgError, ValueError):
        return False
    if s.size == 0:
        return False
    s_max = float(s[0])
    if not np.isfinite(s_max) or s_max <= 0.0:
        return False
    s_min = float(s[-1])
    cond = np.inf if s_min <= 0.0 else s_max / s_min
    if not np.isfinite(cond) or cond > 1e12:
        return False
    # Rank check: singular values above the standard numerical floor
    # (S.max() * max(M, N) * eps), the same tolerance numpy.matrix_rank uses.
    floor = s_max * max(n, p) * np.finfo(float).eps
    if int(np.sum(s > floor)) < p:
        return False
    return True


def fit_linear_model_checked(design: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Gate-wrapped OLS: ``None`` on a rank-deficient / ill-conditioned design.

    Thin convenience wrapper around :func:`ols_fit` for callers (HAR-RV, regime /
    MoE OLS paths) that want the ModelDesignGate without importing ``ols_fit``
    directly; it returns ``None`` so they fail closed (emit NaN) instead of
    accepting a numerically meaningless coefficient.
    """
    return ols_fit(design, y)


def ols_fit(design: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """OLS coefficients, None if design is rank deficient / degenerate."""
    if design.shape[0] < design.shape[1]:
        _set_fit_status(False, "insufficient_sample")
        return None
    if not design_is_well_conditioned(design):
        _set_fit_status(False, "singular")
        return None
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        _set_fit_status(False, "singular")
        return None
    if not np.all(np.isfinite(beta)):
        _set_fit_status(False, "singular")
        return None
    _set_fit_status(True, "converged")
    return beta




def _returned_fit_backward_error(
    design: np.ndarray,
    target: np.ndarray,
    beta: np.ndarray,
    response_max: float,
    response_spread: float,
) -> float:
    """Scale-free residual error of returned original-unit coefficients."""
    residual = target - design @ beta
    if not np.all(np.isfinite(residual)):
        return float("inf")
    normalized = (residual / response_max) / response_spread
    if not np.all(np.isfinite(normalized)):
        return float("inf")
    return float(np.max(np.abs(normalized)))


def huber_fit(
    design: np.ndarray,
    y: np.ndarray,
    *,
    delta: float = _HUBER_DELTA,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> np.ndarray | None:
    """Stable Huber IRLS with fail-closed, multi-criterion convergence."""
    try:
        x = np.asarray(design, dtype=float)
        target = np.asarray(y, dtype=float)
        cutoff, max_iter, tol = float(delta), int(iterations), float(tolerance)
    except (TypeError, ValueError, OverflowError):
        _set_fit_status(False, "invalid_params")
        return None
    if (x.ndim != 2 or target.ndim != 1 or x.shape[0] != target.shape[0]
            or x.shape[1] == 0
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(target))
            or isinstance(delta, (bool, np.bool_)) or not np.isfinite(cutoff) or cutoff <= 0.0
            or isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, (int, np.integer)) or max_iter <= 0
            or isinstance(tolerance, (bool, np.bool_)) or not np.isfinite(tol) or tol <= 0.0):
        _set_fit_status(False, "invalid_params")
        return None

    n, p = x.shape
    if n < p:
        _set_fit_status(False, "insufficient_sample", n=int(n), p=int(p))
        return None
    constant = np.all(x == x[0], axis=0)
    constant_nonzero = constant & (np.abs(x[0]) > 0.0)
    intercept_cols = np.flatnonzero(constant_nonzero)
    if np.any(constant & ~constant_nonzero) or intercept_cols.size > 1:
        _set_fit_status(False, "singular", rank=int(np.linalg.matrix_rank(x)))
        return None
    intercept = int(intercept_cols[0]) if intercept_cols.size == 1 else None

    z = x.copy()
    centers = np.zeros(p)
    scales = np.ones(p)
    x_max = np.max(np.abs(x), axis=0)
    if intercept is not None:
        feature_cols = np.array([j for j in range(p) if j != intercept], dtype=int)
        z[:, intercept] = 1.0
        if feature_cols.size:
            if np.any(x_max[feature_cols] == 0.0):
                _set_fit_status(False, "singular")
                return None
            normalized_x = x[:, feature_cols] / x_max[feature_cols]
            centers[feature_cols] = np.mean(normalized_x, axis=0)
            centered = normalized_x - centers[feature_cols]
            scales[feature_cols] = np.linalg.norm(centered, axis=0) / np.sqrt(n)
            if np.any(~np.isfinite(scales[feature_cols])) or np.any(scales[feature_cols] <= 0.0):
                _set_fit_status(False, "singular")
                return None
            z[:, feature_cols] = centered / scales[feature_cols]
        y_max = float(np.max(np.abs(target)))
        if y_max == 0.0:
            y_max = 1.0
        normalized_y = target / y_max
        y_center = float(np.mean(normalized_y))
        centered_y = normalized_y - y_center
        y_spread = float(np.max(np.abs(centered_y)))
        if y_spread == 0.0:
            y_spread = 1.0
        yn = centered_y / y_spread
    else:
        feature_cols = np.arange(p)
        if np.any(x_max == 0.0):
            _set_fit_status(False, "singular")
            return None
        z = x / x_max
        y_max = float(np.max(np.abs(target)))
        if y_max == 0.0:
            y_max = 1.0
        y_center = 0.0
        y_spread = 1.0
        yn = target / y_max

    if not design_is_well_conditioned(z):
        _set_fit_status(False, "singular", rank=int(np.linalg.matrix_rank(z)))
        return None
    try:
        gamma, *_ = np.linalg.lstsq(z, yn, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        _set_fit_status(False, "singular")
        return None
    if not np.all(np.isfinite(gamma)):
        _set_fit_status(False, "numerical_failure")
        return None

    eps = np.finfo(float).eps

    def _stable_std(values: np.ndarray) -> float:
        centered_values = values - float(np.mean(values))
        magnitude = float(np.max(np.abs(centered_values)))
        if magnitude == 0.0:
            return 0.0
        return magnitude * float(np.linalg.norm(centered_values / magnitude) / np.sqrt(n))

    initial_resid = yn - z @ gamma
    backward_scale = (
        float(np.max(np.abs(yn)))
        + float(np.linalg.norm(z, ord=np.inf) * np.linalg.norm(gamma, ord=np.inf))
    )
    exact_floor = 64.0 * eps * backward_scale
    exact_consensus: np.ndarray | None = None

    def _exact_majority_fit(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return a verified zero-residual majority fit, never a relaxed IRLS fit."""
        centered = residual - float(np.median(residual))
        residual_floor = 128.0 * eps * max(backward_scale, 1.0)
        mask = np.abs(centered) <= residual_floor
        if int(mask.sum()) <= n // 2 or int(mask.sum()) < p + 1:
            return None
        consensus_design = z[mask]
        if np.linalg.matrix_rank(consensus_design) < p:
            return None
        try:
            candidate, *_ = np.linalg.lstsq(consensus_design, yn[mask], rcond=None)
        except (np.linalg.LinAlgError, ValueError):
            return None
        if not np.all(np.isfinite(candidate)):
            return None
        candidate_residual = yn - z @ candidate
        candidate_centered = candidate_residual - float(np.median(candidate_residual))
        stable_mask = np.abs(candidate_centered) <= residual_floor
        if (int(stable_mask.sum()) <= n // 2 or int(stable_mask.sum()) < p + 1
                or not np.all(stable_mask[mask])):
            return None
        consensus_error = float(np.max(np.abs(candidate_residual[stable_mask])))
        if not np.isfinite(consensus_error) or consensus_error > residual_floor:
            return None
        # Zero-scale Huber/LAD limit certificate.  Exact residuals admit a
        # subgradient in [-1, 1]; non-consensus residuals contribute their
        # signs.  Accept only when a bounded consensus subgradient closes the
        # full transformed-design stationarity equation.
        outlier_mask = ~stable_mask
        rhs = -(z[outlier_mask].T @ np.sign(candidate_residual[outlier_mask]))
        try:
            consensus_subgradient, *_ = np.linalg.lstsq(z[stable_mask].T, rhs, rcond=None)
        except (np.linalg.LinAlgError, ValueError):
            return None
        stationarity_error = z[stable_mask].T @ consensus_subgradient - rhs
        stationarity_floor = 512.0 * eps * max(
            1.0, float(np.linalg.norm(rhs, ord=np.inf)))
        if (not np.all(np.isfinite(consensus_subgradient))
                or float(np.max(np.abs(consensus_subgradient))) > 1.0 + 512.0 * eps
                or not np.all(np.isfinite(stationarity_error))
                or float(np.max(np.abs(stationarity_error))) > stationarity_floor):
            return None
        return candidate, stable_mask

    if float(np.max(np.abs(initial_resid))) <= exact_floor:
        reason, final_scale, final_score, used = "exact_fit", 0.0, 0.0, 0
    else:
        reason = "non_converged"
        final_scale, final_score = np.nan, np.inf
        pred_change, param_change = np.inf, np.inf
        used = 0
        for used in range(1, max_iter + 1):
            resid = yn - z @ gamma
            centered_resid = resid - np.median(resid)
            mad = float(np.median(np.abs(centered_resid)))
            residual_extent = float(np.max(np.abs(centered_resid)))
            if (np.isfinite(mad) and residual_extent > exact_floor
                    and mad <= 64.0 * eps * residual_extent):
                consensus = _exact_majority_fit(resid)
                if consensus is None:
                    _set_fit_status(
                        False, "scale_degenerate", iterations=used,
                        mad=mad, residual_extent=residual_extent,
                    )
                    return None
                gamma, exact_consensus = consensus
                reason, final_scale, final_score = "exact_consensus", 0.0, 0.0
                break
            scale = 1.4826 * mad
            if not np.isfinite(scale) or scale == 0.0:
                scale = _stable_std(resid)
            if not np.isfinite(scale) or scale <= 0.0:
                _set_fit_status(False, "numerical_failure", iterations=used)
                return None
            u = resid / scale
            weight = np.ones_like(u)
            outlier = np.abs(u) > cutoff
            weight[outlier] = cutoff / np.abs(u[outlier])
            root_w = np.sqrt(weight)
            wz, wyn = z * root_w[:, None], yn * root_w
            if not design_is_well_conditioned(wz):
                _set_fit_status(False, "singular", iterations=used)
                return None
            try:
                gamma_new, *_ = np.linalg.lstsq(wz, wyn, rcond=None)
            except (np.linalg.LinAlgError, ValueError):
                _set_fit_status(False, "singular", iterations=used)
                return None
            if not np.all(np.isfinite(gamma_new)):
                _set_fit_status(False, "numerical_failure", iterations=used)
                return None
            prediction_delta = z @ (gamma_new - gamma)
            pred_change = float(np.sqrt(np.mean(prediction_delta * prediction_delta))) / max(scale, eps)
            param_change = float(np.linalg.norm(gamma_new - gamma, ord=np.inf)) / max(
                1.0, float(np.linalg.norm(gamma_new, ord=np.inf)))
            gamma = gamma_new

            final_resid = yn - z @ gamma
            centered_final = final_resid - np.median(final_resid)
            final_mad = float(np.median(np.abs(centered_final)))
            final_extent = float(np.max(np.abs(centered_final)))
            if (np.isfinite(final_mad) and final_extent > exact_floor
                    and final_mad <= 64.0 * eps * final_extent):
                consensus = _exact_majority_fit(final_resid)
                if consensus is None:
                    _set_fit_status(
                        False, "scale_degenerate", iterations=used,
                        mad=final_mad, residual_extent=final_extent,
                    )
                    return None
                gamma, exact_consensus = consensus
                reason, final_scale, final_score = "exact_consensus", 0.0, 0.0
                break
            final_scale = 1.4826 * final_mad
            if not np.isfinite(final_scale) or final_scale == 0.0:
                final_scale = _stable_std(final_resid)
            if not np.isfinite(final_scale) or final_scale <= 0.0:
                if float(np.max(np.abs(final_resid))) <= exact_floor:
                    reason, final_score = "exact_fit", 0.0
                    break
                _set_fit_status(False, "numerical_failure", iterations=used)
                return None
            psi = np.clip(final_resid / final_scale, -cutoff, cutoff)
            col_norm = np.sqrt(np.mean(z * z, axis=0))
            final_score = float(np.max(np.abs((z.T @ psi) / n) / np.maximum(col_norm, eps)))
            if pred_change <= tol and param_change <= tol and final_score <= tol:
                reason = "converged"
                break

        if reason == "non_converged":
            _set_fit_status(False, reason, iterations=used, scale=float(final_scale),
                            normalized_score=float(final_score),
                            prediction_change=float(pred_change),
                            parameter_change=float(param_change))
            return None

    def _scaled_ratio(
        value: float, numerator: float, denominator_a: float, denominator_b: float = 1.0
    ) -> float:
        vm, ve = np.frexp(value)
        nm, ne = np.frexp(numerator)
        am, ae = np.frexp(denominator_a)
        bm, be = np.frexp(denominator_b)
        return float(np.ldexp((vm * nm) / (am * bm), ve + ne - ae - be))

    beta = np.empty(p, dtype=float)
    for column in range(p):
        if column == intercept:
            beta[column] = _scaled_ratio(float(gamma[column]), y_max, 1.0)
        else:
            beta[column] = _scaled_ratio(
                float(gamma[column] * y_spread), y_max,
                float(x_max[column]), float(scales[column]))
    if intercept is not None:
        normalized_intercept = y_center + y_spread * gamma[intercept]
        if feature_cols.size:
            normalized_intercept -= y_spread * float(np.sum(
                gamma[feature_cols] * centers[feature_cols] / scales[feature_cols]))
        beta[intercept] = _scaled_ratio(normalized_intercept, y_max, float(x[0, intercept]))
    if not np.all(np.isfinite(beta)):
        _set_fit_status(False, "numerical_failure", iterations=used)
        return None
    # Recompute diagnostics independently from the returned original-unit
    # coefficients.  This catches restoration/cancellation errors that would be
    # invisible if telemetry merely reused transformed-coordinate residuals.
    original_resid = target - x @ beta
    if not np.all(np.isfinite(original_resid)):
        _set_fit_status(False, "numerical_failure", iterations=used)
        return None
    reported_scale = 1.4826 * float(
        np.median(np.abs(original_resid - np.median(original_resid))))
    if not np.isfinite(reported_scale) or reported_scale == 0.0:
        reported_scale = _stable_std(original_resid)
    returned_backward_error = _returned_fit_backward_error(
        x, target, beta, y_max, y_spread)
    if reason == "exact_fit" and returned_backward_error > exact_floor:
        _set_fit_status(
            False,
            "numerical_failure",
            iterations=used,
            scale=reported_scale,
            normalized_backward_error=returned_backward_error,
        )
        return None
    if reason == "exact_consensus":
        if exact_consensus is None:
            _set_fit_status(False, "internal_status_mismatch", iterations=used)
            return None
        original_scale = max(
            1.0, float(np.max(np.abs(target))),
            float(np.linalg.norm(x, ord=np.inf) * np.linalg.norm(beta, ord=np.inf)),
        )
        consensus_floor = 256.0 * eps * original_scale
        if (np.linalg.matrix_rank(x[exact_consensus]) < p
                or not np.all(np.isfinite((x @ beta)[exact_consensus]))
                or float(np.max(np.abs(original_resid[exact_consensus]))) > consensus_floor):
            _set_fit_status(False, "numerical_failure", iterations=used)
            return None
    if reported_scale == 0.0:
        reported_score = 0.0
    elif not np.isfinite(reported_scale):
        _set_fit_status(False, "numerical_failure", iterations=used)
        return None
    else:
        reported_psi = np.clip(original_resid / reported_scale, -cutoff, cutoff)
        reported_col_norm = np.sqrt(np.mean(z * z, axis=0))
        reported_score = float(np.max(
            np.abs((z.T @ reported_psi) / n) / np.maximum(reported_col_norm, eps)))
    if reason == "converged" and reported_score > tol:
        _set_fit_status(False, "non_converged", iterations=used,
                        scale=reported_scale, normalized_score=reported_score)
        return None
    _set_fit_status(True, reason, iterations=used, rank=p,
                    scale=reported_scale, normalized_score=reported_score)
    return beta


def ridge_fit(design: np.ndarray, y: np.ndarray, alpha: float, *, has_intercept: bool = True) -> np.ndarray | None:
    """Stable Ridge fit with an original-feature-unit L2 penalty.

    The intercept column is only exempt from the penalty when the design
    actually starts with an intercept (``has_intercept=True``).  Penalising the
    first *feature* column when no intercept is present would wrongly shrink a
    real regressor.  Features are scaled only as a numerical change of
    variables: the augmented-system penalty is divided by the same scales, so
    ``alpha`` continues to penalise coefficients in the caller's original
    feature units.

    ``alpha == 0`` deliberately delegates to the OLS rank/conditioning policy.
    For positive alpha the augmented system supports ``p > n``; penalised
    constant columns are assigned coefficient zero.  With an intercept, its
    column must be a finite non-zero constant and is never penalised.
    """
    try:
        x = np.asarray(design, dtype=float)
        target = np.asarray(y, dtype=float)
        strength = float(alpha)
    except (TypeError, ValueError, OverflowError):
        _set_fit_status(False, "invalid_params")
        return None
    if (x.ndim != 2 or target.ndim != 1 or x.shape[0] != target.shape[0] or
            x.shape[0] == 0 or x.shape[1] == 0 or not np.isfinite(strength) or
            strength < 0.0 or not np.all(np.isfinite(x)) or
            not np.all(np.isfinite(target))):
        _set_fit_status(False, "invalid_params")
        return None
    if strength == 0.0:
        return ols_fit(x, target)

    if has_intercept:
        intercept_column = x[:, 0]
        intercept_value = float(intercept_column[0])
        if (not np.isfinite(intercept_value) or intercept_value == 0.0 or
                not np.all(intercept_column == intercept_value)):
            _set_fit_status(False, "invalid_params")
            return None
        features = x[:, 1:]
        feature_mean = np.mean(features, axis=0)
        target_mean = float(np.mean(target))
        work_x = features - feature_mean
        work_y = target - target_mean
    else:
        intercept_value = 0.0
        feature_mean = np.zeros(x.shape[1], dtype=float)
        target_mean = 0.0
        work_x = x
        work_y = target

    # Scaling is solely a solver preconditioner.  If theta = scale * beta,
    # alpha * ||beta||^2 becomes alpha * ||theta / scale||^2, preserving the
    # penalty in original feature units rather than silently standardising it.
    scale = np.max(np.abs(work_x), axis=0) if work_x.shape[1] else np.empty(0)
    active = np.isfinite(scale) & (scale > 0.0)
    slopes = np.zeros(work_x.shape[1], dtype=float)
    try:
        if np.any(active):
            scaled_x = work_x[:, active] / scale[active]
            penalty_rows = np.diag(np.sqrt(strength) / scale[active])
            augmented_x = np.vstack((scaled_x, penalty_rows))
            augmented_y = np.concatenate((work_y, np.zeros(penalty_rows.shape[0])))
            theta, *_ = np.linalg.lstsq(augmented_x, augmented_y, rcond=0.0)
            slopes[active] = theta / scale[active]
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        _set_fit_status(False, "singular")
        return None
    if has_intercept:
        intercept = (target_mean - float(feature_mean @ slopes)) / intercept_value
        beta = np.concatenate(([intercept], slopes))
    else:
        beta = slopes
    if not np.all(np.isfinite(beta)):
        _set_fit_status(False, "singular")
        return None
    _set_fit_status(True, "converged")
    return beta


def quantile_fit(
    design: np.ndarray,
    y: np.ndarray,
    q: float,
    iterations: int = 8,
    tolerance: float = 1e-6,
) -> np.ndarray | None:
    """IRLS asymmetric-weighted least squares.

    This minimises an asymmetric *squared*-error objective, i.e. it is an
    **expectile** regression (Neyman--Pearson asymmetric loss), not a
    pinball-loss quantile regression.  It is kept under the historic
    ``quantile_*`` operator names for backward compatibility, with the honest
    ``expectile_*`` names exposed alongside; both must be read as expectiles.

    Convergence is checked (audit M-060): each iteration's max ``|beta_new -
    beta|`` is tracked; if the budget ``iterations`` is exhausted with the last
    change still above ``tolerance`` the fit is treated as NON-converged and
    ``None`` is returned (fail-closed) instead of the half-converged last
    iterate.  The failure reason is exposed via :func:`last_fit_status`.
    """
    beta = ols_fit(design, y)
    if beta is None:
        return None
    for _ in range(iterations):
        resid = y - design @ beta
        weight = np.where(resid > 0, q, 1.0 - q)
        weight = np.clip(weight, 1e-6, None)
        # WLS with asymmetric weights minimises ``sum w_i e_i^2``; the correct
        # design is ``sqrt(w_i) * X_i, sqrt(w_i) * y_i``.  Multiplying by
        # ``weight`` itself would minimise ``sum w_i^2 e_i^2``, over-weighting
        # the asymmetric side (review P0: expectile family).
        sqrt_w = np.sqrt(weight)
        beta_new = ols_fit(design * sqrt_w[:, None], y * sqrt_w)
        if beta_new is None:
            return None
        if np.max(np.abs(beta_new - beta)) < tolerance:
            _set_fit_status(True, "converged")
            return beta_new
        beta = beta_new
    # Iterations exhausted without meeting the convergence tolerance: the last
    # iterate is a half-converged fit and must not be accepted (M-060).  The
    # caller emits NaN (fail-closed) rather than a spurious coefficient.
    _set_fit_status(False, "non_converged")
    return None


# The IRLS asymmetric-weighted fit above is an expectile fit; expose an
# explicitly-named alias so factor authors can call the honest name.
expectile_fit = quantile_fit


def pinball_quantile_fit(design: np.ndarray, y: np.ndarray, q: float) -> np.ndarray | None:
    """Sparse, bounded Koenker-Bassett pinball regression.

    This coefficient-returning kernel rejects rank-deficient designs: solver
    success can certify an optimal prediction/loss without identifying a unique
    coefficient vector.  Resource or solver failure is explicit telemetry and
    fails closed.
    """
    try:
        x = np.asarray(design, dtype=float)
        target_values = np.asarray(y, dtype=float)
        quantile = float(q)
    except (TypeError, ValueError, OverflowError):
        _set_fit_status(False, "invalid_params")
        return None
    if (_linprog is None or x.ndim != 2 or target_values.ndim != 1
            or x.shape[0] != target_values.shape[0] or x.shape[1] == 0
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(target_values))
            or isinstance(q, (bool, np.bool_)) or not np.isfinite(quantile)
            or not 0.0 < quantile < 1.0):
        _set_fit_status(False, "invalid_params")
        return None
    n, p = x.shape
    if n < p + 2:
        _set_fit_status(False, "insufficient_sample")
        return None
    estimated_nnz = n * p + 2 * n
    if n > 10_000 or estimated_nnz > 2_000_000:
        _set_fit_status(False, "resource_limit", estimated_nnz=estimated_nnz)
        return None

    constant = np.all(x == x[0], axis=0)
    constant_nonzero = constant & (x[0] != 0.0)
    intercept_columns = np.flatnonzero(constant_nonzero)
    if np.any(constant & ~constant_nonzero) or intercept_columns.size > 1:
        _set_fit_status(False, "singular")
        return None
    intercept = int(intercept_columns[0]) if intercept_columns.size == 1 else None
    z = x.copy()
    centers = np.zeros(p)
    base_scales = np.ones(p)
    spreads = np.ones(p)
    if intercept is not None:
        feature_columns = np.array([j for j in range(p) if j != intercept], dtype=int)
        z[:, intercept] = 1.0
        for column in feature_columns:
            with np.errstate(over="ignore", invalid="ignore"):
                delta_x = x[:, column] - x[0, column]
            if np.all(np.isfinite(delta_x)):
                base = float(np.max(np.abs(delta_x)))
                normalized_delta = delta_x / base if base > 0.0 else delta_x
            else:
                base = float(np.max(np.abs(x[:, column])))
                normalized_x = x[:, column] / base
                normalized_delta = normalized_x - normalized_x[0]
            center = float(np.mean(normalized_delta))
            centered = normalized_delta - center
            spread = float(np.max(np.abs(centered)))
            if base <= 0.0 or spread <= 0.0:
                _set_fit_status(False, "singular")
                return None
            centers[column] = center + (
                float(x[0, column]) / base if np.all(np.isfinite(delta_x)) else
                float((x[:, column] / base)[0]))
            base_scales[column] = base
            spreads[column] = spread
            z[:, column] = centered / spread
        with np.errstate(over="ignore", invalid="ignore"):
            delta_y = target_values - target_values[0]
        if np.all(np.isfinite(delta_y)):
            y_base = float(np.max(np.abs(delta_y)))
            normalized_delta_y = delta_y / y_base if y_base > 0.0 else delta_y
        else:
            y_base = float(np.max(np.abs(target_values)))
            normalized_y = target_values / y_base
            normalized_delta_y = normalized_y - normalized_y[0]
        y_center_delta = float(np.mean(normalized_delta_y))
        centered_y = normalized_delta_y - y_center_delta
        y_spread = float(np.max(np.abs(centered_y)))
        if y_base <= 0.0 or y_spread <= 0.0:
            y_base, y_spread = max(1.0, float(np.max(np.abs(target_values)))), 1.0
            yn = np.zeros_like(target_values)
            y_location = float(target_values[0] / y_base)
        else:
            yn = centered_y / y_spread
            y_location = y_center_delta + float(target_values[0] / y_base)
    else:
        feature_columns = np.arange(p)
        column_max = np.max(np.abs(x), axis=0)
        if np.any(column_max <= 0.0):
            _set_fit_status(False, "singular")
            return None
        z = x / column_max
        base_scales = column_max
        y_base = float(np.max(np.abs(target_values)))
        if y_base == 0.0:
            y_base = 1.0
        y_spread = 1.0
        y_location = 0.0
        yn = target_values / y_base

    if not design_is_well_conditioned(z):
        _set_fit_status(False, "singular", rank=int(np.linalg.matrix_rank(z)))
        return None
    try:
        solved = _highs_solve_eq(z, yn, quantile)
    except (ImportError, MemoryError, RuntimeError, ValueError, OverflowError):
        solved = None
    if solved is None:
        # Covers both a non-success solver result and a solver/runtime failure,
        # exactly like the legacy ``not result.success or result.status != 0``
        # and ``except`` handling around ``_linprog``.
        _set_fit_status(False, "non_converged")
        return None
    primal_full, dual, fun_value = solved
    if not np.isfinite(fun_value):
        _set_fit_status(False, "non_converged")
        return None
    gamma = np.asarray(primal_full[:p], dtype=float)
    positive = np.asarray(primal_full[p : p + n], dtype=float)
    negative = np.asarray(primal_full[p + n :], dtype=float)
    if (gamma.shape != (p,) or positive.shape != (n,) or negative.shape != (n,)
            or not np.all(np.isfinite(gamma))
            or not np.all(np.isfinite(positive))
            or not np.all(np.isfinite(negative))):
        _set_fit_status(False, "singular")
        return None
    primal_residual = z @ gamma + positive - negative - yn
    certificate_tol = 1e-8 * max(1.0, float(np.max(np.abs(yn))))
    if (float(np.max(np.abs(primal_residual))) > certificate_tol
            or np.any(positive < -certificate_tol)
            or np.any(negative < -certificate_tol)):
        _set_fit_status(False, "non_converged")
        return None
    if dual.shape != (n,) or not np.all(np.isfinite(dual)):
        _set_fit_status(False, "coefficient_not_identified")
        return None
    dual_gap = fun_value - float(yn @ dual)
    dual_tol = 1e-7 * max(1.0, abs(fun_value))
    stationarity = (z.T @ dual) / n
    if (np.any(dual > quantile + dual_tol)
            or np.any(dual < quantile - 1.0 - dual_tol)
            or abs(dual_gap) > dual_tol
            or not np.all(np.isfinite(stationarity))
            or float(np.max(np.abs(stationarity))) > dual_tol):
        _set_fit_status(
            False, "non_converged", duality_gap=dual_gap,
            stationarity=float(np.max(np.abs(stationarity))))
        return None
    fitted_residual = yn - z @ gamma
    strict = ((np.abs(fitted_residual) <= certificate_tol)
              & (dual < quantile - dual_tol)
              & (dual > quantile - 1.0 + dual_tol))
    strict_rank = int(np.linalg.matrix_rank(z[strict])) if np.any(strict) else 0
    if strict_rank < p:
        _set_fit_status(
            False, "coefficient_not_identified", rank=p,
            strict_active_rank=strict_rank, duality_gap=dual_gap)
        return None

    def _ratio(value: float, na: float, nb: float, da: float, db: float = 1.0) -> float:
        vm, ve = np.frexp(value)
        am, ae = np.frexp(na)
        bm, be = np.frexp(nb)
        cm, ce = np.frexp(da)
        dm, de = np.frexp(db)
        return float(np.ldexp((vm * am * bm) / (cm * dm), ve + ae + be - ce - de))

    beta = np.empty(p)
    for column in range(p):
        if column == intercept:
            beta[column] = 0.0
        else:
            beta[column] = _ratio(
                float(gamma[column]), y_base, y_spread,
                float(base_scales[column]), float(spreads[column]))
    if intercept is not None:
        residual_for_intercept = target_values - x[:, feature_columns] @ beta[feature_columns]
        beta[intercept] = float(np.quantile(residual_for_intercept, quantile, method="inverted_cdf"))
        beta[intercept] /= float(x[0, intercept])
    if not np.all(np.isfinite(beta)):
        _set_fit_status(False, "numerical_failure")
        return None

    residual = target_values - x @ beta
    if not np.all(np.isfinite(residual)):
        _set_fit_status(False, "numerical_failure")
        return None
    normalized_residual = (residual / y_base) / y_spread
    restored_objective = float(np.sum(np.where(
        normalized_residual >= 0.0,
        quantile * normalized_residual,
        (quantile - 1.0) * normalized_residual)))
    objective_tolerance = 5e-7 * max(1.0, fun_value)
    if (not np.isfinite(restored_objective)
            or abs(restored_objective - fun_value) > objective_tolerance):
        _set_fit_status(False, "numerical_failure", objective=restored_objective)
        return None
    _set_fit_status(
        True, "converged", rank=p, identifiable=True,
        objective=restored_objective, estimated_nnz=estimated_nnz,
        strict_active_rank=strict_rank, duality_gap=dual_gap)
    return beta


def build_design(features: list[np.ndarray], add_intercept: bool) -> np.ndarray:
    cols: list[np.ndarray] = []
    if add_intercept:
        cols.append(np.ones(features[0].shape[0], dtype=float))
    cols.extend(features)
    return np.column_stack(cols)


def rolling_xy(
    y: np.ndarray, xs: list[np.ndarray], window: int, min_periods: int
) -> np.ndarray:
    """Return per-row beta / intercept-free helper --- not used directly.

    Kept for symmetry with ``microstructure``; see ``rolling_fit`` below.
    """
    raise NotImplementedError("use rolling_fit")


def rolling_fit(
    y: np.ndarray,
    xs: list[np.ndarray],
    window: int,
    min_periods: int,
    *,
    fit_fn: Any = ols_fit,
    add_intercept: bool = True,
    extra: Any = None,
    fit_lag: int = 0,
    fit_scope: FitScope | None = None,
    failure_sink: BoundedFitFailureSink | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling regression over a 1-D series.

    Returns ``(beta, resid, resid_std)`` arrays aligned to ``y``.  ``beta`` has
    shape (len(y), n_coeffs).  Each window uses the last ``window`` rows ending
    at ``row - fit_lag`` (``fit_lag=0`` is the legacy in-sample fit whose
    training set includes the current row; ``fit_lag>=1`` fits only on rows
    strictly before the current one and reports the out-of-sample residual at
    the current row).  If fewer than ``min_periods`` finite rows, NaN.
    """
    n = len(y)
    n_coeffs = len(xs) + (1 if add_intercept else 0)
    beta = np.full((n, n_coeffs), np.nan, dtype=float)
    resid = np.full(n, np.nan, dtype=float)
    resid_std = np.full(n, np.nan, dtype=float)
    if n_coeffs <= 0:
        return beta, resid, resid_std
    w = int(window)
    mp = max(int(min_periods), n_coeffs + 1)
    lag = max(0, int(fit_lag))
    sink = failure_sink if failure_sink is not None else _FIT_FAILURE_SINK.get()

    def record_pre_solver(reason: str, start: int, fit_end: int, row: int) -> None:
        if sink is None:
            return
        base_scope = fit_scope if fit_scope is not None else (current_fit_scope() or FitScope())
        sink.record(
            FitResult(None, FitStatus(False, reason)),
            replace(base_scope, window_start=start, window_end=fit_end,
                    output_row=row, fit_cutoff=fit_end),
        )

    for row in range(n):
        fit_end = row - lag
        if fit_end < 0:
            continue
        start = max(0, fit_end - w + 1)
        seg_y = y[start : fit_end + 1]
        seg_xs = [x[start : fit_end + 1] for x in xs]
        valid = np.isfinite(seg_y)
        for x in seg_xs:
            valid &= np.isfinite(x)
        if valid.sum() < mp:
            record_pre_solver("insufficient_sample", start, fit_end, row)
            continue
        vy = seg_y[valid]
        vxs = [x[valid] for x in seg_xs]
        if any(np.std(vx) <= 0.0 for vx in vxs):
            record_pre_solver("singular", start, fit_end, row)
            continue
        design = build_design(vxs, add_intercept)
        result = (fit_result(fit_fn, design, vy) if extra is None
                  else fit_result(fit_fn, design, vy, extra))
        if sink is not None:
            base_scope = fit_scope if fit_scope is not None else (current_fit_scope() or FitScope())
            sink.record(result, replace(
                base_scope, window_start=start, window_end=fit_end,
                output_row=row, fit_cutoff=fit_end))
        b = result.value
        if b is None:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            pred = design @ b
            e = vy - pred
            ddof = max(design.shape[1], 1)
            sd = float(np.sqrt(np.sum(e * e) / max(len(e) - ddof, 1))) if len(e) > ddof else np.nan
        beta[row] = b
        resid_std[row] = sd
        # Residual at the current row uses current x values, only if current
        # y is finite (current x is implicitly finite because row is finite).
        if np.isfinite(y[row]):
            cur_xs = [x[row] for x in xs]
            pred_cur = current_prediction(cur_xs, b, add_intercept)
            resid[row] = float(y[row] - pred_cur)
    return beta, resid, resid_std


def current_prediction(seg_xs: list[float], b: np.ndarray, add_intercept: bool) -> float:
    terms: list[float] = []
    if add_intercept:
        terms.append(1.0)
    terms.extend(seg_xs)
    return float(np.dot(terms, b))


# ---------------------------------------------------------------------------
# Batched fit kernels (R66).
#
# These evaluate the exact same primitive sequence as the scalar kernels
# above, with the elementwise / reduction / matmul stages batched via NumPy
# primitives whose per-slice results are bit-identical to the corresponding
# 2-D calls (verified on this stack: axis-wise median / sum / mean / std and
# stacked ``matmul`` reproduce the 2-D results bit for bit; 1-D ``dot`` and
# ``np.linalg.norm`` equal the stacked matmul forms).  LAPACK solves use
# per-window 2-D ``np.linalg.lstsq`` — the stacked gufunc variant applies a
# different LAPACK workspace strategy and can differ by 1-2 ulp on weighted
# designs, which IRLS amplifies through its convergence path.  Windows that
# hit a rare branch (exact-fit consensus, MAD-degenerate scale, mid-IRLS
# conditioning failure, non-finite iterates) are re-run individually through
# the scalar kernel, so their results are the original ones by construction.
# Only the intercept-first p=2 design produced by ``build_design`` is
# supported — that is the shape every caller in this package uses.
# ---------------------------------------------------------------------------


def _gate_from_s(s: np.ndarray, n: int, p: int) -> np.ndarray:
    """Vectorised well-conditioning gate from lstsq's singular values.

    The scalar kernel gates the design (``np.linalg.svd``) *before* calling
    ``lstsq``; using the singular values that ``np.linalg.lstsq`` already
    computed from the identical matrix gives the same accept/reject outcome
    (both are the exact singular values up to identical LAPACK rounding, and
    the accept set is identical unless a value sits within 1 ulp of the
    threshold), while halving the LAPACK work per iteration.
    """
    s_max = s[:, 0]
    s_min = s[:, -1]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cond = np.where(s_min > 0.0, s_max / s_min, np.inf)
        floor = s_max * max(n, p) * np.finfo(float).eps
        rank_ok = np.sum(s > floor[:, None], axis=1) >= p
    return (np.isfinite(s_max) & (s_max > 0.0)
            & np.isfinite(cond) & (cond <= 1e12) & rank_ok)


def _batch_huber_fit(
    designs: np.ndarray,
    targets: np.ndarray,
    *,
    delta: float = _HUBER_DELTA,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Stacked IRLS Huber fit, bit-identical to per-window ``huber_fit``.

    ``designs`` is ``(K, n, 2)`` (intercept + one feature), ``targets`` is
    ``(K, n)``.  Returns ``(betas, ok)`` where ``betas`` is ``(K, 2)`` and
    ``ok[i]`` is True iff the scalar kernel would have returned a beta for
    window ``i``.
    """
    if not _IN_PROC_WORKER and _PROC_MAX_WORKERS > 1:
        res = _proc_fanout(
            _batch_huber_fit, (designs, targets),
            {"delta": delta, "iterations": iterations, "tolerance": tolerance})
        if res is not None:
            return res
    K, n, p = designs.shape
    eps = np.finfo(float).eps
    cutoff, max_iter, tol = float(delta), int(iterations), float(tolerance)
    betas = np.full((K, p), np.nan, dtype=float)
    ok = np.zeros(K, dtype=bool)
    fallback: list[int] = []
    x = designs
    target = targets

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(target))):
        for i in range(K):
            b = huber_fit(x[i], target[i], delta=delta, iterations=iterations,
                          tolerance=tolerance)
            if b is not None:
                betas[i] = b
                ok[i] = True
        return betas, ok

    none_mask = np.zeros(K, dtype=bool)   # scalar kernel returns None
    active = np.ones(K, dtype=bool)

    # Intercept detection (mirrors the scalar constant-column logic).
    constant = np.all(x == x[:, :1, :], axis=1)             # (K, p)
    constant_nonzero = constant & (x[:, 0, :] != 0.0)
    zero_const = np.any(constant & ~constant_nonzero, axis=1)
    n_intercepts = np.sum(constant_nonzero, axis=1)
    weird = zero_const | (n_intercepts != 1) | ~constant_nonzero[:, 0]
    if np.any(weird):
        idx = np.flatnonzero(weird)
        fallback.extend(int(i) for i in idx)
        active &= ~weird

    # Normalization (intercept = column 0, features = columns 1..p-1).
    x_max = np.max(np.abs(x), axis=1)                        # (K, p)
    nf = p - 1
    if nf:
        feat_max = x_max[:, 1:]
        bad = active & (~np.all(np.isfinite(feat_max), axis=1)
                        | np.any(feat_max <= 0.0, axis=1))
        if np.any(bad):
            fallback.extend(int(i) for i in np.flatnonzero(bad))
            active &= ~bad

        normalized_x = x[:, :, 1:] / feat_max[:, None, :]    # (K, n, nf)
        centers1 = np.mean(normalized_x, axis=1)             # (K, nf)
        centered = normalized_x - centers1[:, None, :]
        # ``np.linalg.norm(col2d, axis=0)`` reduces via the sum path per
        # column; the axis-wise batched sum is bit-identical (ddot and the
        # stacked gufunc matmul are different kernels — do not use them).
        norm2 = np.full((K, nf), np.nan, dtype=float)
        norm2[active] = np.sum(centered[active] * centered[active], axis=1)
        scales1 = np.sqrt(norm2) / np.sqrt(n)
        bad = active & (~np.all(np.isfinite(scales1), axis=1)
                        | np.any(scales1 <= 0.0, axis=1))
        if np.any(bad):
            fallback.extend(int(i) for i in np.flatnonzero(bad))
            active &= ~bad

        z = np.empty_like(x)
        z[:, :, 0] = 1.0
        z[:, :, 1:] = centered / scales1[:, None, :]
    else:
        centers1 = np.zeros((K, 0), dtype=float)
        scales1 = np.zeros((K, 0), dtype=float)
        z = np.empty_like(x)
        z[:, :, 0] = 1.0

    y_max = np.max(np.abs(target), axis=1)
    y_max = np.where(y_max == 0.0, 1.0, y_max)
    normalized_y = target / y_max[:, None]
    y_center = np.mean(normalized_y, axis=1)
    centered_y = normalized_y - y_center[:, None]
    y_spread = np.max(np.abs(centered_y), axis=1)
    y_spread = np.where(y_spread == 0.0, 1.0, y_spread)
    yn = centered_y / y_spread[:, None]

    # Initial OLS seed.  Per-window 2-D ``np.linalg.lstsq`` reproduces the
    # scalar kernel's LAPACK call bit-for-bit (the stacked gufunc variant uses
    # a different LAPACK workspace strategy and can differ by 1-2 ulp on
    # weighted designs, which IRLS then amplifies through its convergence
    # path).  The conditioning gate reuses each solve's singular values.
    gamma = np.full((K, p), np.nan, dtype=float)
    s_arr = np.full((K, p), np.nan, dtype=float)

    def _initial_solve(i: int) -> bool:
        try:
            sol, _res, _rk, s_i = np.linalg.lstsq(z[i], yn[i], rcond=None)
        except (np.linalg.LinAlgError, ValueError):
            return False
        gamma[i] = sol
        s_arr[i] = s_i
        return True

    lstsq_failed = not _parallel_for(np.flatnonzero(active), _initial_solve)
    if lstsq_failed:
        idx = np.flatnonzero(active)
        fallback.extend(int(i) for i in idx)
        active &= False
    if np.any(active):
        gate = _gate_from_s(s_arr, n, p)
        bad = active & ~gate
        if np.any(bad):
            none_mask |= bad
            active &= ~bad
        bad = active & ~np.all(np.isfinite(gamma), axis=1)
        if np.any(bad):
            none_mask |= bad
            active &= ~bad

    if np.any(active):
        norm_z_inf = np.max(np.sum(np.abs(z), axis=2), axis=1)
        norm_g_inf = np.max(np.abs(gamma), axis=1)
        backward_scale = (np.max(np.abs(yn), axis=1)
                          + norm_z_inf * norm_g_inf)
        exact_floor = 64.0 * eps * backward_scale
        col_norm = np.sqrt(np.mean(z * z, axis=1))           # (K, p)
        converged = np.zeros(K, dtype=bool)

        resid: np.ndarray | None = None
        for used in range(1, max_iter + 1):
            if not np.any(active):
                break
            if resid is None:
                resid = np.full((K, n), np.nan, dtype=float)
                for i in np.flatnonzero(active):
                    # 2-D dgemv — bit-identical to the scalar kernel's
                    # z @ gamma; the stacked gufunc matmul is a different
                    # kernel (1-2 ulp).
                    resid[i] = yn[i] - z[i] @ gamma[i]
            # else: carried over from the previous iteration's final_resid —
            # the identical dgemv on identical inputs (gamma unchanged since),
            # so reusing it is bit-for-bit the reference's next resid.
            centered_resid = resid - np.median(resid, axis=1, keepdims=True)
            mad = np.median(np.abs(centered_resid), axis=1)
            residual_extent = np.max(np.abs(centered_resid), axis=1)
            with np.errstate(invalid="ignore"):
                cons_hit = (active & np.isfinite(mad)
                            & (residual_extent > exact_floor)
                            & (mad <= 64.0 * eps * residual_extent))
            if np.any(cons_hit):
                fallback.extend(int(i) for i in np.flatnonzero(cons_hit))
                active &= ~cons_hit
            if not np.any(active):
                break
            scale = 1.4826 * mad
            scale_bad = active & (~np.isfinite(scale) | (scale == 0.0))
            if np.any(scale_bad):
                fallback.extend(int(i) for i in np.flatnonzero(scale_bad))
                active &= ~scale_bad
            if not np.any(active):
                break
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                u = resid / scale[:, None]
                au = np.abs(u)
                weight = np.where(au > cutoff, cutoff / au, 1.0)
            root_w = np.sqrt(weight)
            wz = z * root_w[:, :, None]
            wyn = yn * root_w
            gamma_new = np.full((K, p), np.nan, dtype=float)
            s_new = np.full((K, p), np.nan, dtype=float)

            def _irls_solve(i: int) -> bool:
                try:
                    sol, _res, _rk, s_i = np.linalg.lstsq(wz[i], wyn[i], rcond=None)
                except (np.linalg.LinAlgError, ValueError):
                    return False
                gamma_new[i] = sol
                s_new[i] = s_i
                return True

            # Window solves are independent; any failure sends the whole
            # group to the scalar fallback (same as the sequential
            # first-failure break it replaces).
            lstsq_failed = not _parallel_for(np.flatnonzero(active), _irls_solve)
            if lstsq_failed:
                idx = np.flatnonzero(active)
                fallback.extend(int(i) for i in idx)
                active &= False
                break
            gate2 = _gate_from_s(s_new, n, p)
            bad2 = active & ~gate2
            if np.any(bad2):
                none_mask |= bad2
                active &= ~bad2
            if not np.any(active):
                break
            bad3 = active & ~np.all(np.isfinite(gamma_new), axis=1)
            if np.any(bad3):
                none_mask |= bad3
                active &= ~bad3
            if not np.any(active):
                break
            prediction_delta = np.full((K, n), np.nan, dtype=float)
            for i in np.flatnonzero(active):
                # 2-D dgemv — bit-identical to the scalar kernel.
                prediction_delta[i] = z[i] @ (gamma_new[i] - gamma[i])
            with np.errstate(over="ignore", invalid="ignore"):
                pred_change = (np.sqrt(np.mean(prediction_delta * prediction_delta,
                                               axis=1))
                               / np.maximum(scale, eps))
                param_change = (np.max(np.abs(gamma_new - gamma), axis=1)
                                / np.maximum(1.0, np.max(np.abs(gamma_new), axis=1)))
            gamma = np.where(active[:, None], gamma_new, gamma)
            final_resid = np.full((K, n), np.nan, dtype=float)
            for i in np.flatnonzero(active):
                # 2-D dgemv — bit-identical to the scalar kernel.
                final_resid[i] = yn[i] - z[i] @ gamma[i]
            centered_final = final_resid - np.median(final_resid, axis=1, keepdims=True)
            final_mad = np.median(np.abs(centered_final), axis=1)
            final_extent = np.max(np.abs(centered_final), axis=1)
            with np.errstate(invalid="ignore"):
                cons_hit2 = (active & np.isfinite(final_mad)
                             & (final_extent > exact_floor)
                             & (final_mad <= 64.0 * eps * final_extent))
            if np.any(cons_hit2):
                fallback.extend(int(i) for i in np.flatnonzero(cons_hit2))
                active &= ~cons_hit2
            if not np.any(active):
                break
            final_scale = 1.4826 * final_mad
            fs_bad = active & (~np.isfinite(final_scale) | (final_scale == 0.0))
            if np.any(fs_bad):
                # Scalar kernel either certifies an exact fit or fails closed;
                # both paths are reproduced by re-running the scalar kernel.
                fallback.extend(int(i) for i in np.flatnonzero(fs_bad))
                active &= ~fs_bad
            if not np.any(active):
                break
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                psi = np.clip(final_resid / final_scale[:, None], -cutoff, cutoff)
                zt_psi = np.full((K, p), np.nan, dtype=float)
                for i in np.flatnonzero(active):
                    # 2-D dgemv (transposed view) — bit-identical to the
                    # scalar kernel's z.T @ psi.
                    zt_psi[i] = z[i].T @ psi[i]
                final_score = np.max(
                    np.abs(zt_psi / n) / np.maximum(col_norm, eps), axis=1)
            conv = (active & (pred_change <= tol) & (param_change <= tol)
                    & (final_score <= tol))
            converged |= conv
            active &= ~conv
            # Next iteration's ``resid`` in the scalar kernel is the same
            # dgemv (``yn - z @ gamma``) on inputs unchanged since
            # ``final_resid`` was computed — reuse it bit-for-bit.
            resid = final_resid
        non_converged = active
        none_mask |= non_converged

        # Beta restoration + tail diagnostics for converged windows (scalar
        # arithmetic, cheap: K is the group size).
        for i in np.flatnonzero(converged):
            g = gamma[i]
            ym = float(y_max[i])
            ys_ = float(y_spread[i])
            yc = float(y_center[i])
            zi = z[i]

            def _scaled_ratio(value: float, numerator: float,
                              denominator_a: float,
                              denominator_b: float = 1.0) -> float:
                vm, ve = np.frexp(value)
                nm, ne = np.frexp(numerator)
                am, ae = np.frexp(denominator_a)
                bm, be = np.frexp(denominator_b)
                return float(np.ldexp((vm * nm) / (am * bm), ve + ne - ae - be))

            beta = np.empty(p, dtype=float)
            beta[0] = _scaled_ratio(float(g[0]), ym, 1.0)
            for c in range(1, p):
                beta[c] = _scaled_ratio(
                    float(g[c] * ys_), ym, float(x_max[i, c]),
                    float(scales1[i, c - 1]))
            normalized_intercept = yc + ys_ * float(g[0])
            if p > 1:
                normalized_intercept -= ys_ * float(
                    np.sum(g[1:] * centers1[i] / scales1[i]))
            beta[0] = _scaled_ratio(normalized_intercept, ym, float(x[i, 0, 0]))
            if not np.all(np.isfinite(beta)):
                none_mask[i] = True
                continue
            original_resid = target[i] - x[i] @ beta
            if not np.all(np.isfinite(original_resid)):
                none_mask[i] = True
                continue
            reported_scale = 1.4826 * float(
                np.median(np.abs(original_resid - np.median(original_resid))))
            if not np.isfinite(reported_scale) or reported_scale == 0.0:
                # Rare ``_stable_std`` fallback — re-run the scalar kernel.
                fallback.append(int(i))
                continue
            reported_psi = np.clip(original_resid / reported_scale, -cutoff, cutoff)
            reported_score = float(np.max(
                np.abs((zi.T @ reported_psi) / n)
                / np.maximum(col_norm[i], eps)))
            if reported_score > tol:
                none_mask[i] = True
                continue
            betas[i] = beta
            ok[i] = True

    # Rare-branch windows: reproduce the scalar kernel exactly.
    for i in fallback:
        b = huber_fit(x[i], target[i], delta=delta, iterations=iterations,
                      tolerance=tolerance)
        if b is not None:
            betas[i] = b
            ok[i] = True
    return betas, ok & ~none_mask


def _batch_expectile_fit(
    designs: np.ndarray,
    targets: np.ndarray,
    q: float,
    iterations: int = 8,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Stacked IRLS expectile fit, bit-identical to per-window ``quantile_fit``.

    Returns ``(betas, ok)`` with the same contract as :func:`_batch_huber_fit`.
    Uses per-window 2-D ``np.linalg.lstsq`` (bit-identical LAPACK calls to the
    scalar kernel); the conditioning gate reuses each solve's singular values.
    """
    if not _IN_PROC_WORKER and _PROC_MAX_WORKERS > 1:
        res = _proc_fanout(
            _batch_expectile_fit, (designs, targets, q),
            {"iterations": iterations, "tolerance": tolerance})
        if res is not None:
            return res
    K, n, p = designs.shape
    betas = np.full((K, p), np.nan, dtype=float)
    ok = np.zeros(K, dtype=bool)
    active = np.ones(K, dtype=bool)
    fallback: list[int] = []

    def _ols_batched(d2: np.ndarray, t2: np.ndarray):
        """Per-window lstsq + conditioning gate; returns (beta, gate, failed)."""
        beta = np.full((K, p), np.nan, dtype=float)
        s_arr = np.full((K, p), np.nan, dtype=float)

        def _solve(i: int) -> bool:
            try:
                sol, _res, _rk, s_i = np.linalg.lstsq(d2[i], t2[i], rcond=None)
            except (np.linalg.LinAlgError, ValueError):
                return False
            beta[i] = sol
            s_arr[i] = s_i
            return True

        # Window solves are independent; any failure sends the whole group
        # to the scalar fallback (same as the sequential first-failure break).
        failed = not _parallel_for(np.flatnonzero(active), _solve)
        gate = _gate_from_s(s_arr, n, p) if not failed else np.zeros(K, dtype=bool)
        return beta, gate, failed

    beta, gate, failed = _ols_batched(designs, targets)
    if failed:
        fallback.extend(int(i) for i in range(K))
        active &= False
    else:
        active &= gate
    for _ in range(int(iterations)):
        if not np.any(active):
            break
        resid = np.full((K, n), np.nan, dtype=float)
        for i in np.flatnonzero(active):
            # 2-D dgemv — bit-identical to the scalar kernel's design @ beta.
            resid[i] = targets[i] - designs[i] @ beta[i]
        weight = np.where(resid > 0, q, 1.0 - q)
        weight = np.clip(weight, 1e-6, None)
        sqrt_w = np.sqrt(weight)
        d2 = designs * sqrt_w[:, :, None]
        t2 = targets * sqrt_w
        beta_new, gate2, failed = _ols_batched(d2, t2)
        if failed:
            fallback.extend(int(i) for i in np.flatnonzero(active))
            active &= False
            break
        bad3 = active & ~gate2
        if np.any(bad3):
            active &= ~bad3
        if not np.any(active):
            break
        bad4 = active & ~np.all(np.isfinite(beta_new), axis=1)
        if np.any(bad4):
            active &= ~bad4
        if not np.any(active):
            break
        with np.errstate(invalid="ignore"):
            change = np.max(np.abs(beta_new - beta), axis=1)
            conv = active & (change < tolerance)
        ok |= conv
        betas[conv] = beta_new[conv]
        active &= ~conv
        beta = np.where(active[:, None], beta_new, beta)
    for i in np.flatnonzero(active):
        fallback.append(int(i))
    for i in fallback:
        b = quantile_fit(designs[i], targets[i], q,
                         iterations=iterations, tolerance=tolerance)
        if b is not None:
            betas[i] = b
            ok[i] = True
    return betas, ok



def _pinball_post_solve(
    z: np.ndarray,
    yn: np.ndarray,
    orig: np.ndarray,
    target: np.ndarray,
    quantile: float,
    base_scales: np.ndarray,
    spreads: np.ndarray,
    y_base: float,
    y_spread: float,
) -> tuple[bool, np.ndarray | None]:
    """Per-window HiGHS solve + certificates, mirroring ``pinball_quantile_fit``.

    ``z`` is the normalized design, ``orig`` the original one.  Returns
    ``(ok, beta)`` — ``ok=False`` whenever the scalar kernel would have
    returned ``None`` (no exception is raised).
    """
    n, p = z.shape
    solved = None
    try:
        solved = _highs_solve_eq(z, yn, quantile)
    except (ImportError, MemoryError, RuntimeError, ValueError, OverflowError):
        solved = None
    if solved is None:
        return False, None
    primal_full, dual, fun_value = solved
    if not np.isfinite(fun_value):
        return False, None
    gamma = np.asarray(primal_full[:p], dtype=float)
    positive = np.asarray(primal_full[p : p + n], dtype=float)
    negative = np.asarray(primal_full[p + n :], dtype=float)
    if (gamma.shape != (p,) or positive.shape != (n,) or negative.shape != (n,)
            or not np.all(np.isfinite(gamma))
            or not np.all(np.isfinite(positive))
            or not np.all(np.isfinite(negative))):
        return False, None
    z_gamma = z @ gamma
    primal_residual = z_gamma + positive - negative - yn
    certificate_tol = 1e-8 * max(1.0, float(np.max(np.abs(yn))))
    if (float(np.max(np.abs(primal_residual))) > certificate_tol
            or np.any(positive < -certificate_tol)
            or np.any(negative < -certificate_tol)):
        return False, None
    if dual.shape != (n,) or not np.all(np.isfinite(dual)):
        return False, None
    dual_gap = fun_value - float(yn @ dual)
    dual_tol = 1e-7 * max(1.0, abs(fun_value))
    stationarity = (z.T @ dual) / n
    if (np.any(dual > quantile + dual_tol)
            or np.any(dual < quantile - 1.0 - dual_tol)
            or abs(dual_gap) > dual_tol
            or not np.all(np.isfinite(stationarity))
            or float(np.max(np.abs(stationarity))) > dual_tol):
        return False, None
    fitted_residual = yn - z_gamma
    strict = ((np.abs(fitted_residual) <= certificate_tol)
              & (dual < quantile - dual_tol)
              & (dual > quantile - 1.0 + dual_tol))
    strict_rank = int(np.linalg.matrix_rank(z[strict])) if np.any(strict) else 0
    if strict_rank < p:
        return False, None

    def _ratio(value: float, na: float, nb: float, da: float, db: float = 1.0) -> float:
        vm, ve = np.frexp(value)
        am, ae = np.frexp(na)
        bm, be = np.frexp(nb)
        cm, ce = np.frexp(da)
        dm, de = np.frexp(db)
        return float(np.ldexp((vm * am * bm) / (cm * dm), ve + ae + be - ce - de))

    beta = np.empty(p)
    for column in range(1, p):
        beta[column] = _ratio(
            float(gamma[column]), y_base, y_spread,
            float(base_scales[column]), float(spreads[column]))
    # Intercept restoration against the ORIGINAL design (scalar kernel lines):
    #   residual_for_intercept = target - x[:, feat] @ beta[feat]
    #   beta[intercept] = quantile(residual, q, inverted_cdf) / x[0, intercept]
    if p > 1:
        residual_for_intercept = target - orig[:, 1:] @ beta[1:]
        beta0 = float(np.quantile(residual_for_intercept, quantile,
                                  method="inverted_cdf"))
        beta[0] = beta0 / float(orig[0, 0])
    if not np.all(np.isfinite(beta)):
        return False, None
    residual = target - orig @ beta
    if not np.all(np.isfinite(residual)):
        return False, None
    normalized_residual = (residual / y_base) / y_spread
    with np.errstate(invalid="ignore"):
        restored_objective = float(np.sum(np.where(
            normalized_residual >= 0.0,
            quantile * normalized_residual,
            (quantile - 1.0) * normalized_residual)))
    objective_tolerance = 5e-7 * max(1.0, fun_value)
    if (not np.isfinite(restored_objective)
            or abs(restored_objective - fun_value) > objective_tolerance):
        return False, None
    return True, beta


def _batch_pinball_fit(
    designs: np.ndarray,
    targets: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched pinball LP fit, bit-identical to per-window ``pinball_quantile_fit``.

    ``designs`` is ``(K, n, p)``, ``targets`` is ``(K, n)`` — every window in a
    group shares the same valid count ``n``.  The per-window normalization is
    vectorised with elementwise / axis-wise reductions (verified bit-identical
    to the scalar 1-D forms); the conditioning gate, the HiGHS solve and the
    dgemv-based optimality certificates stay per-window so every accept/reject
    decision and every returned coefficient is computed exactly like the
    scalar kernel.  Windows on rare branches (no leading intercept column,
    degenerate designs) are re-run through ``pinball_quantile_fit``.

    Returns ``(betas, ok)`` where ``ok[i]`` is True iff the scalar kernel would
    have returned a beta for window ``i``.
    """
    if not _IN_PROC_WORKER and _PROC_MAX_WORKERS > 1:
        res = _proc_fanout(_batch_pinball_fit, (designs, targets, q), {})
        if res is not None:
            return res
    K, n, p = designs.shape
    betas = np.full((K, p), np.nan, dtype=float)
    ok = np.zeros(K, dtype=bool)
    quantile = float(q)

    fallback: list[int] = []
    if n < p + 2 or p < 2:
        return betas, ok          # scalar kernel returns None for every window

    if not (np.all(np.isfinite(designs)) and np.all(np.isfinite(targets))):
        for i in range(K):
            b = pinball_quantile_fit(designs[i], targets[i], quantile)
            if b is not None:
                betas[i] = b
                ok[i] = True
        return betas, ok

    # Intercept detection (mirrors the scalar constant-column logic): the
    # batch path requires the single intercept column to be column 0.
    constant = np.all(designs == designs[:, :1, :], axis=1)            # (K, p)
    constant_nonzero = constant & (designs[:, 0, :] != 0.0)
    weird = (np.any(constant & ~constant_nonzero, axis=1)
             | (np.sum(constant_nonzero, axis=1) != 1)
             | ~constant_nonzero[:, 0])
    if np.any(weird):
        fallback.extend(int(i) for i in np.flatnonzero(weird))
    active = ~np.asarray(weird, dtype=bool)

    orig = designs                     # never mutated — needed for restoration
    z = designs.copy()                 # becomes the normalized design
    base_scales = np.ones((K, p), dtype=float)
    spreads = np.ones((K, p), dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for j in range(1, p):
            col_x = orig[:, :, j]
            delta_x = col_x - col_x[:, :1]
            all_fin = np.all(np.isfinite(delta_x), axis=1)
            base_a = np.max(np.abs(delta_x), axis=1)
            nd_a = delta_x / base_a[:, None]
            base_b = np.max(np.abs(col_x), axis=1)
            nx_b = col_x / base_b[:, None]
            nd_b = nx_b - nx_b[:, :1]
            base = np.where(all_fin, base_a, base_b)
            # all_fin & base>0 -> delta/base;  all_fin & base<=0 -> raw delta
            # (flagged singular below);  ~all_fin -> normalized-minus-first.
            normalized_delta = np.where(
                (all_fin & (base_a > 0.0))[:, None], nd_a,
                np.where(all_fin[:, None], delta_x, nd_b))
            center = np.mean(normalized_delta, axis=1)
            centered = normalized_delta - center[:, None]
            spread = np.max(np.abs(centered), axis=1)
            active &= ~((base <= 0.0) | (spread <= 0.0))
            zj = centered / spread[:, None]
            with np.errstate(invalid="ignore"):
                zj_finite = np.all(np.isfinite(zj), axis=1)
            active &= zj_finite
            base_scales[:, j] = base
            spreads[:, j] = spread
            z[:, :, j] = np.where(active[:, None], zj, z[:, :, j])

        delta_y = targets - targets[:, :1]
        all_fin_y = np.all(np.isfinite(delta_y), axis=1)
        y_base_a = np.max(np.abs(delta_y), axis=1)
        ndy_a = delta_y / y_base_a[:, None]
        y_base_b = np.max(np.abs(targets), axis=1)
        ny_b = targets / y_base_b[:, None]
        ndy_b = ny_b - ny_b[:, :1]
        use_a = all_fin_y & (y_base_a > 0.0)
        normalized_delta_y = np.where(
            use_a[:, None], ndy_a,
            np.where(all_fin_y[:, None], delta_y, ndy_b))
        y_base = np.where(use_a, y_base_a, y_base_b)
        y_center_delta = np.mean(normalized_delta_y, axis=1)
        centered_y = normalized_delta_y - y_center_delta[:, None]
        y_spread_a = np.max(np.abs(centered_y), axis=1)
        # Degenerate y: the scalar kernel resets to a zero normalized target.
        degen = ((~use_a) & (y_base_b <= 0.0)) | (y_spread_a <= 0.0)
        y_base = np.where(degen, np.maximum(1.0, y_base_b), y_base)
        y_spread = np.where(degen, 1.0, y_spread_a)
        yn = np.where(degen[:, None], 0.0, centered_y / y_spread[:, None])

    def _solve_window(i: int) -> bool:
        if not design_is_well_conditioned(z[i]):
            return True
        good, beta_i = _pinball_post_solve(
            z[i], yn[i], orig[i], targets[i], quantile,
            base_scales[i], spreads[i], y_base[i], y_spread[i])
        if good:
            betas[i] = beta_i
            ok[i] = True
        return True

    # Windows are fully independent (fresh HiGHS instance per solve, the LP
    # skeleton cache is thread-local), so parallel scheduling cannot change
    # any accept/reject decision or coefficient.
    _parallel_for(np.flatnonzero(active), _solve_window)

    for i in fallback:
        b = pinball_quantile_fit(designs[i], targets[i], quantile)
        if b is not None:
            betas[i] = b
            ok[i] = True
    return betas, ok


def fit_failure_sink_active() -> bool:
    """True when a caller-owned fit-failure sink is bound in this context.

    The batched dispatch paths skip ``record_current_fit`` bookkeeping (it is a
    no-op without a sink); callers must check this before selecting them.
    """
    return _FIT_FAILURE_SINK.get() is not None
