# -*- coding: utf-8 -*-
"""Shared rolling regression kernels for time-series model operators."""
from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from types import MappingProxyType
from threading import Lock
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata

try:  # HiGHS LP for true pinball-loss quantile regression.
    from scipy.optimize import linprog as _linprog
except Exception:  # pragma: no cover
    _linprog = None

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
        from scipy.sparse import csr_matrix, eye, hstack
        sparse_z = csr_matrix(z)
        identity = eye(n, format="csr")
        a_eq = hstack([sparse_z, identity, -identity], format="csr")
        objective = np.concatenate([
            np.zeros(p), quantile * np.ones(n), (1.0 - quantile) * np.ones(n)])
        bounds = [(None, None)] * p + [(0.0, None)] * (2 * n)
        result = _linprog(
            objective, A_eq=a_eq, b_eq=yn, bounds=bounds, method="highs",
            options={"maxiter": 10_000, "time_limit": 5.0})
    except (ImportError, MemoryError, RuntimeError, ValueError, OverflowError):
        _set_fit_status(False, "non_converged")
        return None
    if (not getattr(result, "success", False) or result.status != 0
            or result.x is None or result.fun is None or not np.isfinite(result.fun)):
        _set_fit_status(False, "non_converged")
        return None
    gamma = np.asarray(result.x[:p], dtype=float)
    positive = np.asarray(result.x[p : p + n], dtype=float)
    negative = np.asarray(result.x[p + n :], dtype=float)
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
    try:
        dual = np.asarray(result.eqlin.marginals, dtype=float)
    except (AttributeError, TypeError, ValueError):
        _set_fit_status(False, "coefficient_not_identified")
        return None
    if dual.shape != (n,) or not np.all(np.isfinite(dual)):
        _set_fit_status(False, "coefficient_not_identified")
        return None
    dual_gap = float(result.fun) - float(yn @ dual)
    dual_tol = 1e-7 * max(1.0, abs(float(result.fun)))
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
    objective_tolerance = 5e-7 * max(1.0, float(result.fun))
    if (not np.isfinite(restored_objective)
            or abs(restored_objective - float(result.fun)) > objective_tolerance):
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
