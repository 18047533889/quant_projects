"""R21-171..183: structured JSON logging, core metrics, and trace spans.

Metrics are name->counter/observer accumulators with bounded label cardinality
(R21-177: factor names are never high-cardinality labels).  Logging emits
single-line JSON with run_id / request_id / execution_id / principal so every
backend/fallback/DQ/materialization line is correlatable (R21-172).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger("factor_engine.service")

_CORE_METRICS = [
    "job_queue_depth",
    "job_wait_seconds",
    "run_latency",
    "compile_latency",
    "source_read_latency",
    "rows_read",
    "bytes_read",
    "peak_rss_mb",
    "cache_hit",
    "cache_miss",
    "backend_routes",
    "fallback_count",
    "dq_failures",
    "materialize_bytes",
    "sql_queries",
    "job_cancel",
    "job_timeout",
    "auth_failures",
    "idempotency_conflicts",
    "error_total",
]


@dataclass
class MetricsRegistry:
    counters: dict[tuple[str, str], int] = field(default_factory=dict)
    gauges: dict[tuple[str, str], float] = field(default_factory=dict)
    histograms: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> tuple[str, str]:
        label_key = "|".join(f"{k}={v}" for k, v in sorted((labels or {}).items()))
        return (name, label_key)

    def incr(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self.gauges[key] = float(value)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self.histograms.setdefault(key, []).append(float(value))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = {
                name: self._aggregate_counters(name) for name, _ in self.counters
            }
            # collapse label variants into totals + per-label breakdown
            return {
                "counters": {n: v for n, v in sorted(counters.items())},
                "gauges": {f"{n}{k}": v for (n, k), v in sorted(self.gauges.items())},
                "histogram_totals": {
                    f"{n}{k}": round(sum(v), 2) for (n, k), v in sorted(self.histograms.items())
                },
            }

    def _aggregate_counters(self, name: str) -> int:
        return sum(v for (n, _), v in self.counters.items() if n == name)

    def counter_value(self, name: str, labels: dict[str, str] | None = None) -> int:
        return self.counters.get(self._key(name, labels), 0)


METRICS = MetricsRegistry()


class _ContextVars(threading.local):
    def __init__(self) -> None:
        self.request_id: str | None = None
        self.run_id: str | None = None
        self.execution_id: str | None = None
        self.principal: str | None = None


_CONTEXT = _ContextVars()


def bind_log_context(*, request_id: str | None = None, run_id: str | None = None,
                     execution_id: str | None = None, principal: str | None = None) -> None:
    if request_id is not None:
        _CONTEXT.request_id = request_id
    if run_id is not None:
        _CONTEXT.run_id = run_id
    if execution_id is not None:
        _CONTEXT.execution_id = execution_id
    if principal is not None:
        _CONTEXT.principal = principal


def _base_ctx() -> dict[str, str]:
    return {
        "request_id": _CONTEXT.request_id or "",
        "run_id": _CONTEXT.run_id or "",
        "execution_id": _CONTEXT.execution_id or "",
        "principal": _CONTEXT.principal or "",
    }


def json_log(level: int, event: str, **fields: Any) -> None:
    """Emit a single-line structured JSON log record (R21-173)."""
    payload = {"event": event, **_base_ctx(), **fields}
    logger.log(level, "%s", json.dumps(payload, ensure_ascii=False, default=str))


def info(event: str, **fields: Any) -> None:
    json_log(logging.INFO, event, **fields)


def warning(event: str, **fields: Any) -> None:
    json_log(logging.WARNING, event, **fields)


def error(event: str, **fields: Any) -> None:
    json_log(logging.ERROR, event, **fields)


def new_execution_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def trace_span(span: str, *, run_id: str | None = None, **fields: Any) -> Iterator[None]:
    """R21-178..180: minimal trace span (OpenTelemetry-shaped) with timings.

    Usage:
        with trace_span("source.read", source=src): ...
    """
    start = time.monotonic()
    info(f"span.start:{span}", **fields)
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        METRICS.observe(f"span_ms:{span}", round(elapsed * 1000, 2))
        info(f"span.end:{span}", duration_ms=round(elapsed * 1000, 2), **fields)
