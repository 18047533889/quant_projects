"""
Distributed tracing support for factor_engine.

Provides span-based tracing with optional OpenTelemetry backend.
All tracing is privacy-safe and opt-in only.
"""

from __future__ import annotations

import contextvars
import functools
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union


class SpanKind(Enum):
    """Type of span."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a completed span."""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass
class SpanContext:
    """Context information for a span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None

    @classmethod
    def create_root(cls) -> SpanContext:
        """Create a root span context."""
        return cls(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=None,
        )

    def create_child(self) -> SpanContext:
        """Create a child span context."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=self.span_id,
        )


@dataclass
class SpanData:
    """Recorded span data."""
    context: SpanContext
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    error: Optional[Exception] = None

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class SpanEvent:
    """An event recorded during a span."""
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)


class Span:
    """A single trace span."""

    def __init__(
        self,
        context: SpanContext,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        tracer: Optional[Tracer] = None,
    ):
        self._context = context
        self._name = name
        self._kind = kind
        self._tracer = tracer
        self._start_time = time.time()
        self._end_time: Optional[float] = None
        self._status = SpanStatus.UNSET
        self._attributes: Dict[str, Any] = {}
        self._events: List[SpanEvent] = []
        self._error: Optional[Exception] = None
        self._lock = threading.Lock()

    @property
    def context(self) -> SpanContext:
        """Get the span context."""
        return self._context

    def set_attribute(self, key: str, value: Any) -> Span:
        """Set a span attribute."""
        with self._lock:
            self._attributes[key] = value
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> Span:
        """Set multiple span attributes."""
        with self._lock:
            self._attributes.update(attributes)
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Span:
        """Add an event to the span."""
        event = SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {},
        )
        with self._lock:
            self._events.append(event)
        return self

    def set_status(self, status: SpanStatus, error: Optional[Exception] = None) -> Span:
        """Set the span status."""
        with self._lock:
            self._status = status
            if error is not None:
                self._error = error
        return self

    def end(self) -> None:
        """End the span."""
        with self._lock:
            if self._end_time is not None:
                return  # Already ended
            self._end_time = time.time()

        if self._tracer is not None:
            span_data = SpanData(
                context=self._context,
                name=self._name,
                kind=self._kind,
                start_time=self._start_time,
                end_time=self._end_time,
                status=self._status,
                attributes=dict(self._attributes),
                events=list(self._events),
                error=self._error,
            )
            self._tracer._record_span(span_data)

    def __enter__(self) -> Span:
        """Enter span context."""
        if self._tracer is not None:
            self._tracer._push_span(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit span context."""
        if exc_type is not None:
            self.set_status(SpanStatus.ERROR, exc_val)
        elif self._status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)

        self.end()

        if self._tracer is not None:
            self._tracer._pop_span()


# Context variable to track current span
_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "current_span", default=None
)


class Tracer:
    """A tracer for creating spans."""

    def __init__(self, name: str, provider: Optional[TracerProvider] = None):
        self._name = name
        self._provider = provider

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        if self._provider is None or not self._provider.is_enabled():
            # Return a no-op span
            return Span(
                context=SpanContext.create_root(),
                name=name,
                kind=kind,
                tracer=None,
            )

        # Get parent context
        parent_span = _current_span.get()
        if parent_span is not None:
            context = parent_span.context.create_child()
        else:
            context = SpanContext.create_root()

        span = Span(context, name, kind, tracer=self)

        if attributes:
            span.set_attributes(attributes)

        return span

    @contextmanager
    def trace(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """Context manager for tracing a block of code."""
        span = self.start_span(name, kind, attributes)
        with span:
            yield span

    def _push_span(self, span: Span) -> None:
        """Push a span onto the context stack."""
        _current_span.set(span)

    def _pop_span(self) -> None:
        """Pop a span from the context stack."""
        current = _current_span.get()
        if current is not None:
            parent_id = current.context.parent_span_id
            # Restore parent span (simplified - in production would need a stack)
            _current_span.set(None)

    def _record_span(self, span_data: SpanData) -> None:
        """Record a completed span."""
        if self._provider is not None:
            self._provider._record_span(span_data)


class SpanProcessor:
    """Base class for span processors."""

    def on_start(self, span: Span) -> None:
        """Called when a span starts."""
        pass

    def on_end(self, span_data: SpanData) -> None:
        """Called when a span ends."""
        pass

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass


class SimpleSpanProcessor(SpanProcessor):
    """A simple span processor that immediately exports spans."""

    def __init__(self, exporter: SpanExporter):
        self._exporter = exporter

    def on_end(self, span_data: SpanData) -> None:
        """Export span immediately."""
        try:
            self._exporter.export([span_data])
        except Exception:
            pass  # Don't let exporter failures break the application

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._exporter.shutdown()


class BatchSpanProcessor(SpanProcessor):
    """A span processor that batches spans before export."""

    def __init__(
        self,
        exporter: SpanExporter,
        max_batch_size: int = 512,
        export_interval: float = 5.0,
    ):
        self._exporter = exporter
        self._max_batch_size = max_batch_size
        self._export_interval = export_interval
        self._batch: List[SpanData] = []
        self._lock = threading.Lock()
        self._shutdown = False

        # Start background export thread
        self._thread = threading.Thread(target=self._export_loop, daemon=True)
        self._thread.start()

    def on_end(self, span_data: SpanData) -> None:
        """Add span to batch."""
        with self._lock:
            if self._shutdown:
                return

            self._batch.append(span_data)

            if len(self._batch) >= self._max_batch_size:
                self._flush()

    def _flush(self) -> None:
        """Flush the current batch (must be called with lock held)."""
        if not self._batch:
            return

        batch = self._batch[:]
        self._batch.clear()

        try:
            self._exporter.export(batch)
        except Exception:
            pass  # Don't let exporter failures break the application

    def _export_loop(self) -> None:
        """Background thread that exports batches periodically."""
        while not self._shutdown:
            time.sleep(self._export_interval)
            with self._lock:
                self._flush()

    def shutdown(self) -> None:
        """Shutdown the processor."""
        with self._lock:
            self._shutdown = True
            self._flush()

        self._thread.join(timeout=5.0)
        self._exporter.shutdown()


class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: List[SpanData]) -> None:
        """Export a batch of spans."""
        pass

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """A span exporter that prints to console."""

    def export(self, spans: List[SpanData]) -> None:
        """Print spans to console."""
        for span in spans:
            duration = span.duration or 0.0
            print(
                f"Span: {span.name} "
                f"[{span.kind.value}] "
                f"trace_id={span.context.trace_id[:8]}... "
                f"span_id={span.context.span_id[:8]}... "
                f"duration={duration*1000:.2f}ms "
                f"status={span.status.value}"
            )
            if span.attributes:
                print(f"  Attributes: {span.attributes}")
            for event in span.events:
                print(f"  Event: {event.name} @ {event.timestamp}")


class InMemorySpanExporter(SpanExporter):
    """A span exporter that stores spans in memory (for testing)."""

    def __init__(self):
        self._spans: List[SpanData] = []
        self._lock = threading.Lock()

    def export(self, spans: List[SpanData]) -> None:
        """Store spans in memory."""
        with self._lock:
            self._spans.extend(spans)

    def get_spans(self) -> List[SpanData]:
        """Get all recorded spans."""
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """Clear all recorded spans."""
        with self._lock:
            self._spans.clear()


class TracerProvider:
    """Provider for creating tracers."""

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._tracers: Dict[str, Tracer] = {}
        self._processors: List[SpanProcessor] = []
        self._lock = threading.Lock()

    def enable(self) -> None:
        """Enable tracing (opt-in)."""
        self._enabled = True

    def disable(self) -> None:
        """Disable tracing."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    def get_tracer(self, name: str) -> Tracer:
        """Get or create a tracer."""
        with self._lock:
            if name not in self._tracers:
                self._tracers[name] = Tracer(name, provider=self)
            return self._tracers[name]

    def add_span_processor(self, processor: SpanProcessor) -> None:
        """Add a span processor."""
        with self._lock:
            self._processors.append(processor)

    def _record_span(self, span_data: SpanData) -> None:
        """Record a completed span."""
        if not self._enabled:
            return

        for processor in self._processors:
            try:
                processor.on_end(span_data)
            except Exception:
                pass  # Don't let processor failures break the application

    def shutdown(self) -> None:
        """Shutdown all processors."""
        for processor in self._processors:
            try:
                processor.shutdown()
            except Exception:
                pass


# Global tracer provider
_global_provider: Optional[TracerProvider] = None
_provider_lock = threading.Lock()


def get_tracer_provider() -> TracerProvider:
    """Get the global tracer provider."""
    global _global_provider
    if _global_provider is None:
        with _provider_lock:
            if _global_provider is None:
                _global_provider = TracerProvider(enabled=False)
    return _global_provider


def get_tracer(name: str) -> Tracer:
    """Get a tracer from the global provider."""
    return get_tracer_provider().get_tracer(name)


def configure_tracing(enabled: bool = False) -> TracerProvider:
    """Configure the global tracer provider."""
    provider = get_tracer_provider()
    if enabled:
        provider.enable()
    else:
        provider.disable()
    return provider


# Decorator for tracing functions
F = TypeVar("F", bound=Callable[..., Any])


def trace(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to trace a function."""
    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        tracer = get_tracer(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace(span_name, kind, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.add_event("exception", {"exception": str(e)})
                    raise

        return wrapper  # type: ignore

    return decorator
