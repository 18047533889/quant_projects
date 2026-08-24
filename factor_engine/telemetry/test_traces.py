"""
Tests for telemetry.traces module.
"""

import time
from factor_engine.telemetry.traces import (
    Span,
    SpanKind,
    SpanStatus,
    Tracer,
    TracerProvider,
    InMemorySpanExporter,
    SimpleSpanProcessor,
    get_tracer,
    get_tracer_provider,
    configure_tracing,
    trace,
)


def test_span_context_creation():
    """Test span context creation."""
    from factor_engine.telemetry.traces import SpanContext

    root = SpanContext.create_root()
    assert root.trace_id
    assert root.span_id
    assert root.parent_span_id is None

    child = root.create_child()
    assert child.trace_id == root.trace_id
    assert child.span_id != root.span_id
    assert child.parent_span_id == root.span_id


def test_span_basic():
    """Test basic span operations."""
    from factor_engine.telemetry.traces import SpanContext

    context = SpanContext.create_root()
    span = Span(context, "test_span")

    assert span.context == context
    span.set_attribute("key", "value")
    span.add_event("test_event")
    span.set_status(SpanStatus.OK)
    span.end()


def test_span_context_manager():
    """Test span as context manager."""
    from factor_engine.telemetry.traces import SpanContext

    context = SpanContext.create_root()
    span = Span(context, "test_span")

    with span:
        span.set_attribute("inside", "context")

    # Span should be ended after context exit


def test_span_error_handling():
    """Test span error handling in context manager."""
    from factor_engine.telemetry.traces import SpanContext

    context = SpanContext.create_root()
    span = Span(context, "test_span")

    try:
        with span:
            raise ValueError("test error")
    except ValueError:
        pass

    # Span should have error status (but we can't easily check without tracer)


def test_tracer_disabled():
    """Test tracer when provider is disabled."""
    provider = TracerProvider(enabled=False)
    tracer = provider.get_tracer("test")

    span = tracer.start_span("test_span")
    span.set_attribute("key", "value")
    span.end()

    # Should not record anything when disabled


def test_tracer_enabled():
    """Test tracer when provider is enabled."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider(enabled=True)
    provider.add_span_processor(processor)

    tracer = provider.get_tracer("test")

    span = tracer.start_span("test_span")
    span.set_attribute("key", "value")
    span.end()

    spans = exporter.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_span"
    assert spans[0].attributes["key"] == "value"


def test_tracer_context_manager():
    """Test tracer with context manager."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider(enabled=True)
    provider.add_span_processor(processor)

    tracer = provider.get_tracer("test")

    with tracer.trace("test_span") as span:
        span.set_attribute("inside", "context")

    spans = exporter.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_span"
    assert spans[0].attributes["inside"] == "context"
    assert spans[0].status == SpanStatus.OK


def test_tracer_nested_spans():
    """Test nested spans."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider(enabled=True)
    provider.add_span_processor(processor)

    tracer = provider.get_tracer("test")

    with tracer.trace("parent") as parent:
        parent.set_attribute("level", "parent")

        with tracer.trace("child") as child:
            child.set_attribute("level", "child")

    spans = exporter.get_spans()
    assert len(spans) == 2

    # Find parent and child
    parent_span = next(s for s in spans if s.name == "parent")
    child_span = next(s for s in spans if s.name == "child")

    assert parent_span.attributes["level"] == "parent"
    assert child_span.attributes["level"] == "child"

    # Child should reference parent
    assert child_span.context.parent_span_id == parent_span.context.span_id
    assert child_span.context.trace_id == parent_span.context.trace_id


def test_span_events():
    """Test span events."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider(enabled=True)
    provider.add_span_processor(processor)

    tracer = provider.get_tracer("test")

    with tracer.trace("test_span") as span:
        span.add_event("event1", {"detail": "first"})
        span.add_event("event2", {"detail": "second"})

    spans = exporter.get_spans()
    assert len(spans) == 1

    events = spans[0].events
    assert len(events) == 2
    assert events[0].name == "event1"
    assert events[0].attributes["detail"] == "first"
    assert events[1].name == "event2"
    assert events[1].attributes["detail"] == "second"


def test_span_duration():
    """Test span duration calculation."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider(enabled=True)
    provider.add_span_processor(processor)

    tracer = provider.get_tracer("test")

    with tracer.trace("test_span"):
        time.sleep(0.01)

    spans = exporter.get_spans()
    assert len(spans) == 1

    duration = spans[0].duration
    assert duration is not None
    assert duration >= 0.01


def test_span_kinds():
    """Test different span kinds."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider(enabled=True)
    provider.add_span_processor(processor)

    tracer = provider.get_tracer("test")

    for kind in SpanKind:
        with tracer.trace(f"span_{kind.value}", kind=kind):
            pass

    spans = exporter.get_spans()
    assert len(spans) == len(SpanKind)

    for kind in SpanKind:
        span = next(s for s in spans if s.name == f"span_{kind.value}")
        assert span.kind == kind


def test_global_tracer_provider():
    """Test global tracer provider singleton."""
    provider1 = get_tracer_provider()
    provider2 = get_tracer_provider()

    assert provider1 is provider2


def test_get_tracer():
    """Test getting a tracer from global provider."""
    tracer1 = get_tracer("module1")
    tracer2 = get_tracer("module1")
    tracer3 = get_tracer("module2")

    assert tracer1 is tracer2
    assert tracer1 is not tracer3


def test_configure_tracing():
    """Test configuring global tracing."""
    provider = configure_tracing(enabled=True)
    assert provider.is_enabled()

    # Add exporter
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    tracer = get_tracer("test")

    with tracer.trace("test_span"):
        pass

    spans = exporter.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_span"

    # Clean up
    configure_tracing(enabled=False)


def test_trace_decorator():
    """Test trace decorator."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = configure_tracing(enabled=True)
    provider.add_span_processor(processor)

    @trace(name="custom_span")
    def my_function(x, y):
        return x + y

    result = my_function(2, 3)
    assert result == 5

    spans = exporter.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "custom_span"

    # Clean up
    exporter.clear()
    configure_tracing(enabled=False)


def test_trace_decorator_with_exception():
    """Test trace decorator with exception."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = configure_tracing(enabled=True)
    provider.add_span_processor(processor)

    @trace()
    def failing_function():
        raise ValueError("test error")

    try:
        failing_function()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    spans = exporter.get_spans()
    assert len(spans) == 1
    assert spans[0].status == SpanStatus.ERROR

    # Should have an exception event
    exception_events = [e for e in spans[0].events if e.name == "exception"]
    assert len(exception_events) == 1

    # Clean up
    exporter.clear()
    configure_tracing(enabled=False)


def test_in_memory_exporter():
    """Test in-memory span exporter."""
    exporter = InMemorySpanExporter()

    from factor_engine.telemetry.traces import SpanContext, SpanData

    span1 = SpanData(
        context=SpanContext.create_root(),
        name="span1",
        kind=SpanKind.INTERNAL,
        start_time=time.time(),
        end_time=time.time(),
    )

    span2 = SpanData(
        context=SpanContext.create_root(),
        name="span2",
        kind=SpanKind.INTERNAL,
        start_time=time.time(),
        end_time=time.time(),
    )

    exporter.export([span1, span2])

    spans = exporter.get_spans()
    assert len(spans) == 2

    exporter.clear()
    assert len(exporter.get_spans()) == 0


if __name__ == "__main__":
    import sys

    test_functions = [
        test_span_context_creation,
        test_span_basic,
        test_span_context_manager,
        test_span_error_handling,
        test_tracer_disabled,
        test_tracer_enabled,
        test_tracer_context_manager,
        test_tracer_nested_spans,
        test_span_events,
        test_span_duration,
        test_span_kinds,
        test_global_tracer_provider,
        test_get_tracer,
        test_configure_tracing,
        test_trace_decorator,
        test_trace_decorator_with_exception,
        test_in_memory_exporter,
    ]

    failed = []
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"✓ {test_fn.__name__}")
        except Exception as e:
            print(f"✗ {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test_fn.__name__)

    if failed:
        print(f"\n{len(failed)} tests failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"\nAll {len(test_functions)} tests passed!")
        sys.exit(0)
