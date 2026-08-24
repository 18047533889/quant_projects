"""
Optional OpenTelemetry exporter for traces and metrics.

This module provides OpenTelemetry export functionality if opentelemetry is installed.
It is an optional dependency and gracefully degrades if not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from factor_engine.telemetry.traces import SpanData

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter as OTelSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import SpanKind as OTelSpanKind
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class OpenTelemetryBridge:
    """Bridge between our tracing API and OpenTelemetry."""

    def __init__(
        self,
        service_name: str = "factor-engine",
        exporter: Optional[OTelSpanExporter] = None,
    ):
        if not OPENTELEMETRY_AVAILABLE:
            raise RuntimeError(
                "opentelemetry is not installed. "
                "Install it with: pip install opentelemetry-api opentelemetry-sdk"
            )

        # Create resource
        resource = Resource.create({"factor_engine.service.name": service_name})

        # Create tracer provider
        self._provider = OTelTracerProvider(resource=resource)

        # Add span processor
        if exporter is None:
            exporter = ConsoleSpanExporter()

        span_processor = BatchSpanProcessor(exporter)
        self._provider.add_span_processor(span_processor)

        # Set as global provider
        trace.set_tracer_provider(self._provider)

        self._tracer = trace.get_tracer(__name__)

    def export_span(self, span_data: SpanData) -> None:
        """Export a span to OpenTelemetry."""
        from factor_engine.telemetry.traces import SpanKind as OurSpanKind, SpanStatus

        # Map span kind
        kind_mapping = {
            OurSpanKind.INTERNAL: OTelSpanKind.INTERNAL,
            OurSpanKind.SERVER: OTelSpanKind.SERVER,
            OurSpanKind.CLIENT: OTelSpanKind.CLIENT,
            OurSpanKind.PRODUCER: OTelSpanKind.PRODUCER,
            OurSpanKind.CONSUMER: OTelSpanKind.CONSUMER,
        }
        otel_kind = kind_mapping.get(span_data.kind, OTelSpanKind.INTERNAL)

        # Create span
        with self._tracer.start_as_current_span(
            span_data.name,
            kind=otel_kind,
            start_time=int(span_data.start_time * 1e9),  # Convert to nanoseconds
        ) as span:
            # Set attributes
            for key, value in span_data.attributes.items():
                span.set_attribute(key, value)

            # Add events
            for event in span_data.events:
                span.add_event(
                    event.name,
                    attributes=event.attributes,
                    timestamp=int(event.timestamp * 1e9),
                )

            # Set status
            if span_data.status == SpanStatus.OK:
                span.set_status(Status(StatusCode.OK))
            elif span_data.status == SpanStatus.ERROR:
                span.set_status(Status(StatusCode.ERROR))
                if span_data.error:
                    span.record_exception(span_data.error)

            # End span with the recorded end time
            if span_data.end_time:
                span.end(end_time=int(span_data.end_time * 1e9))

    def shutdown(self) -> None:
        """Shutdown the OpenTelemetry provider."""
        if hasattr(self._provider, "shutdown"):
            self._provider.shutdown()


class OpenTelemetrySpanExporter:
    """Span exporter that forwards to OpenTelemetry."""

    def __init__(self, bridge: OpenTelemetryBridge):
        self._bridge = bridge

    def export(self, spans: List[SpanData]) -> None:
        """Export spans to OpenTelemetry."""
        for span in spans:
            try:
                self._bridge.export_span(span)
            except Exception:
                pass  # Don't let export failures break the application

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._bridge.shutdown()


def create_opentelemetry_bridge(
    service_name: str = "factor-engine",
    exporter: Optional[OTelSpanExporter] = None,
) -> Optional[OpenTelemetryBridge]:
    """Create an OpenTelemetry bridge if available."""
    if not OPENTELEMETRY_AVAILABLE:
        return None

    return OpenTelemetryBridge(service_name, exporter)


def create_jaeger_exporter(
    agent_host: str = "localhost",
    agent_port: int = 6831,
) -> Optional[OTelSpanExporter]:
    """Create a Jaeger exporter if available."""
    if not OPENTELEMETRY_AVAILABLE:
        return None

    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter

        return JaegerExporter(
            agent_host_name=agent_host,
            agent_port=agent_port,
        )
    except ImportError:
        return None


def create_otlp_exporter(
    endpoint: str = "http://localhost:4317",
) -> Optional[OTelSpanExporter]:
    """Create an OTLP exporter if available."""
    if not OPENTELEMETRY_AVAILABLE:
        return None

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=endpoint)
    except ImportError:
        return None
