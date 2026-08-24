"""Oracle for the canonical availability/latency exception identity."""


def test_cos_runtime_uses_canonical_availability_latency_error() -> None:
    from data_access.core.exceptions import AvailabilityLatencyError
    from data_access.cos_event_runtime import (
        AvailabilityLatencyError as RuntimeAvailabilityLatencyError,
    )

    assert RuntimeAvailabilityLatencyError is AvailabilityLatencyError
    assert AvailabilityLatencyError.__module__ == "data_access.core.exceptions"
