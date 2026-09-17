"""Transfer observations must not reuse optimizer estimates as actuals."""
import pandas as pd
import pyarrow as pa

from factor_engine.runtime.region_telemetry import RegionTelemetryCollector
from factor_engine.runtime.transfer_fallback import TransferFallback


def _execute(value, transform="same_backend_native", executor=None):
    telemetry = RegionTelemetryCollector()
    fallback = TransferFallback(executor=executor, telemetry=telemetry)
    result = fallback.execute_edge(
        edge_id="edge", producer_region="p", consumer_region="c",
        source_repr="pandas_long" if transform == "same_backend_native" else "arrow_table",
        target_repr="pandas_long", transform=transform, data=value,
        estimated_bytes=612466512,
    )
    return result, telemetry.transfers[-1]


def test_identity_transfer_is_zero_copy_and_has_real_payload_shape():
    value = pd.Series([1., 2., 3.])
    result, event = _execute(value)
    assert result is value
    assert event.predicted_bytes == 612466512
    assert event.actual_bytes == 0
    assert event.row_count == 3 and event.column_count == 1
    assert event.payload_bytes == int(value.memory_usage(index=True, deep=True))
    assert event.actual_bytes_basis == "identity_zero_copy"


def test_conversion_records_materialized_payload_not_prediction():
    result, event = _execute(pa.table({"x": [1., 2., 3.]}), "arrow_to_pandas")
    expected = int(result.memory_usage(index=True, deep=True).sum())
    assert event.actual_bytes == expected
    assert event.payload_bytes == expected
    assert event.row_count == 3 and event.column_count == 1
    assert event.actual_bytes_basis == "materialized_payload"


def test_unknown_payload_does_not_invent_bytes_or_force_materialization():
    class Opaque:
        def __getattr__(self, name):
            raise AssertionError("telemetry must not materialize unknown objects")
    class Executor:
        def execute(self, *args, **kwargs):
            return Opaque()
    _, event = _execute(None, "arrow_to_pandas", Executor())
    assert event.actual_bytes is None
    assert event.payload_bytes is None
    assert event.row_count is None and event.column_count is None
    assert event.actual_bytes_basis == "unmeasured"


def test_real_transfer_event_survives_compact_execution_ledger():
    from dataclasses import asdict
    from factor_engine.runtime.execution_ledger import summarize_execution_ledger
    _, event = _execute(pd.Series([1., 2., 3.]))
    ledger = summarize_execution_ledger(
        {"backend_paths": {"factor": {"physical_plan": {"transfers": [asdict(event)]}}}},
        run_id="run", evidence_id="evidence",
    )
    group = ledger["actual_physical"]["transfer_groups"][0]
    assert group["source_backend"] == "unreported"
    assert group["target_backend"] == "unreported"
    assert group["actual_bytes"] == 0
    assert group["actual_rows"] == 3
    assert group["actual_transfer_ms"] == event.actual_ms
    assert group["payload_bytes"] == event.payload_bytes
    assert group["actual_bytes_basis_counts"] == {"identity_zero_copy": 1}
