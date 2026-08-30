# -*- coding: utf-8 -*-
"""R50 P0-11: Transfer fallback unit tests.

Forces an unsupported transform (no direct executor path) and asserts the
safe fallback fires (materialize-then-convert) and a telemetry counter is
recorded; a transform with no representation-level resolution at all fails
closed with ``UnsupportedTransferTransform`` (never a silent no-op).
"""
from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from factor_engine.planning.transfer_edge import (
    UnsupportedTransferTransform,
)
from factor_engine.runtime.multibackend.batch_transfer_optimizer import (
    TransferTransform,
)
from factor_engine.runtime.transfer_fallback import (
    TransferFallback,
    TransferFallbackMetrics,
)


def _arrow_table() -> pa.Table:
    return pa.table(
        {
            "date": pa.array(["2024-01-01", "2024-01-02"]),
            "instrument": pa.array([1, 2], type=pa.int64()),
            "value": pa.array([1.5, 2.5], type=pa.float64()),
        }
    )


def _metric_sink() -> TransferFallbackMetrics:
    return TransferFallbackMetrics()


def _run_edge(
    fb: TransferFallback,
    *,
    transform: TransferTransform,
    target_repr: str,
    data: Any,
    source_repr: str = "arrow_table",
) -> Any:
    return fb.execute_edge(
        edge_id="edge_test",
        producer_region="p",
        consumer_region="c",
        source_repr=type("R", (), {"value": source_repr})(),
        target_repr=type("R", (), {"value": target_repr})(),
        transform=transform,
        data=data,
    )


class TestUnsupportedTransformForcesFallback:
    def test_wide_to_long_routes_to_safe_fallback(self) -> None:
        """A transform with no direct executor path fires the safe fallback."""
        metrics = _metric_sink()
        fb = TransferFallback(metrics=metrics)
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "instrument": [1],
                "v1": [1.0],
                "v2": [2.0],
            }
        ).convert_dtypes()
        out = _run_edge(
            fb,
            transform=TransferTransform.WIDE_TO_LONG,
            target_repr="pandas_long",
            data=df,
            source_repr="pandas_wide",
        )
        assert isinstance(out, pd.DataFrame)
        assert "field" in out.columns and "value" in out.columns
        assert metrics.fallback_count == 1
        assert metrics.direct_count == 0
        assert metrics.last_fallback_transform == "wide_to_long"

    def test_arrow_to_pandas_executes_directly(self) -> None:
        """A natively-executable transform records a direct (non-fallback) hit."""
        metrics = _metric_sink()
        fb = TransferFallback(metrics=metrics)
        out = _run_edge(
            fb,
            transform=TransferTransform.ARROW_TO_PANDAS,
            target_repr="pandas_long",
            data=_arrow_table(),
        )
        assert isinstance(out, pd.DataFrame)
        assert metrics.fallback_count == 0
        assert metrics.direct_count == 1

    def test_same_backend_native_is_direct(self) -> None:
        metrics = _metric_sink()
        fb = TransferFallback(metrics=metrics)
        out = _run_edge(
            fb,
            transform=TransferTransform.SAME_BACKEND_NATIVE,
            target_repr="arrow_table",
            data=_arrow_table(),
        )
        assert isinstance(out, pa.Table)
        assert metrics.direct_count == 1
        assert metrics.fallback_count == 0

    def test_unresolvable_transform_fails_closed(self) -> None:
        """Unknown transform -> UnsupportedTransferTransform, never silent."""
        metrics = _metric_sink()
        fb = TransferFallback(metrics=metrics)
        with pytest.raises(UnsupportedTransferTransform):
            _run_edge(
                fb,
                transform=TransferTransform.SORT,
                target_repr="pandas_long",
                data=_arrow_table(),
            )
        assert metrics.failures == 1


class TestTelemetrySink:
    def test_telemetry_recorded_on_fallback(self) -> None:
        from factor_engine.runtime.region_telemetry import (
            RegionTelemetryCollector,
        )

        collector = RegionTelemetryCollector()
        fb = TransferFallback(
            metrics=_metric_sink(), telemetry=collector
        )
        df = pd.DataFrame(
            {"date": ["2024-01-01"], "instrument": [1], "v1": [1.0]}
        ).convert_dtypes()
        _run_edge(
            fb,
            transform=TransferTransform.WIDE_TO_LONG,
            target_repr="pandas_long",
            data=df,
            source_repr="pandas_wide",
        )
        assert len(collector.transfers) == 1
        t = collector.transfers[0]
        assert t.edge_id == "edge_test"
        assert t.schema_version == "fallback"
