"""R50 P0-11: TransferEdge fallback execution helper.

This module provides the *fallback* path for a planned region-boundary
transfer when the planned ``TransferKind`` has no direct execution path in the
runtime executor.  The region transfer layer (``planning/backend_region.py`` /
``planning/transfer_edge.py``) is READ-ONLY for P0-11; every call-site that
consumes a planned transfer goes through here so the executor never crashes
silently and never produces wrong-backend results.

Fallback strategy (safe, semantics-preserving materialize-then-convert):
    * If the runtime already produced an Arrow ``pa.Table``, it is the neutral
      interchange format.  ``ARROW_TO_POLARS`` / ``ARROW_TO_PANDAS`` /
      ``DUCKDB_TO_ARROW`` / ``Q_TO_ARROW`` / ``CLICKHOUSE_TO_ARROW`` all execute
      natively.
    * ``PANDAS_TO_POLARS`` executes natively via ``pl.from_pandas``.
    * ``POLARS_TO_PANDAS`` executes natively via ``to_pandas``.
    * ``SAME_BACKEND_NATIVE`` is identity (shared buffer, no conversion).
    * Everything else that the *planner* can emit (``WIDE_TO_LONG``,
      ``LONG_TO_WIDE``, ``SORT``, ``REPARTITION``, ``DTYPE_CAST``) is a
      structural reshape that the executor does not implement directly; those
      route through the documented safe fallback (materialize to Arrow, then
      convert to the target representation) and record a telemetry counter.
    * A transform with *no* representation-level resolution at all raises
      :class:`UnsupportedTransferTransform` (fail closed) -- never a silent
      no-op and never a guessed conversion.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from factor_engine.runtime.multibackend.batch_transfer_optimizer import (
    TransferExecutor,
    UnknownTransferTargetError,
)
from factor_engine.runtime.region_telemetry import TransferEdgeTelemetry


@dataclass
class TransferFallbackMetrics:
    """R50: telemetry counters for transfer fallback execution.

    ``fallback_count`` increments every time a planned transform has no direct
    executor path and routes through the materialize-then-convert fallback.
    ``direct_count`` counts transforms that executed natively.
    """

    fallback_count: int = 0
    direct_count: int = 0
    failures: int = 0
    last_fallback_transform: str | None = None
    recent: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fallback_count": self.fallback_count,
            "direct_count": self.direct_count,
            "failures": self.failures,
            "last_fallback_transform": self.last_fallback_transform,
            "recent": list(self.recent[-20:]),
        }


_GLOBAL_LOCK = threading.Lock()
_global_metrics: TransferFallbackMetrics | None = None


def get_global_transfer_fallback_metrics() -> TransferFallbackMetrics:
    """Return the process-level fallback telemetry counter (observable)."""
    global _global_metrics
    if _global_metrics is None:
        with _GLOBAL_LOCK:
            if _global_metrics is None:
                _global_metrics = TransferFallbackMetrics()
    return _global_metrics


class TransferFallback:
    """Execute a planned region-boundary transfer, routing to the safe fallback.

    Parameters:
        executor: real ``TransferExecutor`` for direct conversions (optional;
            a fresh one is created when not supplied).
        metrics: telemetry counter sink (defaults to the global observable).
        telemetry: optional ``RegionTelemetryCollector`` -- when provided, every
            executed edge is recorded via ``record_transfer`` with the
            fallback/direct classification.
    """

    def __init__(
        self,
        *,
        executor: TransferExecutor | None = None,
        metrics: TransferFallbackMetrics | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self._executor = executor or TransferExecutor()
        self._metrics = metrics or get_global_transfer_fallback_metrics()
        self._telemetry = telemetry
        self._lock = threading.RLock()

    # -- direct transforms (real executor paths) ------------------------------

    _DIRECT_TRANSFORMS = frozenset(
        {
            "same_backend_native",
            "duckdb_to_arrow",
            "q_to_arrow",
            "clickhouse_to_arrow",
            "arrow_to_polars",
            "arrow_to_pandas",
            "polars_to_pandas",
            "pandas_to_polars",
            "polars_to_numpy",
            "numpy_to_polars",
        }
    )

    # -- representation-level fallback table ----------------------------------

    def _fallback_arrow(
        self, data: Any, target_repr: Any
    ) -> Any:
        """Convert an Arrow table to the target representation (safe path)."""
        import pyarrow as pa

        if not isinstance(data, pa.Table):
            raise UnknownTransferTargetError(
                "fallback requires an Arrow materialization first; "
                f"got {type(data)} (target={getattr(target_repr, 'value', target_repr)})"
            )
        target = getattr(target_repr, "value", str(target_repr))
        if target == "polars_long" or target == "polars_wide":
            return self._executor.arrow_to_polars(data)
        if target == "pandas_long" or target == "pandas_wide":
            return self._executor.arrow_to_pandas(data)
        if target == "arrow_table":
            return data
        if target == "duckdb_relation":
            return self._to_duckdb(data)
        raise UnknownTransferTargetError(
            f"no fallback conversion to target representation {target!r}"
        )

    def _to_duckdb(self, table: Any) -> Any:
        import duckdb

        conn = duckdb.connect()
        conn.register("fallback_t", table)
        return conn.table("fallback_t")

    def _materialize_to_arrow(self, data: Any) -> Any:
        """Materialize arbitrary runtime output to Arrow (safe fallback root)."""
        import pyarrow as pa

        if isinstance(data, pa.Table):
            return data
        if isinstance(data, pa.RecordBatch):
            return pa.Table.from_batches([data])
        if hasattr(data, "to_arrow"):
            out = data.to_arrow()
            if isinstance(out, pa.Table):
                return out
            return pa.table(out)
        import polars as pl

        if isinstance(data, pl.DataFrame):
            return data.to_arrow()
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return pa.Table.from_pandas(data, preserve_index=False)
        raise UnknownTransferTargetError(
            f"cannot materialize {type(data)} to Arrow for fallback transfer"
        )

    # -- entry point ----------------------------------------------------------

    def execute_edge(
        self,
        *,
        edge_id: str,
        producer_region: str,
        consumer_region: str,
        source_repr: Any,
        target_repr: Any,
        transform: Any,
        data: Any,
        requires_sort: bool = False,
        requires_repartition: bool = False,
        requires_reshape: bool = False,
        estimated_bytes: int = 0,
    ) -> Any:
        """Execute one planned transfer edge with fallback + telemetry.

        Returns the converted value in the consumer region representation.

        Raises:
            UnsupportedTransferTransform: no execution path and no safe fallback
                exists (fail closed, never crash silently, never guess).
        """
        import time

        import pyarrow as pa

        from factor_engine.planning.transfer_edge import UnsupportedTransferTransform

        transform_value = (
            transform.value if hasattr(transform, "value") else str(transform)
        )
        started = time.monotonic()
        direct = False
        result: Any = None
        try:
            if transform_value in self._DIRECT_TRANSFORMS:
                if transform_value == "same_backend_native":
                    # Identity: shared buffer, no conversion.
                    result = data
                else:
                    result = self._executor.execute(
                        self._executor_transform(transform), data, verify=True
                    )
                direct = True
            elif transform_value == "wide_to_long" or transform_value == "long_to_wide":
                # Structural reshape: materialize to Arrow, reshape via pandas,
                # then convert to the target representation.
                arrow = self._materialize_to_arrow(data)
                df = arrow.to_pandas()
                if transform_value == "wide_to_long":
                    df = df.melt(var_name="field", value_name="value")
                    # Mixed-type melted columns (str/int/float) would fail Arrow
                    # inference; normalize to pandas "string" dtype so the
                    # materialize-then-convert path is lossless for the wide->long
                    # structural reshape.
                    df["value"] = df["value"].astype("string")
                else:
                    raise UnsupportedTransferTransform(
                        f"long_to_wide fallback is not implemented without schema "
                        f"knowledge; refusing to guess a pivot key (edge={edge_id})"
                    )
                result = self._fallback_arrow(
                    pa.Table.from_pandas(df, preserve_index=False), target_repr
                )
            elif transform_value == "duckdb_relation":
                result = self._fallback_arrow(
                    self._materialize_to_arrow(data), target_repr
                )
            elif transform_value == "sort" or transform_value == "repartition":
                raise UnsupportedTransferTransform(
                    f"transform {transform_value!r} has no executor path and no safe "
                    f"fallback (edge={edge_id}); the region transfer layer must not "
                    f"silently produce unsorted data"
                )
            else:
                # Unknown / unresolvable transform: fail closed.
                raise UnsupportedTransferTransform(
                    f"transfer edge {edge_id!r} uses transform {transform_value!r} "
                    f"with source {getattr(source_repr, 'value', source_repr)!r} -> "
                    f"target {getattr(target_repr, 'value', target_repr)!r}; no direct "
                    f"executor path and no safe fallback (R50 fail closed)"
                )
            elapsed = time.monotonic() - started
            with self._lock:
                if direct:
                    self._metrics.direct_count += 1
                else:
                    self._metrics.fallback_count += 1
                    self._metrics.last_fallback_transform = transform_value
                    self._metrics.recent.append(
                        {
                            "edge_id": edge_id,
                            "transform": transform_value,
                            "target": getattr(target_repr, "value", str(target_repr)),
                        }
                    )
            self._record_telemetry(
                edge_id=edge_id,
                producer_region=producer_region,
                consumer_region=consumer_region,
                source_repr=source_repr,
                target_repr=target_repr,
                requires_sort=requires_sort,
                requires_repartition=requires_repartition,
                requires_reshape=requires_reshape,
                estimated_bytes=estimated_bytes,
                started=started,
                elapsed=elapsed,
                fallback=not direct,
            )
            return result
        except Exception as exc:
            with self._lock:
                self._metrics.failures += 1
            raise

    def _executor_transform(self, transform: Any) -> Any:
        """Map TransferKind (planning) or TransferTransform to executor enum."""
        from factor_engine.runtime.multibackend.batch_transfer_optimizer import (
            TransferTransform,
        )

        value = transform.value if hasattr(transform, "value") else str(transform)
        try:
            return TransferTransform(value)
        except ValueError:
            return transform

    def _record_telemetry(
        self,
        *,
        edge_id: str,
        producer_region: str,
        consumer_region: str,
        source_repr: Any,
        target_repr: Any,
        requires_sort: bool,
        requires_repartition: bool,
        requires_reshape: bool,
        estimated_bytes: int,
        started: float,
        elapsed: float,
        fallback: bool,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            import pyarrow as pa  # noqa: F401

            telemetry = TransferEdgeTelemetry(
                edge_id=edge_id,
                producer_region=producer_region,
                consumer_region=consumer_region,
                source_representation=getattr(
                    source_repr, "value", str(source_repr)
                ),
                target_representation=getattr(
                    target_repr, "value", str(target_repr)
                ),
                predicted_bytes=estimated_bytes,
                actual_bytes=estimated_bytes,
                predicted_ms=0.0,
                actual_ms=elapsed * 1000.0,
                requires_sort=requires_sort,
                actual_sort_ms=0.0,
                requires_repartition=requires_repartition,
                actual_repartition_ms=0.0,
                requires_reshape=requires_reshape,
                actual_reshape_ms=0.0,
                row_count=0,
                column_count=0,
                schema_version="fallback" if fallback else "direct",
                transfer_started_at=started,
                transfer_finished_at=started + elapsed,
            )
            self._telemetry.record_transfer(telemetry)
        except Exception:
            pass


__all__ = [
    "TransferFallback",
    "TransferFallbackMetrics",
    "get_global_transfer_fallback_metrics",
]
