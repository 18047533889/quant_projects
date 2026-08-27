"""``MetricsRegistry`` — thread-safe, pure in-memory metrics store.

P4 observability layer. Kept deliberately standard-library-only and standalone:
counters, gauges (min/max/sum/count box), and an immutable ``snapshot`` exported
as a plain ``Mapping``. No Prometheus / OpenTelemetry / third-party imports —
the store shapes its interface after those semantics so wiring up an external
collector later is a mapping exercise, not a rewrite.

Metric naming is a security/injection surface: only ``[a-z0-9_.]`` is allowed
(no uppercase, no whitespace, no ``-``) so a name can never smuggle label keys
or newlines into a downstream text exposition. Illegal names are rejected with a
``ValueError`` before any state is touched.

Thread-safety model: ``incr`` / ``observe`` / ``snapshot`` are guarded by a
single ``RLock``. Counters/probes read-modify-write atomically under the lock;
the min/max/sum/count box accumulates per observation; ``snapshot`` fences with
the lock and copies.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "MetricsRegistry",
    "MetricValue",
    "INVALID_METRIC_NAME_RE",
    "InvalidMetricNameError",
]

#: matches ANY character that is NOT in the allowed set ``[a-z0-9_.]`` — a name
#: containing anything outside that set is rejected (see ``MetricsRegistry``).
INVALID_METRIC_NAME_RE = r"[^a-z0-9_.]"


class InvalidMetricNameError(ValueError):
    """Raised when a metric name violates the ``[a-z0-9_.]`` grammar."""


@dataclass(frozen=True)
class MetricValue:
    """One metric's immutable snapshot.

    ``value`` is the current gauge value / counter total; ``min``/``max``/``sum``/
    ``count`` are the accumulated observation statistics (for a plain counter
    ``count`` increments per ``incr`` call and ``sum`` mirrors ``value``).
    """

    name: str
    kind: str  # "counter" | "gauge"
    value: float
    min: float
    max: float
    sum: float
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "min", float(self.min))
        object.__setattr__(self, "max", float(self.max))
        object.__setattr__(self, "sum", float(self.sum))
        object.__setattr__(self, "count", int(self.count))


class MetricsRegistry:
    """Thread-safe in-memory metric store.

    - ``incr(name, value=1, labels=None)`` — accumulate a counter.
      ``value`` can be ``-1`` (decrement) or any number.
    - ``observe(name, value, labels=None)`` — record one gauge-like observation:
      updates ``min``/``max``/``sum``/``count`` and sets the current ``value``.
    - ``set_value(name, value, labels=None)`` — overwrite the current gauge
      value without touching the box.
    - ``get(name, labels=None) -> MetricValue | None`` and
      ``snapshot() -> Mapping[str, MetricValue]`` — reads (immutable export).
    - ``dimensions()`` — label dimension names per metric, for operator docs.

    Labels are an ordered ``Mapping[str, hashable]``; the label KEY SET must be
    stable for any name (a label-dimension mismatch on a later call raises
    ``ValueError``). An unlabeled row and a labeled row of the same name are
    distinct metric rows.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # name -> {labels-key: MetricValue}; labels-key is ``None`` for the
        # unlabeled row, else ``tuple(sorted(labels.items()))``.
        self._metrics: dict[str, dict[Any, MetricValue]] = {}
        self._dimensions: dict[str, tuple[str, ...]] = {}

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise InvalidMetricNameError("metric name must be a non-empty string")
        if re.search(INVALID_METRIC_NAME_RE, name):
            raise InvalidMetricNameError(
                f"invalid metric name {name!r}: only [a-z0-9_.] allowed"
            )

    @staticmethod
    def _labels_key(labels: Mapping[str, object] | None) -> Any:
        if labels is None or not labels:
            return None
        try:
            return tuple(sorted(labels.items()))
        except TypeError as exc:  # unhashable values
            raise TypeError(f"label values must be hashable, got {labels!r}") from exc

    def _labels_dims(self, labels: Mapping[str, object] | None) -> tuple[str, ...]:
        return tuple(sorted(labels)) if labels else ()

    def _row_bucket(self, name: str, labels: Mapping[str, object] | None) -> dict[Any, MetricValue]:
        dims = self._labels_dims(labels)
        prev = self._dimensions.setdefault(name, dims)
        if dims != prev:
            raise ValueError(f"metric {name!r} labels mismatch: {prev!r} != {dims!r}")
        return self._metrics.setdefault(name, {})

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def incr(
        self,
        name: str,
        value: int | float = 1,
        labels: Mapping[str, object] | None = None,
    ) -> MetricValue:
        """Accumulate ``value`` (default ``1``) into counter ``name``.

        The first call creates a ``counter``; a later ``observe`` on the same
        name keeps the counter kind (observations accumulate into its box).
        """
        self._check_name(name)
        try:
            fval = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"incr value must be numeric, got {value!r}")
        k = self._labels_key(labels)
        with self._lock:
            rows = self._row_bucket(name, labels)
            row = rows.get(k)
            if row is None:
                row = MetricValue(
                    name=name, kind="counter", value=fval,
                    min=fval, max=fval, sum=fval, count=1,
                )
            else:
                row = MetricValue(
                    name=name, kind=row.kind, value=row.value + fval,
                    min=min(row.min, fval), max=max(row.max, fval),
                    sum=row.sum + fval, count=row.count + 1,
                )
            rows[k] = row
            return row

    def observe(
        self,
        name: str,
        value: int | float,
        labels: Mapping[str, object] | None = None,
    ) -> MetricValue:
        """Record one gauge observation on ``name``.

        Updates min/max/sum/count and sets the current ``value`` to the
        observation. Counter-kind semantics: an ``observe`` on a name that was
        first created via ``incr`` accumulates into the counter's box (value
        stays the counter total).
        """
        self._check_name(name)
        try:
            fval = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"observe value must be numeric, got {value!r}")
        k = self._labels_key(labels)
        with self._lock:
            rows = self._row_bucket(name, labels)
            row = rows.get(k)
            if row is None:
                row = MetricValue(
                    name=name, kind="gauge", value=fval,
                    min=fval, max=fval, sum=fval, count=1,
                )
            elif row.kind == "gauge":
                row = MetricValue(
                    name=name, kind="gauge", value=fval,
                    min=min(row.min, fval), max=max(row.max, fval),
                    sum=row.sum + fval, count=row.count + 1,
                )
            else:  # existing counter row — keep counter kind, accumulate box
                row = MetricValue(
                    name=name, kind="counter", value=row.value,
                    min=row.min, max=row.max,
                    sum=row.sum + fval, count=row.count + 1,
                )
            rows[k] = row
            return row

    def set_value(
        self,
        name: str,
        value: int | float,
        labels: Mapping[str, object] | None = None,
    ) -> MetricValue:
        """Explicitly set the current gauge value (no box accumulation)."""
        self._check_name(name)
        try:
            fval = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"set_value value must be numeric, got {value!r}")
        k = self._labels_key(labels)
        with self._lock:
            rows = self._row_bucket(name, labels)
            row = rows.get(k)
            if row is None:
                row = MetricValue(
                    name=name, kind="gauge", value=fval,
                    min=fval, max=fval, sum=fval, count=1,
                )
            else:
                row = MetricValue(
                    name=name, kind="gauge", value=fval,
                    min=row.min, max=row.max, sum=row.sum, count=row.count,
                )
            rows[k] = row
            return row

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def get(self, name: str, labels: Mapping[str, object] | None = None) -> MetricValue | None:
        """Snapshot for ``(name, labels)`` or ``None`` when absent."""
        self._check_name(name)
        k = self._labels_key(labels)
        with self._lock:
            return self._metrics.get(name, {}).get(k)

    def snapshot(self) -> Mapping[str, MetricValue]:
        """IMMUTABLE full export: metric name → ``MetricValue``.

        No caller can mutate the returned mapping (structural ``MappingProxyType``
        wrapper) or a ``MetricValue`` (frozen dataclass); re-snapshotting is the
        only way to observe newer state.

        For a name with labeled variants the unlabeled row (if any) is the
        top-level entry; otherwise the first labeled row by labels-key sort
        order acts as the aggregate proxy. Use ``snapshot_labels`` for the full
        per-label breakdown.
        """
        with self._lock:
            out: dict[str, MetricValue] = {}
            for name, rows in self._metrics.items():
                if None in rows:
                    out[name] = rows[None]
                else:
                    out[name] = rows[sorted(rows.keys())[0]]
            return MappingProxyType(out)

    def snapshot_labels(self) -> Mapping[str, Mapping[Any, MetricValue]]:
        """Immutable per-label view: name → {labels-key: MetricValue}."""
        with self._lock:
            return {name: dict(rows) for name, rows in self._metrics.items()}

    def dimensions(self) -> Mapping[str, tuple[str, ...]]:
        """Label dimension names per metric (empty tuple when unlabeled)."""
        with self._lock:
            return dict(self._dimensions)