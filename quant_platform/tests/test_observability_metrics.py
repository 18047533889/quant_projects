"""Tests for quant_platform.app.observability.metrics.

Covers the thread-safe in-memory ``MetricsRegistry``: counter accumulation,
gauge/histogram box statistics, name validation (illegal names rejected),
label handling, and immutable snapshots.
"""

from __future__ import annotations

import threading

import pytest

from quant_platform.app.observability.metrics import (
    InvalidMetricNameError,
    MetricsRegistry,
)


# ---- basics / counter ----
def test_empty_registry_snapshot_is_immutable_and_empty():
    reg = MetricsRegistry()
    snap = reg.snapshot()
    assert snap == {}
    with pytest.raises(TypeError):
        snap["jobs_total"] = 1  # immutable mapping


def test_counter_incr_default_is_one():
    reg = MetricsRegistry()
    reg.incr("jobs_total")
    assert reg.get("jobs_total").value == 1.0
    assert reg.get("jobs_total").count == 1


def test_counter_accumulates():
    reg = MetricsRegistry()
    reg.incr("jobs_total", 2)
    reg.incr("jobs_total", 3)
    mv = reg.get("jobs_total")
    assert mv.kind == "counter"
    assert mv.value == 5.0
    assert mv.count == 2
    assert mv.sum == 5.0
    assert mv.min == 2.0
    assert mv.max == 3.0


def test_counter_observes_on_counter_keeps_kind():
    reg = MetricsRegistry()
    reg.incr("events", 4)
    reg.observe("events", 6)  # same name, observe path -> stays a counter
    mv = reg.get("events")
    assert mv.kind == "counter"
    assert mv.value == 4.0
    assert mv.sum == 10.0
    assert mv.count == 2


# ---- gauge / histogram ----
def test_gauge_observe_updates_box():
    reg = MetricsRegistry()
    reg.observe("job_duration_seconds", 1.0)
    reg.observe("job_duration_seconds", 5.0)
    reg.observe("job_duration_seconds", 3.0)
    mv = reg.get("job_duration_seconds")
    assert mv.kind == "gauge"
    assert mv.value == 3.0
    assert mv.min == 1.0
    assert mv.max == 5.0
    assert mv.sum == 9.0
    assert mv.count == 3


def test_set_value_overrides_gauge_without_touching_box():
    reg = MetricsRegistry()
    reg.observe("temp", 2.0)
    reg.observe("temp", 4.0)
    reg.set_value("temp", 10.0)
    mv = reg.get("temp")
    assert mv.value == 10.0
    assert mv.min == 2.0
    assert mv.max == 4.0
    assert mv.sum == 6.0
    assert mv.count == 2


# ---- name validation ----
@pytest.mark.parametrize(
    "bad",
    ["JobsTotal", "jobs-total", "jobs total", "jobs:total", "jobs!", "jobs/ok", "名字"],
)
def test_illegal_metric_names_rejected(bad):
    reg = MetricsRegistry()
    with pytest.raises(InvalidMetricNameError):
        reg.incr(bad)
    with pytest.raises(InvalidMetricNameError):
        reg.observe(bad, 1)
    with pytest.raises(InvalidMetricNameError):
        reg.set_value(bad, 1)
    with pytest.raises(InvalidMetricNameError):
        reg.get(bad)


@pytest.mark.parametrize("good", ["jobs_total", "jobs.total", "jobs_total.count", "a1.b2_c3"])
def test_legal_metric_names_accepted(good):
    reg = MetricsRegistry()
    reg.incr(good)
    assert reg.get(good).value == 1.0
    assert good in reg.snapshot()


def test_empty_and_non_string_names_rejected():
    reg = MetricsRegistry()
    with pytest.raises(InvalidMetricNameError):
        reg.incr("")
    with pytest.raises(InvalidMetricNameError):
        reg.incr(123)  # type: ignore[arg-type]
    with pytest.raises(InvalidMetricNameError):
        reg.incr(None)  # type: ignore[arg-type]


def test_observe_non_numeric_value_rejected():
    reg = MetricsRegistry()
    with pytest.raises(ValueError):
        reg.observe("lat", "not-a-number")
    with pytest.raises(ValueError):
        reg.incr("jobs_total", "x")


# ---- labels ----
def test_labeled_metric_aggregates_within_label():
    reg = MetricsRegistry()
    reg.set_value("jobs_total", 3, labels={"status": "SUCCEEDED"})
    reg.set_value("jobs_total", 1, labels={"status": "FAILED"})
    assert reg.get("jobs_total", labels={"status": "SUCCEEDED"}).value == 3.0
    assert reg.get("jobs_total", labels={"status": "FAILED"}).value == 1.0
    assert len(reg.snapshot_labels()["jobs_total"]) == 2


def test_label_dimension_mismatch_rejected():
    reg = MetricsRegistry()
    reg.set_value("jobs_total", 1, labels={"status": "SUCCEEDED"})
    with pytest.raises(ValueError):
        reg.set_value("jobs_total", 1, labels={"other": "x"})


def test_label_values_must_be_hashable():
    reg = MetricsRegistry()
    with pytest.raises(TypeError):
        reg.set_value("jobs_total", 1, labels={"status": ["SUCCEEDED"]})


# ---- thread safety ----
def test_registry_is_thread_safe():
    reg = MetricsRegistry()
    n = 8
    per = 500
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for i in range(per):
                reg.incr("thread_jobs", 2)
                reg.observe("thread_lat", float(i))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert reg.get("thread_jobs").value == n * per * 2
    assert reg.get("thread_jobs").count == n * per
    lat = reg.get("thread_lat")
    assert lat.count == n * per
    assert lat.min == 0.0
    assert lat.max == float(per - 1)


def test_snapshot_returns_fresh_immutable_copy():
    reg = MetricsRegistry()
    reg.incr("jobs_total", 3)
    snap = reg.snapshot()
    reg.incr("jobs_total", 1)
    assert snap["jobs_total"].value == 3.0
    assert reg.get("jobs_total").value == 4.0