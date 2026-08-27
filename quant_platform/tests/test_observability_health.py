"""Tests for quant_platform.app.observability.health.

Covers the ``assess`` rule (healthy / degraded / unhealthy with injectable
threshold), immutable ``HealthReport`` semantics, and ``HealthCheck`` behavior.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from quant_platform.app.observability.health import HealthCheck, HealthReport, HealthStatus, assess


def _failing_names(report: HealthReport) -> tuple[str, ...]:
    return tuple(c.name for c in report.failing_checks)


# ---- healthy ----
def test_assess_all_ok_is_healthy():
    report = assess(job_errors=False, outbox_pending=False, registry_ok=True)
    assert report.status == "HEALTHY"
    assert report.status == HealthStatus.HEALTHY
    assert report.ok and report.healthy
    assert not report.degraded and not report.unhealthy
    assert report.failing_checks == ()
    assert "all" in report.summary


# ---- degraded ----
def test_assess_one_failure_is_degraded():
    report = assess(job_errors=True, outbox_pending=False, registry_ok=True)
    assert report.status == "DEGRADED"
    assert report.degraded
    assert not report.unhealthy
    assert _failing_names(report) == ("jobs",)


def test_assess_other_single_failures_are_degraded():
    assert assess(job_errors=False, outbox_pending=True, registry_ok=True).status == "DEGRADED"
    assert assess(job_errors=False, outbox_pending=False, registry_ok=False).status == "DEGRADED"


# ---- unhealthy / threshold ----
def test_assess_two_failures_is_unhealthy():
    report = assess(job_errors=True, outbox_pending=True, registry_ok=True)
    assert report.status == "UNHEALTHY"
    assert report.unhealthy
    assert not report.degraded
    assert set(_failing_names(report)) == {"jobs", "outbox"}


def test_assess_all_three_failures_is_unhealthy():
    report = assess(job_errors=True, outbox_pending=True, registry_ok=False)
    assert report.status == "UNHEALTHY"
    assert report.unhealthy


def test_assess_unhealthy_mentions_all_failing():
    report = assess(job_errors=True, outbox_pending=True, registry_ok=False)
    summary = report.summary
    assert "jobs" in summary and "outbox" in summary and "registry" in summary
    assert "3 of 3" in summary


def test_assess_injectable_threshold():
    # default threshold=2: exactly 2 -> UNHEALTHY
    assert assess(job_errors=True, outbox_pending=True, registry_ok=True).status == "UNHEALTHY"
    # threshold=3: exactly 2 -> still DEGRADED
    assert (
        assess(job_errors=True, outbox_pending=True, registry_ok=True, degraded_threshold=3).status
        == "DEGRADED"
    )
    # threshold=1: any single failure -> UNHEALTHY
    assert (
        assess(job_errors=True, outbox_pending=False, registry_ok=True, degraded_threshold=1).status
        == "UNHEALTHY"
    )


def test_assess_rejects_bad_threshold():
    with pytest.raises(ValueError):
        assess(True, False, True, degraded_threshold=0)
    with pytest.raises(ValueError):
        assess(False, False, True, degraded_threshold=-1)


# ---- HealthReport immutability ----
def test_health_report_is_immutable():
    report = assess(job_errors=True, outbox_pending=False, registry_ok=True)
    assert isinstance(report.checks, tuple)
    with pytest.raises(FrozenInstanceError):
        report.status = "HEALTHY"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.checks = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.summary = "nope"  # type: ignore[misc]


def test_health_report_checks_field_is_deeply_immutable():
    report = assess(False, False, True)
    assert all(isinstance(c, HealthCheck) for c in report.checks)
    # a list passed as checks is coerced to a tuple
    manual = HealthReport(status="HEALTHY", checks=[HealthCheck("x", True)], summary="")
    assert isinstance(manual.checks, tuple)


def test_health_report_properties():
    report = assess(job_errors=True, outbox_pending=False, registry_ok=True)
    assert report.check_names == ("jobs", "outbox", "registry")
    assert report.failing_checks[0].name == "jobs"


def test_health_report_bad_status_rejected():
    with pytest.raises(ValueError):
        HealthReport(status="UNKNOWN", checks=(), summary="")


def test_health_check_defaults_and_validation():
    hc = HealthCheck("jobs", True)
    assert hc.severity == "high"
    assert hc.message == ""
    assert HealthCheck("", True).name == "unnamed"
    assert HealthCheck("x", True, severity="urgent").severity == "high"  # unknown -> high


def test_health_report_summary_auto_filled():
    report = HealthReport(status="DEGRADED", checks=(HealthCheck("jobs", False),), summary="")
    assert report.summary
    assert "jobs" in report.summary


def test_health_report_with_override_returns_new_instance():
    report = assess(False, False, True)
    override = report.with_override(status="DEGRADED")
    assert override is not report
    assert report.status == "HEALTHY"
    assert override.status == "DEGRADED"
    assert override.summary == report.summary