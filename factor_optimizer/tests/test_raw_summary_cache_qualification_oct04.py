"""Fail-closed qualification checks for the real raw-summary cache A/B."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_FIXTURE_PATH = Path(__file__).with_name("test_raw_summary_cache_benchmark_oct04.py")
_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "raw_summary_qualification_fixtures", _FIXTURE_PATH)
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
_FIXTURES = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURES)
BENCH = _FIXTURES.BENCH


def _patch_stable_admissions(monkeypatch):
    monkeypatch.setattr(BENCH, "source_hashes", lambda: {"code": "stable"})
    monkeypatch.setattr(BENCH, "runtime_context", lambda: {"runtime": "stable"})



@pytest.mark.parametrize(
    "status", ["error_raw_retained", "budget_exceeded_raw_retained", "invalid_raw"])
def test_result_signature_rejects_optimizer_nonqualification_status(status):
    """The same raw fallback across modes must not qualify a failed/over-budget search."""
    batch, labels, _, _ = _FIXTURES._cohort()
    result = _FIXTURES._fake_result(batch, labels, BENCH._default_config())
    result.factors[batch.factor_ids[0]].status = status

    automatic_audit = BENCH._load_audit_modules()[0]
    with pytest.raises(ValueError, match="qualification|failed|budget"):
        BENCH._result_signature(result, batch, result.split, automatic_audit)


def test_benchmark_rejects_runs_with_no_raw_summary_requests(tmp_path, monkeypatch):
    """A no-op optimizer runner cannot produce a cache speed comparison."""
    cohort = _FIXTURES._cohort()

    def runner(batch, labels, *, config, **kwargs):
        return _FIXTURES._fake_result(batch, labels, config)

    _patch_stable_admissions(monkeypatch)
    report = tmp_path / "no-requests.json"
    with pytest.raises(ValueError, match="cache qualification"):
        BENCH.run_benchmark(
            report=report, source_loader=lambda **kwargs: cohort,
            auto_runner=runner, resource_check=lambda: None,
            environment_check=lambda: None,
        )
    assert not report.exists()


def test_benchmark_rejects_runs_with_zero_cached_hits(tmp_path, monkeypatch):
    """One distinct summary per run exercises calls but never the memoized path."""
    from factor_optimizer import research_summary_cache

    cohort = _FIXTURES._cohort()
    panel = np.column_stack((np.linspace(-.1, .1, 60),
                             np.sin(np.arange(60)) * .01,
                             np.full(60, .4)))

    def runner(batch, labels, *, config, **kwargs):
        research_summary_cache.RawMetricSummaryCache().summarize(panel)
        return _FIXTURES._fake_result(batch, labels, config)

    _patch_stable_admissions(monkeypatch)
    report = tmp_path / "no-hits.json"
    with pytest.raises(ValueError, match="cache qualification"):
        BENCH.run_benchmark(
            report=report, source_loader=lambda **kwargs: cohort,
            auto_runner=runner, resource_check=lambda: None,
            environment_check=lambda: None,
        )
    assert not report.exists()


def test_benchmark_rejects_runs_without_uncached_summary_recomputes(
        tmp_path, monkeypatch):
    """Cached hits without an uncached baseline do not form an A/B pair."""
    from factor_optimizer import research_summary_cache

    cohort = _FIXTURES._cohort()
    panel = np.column_stack((np.linspace(-.1, .1, 60),
                             np.sin(np.arange(60)) * .01,
                             np.full(60, .4)))

    def runner(batch, labels, *, config, **kwargs):
        cache_type = research_summary_cache.RawMetricSummaryCache
        if cache_type.__name__ == "ObservedSummaryCache":
            cache = cache_type()
            cache.summarize(panel)
            cache.summarize(panel.copy())
        return _FIXTURES._fake_result(batch, labels, config)

    _patch_stable_admissions(monkeypatch)
    report = tmp_path / "no-uncached-computes.json"
    with pytest.raises(ValueError, match="cache qualification"):
        BENCH.run_benchmark(
            report=report, source_loader=lambda **kwargs: cohort,
            auto_runner=runner, resource_check=lambda: None,
            environment_check=lambda: None,
        )
    assert not report.exists()


_COUNTER_FIELDS = (
    "raw_summary_requests", "raw_summary_computes",
    "raw_summary_cache_hits", "raw_summary_failures",
    "all_summarize_computes",
)


def _qualified_counts(mode):
    cached = mode == "cached"
    return {
        "raw_summary_requests": 2,
        "raw_summary_computes": 1 if cached else 2,
        "raw_summary_cache_hits": 1 if cached else 0,
        "raw_summary_failures": 0,
        "all_summarize_computes": 1 if cached else 2,
    }


@pytest.mark.parametrize("mode", ["cached", "uncached"])
def test_summary_activity_accepts_qualified_mode_counts(mode):
    BENCH._validate_summary_activity(mode, _qualified_counts(mode))


@pytest.mark.parametrize("field", _COUNTER_FIELDS)
@pytest.mark.parametrize("invalid", [True, 1.0])
def test_summary_activity_requires_exact_integer_counters(field, invalid):
    counts = _qualified_counts("cached")
    counts[field] = invalid
    with pytest.raises(ValueError, match="nonnegative integer counters"):
        BENCH._validate_summary_activity("cached", counts)


@pytest.mark.parametrize("field", _COUNTER_FIELDS)
def test_summary_activity_rejects_negative_counters(field):
    counts = _qualified_counts("cached")
    counts[field] = -1
    with pytest.raises(ValueError, match="nonnegative integer counters"):
        BENCH._validate_summary_activity("cached", counts)


@pytest.mark.parametrize(
    ("mode", "updates"),
    [
        ("cached", {"raw_summary_cache_hits": 0}),
        ("cached", {"raw_summary_computes": 0, "raw_summary_cache_hits": 2}),
        ("cached", {"raw_summary_requests": 3}),
        ("uncached", {"raw_summary_computes": 1}),
        ("uncached", {"raw_summary_cache_hits": 1}),
        ("cached", {"raw_summary_failures": 1}),
        ("cached", {"all_summarize_computes": 0}),
    ],
)
def test_summary_activity_rejects_nonconserving_or_failed_counts(mode, updates):
    counts = _qualified_counts(mode)
    counts.update(updates)
    with pytest.raises(ValueError, match="cache qualification"):
        BENCH._validate_summary_activity(mode, counts)


def test_summary_activity_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        BENCH._validate_summary_activity("skipped", _qualified_counts("cached"))
