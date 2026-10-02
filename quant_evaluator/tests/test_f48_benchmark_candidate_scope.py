"""Isolation and fail-closed tests for the benchmark-only F48 route."""
import subprocess
import sys
from pathlib import Path

import pytest

from quant_evaluator.api import factor_source
from quant_evaluator.runtime import source_auto_evidence
from quant_evaluator.scripts.f48_benchmark_candidate import (
    F48_BENCHMARK_CANDIDATE_STATUS,
    F48_EVIDENCE_ID,
    F48_METRICS,
    F48_SHAPE,
    f48_benchmark_candidate_scope,
)


def _args(**overrides):
    values = dict(shape=F48_SHAPE, metrics=F48_METRICS,
                  source_dtype="float64", label_dtype="float64",
                  requested_tile_width=2)
    values.update(overrides)
    return values


def _select():
    return factor_source.select_source_auto_route(
        shape=F48_SHAPE, metrics=F48_METRICS, source_dtype="float64",
        label_dtype="float64", requested_tile_width=2,
    )


def test_default_route_is_unavailable_and_candidate_has_benchmark_provenance():
    assert _select() is None
    original = factor_source.select_source_auto_route
    with f48_benchmark_candidate_scope(**_args()) as provenance:
        route = _select()
        assert route.evidence_id == F48_EVIDENCE_ID
        assert route.effective_tile_width == 2
        assert route.evidence_status == F48_BENCHMARK_CANDIDATE_STATUS
        assert provenance["provenance"] == "benchmark_only_pending_auto_candidate"
        assert provenance["evidence_status"] == F48_BENCHMARK_CANDIDATE_STATUS
        assert provenance["source_ab_artifacts"]
    assert factor_source.select_source_auto_route is original
    assert _select() is None


def test_candidate_route_materializes_metric_iterable_once():
    with f48_benchmark_candidate_scope(**_args(metrics=iter(F48_METRICS))):
        route = factor_source.select_source_auto_route(
            shape=F48_SHAPE, metrics=iter(F48_METRICS), source_dtype="float64",
            label_dtype="float64", requested_tile_width=2,
        )
    assert route.evidence_id == F48_EVIDENCE_ID
    assert route.evidence_status == F48_BENCHMARK_CANDIDATE_STATUS


def test_scope_is_non_reentrant():
    with f48_benchmark_candidate_scope(**_args()):
        with pytest.raises(RuntimeError, match="non-reentrant"):
            with f48_benchmark_candidate_scope(**_args()):
                pytest.fail("nested candidate scope unexpectedly succeeded")
        assert _select().evidence_status == F48_BENCHMARK_CANDIDATE_STATUS


def test_scope_reads_current_registry_binding_and_fails_closed(monkeypatch):
    monkeypatch.setattr(source_auto_evidence, "SOURCE_AUTO_EVIDENCE", ())
    with pytest.raises(RuntimeError, match="envelope is unavailable"):
        with f48_benchmark_candidate_scope(**_args()):
            pytest.fail("scope accepted a rebound registry without F48")


@pytest.mark.parametrize("override", [
    {"shape": (2586, 5461, 47)},
    {"metrics": ("rank_ic", "factor_turnover_rate", "quantile_spread")},
    {"source_dtype": "float32"},
    {"label_dtype": "float32"},
    {"requested_tile_width": 4},
])
def test_candidate_scope_refuses_non_exact_profile(override):
    with pytest.raises(ValueError, match="exact F48 tile-2"):
        with f48_benchmark_candidate_scope(**_args(**override)):
            pytest.fail("candidate scope unexpectedly admitted a changed profile")


def test_candidate_patch_is_restored_after_exception():
    original = factor_source.select_source_auto_route
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with f48_benchmark_candidate_scope(**_args()):
            assert _select().evidence_status == F48_BENCHMARK_CANDIDATE_STATUS
            raise RuntimeError("synthetic failure")
    assert factor_source.select_source_auto_route is original
    assert source_auto_evidence.select_source_auto_route(
        shape=F48_SHAPE, metrics=F48_METRICS, source_dtype="float64",
        label_dtype="float64", requested_tile_width=2,
    ) is None


def test_candidate_override_does_not_leak_across_processes():
    project_root = Path(__file__).resolve().parents[2]
    code = """
from quant_evaluator.api import factor_source
from quant_evaluator.scripts.f48_benchmark_candidate import f48_benchmark_candidate_scope
shape = (2586, 5461, 48)
metrics = ('rank_ic', 'quantile_spread', 'factor_turnover_rate')
args = dict(shape=shape, metrics=metrics, source_dtype='float64',
            label_dtype='float64', requested_tile_width=2)
assert factor_source.select_source_auto_route(**args) is None
with f48_benchmark_candidate_scope(**args):
    route = factor_source.select_source_auto_route(**args)
    assert route.evidence_status == 'benchmark_only_pending_auto'
assert factor_source.select_source_auto_route(**args) is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=project_root,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert completed.returncode == 0, completed.stderr
