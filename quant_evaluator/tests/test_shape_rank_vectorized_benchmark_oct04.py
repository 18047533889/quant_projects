"""Cheap contract tests for the vectorized quantile-rank CPU benchmark.

These tests replace the expensive numerical callables and environmental seams;
they verify benchmark qualification/receipt behavior without running the
Q=512, F=4096 SciPy baseline.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_shape_rank_vectorized_oct04.py"
)
SPEC = importlib.util.spec_from_file_location(
    "shape_rank_vectorized_benchmark", SCRIPT,
)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(benchmark)


def _stable_sources(_root=None):
    return {"benchmark.py": "a" * 64, "quantile_shape.py": "b" * 64}


def _stable_runtime():
    return {"pid": 1234, "python": "3.test", "numpy": "test", "scipy": "test"}


def _install_fast_callables(monkeypatch, calls=None):
    calls = [] if calls is None else calls

    def fake(name):
        def call(qr):
            # The harness must use the same prebuilt, read-only profile for both
            # implementations; producing a fixed vector avoids numerical work.
            assert qr.flags.c_contiguous
            assert not qr.flags.writeable
            calls.append(name)
            return np.zeros(qr.shape[1], dtype=np.float64)
        return call

    monkeypatch.setattr(benchmark, "_legacy", fake("legacy"))
    monkeypatch.setattr(benchmark, "_candidate", fake("candidate"))
    monkeypatch.setattr(benchmark, "_source_hashes", _stable_sources)
    monkeypatch.setattr(benchmark, "_runtime_identity", _stable_runtime)
    tick = iter(i / 1000 for i in range(10000))
    monkeypatch.setattr(benchmark, "_now", lambda: next(tick))
    return calls


def _one_case(**kwargs):
    options = {
        "cases": ["q5_f48_mixed"], "rounds": 3, "seed": 20261004,
        "deadline_seconds": 120.0,
    }
    options.update(kwargs)
    return benchmark.run_benchmark(**options)


def _only_result(receipt):
    assert len(receipt["results"]) == 1
    return next(iter(receipt["results"].values()))


def test_cli_defaults_to_dry_run_without_clock_or_rank_calls(monkeypatch, capsys):
    # Catches a CLI default that accidentally starts timing or imports/executes
    # the large SciPy baseline unless `--run` is explicitly supplied.
    monkeypatch.setattr(
        benchmark, "_now", lambda: pytest.fail("dry-run read the timing clock"),
    )
    monkeypatch.setattr(
        benchmark, "_legacy", lambda _qr: pytest.fail("dry-run ran legacy"),
    )
    monkeypatch.setattr(
        benchmark, "_candidate", lambda _qr: pytest.fail("dry-run ran candidate"),
    )

    exit_code = benchmark.main([])
    assert exit_code in (None, 0)
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "dry_run"
    assert receipt["winner_eligible"] is False


def test_small_mocked_run_has_frozen_head_identity_and_balanced_abba(monkeypatch):
    # Catches lost HEAD pinning, an unbalanced/misordered schedule, or warming
    # calls being mixed into the measured sample set.
    calls = _install_fast_callables(monkeypatch)
    receipt = _one_case()

    assert receipt["status"] == "complete"
    assert receipt["winner_eligible"] is True
    assert receipt["baseline"]["revision"] == receipt["environment"]["frozen_head"]
    assert re.fullmatch(r"[0-9a-f]{40}", receipt["baseline"]["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", receipt["baseline"]["source_sha256"])
    assert receipt["candidate_source_sha256_before"] == receipt["candidate_source_sha256_after"]
    assert receipt["runtime_before"] == receipt["runtime_after"]

    schedule = receipt["schedule"]
    assert schedule["warmup_calls"] == 1
    assert schedule["rounds"] == 3
    assert schedule["blocks_per_round"] == 4
    assert schedule["calls_per_block"] == 1
    assert schedule["order"] == "ABBA"
    result = _only_result(receipt)
    assert result["warmup"]["legacy"]["parity"] is True
    assert result["warmup"]["candidate"]["parity"] is True
    assert len(result["samples"]) == 12
    expected_mode_order = ["legacy", "candidate", "candidate", "legacy"] * 3
    assert [sample["mode"] for sample in result["samples"]] == expected_mode_order
    assert all(sample["bitwise_parity_each_call"] is True for sample in result["samples"])
    assert all(sample["shape_ok"] and sample["dtype_ok"] for sample in result["samples"])
    assert len(calls) == 14  # two warmups plus twelve timed calls
    assert calls == ["legacy", "candidate"] + expected_mode_order
    assert result["bitwise_parity_each_call"] is True
    assert np.isfinite(result["candidate_over_legacy"]) and result["candidate_over_legacy"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", receipt["cases"][0]["input_sha256"])


def test_same_seed_yields_same_prebuilt_case_hash(monkeypatch):
    # Catches accidental nondeterministic panel generation or unstable byte
    # canonicalization in the receipt input fingerprint.
    _install_fast_callables(monkeypatch)
    first = _one_case()
    second = _one_case()
    assert first["cases"][0]["input_sha256"] == second["cases"][0]["input_sha256"]


@pytest.mark.parametrize(
    "bad_cases,bad_seed",
    [(["not-an-allowed-case"], 20261004), (["q2_f48_mixed"], 20261004),
     (["q5_f48_mixed"], -1), (["q5_f48_mixed"], True)],
)
def test_invalid_case_or_seed_is_rejected_before_timing(monkeypatch, bad_cases, bad_seed):
    # Catches accepting unknown/out-of-range workload cases or bool/negative
    # seeds after allocating data or entering the timed loop.
    monkeypatch.setattr(
        benchmark, "_now", lambda: pytest.fail("invalid request entered timing"),
    )
    with pytest.raises(ValueError):
        benchmark.run_benchmark(
            cases=bad_cases, rounds=1, seed=bad_seed, deadline_seconds=120.0,
        )


def test_source_drift_disqualifies_receipt(monkeypatch):
    # Catches continuing to qualify timings after the candidate's source bytes
    # change during the run.
    _install_fast_callables(monkeypatch)
    snapshots = iter((
        {"candidate.py": "a" * 64},
        {"candidate.py": "c" * 64},
    ))
    monkeypatch.setattr(benchmark, "_source_hashes", lambda _root=None: next(snapshots))
    receipt = _one_case()
    assert receipt["status"] == "partial"
    assert receipt["winner_eligible"] is False
    assert receipt["candidate_source_sha256_before"] != receipt["candidate_source_sha256_after"]


def test_runtime_drift_disqualifies_receipt(monkeypatch):
    # Catches accepting a CPU comparison when interpreter/numeric runtime
    # provenance changed between the initial and final snapshots.
    _install_fast_callables(monkeypatch)
    snapshots = iter((
        {"pid": 1234, "python": "3.test", "numpy": "1", "scipy": "1"},
        {"pid": 1234, "python": "3.test", "numpy": "2", "scipy": "1"},
    ))
    monkeypatch.setattr(benchmark, "_runtime_identity", lambda: next(snapshots))
    receipt = _one_case()
    assert receipt["status"] == "partial"
    assert receipt["winner_eligible"] is False
    assert receipt["runtime_before"] != receipt["runtime_after"]


def test_mutated_readonly_input_is_detected_and_disqualified(monkeypatch):
    # Catches a candidate that circumvents the read-only flag and changes the
    # shared profile, which could invalidate subsequent A/B comparisons.
    _install_fast_callables(monkeypatch)

    def mutating_candidate(qr):
        qr.setflags(write=True)
        qr[0, 0] += 1.0
        return np.zeros(qr.shape[1], dtype=np.float64)

    monkeypatch.setattr(benchmark, "_candidate", mutating_candidate)
    receipt = _one_case()
    assert receipt["status"] == "partial"
    assert receipt["winner_eligible"] is False
    assert _only_result(receipt)["bitwise_parity_each_call"] is True


def test_memory_preflight_skips_case_without_timing_or_qualification(monkeypatch):
    # Catches allocating/timing a case whose conservative QxF working-set
    # estimate exceeds the benchmark cap.
    calls = _install_fast_callables(monkeypatch)
    monkeypatch.setattr(
        benchmark, "_estimate_peak_memory_bytes",
        lambda _shape, *, tile_factors=128: 536870913,
    )
    receipt = _one_case()
    assert receipt["status"] == "partial"
    assert receipt["winner_eligible"] is False
    assert receipt["cases"][0]["status"] == "skipped_memory_cap"
    assert calls == []
    assert _only_result(receipt)["samples"] == []


def test_deadline_preflight_returns_partial_without_qualification(monkeypatch):
    # Catches a benchmark that continues once the hard wall-clock budget is
    # already exhausted.
    calls = _install_fast_callables(monkeypatch)
    times = iter((0.0, 121.0, 121.0, 121.0))
    monkeypatch.setattr(benchmark, "_now", lambda: next(times))
    receipt = _one_case(deadline_seconds=120.0)
    assert receipt["status"] == "partial"
    assert receipt["winner_eligible"] is False
    assert receipt["budget"]["deadline_exceeded"] is True
    assert calls == []


def test_exception_path_returns_failed_receipt_without_global_leaks(monkeypatch):
    # Catches leaked process-global mutations when an implementation callable
    # raises midway through benchmark orchestration.
    _install_fast_callables(monkeypatch)
    legacy = benchmark._legacy
    before_env = dict(os.environ)

    def raising_candidate(_qr):
        raise RuntimeError("sentinel candidate failure")

    monkeypatch.setattr(benchmark, "_candidate", raising_candidate)
    receipt = _one_case()
    assert receipt["status"] == "failed"
    assert receipt["winner_eligible"] is False
    assert "sentinel candidate failure" in receipt["results"]["q5_f48_mixed"]["failure"]
    assert dict(os.environ) == before_env
    assert benchmark._legacy is legacy
    assert benchmark._candidate is raising_candidate
