"""Mock-only contracts for the fresh-process member-scan benchmark."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_chunked_member_scan_oct04.py"
SPEC = importlib.util.spec_from_file_location("benchmark_chunked_member_scan_oct04", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_legacy_and_bounded_modes_choose_same_seeded_winners_within_epsilon():
    members, queries, mapping = BENCH._make_synthetic_inputs(
        members=43, dimensions=13, queries=4, seed=20261004)

    legacy = BENCH._run_mode("legacy", members, queries, mapping)
    bounded = BENCH._run_mode(
        "bounded", members, queries, mapping,
        max_chunk_rows=7, max_chunk_bytes=13 * 8 * 3 * 4)

    assert [row[0] for row in bounded] == [row[0] for row in legacy]
    assert [row[1] for row in bounded] == pytest.approx(
        [row[1] for row in legacy], rel=0.0, abs=BENCH.DEFAULT_SIMILARITY_EPSILON)


def test_both_modes_preserve_first_member_on_exact_similarity_tie():
    members = ["first", "opposite", "tied-later", "orthogonal"]
    queries = [BENCH._make_fingerprint("query", [1.0, 0.0, 0.0], seed=1)]
    mapping = {
        "first": BENCH._make_fingerprint("first", [1.0, 0.0, 0.0], seed=1),
        "opposite": BENCH._make_fingerprint("opposite", [-1.0, 0.0, 0.0], seed=1),
        "tied-later": BENCH._make_fingerprint("tied-later", [1.0, 0.0, 0.0], seed=1),
        "orthogonal": BENCH._make_fingerprint("orthogonal", [0.0, 1.0, 0.0], seed=1),
        "query": queries[0],
    }

    for mode in ("legacy", "bounded"):
        result = BENCH._run_mode(
            mode, members, queries, mapping, max_chunk_rows=2)
        assert result[0][0] == "first"
        assert result[0][1] == 1.0


def test_input_envelope_rejects_estimated_payload_above_cap():
    with pytest.raises(ValueError, match="input cap"):
        BENCH._validate_input_envelope(
            members=BENCH.DEFAULT_MEMBERS, dimensions=64, queries=4,
            max_input_bytes=1)


def test_default_workload_is_oversized_for_existing_member_matrix_cache():
    estimate = BENCH._legacy_cache_estimate_bytes(
        BENCH.DEFAULT_MEMBERS, BENCH.DEFAULT_DIMENSIONS)

    assert estimate > 32 * 1024**2
    assert BENCH._legacy_requires_per_query_preparation(
        BENCH.DEFAULT_MEMBERS, BENCH.DEFAULT_DIMENSIONS)


def test_legacy_prepares_once_when_cached_and_once_per_query_when_oversized(monkeypatch):
    members, queries, mapping = BENCH._make_synthetic_inputs(
        members=5, dimensions=2, queries=4, seed=7)
    original = BENCH.incremental._prepare_unit_members
    calls = []

    def observed(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(BENCH.incremental, "_prepare_unit_members", observed)
    BENCH._run_mode("legacy", members, queries, mapping,
                    cache_budget_bytes=10**9)
    assert len(calls) == 1

    calls.clear()
    BENCH._run_mode("legacy", members, queries, mapping,
                    cache_budget_bytes=0)
    assert len(calls) == len(queries)


def test_exclusive_report_writer_does_not_replace_existing_evidence(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        BENCH._write_report_exclusive(report, {"status": "new"})

    assert report.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), True, np.int64(42), None, -1,
])
def test_resource_probes_reject_non_builtin_or_negative_values(bad):
    with pytest.raises(RuntimeError, match="available-memory probe"):
        BENCH._check_resource_headroom(
            memory_probe=lambda: bad,
            disk_probe=lambda: BENCH.MIN_FREE_DISK_BYTES)
    with pytest.raises(RuntimeError, match="free-disk probe"):
        BENCH._check_resource_headroom(
            memory_probe=lambda: BENCH.MIN_AVAILABLE_MEMORY_BYTES,
            disk_probe=lambda: bad)


@pytest.mark.parametrize("field,bad", [
    ("mode", "other"),
    ("elapsed_seconds", float("nan")),
    ("process_peak_rss_bytes", True),
    ("input_estimate_bytes", 1),
    ("winners", []),
    ("winners", [("M00000", float("inf"))]),
])
def test_worker_receipt_validator_rejects_bad_mode_time_rss_or_envelope(field, bad):
    row = {
        "mode": "legacy", "elapsed_seconds": 0.01,
        "process_peak_rss_bytes": 10,
        "input_estimate_bytes": BENCH._estimate_input_bytes(
            members=5, dimensions=2, queries=1),
        "input_fingerprint": "a" * 64,
        "winners": [("M00000", 0.5)],
        "source_hashes": {}, "runtime": {},
    }
    row[field] = bad

    with pytest.raises(RuntimeError):
        BENCH._validate_worker_receipt(
            row, expected_mode="legacy", expected_queries=1,
            expected_members=5,
            expected_input_bytes=BENCH._estimate_input_bytes(
                members=5, dimensions=2, queries=1),
            max_input_bytes=BENCH.MAX_INPUT_BYTES)


def test_worker_receipt_rejects_winner_outside_synthetic_member_range():
    row = {
        "mode": "bounded", "elapsed_seconds": 0.01,
        "process_peak_rss_bytes": 10,
        "input_estimate_bytes": BENCH._estimate_input_bytes(
            members=5, dimensions=2, queries=1),
        "input_fingerprint": "a" * 64,
        "winners": [("M99999", 0.5)],
        "source_hashes": {}, "runtime": {},
    }

    with pytest.raises(RuntimeError, match="outside the configured member set"):
        BENCH._validate_worker_receipt(
            row, expected_mode="bounded", expected_queries=1,
            expected_members=5,
            expected_input_bytes=BENCH._estimate_input_bytes(
                members=5, dimensions=2, queries=1),
            max_input_bytes=BENCH.MAX_INPUT_BYTES)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_winner_parity_never_accepts_nonfinite_similarity(bad):
    with pytest.raises(ValueError, match="non-finite"):
        BENCH._assert_winner_parity([("member", 0.5)], [("member", bad)], 1e-12)


def test_chunk_bounds_reject_larger_than_declared_caps_at_every_seam(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="chunk bounds"):
        BENCH._validate_chunk_bounds(
            BENCH.DEFAULT_MAX_CHUNK_ROWS + 1, BENCH.DEFAULT_MAX_CHUNK_BYTES)
    with pytest.raises(ValueError, match="chunk bounds"):
        BENCH._validate_chunk_bounds(
            BENCH.DEFAULT_MAX_CHUNK_ROWS, BENCH.DEFAULT_MAX_CHUNK_BYTES + 1)

    def forbidden(*args, **kwargs):
        pytest.fail("oversized chunks must be rejected before running workers")

    with pytest.raises(ValueError, match="chunk bounds"):
        BENCH.run_benchmark(
            report=tmp_path / "invalid-chunks.json", members=5, dimensions=2,
            queries=1, max_chunk_rows=BENCH.DEFAULT_MAX_CHUNK_ROWS + 1,
            worker_runner=forbidden, resource_check=forbidden)


def test_chunk_scan_entry_rejects_unbounded_override():
    members, queries, mapping = BENCH._make_synthetic_inputs(
        members=5, dimensions=2, queries=1, seed=11)
    with pytest.raises(ValueError, match="chunk bounds"):
        BENCH._run_mode(
            "bounded", members, queries, mapping,
            max_chunk_rows=BENCH.DEFAULT_MAX_CHUNK_ROWS + 1)


def test_worker_rejects_unbounded_chunk_before_resource_or_input_work(monkeypatch):
    monkeypatch.setattr(
        BENCH, "_make_synthetic_inputs",
        lambda **kwargs: pytest.fail("invalid chunk bounds must precede input allocation"))

    with pytest.raises(ValueError, match="chunk bounds"):
        BENCH._worker(
            mode="bounded", members=5, dimensions=2, queries=1, seed=11,
            max_input_bytes=BENCH.MAX_INPUT_BYTES,
            max_chunk_rows=BENCH.DEFAULT_MAX_CHUNK_ROWS + 1,
            max_chunk_bytes=BENCH.DEFAULT_MAX_CHUNK_BYTES,
            resource_check=lambda: pytest.fail(
                "invalid chunk bounds must precede resource admission"))


def _cgroup_files(tmp_path, *, version, current_path, mount_root="/"):
    mount = tmp_path / f"cgroup-{version}"
    relative = current_path.removeprefix(mount_root).strip("/")
    leaf = mount / relative
    leaf.mkdir(parents=True)
    limit_name, current_name, unlimited = (
        ("memory.max", "memory.current", "max") if version == 2 else
        ("memory.limit_in_bytes", "memory.usage_in_bytes", str(2**63 - 4096)))
    ancestor = leaf
    while ancestor != mount:
        (ancestor / limit_name).write_text(unlimited, encoding="utf-8")
        (ancestor / current_name).write_text("0", encoding="utf-8")
        ancestor = ancestor.parent
    # The mounted root may have accounting but no local limit/controller files.
    (mount / current_name).write_text("0", encoding="utf-8")
    cgroup_file = tmp_path / "proc-self-cgroup"
    mountinfo_file = tmp_path / "proc-self-mountinfo"
    if version == 2:
        cgroup_file.write_text(f"0::{current_path}\n", encoding="utf-8")
        mountinfo_file.write_text(
            f"42 25 0:32 {mount_root} {mount} rw - cgroup2 cgroup rw\n",
            encoding="utf-8")
    else:
        cgroup_file.write_text(f"5:memory:{current_path}\n", encoding="utf-8")
        mountinfo_file.write_text(
            f"43 25 0:33 {mount_root} {mount} rw - cgroup cgroup rw,memory\n",
            encoding="utf-8")
    return leaf, cgroup_file, mountinfo_file


def test_cgroup_v2_finite_limit_uses_nested_scope_remaining_bytes(tmp_path):
    leaf, cgroup_file, mountinfo_file = _cgroup_files(
        tmp_path, version=2, current_path="/scope/job/worker", mount_root="/scope")
    (leaf / "memory.max").write_text(str(10 * 1024**3), encoding="utf-8")
    (leaf / "memory.current").write_text(str(4 * 1024**3), encoding="utf-8")

    available = BENCH._cgroup_memory_available_bytes(
        cgroup_file=cgroup_file, mountinfo_file=mountinfo_file)

    assert available == 6 * 1024**3


def test_cgroup_v2_max_is_unlimited_but_still_requires_valid_current(tmp_path):
    leaf, cgroup_file, mountinfo_file = _cgroup_files(
        tmp_path, version=2, current_path="/worker")
    (leaf / "memory.max").write_text("max", encoding="utf-8")
    (leaf / "memory.current").write_text("123", encoding="utf-8")

    assert BENCH._cgroup_memory_available_bytes(
        cgroup_file=cgroup_file, mountinfo_file=mountinfo_file) is None

    (leaf / "memory.current").unlink()
    with pytest.raises(RuntimeError, match="memory.current"):
        BENCH._cgroup_memory_available_bytes(
            cgroup_file=cgroup_file, mountinfo_file=mountinfo_file)


def test_cgroup_v1_finite_and_unlimited_limits(tmp_path):
    leaf, cgroup_file, mountinfo_file = _cgroup_files(
        tmp_path, version=1, current_path="/memory/job")
    (leaf / "memory.limit_in_bytes").write_text(str(9 * 1024**3), encoding="utf-8")
    (leaf / "memory.usage_in_bytes").write_text(str(2 * 1024**3), encoding="utf-8")

    assert BENCH._cgroup_memory_available_bytes(
        cgroup_file=cgroup_file, mountinfo_file=mountinfo_file) == 7 * 1024**3

    (leaf / "memory.limit_in_bytes").write_text(str(2**63 - 4096), encoding="utf-8")
    assert BENCH._cgroup_memory_available_bytes(
        cgroup_file=cgroup_file, mountinfo_file=mountinfo_file) is None


def test_missing_cgroup_controller_mount_fails_closed(tmp_path):
    cgroup_file = tmp_path / "proc-self-cgroup"
    mountinfo_file = tmp_path / "proc-self-mountinfo"
    cgroup_file.write_text("0::/job\n", encoding="utf-8")
    mountinfo_file.write_text("42 25 0:32 / /not-a-cgroup rw - tmpfs tmpfs rw\n",
                              encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot locate current cgroup"):
        BENCH._cgroup_memory_available_bytes(
            cgroup_file=cgroup_file, mountinfo_file=mountinfo_file)


@pytest.mark.parametrize("limit,current", [("garbage", "1"), ("10", "bad"), ("10", None)])
def test_cgroup_malformed_or_missing_values_fail_closed(tmp_path, limit, current):
    leaf, cgroup_file, mountinfo_file = _cgroup_files(
        tmp_path, version=2, current_path="/job")
    (leaf / "memory.max").write_text(limit, encoding="utf-8")
    if current is not None:
        (leaf / "memory.current").write_text(current, encoding="utf-8")
    else:
        (leaf / "memory.current").unlink()

    with pytest.raises(RuntimeError, match="cgroup memory"):
        BENCH._cgroup_memory_available_bytes(
            cgroup_file=cgroup_file, mountinfo_file=mountinfo_file)


def test_cgroup_current_above_limit_and_host_tighter_both_gate(tmp_path):
    leaf, cgroup_file, mountinfo_file = _cgroup_files(
        tmp_path, version=2, current_path="/job")
    (leaf / "memory.max").write_text(str(5 * 1024**3), encoding="utf-8")
    (leaf / "memory.current").write_text(str(6 * 1024**3), encoding="utf-8")
    cgroup_probe = lambda: BENCH._cgroup_memory_available_bytes(
        cgroup_file=cgroup_file, mountinfo_file=mountinfo_file)
    with pytest.raises(MemoryError, match="available RAM"):
        BENCH._check_resource_headroom(
            memory_probe=lambda: 8 * 1024**3, cgroup_probe=cgroup_probe,
            disk_probe=lambda: BENCH.MIN_FREE_DISK_BYTES)

    (leaf / "memory.max").write_text(str(10 * 1024**3), encoding="utf-8")
    (leaf / "memory.current").write_text(str(2 * 1024**3), encoding="utf-8")
    with pytest.raises(MemoryError, match="available RAM"):
        BENCH._check_resource_headroom(
            memory_probe=lambda: 3 * 1024**3, cgroup_probe=cgroup_probe,
            disk_probe=lambda: BENCH.MIN_FREE_DISK_BYTES)
    BENCH._check_resource_headroom(
        memory_probe=lambda: 8 * 1024**3, cgroup_probe=cgroup_probe,
        disk_probe=lambda: BENCH.MIN_FREE_DISK_BYTES)


def test_cgroup_v2_walks_parent_limits_when_child_is_max_and_root_has_no_files(tmp_path):
    leaf, cgroup_file, mountinfo_file = _cgroup_files(
        tmp_path, version=2, current_path="/job/child/worker")
    parent = leaf.parent
    intermediate = leaf.parent.parent
    (leaf / "memory.max").write_text("max", encoding="utf-8")
    (leaf / "memory.current").write_text(str(1 * 1024**3), encoding="utf-8")
    (parent / "memory.max").write_text(str(6 * 1024**3), encoding="utf-8")
    (parent / "memory.current").write_text(str(4 * 1024**3), encoding="utf-8")
    (intermediate / "memory.max").write_text("max", encoding="utf-8")
    (intermediate / "memory.current").write_text(str(2 * 1024**3), encoding="utf-8")
    # The mounted controller root can lack these files when it is not enabled there.

    assert BENCH._cgroup_memory_available_bytes(
        cgroup_file=cgroup_file, mountinfo_file=mountinfo_file) == 2 * 1024**3


def test_cgroup_v2_skips_child_without_controller_and_uses_finite_parent(tmp_path):
    leaf, cgroup_file, mountinfo_file = _cgroup_files(
        tmp_path, version=2, current_path="/job/worker")
    (leaf / "memory.max").unlink()
    (leaf / "memory.current").unlink()
    parent = leaf.parent
    (parent / "memory.max").write_text(str(7 * 1024**3), encoding="utf-8")
    (parent / "memory.current").write_text(str(2 * 1024**3), encoding="utf-8")

    assert BENCH._cgroup_memory_available_bytes(
        cgroup_file=cgroup_file, mountinfo_file=mountinfo_file) == 5 * 1024**3


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_child_output_limit_bounds_both_streams_and_reaps_child(stream):
    code = f"import sys; sys.{stream}.write('x' * 10000); sys.{stream}.flush()"
    with pytest.raises(RuntimeError, match="output exceeded"):
        BENCH._run_bounded_subprocess(
            [sys.executable, "-c", code], cwd=BENCH.PROJECT_ROOT,
            environment=os.environ.copy(), timeout=5,
            stdout_limit=128, stderr_limit=128)


def test_child_timeout_kills_and_reaps_process():
    code = "import time; time.sleep(5)"
    with pytest.raises(subprocess.TimeoutExpired):
        BENCH._run_bounded_subprocess(
            [sys.executable, "-c", code], cwd=BENCH.PROJECT_ROOT,
            environment=os.environ.copy(), timeout=0.05,
            stdout_limit=128, stderr_limit=128)


def test_child_timeout_kills_process_group_once(monkeypatch):
    original_killpg = BENCH.os.killpg
    killed_groups = []

    def tracked_killpg(pid, sig):
        killed_groups.append((pid, sig))
        return original_killpg(pid, sig)

    monkeypatch.setattr(BENCH.os, "killpg", tracked_killpg)
    with pytest.raises(subprocess.TimeoutExpired):
        BENCH._run_bounded_subprocess(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=BENCH.PROJECT_ROOT, environment=os.environ.copy(), timeout=0.05,
            stdout_limit=128, stderr_limit=128)
    assert len(killed_groups) == 1


def test_child_timeout_applies_after_both_output_pipes_close():
    code = "import os, time; os.close(1); os.close(2); time.sleep(1.0)"
    with pytest.raises(subprocess.TimeoutExpired):
        BENCH._run_bounded_subprocess(
            [sys.executable, "-c", code], cwd=BENCH.PROJECT_ROOT,
            environment=os.environ.copy(), timeout=0.4,
            stdout_limit=128, stderr_limit=128)


def test_selector_setup_failure_kills_and_reaps_child(monkeypatch):
    original_popen = BENCH.subprocess.Popen
    children = []

    def tracked_popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    def fail_selector():
        raise OSError("selector setup failed")

    monkeypatch.setattr(BENCH.subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(BENCH.selectors, "DefaultSelector", fail_selector)
    reaped_by_helper = False
    try:
        with pytest.raises(OSError, match="selector setup failed"):
            BENCH._run_bounded_subprocess(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=BENCH.PROJECT_ROOT, environment=os.environ.copy(), timeout=5,
                stdout_limit=128, stderr_limit=128)
    finally:
        reaped_by_helper = bool(children and children[0].poll() is not None)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
    assert reaped_by_helper


def test_orchestrator_uses_fixed_abba_and_writes_only_after_parity(tmp_path):
    calls = []
    resource_calls = []

    def worker(_script, *, mode, **kwargs):
        calls.append(mode)
        member_ids, queries, mapping = BENCH._make_synthetic_inputs(
            members=19, dimensions=5, queries=2, seed=kwargs["seed"])
        winners = BENCH._run_mode(mode, member_ids, queries, mapping,
                                  max_chunk_rows=kwargs["max_chunk_rows"])
        return {
            "mode": mode,
            "elapsed_seconds": 1.0 if mode == "legacy" else 0.5,
            "process_peak_rss_bytes": 100 if mode == "legacy" else 80,
            "input_estimate_bytes": BENCH._estimate_input_bytes(
                members=19, dimensions=5, queries=2),
            "input_fingerprint": BENCH._fingerprint_map_hash(
                member_ids, queries, mapping),
            "winners": winners,
            "source_hashes": BENCH._source_hashes(),
            "runtime": BENCH._runtime_context(),
        }

    report = tmp_path / "abba.json"
    result = BENCH.run_benchmark(
        report=report, members=19, dimensions=5, queries=2,
        worker_runner=worker, resource_check=lambda: resource_calls.append(1))

    assert calls == ["legacy", "bounded", "bounded", "legacy"]
    assert result["equivalence"]["winner_identity_matches"] is True
    assert result["equivalence"]["bitwise_similarity_promise"] is False
    assert report.exists()
    assert resource_calls == [1, 1, 1, 1]


@pytest.mark.parametrize("drift,expected_error", [
    ("source", "declared benchmark sources changed"),
    ("runtime", "benchmark parent runtime changed"),
    ("input", "identical seeded fingerprints"),
])
def test_orchestrator_rejects_source_runtime_or_input_fingerprint_drift(
        drift, expected_error, tmp_path, monkeypatch):
    calls = []
    static_hashes = {"source": "stable"}
    static_runtime = {"runtime": "stable"}
    if drift == "source":
        hash_calls = []

        def changed_hashes():
            hash_calls.append(1)
            return {"source": "stable" if len(hash_calls) == 1 else "changed"}

        monkeypatch.setattr(BENCH, "_source_hashes", changed_hashes)
    if drift == "runtime":
        runtime_calls = []

        def changed_runtime():
            runtime_calls.append(1)
            return {"runtime": "stable" if len(runtime_calls) == 1 else "changed"}

        monkeypatch.setattr(BENCH, "_runtime_context", changed_runtime)

    def worker(_script, *, mode, **kwargs):
        calls.append(mode)
        index = len(calls)
        return {
            "mode": mode, "elapsed_seconds": 0.01,
            "process_peak_rss_bytes": 10,
            "input_estimate_bytes": BENCH._estimate_input_bytes(
                members=5, dimensions=2, queries=1),
            "input_fingerprint": (
                "b" * 64 if drift == "input" and index == 2 else "a" * 64),
            "winners": [("M00000", 0.5)],
            "source_hashes": static_hashes,
            "runtime": static_runtime,
        }

    with pytest.raises(RuntimeError, match=expected_error):
        BENCH.run_benchmark(
            report=tmp_path / f"{drift}.json", members=5, dimensions=2,
            queries=1, resource_check=lambda: None, worker_runner=worker)

    assert not (tmp_path / f"{drift}.json").exists()


def test_parent_resource_guard_stops_before_spawning_workers(tmp_path):
    def insufficient():
        raise MemoryError("resource admission fixture")

    def forbidden_worker(*args, **kwargs):
        pytest.fail("resource failure must precede worker process launch")

    with pytest.raises(MemoryError, match="resource admission fixture"):
        BENCH.run_benchmark(
            report=tmp_path / "not-written.json", members=5, dimensions=2,
            queries=1, resource_check=insufficient, worker_runner=forbidden_worker)

    assert not (tmp_path / "not-written.json").exists()


def test_worker_resource_guard_stops_before_synthetic_allocation(monkeypatch):
    def insufficient():
        raise MemoryError("worker resource fixture")

    monkeypatch.setattr(
        BENCH, "_make_synthetic_inputs",
        lambda **kwargs: pytest.fail("resource failure must precede input allocation"))

    with pytest.raises(MemoryError, match="worker resource fixture"):
        BENCH._worker(
            mode="bounded", members=5, dimensions=2, queries=1, seed=3,
            max_input_bytes=BENCH.MAX_INPUT_BYTES,
            max_chunk_rows=2, max_chunk_bytes=1024,
            resource_check=insufficient)
