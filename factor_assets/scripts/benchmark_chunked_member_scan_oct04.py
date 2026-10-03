"""Fresh-process ABBA benchmark for full and bounded exact member scans.

Both paths receive independently regenerated, byte-identical seeded synthetic
fingerprints. The benchmark is descriptive only: it reports per-process peak
RSS and elapsed preparation-plus-query time, not a production performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import posixpath
import re
import resource
import selectors
import shutil
import signal
import statistics
import subprocess
import sys
import time

import numpy as np


FA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FA_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_assets.clustering import incremental
from factor_assets.clustering.incremental_recall import scan_exact_member_winner
from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact


DEFAULT_MEMBERS = 17_000
DEFAULT_DIMENSIONS = 256
DEFAULT_QUERIES = 4
DEFAULT_SEED = 20261004
DEFAULT_SIMILARITY_EPSILON = 2e-12
DEFAULT_MAX_CHUNK_ROWS = 256
DEFAULT_MAX_CHUNK_BYTES = 4 * 1024**2
MAX_INPUT_BYTES = 256 * 1024**2
MAX_REPORT_BYTES = 1 * 1024**2
MIN_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
MIN_FREE_DISK_BYTES = 1 * 1024**3
ABBA_MODES = ("legacy", "bounded", "bounded", "legacy")
SOURCE_PATHS = (
    "factor_assets/scripts/benchmark_chunked_member_scan_oct04.py",
    "factor_assets/clustering/incremental.py",
    "factor_assets/clustering/incremental_recall.py",
    "factor_assets/similarity/unit_vectors.py",
    "factor_assets/contracts/fingerprint.py",
)


def _make_fingerprint(identifier: str, vector, *, seed: int):
    return SimilarityFingerprintArtifact(
        factor_id=identifier,
        embedding=tuple(map(float, vector)),
        embedding_spec=f"member-scan-ab-seed-{seed}",
        snapshot="synthetic-member-scan-oct04",
        universe="synthetic-embedding-library",
        window="fixed-ab-window",
        preprocessing_ref="none",
        mask_policy="finite-nonzero",
        direction="signed",
        aggregation_method="embedding-cosine",
        embedding_model_version="seeded-normal-v1",
        value_ref=f"synthetic:{identifier}",
        profile_ref=f"synthetic-profile:{identifier}",
    )


def _make_synthetic_inputs(*, members=DEFAULT_MEMBERS,
                           dimensions=DEFAULT_DIMENSIONS,
                           queries=DEFAULT_QUERIES, seed=DEFAULT_SEED):
    """Build one deterministic object set without retaining a source matrix."""
    rng = np.random.default_rng(seed)
    member_ids = []
    fingerprints = {}
    for index in range(members):
        identifier = f"M{index:05d}"
        fingerprint = _make_fingerprint(
            identifier, rng.standard_normal(dimensions), seed=seed)
        member_ids.append(identifier)
        fingerprints[identifier] = fingerprint
    query_objects = []
    for index in range(queries):
        identifier = f"Q{index:04d}"
        fingerprint = _make_fingerprint(
            identifier, rng.standard_normal(dimensions), seed=seed)
        query_objects.append(fingerprint)
        fingerprints[identifier] = fingerprint
    return member_ids, query_objects, fingerprints


def _fingerprint_map_hash(member_ids, query_objects, fingerprints):
    digest = hashlib.sha256()
    identifiers = iter(member_ids)
    for identifier in identifiers:
        fingerprint = fingerprints[identifier]
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(fingerprint.embedding, dtype=np.float64).tobytes())
    for query in query_objects:
        identifier = query.factor_id
        fingerprint = fingerprints[identifier]
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(fingerprint.embedding, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _estimate_input_bytes(*, members, dimensions, queries):
    # Conservative estimate for tuple-held Python floats and per-artifact
    # metadata, plus one transient float64 source vector during construction.
    row_estimate = dimensions * 48 + 2048
    return int((members + queries) * row_estimate + dimensions * 8)


def _validate_input_envelope(*, members, dimensions, queries,
                             max_input_bytes=MAX_INPUT_BYTES):
    if (type(members) is not int or type(dimensions) is not int
            or type(queries) is not int or not 1 <= members <= DEFAULT_MEMBERS
            or not 1 <= dimensions <= DEFAULT_DIMENSIONS
            or not 1 <= queries <= DEFAULT_QUERIES):
        raise ValueError("synthetic dimensions exceed the bounded benchmark envelope")
    if (type(max_input_bytes) is not int or not 1 <= max_input_bytes <= MAX_INPUT_BYTES
            or _estimate_input_bytes(members=members, dimensions=dimensions,
                                     queries=queries) > max_input_bytes):
        raise ValueError("estimated synthetic input exceeds the input cap")
    return _estimate_input_bytes(members=members, dimensions=dimensions,
                                 queries=queries)


def _legacy_cache_estimate_bytes(members, dimensions):
    return int(members * (dimensions * 8 + 64) + 256)


def _legacy_requires_per_query_preparation(members, dimensions):
    return (_legacy_cache_estimate_bytes(members, dimensions)
            > incremental._EXACT_MEMBER_CACHE_BYTES)


def _validate_chunk_bounds(max_chunk_rows, max_chunk_bytes):
    if (type(max_chunk_rows) is not int
            or not 1 <= max_chunk_rows <= DEFAULT_MAX_CHUNK_ROWS
            or type(max_chunk_bytes) is not int
            or not 1 <= max_chunk_bytes <= DEFAULT_MAX_CHUNK_BYTES):
        raise ValueError("chunk bounds must be positive and no larger than the declared caps")


def _available_memory_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _mountinfo_unescape(value):
    return re.sub(r"\\([0-7]{3})",
                  lambda match: chr(int(match.group(1), 8)), value)


def _cgroup_memory_available_bytes(*, cgroup_file=Path("/proc/self/cgroup"),
                                   mountinfo_file=Path("/proc/self/mountinfo")):
    try:
        cgroup_lines = Path(cgroup_file).read_text().splitlines()
        mount_lines = Path(mountinfo_file).read_text().splitlines()
    except OSError as exc:
        raise RuntimeError("cannot inspect cgroup memory controller") from exc

    version, current_path = None, None
    for line in cgroup_lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        if fields[1] == "":
            version, current_path = 2, fields[2]
            break
        if "memory" in fields[1].split(","):
            version, current_path = 1, fields[2]
            break
    if version is None or current_path is None:
        raise RuntimeError("cgroup memory controller is unsupported or unavailable")

    matched_path = None
    expected_type = "cgroup2" if version == 2 else "cgroup"
    for line in mount_lines:
        fields = line.split()
        if "-" not in fields:
            continue
        separator = fields.index("-")
        if len(fields) <= separator + 3 or len(fields) < 5:
            continue
        if fields[separator + 1] != expected_type:
            continue
        if version == 1:
            options = set(fields[separator + 3].split(","))
            if "memory" not in options and fields[separator + 2] != "memory":
                continue
        mount_root = posixpath.normpath(_mountinfo_unescape(fields[3]))
        process_path = posixpath.normpath(current_path)
        if process_path == mount_root:
            suffix = ""
        elif mount_root == "/" and process_path.startswith("/"):
            suffix = process_path.lstrip("/")
        elif process_path.startswith(mount_root.rstrip("/") + "/"):
            suffix = process_path[len(mount_root.rstrip("/")) + 1:]
        else:
            continue
        mountpoint = Path(_mountinfo_unescape(fields[4])).resolve()
        candidate = (mountpoint / suffix).resolve()
        if candidate != mountpoint and mountpoint not in candidate.parents:
            raise RuntimeError("cgroup path escapes its mounted controller")
        matched_path = candidate
        break
    if matched_path is None:
        raise RuntimeError("cannot locate current cgroup memory mount")

    limit_name, current_name = (
        ("memory.max", "memory.current") if version == 2 else
        ("memory.limit_in_bytes", "memory.usage_in_bytes"))
    available = None
    directory = matched_path
    while True:
        limit_path = directory / limit_name
        current_path = directory / current_name
        limit_exists, current_exists = limit_path.exists(), current_path.exists()
        if not limit_exists and not current_exists:
            # This cgroup has no memory controller files; ancestors still apply.
            pass
        elif not current_exists or (not limit_exists and directory != mountpoint):
            raise RuntimeError(
                f"cgroup memory {limit_name} or {current_name} file is missing")
        else:
            try:
                current_text = current_path.read_text().strip()
                limit_text = limit_path.read_text().strip() if limit_exists else None
            except OSError as exc:
                raise RuntimeError(
                    f"cannot read cgroup memory {limit_name} or {current_name}") from exc
            try:
                current = int(current_text)
                if current < 0:
                    raise ValueError("negative current usage")
                if limit_text is None or (version == 2 and limit_text == "max"):
                    limit = None
                else:
                    limit = int(limit_text)
                    if limit < 0:
                        raise ValueError("negative memory limit")
            except ValueError as exc:
                raise RuntimeError(
                    "cgroup memory limit/current value is malformed") from exc
            if version == 1 and limit is not None and limit >= (1 << 60):
                limit = None
            if limit is not None:
                remaining = max(0, limit - current)
                available = remaining if available is None else min(available, remaining)
        if directory == mountpoint:
            break
        directory = directory.parent
    return available


def _check_resource_headroom(*, memory_probe=None, disk_probe=None,
                             cgroup_probe=None):
    available = (memory_probe or _available_memory_bytes)()
    free = (disk_probe or (lambda: shutil.disk_usage(PROJECT_ROOT).free))()
    if type(available) is not int or available < 0:
        raise RuntimeError("available-memory probe must return a non-negative built-in int")
    if type(free) is not int or free < 0:
        raise RuntimeError("free-disk probe must return a non-negative built-in int")
    cgroup_available = (cgroup_probe or _cgroup_memory_available_bytes)()
    if cgroup_available is not None and (
            type(cgroup_available) is not int or cgroup_available < 0):
        raise RuntimeError("cgroup-memory probe must return None or a non-negative built-in int")
    if cgroup_available is not None:
        available = min(available, cgroup_available)
    if available < MIN_AVAILABLE_MEMORY_BYTES:
        raise MemoryError("benchmark requires at least 4 GiB available RAM")
    if free < MIN_FREE_DISK_BYTES:
        raise OSError("benchmark requires at least 1 GiB free disk")


def _run_mode(mode, member_ids, query_objects, fingerprints, *,
              max_chunk_rows=DEFAULT_MAX_CHUNK_ROWS,
              max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES,
              cache_budget_bytes=None):
    if mode not in {"legacy", "bounded"}:
        raise ValueError("mode must be legacy or bounded")
    _validate_chunk_bounds(max_chunk_rows, max_chunk_bytes)
    budget = (incremental._EXACT_MEMBER_CACHE_BYTES if cache_budget_bytes is None
              else cache_budget_bytes)
    if type(budget) is not int or budget < 0:
        raise ValueError("cache budget must be a non-negative integer")
    prepare_per_query = (
        _legacy_cache_estimate_bytes(len(member_ids),
                                     len(query_objects[0].embedding)) > budget
        if query_objects else False
    )
    prepared = None
    if mode == "legacy" and not prepare_per_query:
        present, unit_matrix = incremental._prepare_unit_members(
            fingerprints, member_ids)
        prepared = (present, unit_matrix)
    winners = []
    for query in query_objects:
        raw_query = incremental._to_embedding(query)
        unit_query = incremental._normalize_rows(raw_query.reshape(1, -1))[0]
        if mode == "legacy":
            if prepare_per_query:
                present, unit_matrix = incremental._prepare_unit_members(
                    fingerprints, member_ids)
            else:
                present, unit_matrix = prepared
            if unit_matrix is None:
                winners.append(None)
                continue
            scores = incremental.unit_cosine_scores(unit_query, unit_matrix)
            position = int(np.argmax(scores))
            winners.append((present[position], float(scores[position])))
        else:
            winners.append(scan_exact_member_winner(
                fingerprints, member_ids, unit_query,
                to_embedding=incremental._to_embedding,
                normalize_rows=incremental._normalize_rows,
                max_chunk_rows=max_chunk_rows,
                max_chunk_bytes=max_chunk_bytes,
            ))
    return winners


def _source_hashes():
    return {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }


def _runtime_context():
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "host": platform.node(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            )
        },
    }


def _peak_rss_bytes():
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return raw if sys.platform == "darwin" else raw * 1024


def _worker(*, mode, members, dimensions, queries, seed, max_input_bytes,
            max_chunk_rows, max_chunk_bytes,
            resource_check=_check_resource_headroom):
    input_bytes = _validate_input_envelope(
        members=members, dimensions=dimensions, queries=queries,
        max_input_bytes=max_input_bytes)
    _validate_chunk_bounds(max_chunk_rows, max_chunk_bytes)
    resource_check()
    hashes_before = _source_hashes()
    runtime_before = _runtime_context()
    member_ids, query_objects, fingerprints = _make_synthetic_inputs(
        members=members, dimensions=dimensions, queries=queries, seed=seed)
    input_fingerprint = _fingerprint_map_hash(
        member_ids, query_objects, fingerprints)
    started = time.perf_counter()
    winners = _run_mode(
        mode, member_ids, query_objects, fingerprints,
        max_chunk_rows=max_chunk_rows, max_chunk_bytes=max_chunk_bytes)
    elapsed = time.perf_counter() - started
    hashes_after = _source_hashes()
    runtime_after = _runtime_context()
    if hashes_before != hashes_after:
        raise RuntimeError("declared benchmark sources changed during worker timing")
    if runtime_before != runtime_after:
        raise RuntimeError("benchmark runtime changed during worker timing")
    return {
        "mode": mode,
        "elapsed_seconds": elapsed,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "input_estimate_bytes": input_bytes,
        "input_fingerprint": input_fingerprint,
        "winners": winners,
        "source_hashes": hashes_after,
        "runtime": runtime_after,
    }


def _assert_winner_parity(expected, actual, epsilon):
    if (type(epsilon) not in (int, float) or not math.isfinite(epsilon)
            or epsilon < 0):
        raise ValueError("similarity epsilon must be finite and non-negative")
    if len(expected) != len(actual):
        raise RuntimeError("legacy and bounded query coverage differs")
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if left is None or right is None:
            if left is not right:
                raise RuntimeError(f"winner presence differs at query {index}")
            continue
        if left[0] != right[0]:
            raise RuntimeError(f"winner identity differs at query {index}")
        left_score, right_score = float(left[1]), float(right[1])
        if not math.isfinite(left_score) or not math.isfinite(right_score):
            raise ValueError(f"winner similarity is non-finite at query {index}")
        if abs(left_score - right_score) > epsilon:
            raise RuntimeError(f"winner similarity exceeds epsilon at query {index}")


def _validate_worker_receipt(row, *, expected_mode, expected_queries,
                             expected_members, expected_input_bytes,
                             max_input_bytes):
    if not isinstance(row, dict):
        raise RuntimeError("worker receipt must be an object")
    if row.get("mode") != expected_mode:
        raise RuntimeError("worker receipt mode differs from requested ABBA mode")
    elapsed = row.get("elapsed_seconds")
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
            or elapsed <= 0):
        raise RuntimeError("worker elapsed time must be positive and finite")
    peak_rss = row.get("process_peak_rss_bytes")
    if type(peak_rss) is not int or peak_rss < 0:
        raise RuntimeError("worker RSS must be a non-negative built-in int")
    input_bytes = row.get("input_estimate_bytes")
    if (type(input_bytes) is not int or input_bytes < 0
            or input_bytes != expected_input_bytes or input_bytes > max_input_bytes):
        raise RuntimeError("worker input estimate is outside the admitted envelope")
    input_fingerprint = row.get("input_fingerprint")
    if (not isinstance(input_fingerprint, str) or len(input_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in input_fingerprint)):
        raise RuntimeError("worker input fingerprint is malformed")
    winners = row.get("winners")
    if not isinstance(winners, (list, tuple)) or len(winners) != expected_queries:
        raise RuntimeError("worker query coverage differs from configured query count")
    for index, winner in enumerate(winners):
        if (not isinstance(winner, (list, tuple)) or len(winner) != 2
                or not isinstance(winner[0], str) or not winner[0]):
            raise RuntimeError(f"worker winner envelope is malformed at query {index}")
        member_id = winner[0]
        if (not member_id.startswith("M") or not member_id[1:].isdigit()
                or member_id != f"M{int(member_id[1:]):05d}"
                or not 0 <= int(member_id[1:]) < expected_members):
            raise RuntimeError(f"worker winner is outside the configured member set at query {index}")
        score = winner[1]
        if type(score) not in (int, float) or not math.isfinite(score):
            raise RuntimeError(f"worker winner score is non-finite at query {index}")
        if abs(score) > 1.0 + 64 * np.finfo(np.float64).eps:
            raise RuntimeError(f"worker cosine score is outside the unit envelope at query {index}")
    if not isinstance(row.get("source_hashes"), dict) or not isinstance(row.get("runtime"), dict):
        raise RuntimeError("worker receipt lacks source/runtime identity")


def _write_report_exclusive(destination, payload):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite benchmark report: {destination}")
    encoded = (json.dumps(payload, sort_keys=True, indent=2,
                           ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) >= MAX_REPORT_BYTES:
        raise ValueError("benchmark report must be smaller than 1 MiB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(encoded.decode("utf-8"))
    return destination


def _run_bounded_subprocess(command, *, cwd, environment, timeout,
                           stdout_limit, stderr_limit):
    """Collect both pipes without allowing a noisy child to exhaust memory."""
    if (type(stdout_limit) is not int or stdout_limit < 0
            or type(stderr_limit) is not int or stderr_limit < 0):
        raise ValueError("subprocess output limits must be non-negative integers")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True)
    selector = None
    pipes = ()
    output = {}
    stopped = False

    def stop_child():
        nonlocal stopped
        if stopped:
            return
        stopped = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    try:
        pipes = (process.stdout, process.stderr)
        output = {pipe: bytearray() for pipe in pipes}
        limits = {process.stdout: stdout_limit, process.stderr: stderr_limit}
        selector = selectors.DefaultSelector()
        for pipe in pipes:
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop_child()
                raise subprocess.TimeoutExpired(command, timeout,
                                                output=bytes(output[process.stdout]),
                                                stderr=bytes(output[process.stderr]))
            events = selector.select(min(remaining, 0.1))
            for key, _ in events:
                pipe = key.fileobj
                try:
                    chunk = os.read(pipe.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(pipe)
                    pipe.close()
                    continue
                captured = output[pipe]
                captured.extend(chunk)
                if len(captured) > limits[pipe]:
                    stop_child()
                    raise RuntimeError(
                        f"child {pipe.name} output exceeded {limits[pipe]} bytes")
        remaining = max(0, deadline - time.monotonic())
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            stop_child()
            raise subprocess.TimeoutExpired(
                command, timeout, output=bytes(output[process.stdout]),
                stderr=bytes(output[process.stderr]))
        return subprocess.CompletedProcess(
            command, returncode,
            stdout=bytes(output[process.stdout]),
            stderr=bytes(output[process.stderr]))
    except BaseException:
        stop_child()
        raise
    finally:
        if selector is not None:
            selector.close()
        for pipe in pipes:
            if not pipe.closed:
                pipe.close()


def _run_worker_process(script, *, mode, members, dimensions, queries, seed,
                        max_input_bytes, max_chunk_rows, max_chunk_bytes):
    command = [
        sys.executable, str(script), "--worker-mode", mode,
        "--members", str(members), "--dimensions", str(dimensions),
        "--queries", str(queries), "--seed", str(seed),
        "--max-input-bytes", str(max_input_bytes),
        "--max-chunk-rows", str(max_chunk_rows),
        "--max-chunk-bytes", str(max_chunk_bytes),
    ]
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1"})
    completed = _run_bounded_subprocess(
        command, cwd=PROJECT_ROOT, environment=environment, timeout=300,
        stdout_limit=32 * 1024, stderr_limit=32 * 1024)
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout,
            stderr=completed.stderr)
    try:
        return json.loads(completed.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("benchmark worker returned malformed JSON") from exc


def run_benchmark(*, report, members=DEFAULT_MEMBERS,
                  dimensions=DEFAULT_DIMENSIONS, queries=DEFAULT_QUERIES,
                  seed=DEFAULT_SEED, max_input_bytes=MAX_INPUT_BYTES,
                  max_chunk_rows=DEFAULT_MAX_CHUNK_ROWS,
                  max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES,
                  resource_check=_check_resource_headroom,
                  worker_runner=_run_worker_process):
    input_bytes = _validate_input_envelope(
        members=members, dimensions=dimensions, queries=queries,
        max_input_bytes=max_input_bytes)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _validate_chunk_bounds(max_chunk_rows, max_chunk_bytes)
    destination = Path(report)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite benchmark report: {destination}")
    source_before = _source_hashes()
    runtime_before = _runtime_context()
    samples = []
    for mode in ABBA_MODES:
        resource_check()
        row = worker_runner(
            Path(__file__).resolve(), mode=mode, members=members,
            dimensions=dimensions, queries=queries, seed=seed,
            max_input_bytes=max_input_bytes, max_chunk_rows=max_chunk_rows,
            max_chunk_bytes=max_chunk_bytes)
        _validate_worker_receipt(
            row, expected_mode=mode, expected_queries=queries,
            expected_members=members,
            expected_input_bytes=input_bytes, max_input_bytes=max_input_bytes)
        samples.append(row)
    source_after = _source_hashes()
    runtime_after = _runtime_context()
    if source_before != source_after:
        raise RuntimeError("declared benchmark sources changed during ABBA run")
    if runtime_before != runtime_after:
        raise RuntimeError("benchmark parent runtime changed during ABBA run")
    expected_hash = samples[0]["input_fingerprint"]
    if any(row["input_fingerprint"] != expected_hash for row in samples):
        raise RuntimeError("fresh workers did not build identical seeded fingerprints")
    if any(row["source_hashes"] != source_before for row in samples):
        raise RuntimeError("worker source fingerprint differs from parent")
    worker_runtimes = [row["runtime"] for row in samples]
    if any(runtime != worker_runtimes[0] for runtime in worker_runtimes[1:]):
        raise RuntimeError("worker runtime contexts differ across ABBA measurements")
    legacy = next(row for row in samples if row["mode"] == "legacy")
    for row in samples[1:]:
        _assert_winner_parity(legacy["winners"], row["winners"],
                              DEFAULT_SIMILARITY_EPSILON)
    medians = {}
    for mode in ("legacy", "bounded"):
        values = [row["elapsed_seconds"] for row in samples if row["mode"] == mode]
        medians[mode] = statistics.median(values)
    report_payload = {
        "schema": "factor_assets.chunked_member_scan_abba.v1",
        "status": "complete",
        "scope": {
            "members": members, "dimensions": dimensions, "queries": queries,
            "seed": seed, "input_estimate_bytes": input_bytes,
            "input_cap_bytes": max_input_bytes,
            "source_dataset_copy": False,
            "fresh_process_per_measurement": True,
            "legacy_full_preparation_per_query": _legacy_requires_per_query_preparation(
                members, dimensions),
            "legacy_cache_estimate_bytes": _legacy_cache_estimate_bytes(
                members, dimensions),
            "timing": "member preparation plus exact winner queries; synthetic only",
        },
        "ordering": list(ABBA_MODES),
        "samples": samples,
        "medians_seconds": medians,
        "bounded_over_legacy_time_ratio": (
            medians["bounded"] / medians["legacy"]
            if medians["legacy"] > 0 else None),
        "median_process_peak_rss_bytes": {
            mode: int(statistics.median(
                row["process_peak_rss_bytes"] for row in samples
                if row["mode"] == mode))
            for mode in ("legacy", "bounded")
        },
        "equivalence": {
            "winner_identity_matches": True,
            "similarity_within_epsilon": True,
            "similarity_epsilon": DEFAULT_SIMILARITY_EPSILON,
            "bitwise_similarity_promise": False,
            "query_count": len(legacy["winners"]),
            "input_fingerprint": expected_hash,
        },
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "runtime_parent_before": runtime_before,
        "runtime_parent_after": runtime_after,
        "runtime_workers": worker_runtimes[0],
        "report_limit_bytes_strictly_less_than": MAX_REPORT_BYTES,
        "report_overwrite": "refused via exclusive create",
        "limitations": [
            "Synthetic embeddings; not a production-data performance claim.",
            "Fresh processes regenerate identical seeded fingerprints; no source dataset is copied.",
            "The legacy comparison prepares one full matrix per query when the existing member-cache admission estimate exceeds 32 MiB.",
            "Peak RSS includes imports and synthetic input objects, not only scan scratch.",
            "Declared source hashes are not a complete transitive runtime closure.",
        ],
    }
    written = _write_report_exclusive(destination, report_payload)
    return {"report_path": str(written), **report_payload}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=int, default=DEFAULT_MEMBERS)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-input-bytes", type=int, default=MAX_INPUT_BYTES)
    parser.add_argument("--max-chunk-rows", type=int, default=DEFAULT_MAX_CHUNK_ROWS)
    parser.add_argument("--max-chunk-bytes", type=int, default=DEFAULT_MAX_CHUNK_BYTES)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-mode", choices=("legacy", "bounded"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_mode:
        result = _worker(
            mode=args.worker_mode, members=args.members,
            dimensions=args.dimensions, queries=args.queries, seed=args.seed,
            max_input_bytes=args.max_input_bytes,
            max_chunk_rows=args.max_chunk_rows,
            max_chunk_bytes=args.max_chunk_bytes)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"),
                         allow_nan=False))
        return 0
    if args.report is None:
        parser.error("--report is required")
    result = run_benchmark(
        report=args.report, members=args.members, dimensions=args.dimensions,
        queries=args.queries, seed=args.seed,
        max_input_bytes=args.max_input_bytes,
        max_chunk_rows=args.max_chunk_rows,
        max_chunk_bytes=args.max_chunk_bytes)
    print(json.dumps({
        "report_path": result["report_path"],
        "medians_seconds": result["medians_seconds"],
        "median_process_peak_rss_bytes": result["median_process_peak_rss_bytes"],
        "equivalence": result["equivalence"],
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
