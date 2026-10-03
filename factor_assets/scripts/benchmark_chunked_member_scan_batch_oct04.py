"""Bounded fresh-process benchmark for repeated and batched exact member scans.

Synthetic fingerprints are regenerated from one seed in each worker. Results
include fixed-rowwise winner parity, preparation call counts, process peak RSS,
and source fingerprints. This does not benchmark or certify real factor data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np

FA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FA_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_assets.clustering import incremental
from factor_assets.clustering.incremental_recall import (
    scan_exact_member_winner, scan_exact_member_winners,
)
from factor_assets.scripts import benchmark_chunked_member_scan_oct04 as support

DEFAULT_MEMBERS = 17_000
DEFAULT_DIMENSIONS = 256
DEFAULT_QUERIES = 64
DEFAULT_SEED = 20261004
DEFAULT_MAX_CHUNK_ROWS = 256
DEFAULT_MAX_CHUNK_BYTES = 4 * 1024**2
MAX_MEMBERS = 17_000
MAX_DIMENSIONS = 256
MAX_QUERIES = 256
MAX_INPUT_BYTES = 256 * 1024**2
MAX_REPORT_BYTES = 1024**2
MAX_REPEATS = 5
MAX_WORKER_SECONDS = 600
MAX_RUN_SECONDS = 3600
SOURCE_PATHS = (
    "factor_assets/scripts/benchmark_chunked_member_scan_batch_oct04.py",
    "factor_assets/scripts/benchmark_chunked_member_scan_oct04.py",
    "factor_assets/clustering/incremental.py",
    "factor_assets/clustering/incremental_batch_recall.py",
    "factor_assets/clustering/incremental_recall.py",
    "factor_assets/similarity/unit_vectors.py",
    "factor_assets/contracts/fingerprint.py",
)

def _source_hashes():
    return {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }


def _input_estimate(members, dimensions, queries):
    return support._estimate_input_bytes(
        members=members, dimensions=dimensions, queries=queries)


def _validate_envelope(*, members, dimensions, queries, max_input_bytes):
    if (type(members) is not int or not 1 <= members <= MAX_MEMBERS
            or type(dimensions) is not int or not 1 <= dimensions <= MAX_DIMENSIONS
            or type(queries) is not int or not 1 <= queries <= MAX_QUERIES):
        raise ValueError("benchmark dimensions exceed the declared synthetic envelope")
    if (type(max_input_bytes) is not int or not 1 <= max_input_bytes <= MAX_INPUT_BYTES):
        raise ValueError("max_input_bytes exceeds the 256 MiB cap")
    estimated = _input_estimate(members, dimensions, queries)
    if estimated > max_input_bytes:
        raise ValueError("estimated synthetic fingerprints exceed max_input_bytes")
    return estimated


def _oversized_member_estimate(members, dimensions):
    return support._legacy_cache_estimate_bytes(members, dimensions)


def _runtime_context():
    return support._runtime_context()


def _worker(*, mode, members, dimensions, queries, seed, max_input_bytes,
            max_chunk_rows, max_chunk_bytes):
    if mode not in {"single", "batch"}:
        raise ValueError("worker mode must be single or batch")
    input_bytes = _validate_envelope(
        members=members, dimensions=dimensions, queries=queries,
        max_input_bytes=max_input_bytes)
    support._validate_chunk_bounds(max_chunk_rows, max_chunk_bytes)
    hashes_before = _source_hashes()
    runtime_before = _runtime_context()
    member_ids, query_objects, fingerprints = support._make_synthetic_inputs(
        members=members, dimensions=dimensions, queries=queries, seed=seed)
    fingerprint = support._fingerprint_map_hash(member_ids, query_objects, fingerprints)
    unit_queries = np.ascontiguousarray(np.vstack([
        incremental._normalize_rows(
            np.asarray(query.embedding, dtype=np.float64).reshape(1, -1))[0]
        for query in query_objects
    ]))

    counts = {"member_embedding_conversion_calls": 0,
              "member_normalize_rows_calls": 0,
              "member_normalized_rows": 0}

    def counted_embedding(value):
        counts["member_embedding_conversion_calls"] += 1
        return incremental._to_embedding(value)

    def counted_normalize(rows):
        counts["member_normalize_rows_calls"] += 1
        counts["member_normalized_rows"] += int(rows.shape[0])
        return incremental._normalize_rows(rows)

    started = time.perf_counter()
    if mode == "single":
        winners = [
            scan_exact_member_winner(
                fingerprints, member_ids, query,
                to_embedding=counted_embedding,
                normalize_rows=counted_normalize,
                max_chunk_rows=max_chunk_rows, max_chunk_bytes=max_chunk_bytes)
            for query in unit_queries
        ]
    else:
        winners = scan_exact_member_winners(
            fingerprints, member_ids, unit_queries,
            to_embedding=counted_embedding,
            normalize_rows=counted_normalize,
            max_chunk_rows=max_chunk_rows, max_chunk_bytes=max_chunk_bytes,
            max_batch_queries=MAX_QUERIES)
    chunk_capacity = min(
        max_chunk_rows,
        max(1, max_chunk_bytes // max(1, dimensions * 8 * 3)))
    expected_chunks = math.ceil(members / chunk_capacity)
    expected_calls = expected_chunks * (queries if mode == "single" else 1)
    expected_rows = members * (queries if mode == "single" else 1)
    if (counts["member_embedding_conversion_calls"] != expected_rows
            or counts["member_normalize_rows_calls"] != expected_calls
            or counts["member_normalized_rows"] != expected_rows):
        raise RuntimeError("member preparation did not process the expected query/member rows")
    elapsed = time.perf_counter() - started
    hashes_after = _source_hashes()
    runtime_after = _runtime_context()
    if hashes_before != hashes_after:
        raise RuntimeError("declared benchmark sources changed during worker scan")
    if runtime_before != runtime_after:
        raise RuntimeError("worker runtime changed during scan")
    return {
        "mode": mode,
        "elapsed_seconds": elapsed,
        "process_peak_rss_bytes": support._peak_rss_bytes(),
        "input_estimate_bytes": input_bytes,
        "input_fingerprint": fingerprint,
        "winners": winners,
        "preparation_counts": counts,
        "source_hashes": hashes_after,
        "runtime": runtime_after,
    }


def _assert_parity(expected, actual):
    if len(expected) != len(actual):
        raise RuntimeError("single and batch query counts differ")
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if left is None or right is None:
            if left is not right:
                raise RuntimeError(f"winner presence differs at query {index}")
            continue
        if left[0] != right[0]:
            raise RuntimeError(f"winner identity differs at query {index}")
        left_score, right_score = float(left[1]), float(right[1])
        if (not math.isfinite(left_score) or not math.isfinite(right_score) or left_score != right_score):
            raise RuntimeError(f"fixed-rowwise score differs at query {index}")


def _validate_worker_receipt(row, *, mode, members, queries, input_bytes,
                             max_input_bytes, dimensions=DEFAULT_DIMENSIONS,
                             max_chunk_rows=DEFAULT_MAX_CHUNK_ROWS,
                             max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES):
    if not isinstance(row, dict) or row.get("mode") != mode:
        raise RuntimeError("worker returned an invalid mode receipt")
    elapsed = row.get("elapsed_seconds")
    peak = row.get("process_peak_rss_bytes")
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("worker elapsed time is not positive and finite")
    if type(peak) is not int or peak < 0:
        raise RuntimeError("worker process peak RSS is invalid")
    if row.get("input_estimate_bytes") != input_bytes or input_bytes > max_input_bytes:
        raise RuntimeError("worker input estimate differs from the admitted envelope")
    fingerprint = row.get("input_fingerprint")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)):
        raise RuntimeError("worker input fingerprint is malformed")
    winners = row.get("winners")
    if not isinstance(winners, list) or len(winners) != queries:
        raise RuntimeError("worker winner coverage differs from query count")
    valid_ids = {f"M{i:05d}" for i in range(members)}
    for index, winner in enumerate(winners):
        if (not isinstance(winner, (list, tuple)) or len(winner) != 2
                or not isinstance(winner[0], str)
                or winner[0] not in valid_ids
                or type(winner[1]) not in (int, float)
                or not math.isfinite(winner[1])):
            raise RuntimeError(f"worker winner receipt is invalid at query {index}")
    counts = row.get("preparation_counts")
    if (not isinstance(counts, dict)
            or any(type(counts.get(key)) is not int or counts[key] < 0 for key in (
                "member_embedding_conversion_calls", "member_normalize_rows_calls",
                "member_normalized_rows"))):
        raise RuntimeError("worker preparation-call receipt is invalid")
    chunk_capacity = min(max_chunk_rows, max(1, max_chunk_bytes // max(1, dimensions * 8 * 3)))
    expected_chunks = math.ceil(members / chunk_capacity)
    multiplier = queries if mode == "single" else 1
    expected_rows = members * multiplier
    if (counts["member_embedding_conversion_calls"] != expected_rows
            or counts["member_normalized_rows"] != expected_rows
            or counts["member_normalize_rows_calls"] != expected_chunks * multiplier):
        raise RuntimeError("worker preparation-call counts differ from scan contract")
    if not isinstance(row.get("source_hashes"), dict) or not isinstance(row.get("runtime"), dict):
        raise RuntimeError("worker omitted source/runtime identity")


def _run_worker_process(script, *, mode, members, dimensions, queries, seed,
                        max_input_bytes, max_chunk_rows, max_chunk_bytes,
                        timeout=MAX_WORKER_SECONDS):
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
    completed = support._run_bounded_subprocess(
        command, cwd=PROJECT_ROOT, environment=environment, timeout=timeout,
        stdout_limit=64 * 1024, stderr_limit=64 * 1024)
    if completed.returncode:
        raise RuntimeError(
            f"{mode} worker exited {completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace')[:4000]}")
    try:
        return json.loads(completed.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("benchmark worker returned malformed JSON") from exc


def _write_report_exclusive(destination, payload):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {destination}")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) >= MAX_REPORT_BYTES:
        raise ValueError("benchmark report exceeds the 1 MiB cap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(encoded.decode("utf-8"))


def run_benchmark(*, report, members=DEFAULT_MEMBERS,
                  dimensions=DEFAULT_DIMENSIONS, queries=DEFAULT_QUERIES,
                  seed=DEFAULT_SEED, repeats=3,
                  max_input_bytes=MAX_INPUT_BYTES,
                  max_chunk_rows=DEFAULT_MAX_CHUNK_ROWS,
                  max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES,
                  resource_check=support._check_resource_headroom,
                  worker_runner=_run_worker_process):
    input_bytes = _validate_envelope(
        members=members, dimensions=dimensions, queries=queries,
        max_input_bytes=max_input_bytes)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative built-in integer")
    if type(repeats) is not int or not 3 <= repeats <= MAX_REPEATS:
        raise ValueError("repeats must be an integer in 3..5 to report medians")
    support._validate_chunk_bounds(max_chunk_rows, max_chunk_bytes)
    cache_estimate = _oversized_member_estimate(members, dimensions)
    if cache_estimate <= incremental._EXACT_MEMBER_CACHE_BYTES:
        raise ValueError("benchmark member set must exceed the request-local cache budget")
    destination = Path(report)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {destination}")
    source_before = _source_hashes()
    load_before = os.getloadavg()
    runtime_before = _runtime_context()
    ordering = ["single"] * repeats + ["batch"] * repeats
    random.Random(seed ^ 0xB47C_2026).shuffle(ordering)
    started = time.time()
    samples = []
    for index, mode in enumerate(ordering):
        elapsed_total = time.time() - started
        load_average = os.getloadavg()[0]
        if load_average > (os.cpu_count() or 1):
            raise RuntimeError("host load exceeds CPU count; defer controlled timing")
        remaining = MAX_RUN_SECONDS - elapsed_total
        if remaining <= 0:
            raise TimeoutError("benchmark exceeded the one-hour total run cap")
        resource_check()
        row = worker_runner(
            Path(__file__).resolve(), mode=mode, members=members,
            dimensions=dimensions, queries=queries, seed=seed,
            max_input_bytes=max_input_bytes, max_chunk_rows=max_chunk_rows,
            max_chunk_bytes=max_chunk_bytes,
            timeout=min(MAX_WORKER_SECONDS, remaining))
        _validate_worker_receipt(
            row, mode=mode, members=members, queries=queries,
            input_bytes=input_bytes, max_input_bytes=max_input_bytes,
            dimensions=dimensions, max_chunk_rows=max_chunk_rows,
            max_chunk_bytes=max_chunk_bytes)
        row["measurement_index"] = index
        samples.append(row)
    finished = time.time()
    source_after = _source_hashes()
    runtime_after = _runtime_context()
    if source_before != source_after:
        raise RuntimeError("declared sources changed during benchmark")
    if runtime_before != runtime_after:
        raise RuntimeError("parent runtime changed during benchmark")
    expected_input = samples[0]["input_fingerprint"]
    for row in samples:
        if row["input_fingerprint"] != expected_input or row["source_hashes"] != source_before:
            raise RuntimeError("seeded input or source hashes differ between workers")
        if row["runtime"] != samples[0]["runtime"]:
            raise RuntimeError("worker runtime contexts differ")
    single = next(row for row in samples if row["mode"] == "single")
    batch = next(row for row in samples if row["mode"] == "batch")
    _assert_parity(single["winners"], batch["winners"])
    for row in samples:
        _assert_parity(single["winners"], row["winners"])
    medians = {
        mode: statistics.median(row["elapsed_seconds"] for row in samples
                                if row["mode"] == mode)
        for mode in ("single", "batch")
    }
    report_payload = {
        "schema": "factor_assets.chunked_member_scan_batch_ab.v1",
        "status": "complete",
        "scope": {
            "members": members, "dimensions": dimensions, "queries": queries,
            "seed": seed, "repeats_per_arm": repeats,
            "input_estimate_bytes": input_bytes, "input_cap_bytes": max_input_bytes,
            "oversized_member_cache_estimate_bytes": cache_estimate,
            "member_cache_budget_bytes": incremental._EXACT_MEMBER_CACHE_BYTES,
            "fresh_process_per_measurement": True,
            "timing": "low-level exact scans only; excludes public request/query-policy overhead",
        },
        "randomized_order": ordering,
        "samples": samples,
        "median_elapsed_seconds": medians,
        "batch_over_single_elapsed_ratio": medians["batch"] / medians["single"],
        "median_process_peak_rss_bytes": {
            mode: int(statistics.median(row["process_peak_rss_bytes"] for row in samples
                                        if row["mode"] == mode))
            for mode in ("single", "batch")},
        "median_preparation_counts": {
            mode: {key: int(statistics.median(row["preparation_counts"][key]
                   for row in samples if row["mode"] == mode))
                   for key in samples[0]["preparation_counts"]}
            for mode in ("single", "batch")},
        "equivalence": {
            "fixed_rowwise_winner_identity_matches": True,
            "fixed_rowwise_score_exact": True,
            "score_epsilon": 0.0,
            "input_fingerprint": expected_input,
        },
        "host_load_before": load_before,
        "host_load_after": os.getloadavg(),
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "runtime_parent": runtime_before,
        "unix_started": started, "unix_finished": finished,
        "limitations": [
            "Synthetic embeddings do not represent production factor-value scoring.",
            "Per-process peak RSS includes imports and synthetic input construction.",
            "Preparation counts describe callback invocations, not allocator bytes.",
            "The member cache estimate predicts an oversized set; benchmark scans bypass cache.",
            "Shared-host load varies; this does not certify all sizes or backends.",
            "Low-level scanner timing excludes public request/policy/query-preparation overhead.",
        ],
    }
    _write_report_exclusive(destination, report_payload)
    return report_payload


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--members", type=int, default=DEFAULT_MEMBERS)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-input-mib", type=int, default=256)
    parser.add_argument("--max-chunk-rows", type=int, default=DEFAULT_MAX_CHUNK_ROWS)
    parser.add_argument("--max-chunk-mib", type=int, default=4)
    parser.add_argument("--max-input-bytes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-chunk-bytes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=("single", "batch"))
    args = parser.parse_args(argv)
    return parser, args


def main(argv=None):
    parser, args = _parse_args(argv)
    try:
        max_input_bytes = (args.max_input_bytes if args.max_input_bytes is not None
                           else args.max_input_mib * 1024**2)
        max_chunk_bytes = (args.max_chunk_bytes if args.max_chunk_bytes is not None
                           else args.max_chunk_mib * 1024**2)
        if args.worker_mode:
            row = _worker(
                mode=args.worker_mode, members=args.members,
                dimensions=args.dimensions, queries=args.queries, seed=args.seed,
                max_input_bytes=max_input_bytes,
                max_chunk_rows=args.max_chunk_rows, max_chunk_bytes=max_chunk_bytes)
            print(json.dumps(row, allow_nan=False))
            return 0
        if args.report is None:
            parser.error("--report is required for a benchmark run")
        report = run_benchmark(
            report=args.report, members=args.members, dimensions=args.dimensions,
            queries=args.queries, seed=args.seed, repeats=args.repeats,
            max_input_bytes=max_input_bytes, max_chunk_rows=args.max_chunk_rows,
            max_chunk_bytes=max_chunk_bytes)
        print(json.dumps({
            "status": report["status"], "schema": report["schema"],
            "median_elapsed_seconds": report["median_elapsed_seconds"],
            "median_process_peak_rss_bytes": report["median_process_peak_rss_bytes"],
            "parity": report["equivalence"],
        }))
        return 0
    except (ValueError, RuntimeError, MemoryError, OSError, TimeoutError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
