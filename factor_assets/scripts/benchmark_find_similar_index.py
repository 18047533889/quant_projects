#!/usr/bin/env python3
"""Bounded in-memory A/B benchmark for QEPairwiseSimilarity.find_similar.

Run from the repository root:
    PYTHONPATH=. python3 factor_assets/scripts/benchmark_find_similar_index.py

The default workload is deterministic, creates no files, and does not use QE,
DataAccess, or COS.
"""
from __future__ import annotations

import argparse
import random
import statistics
import timeit
from collections.abc import Sequence

from factor_assets.similarity.exact import (
    QEPairwiseSimilarity,
    SimilarityMethod,
    SimilarityResult,
)

SEED = 20260929
TARGET_FACTOR = "HOT"
TARGET_DEGREE = 24
DEFAULT_SIZES = (1_000, 10_000, 30_000)
MAX_UNRELATED_PAIRS = 30_000
QUERY_NUMBER = 80
REPEATS = 3


def _result(
    factor_id_a: str,
    factor_id_b: str,
    score: float,
    *,
    universe: str | None = None,
    period: str | None = None,
) -> SimilarityResult:
    return SimilarityResult(
        factor_id_a=factor_id_a,
        factor_id_b=factor_id_b,
        similarity_score=score,
        method=SimilarityMethod.PEARSON,
        timestamp="2026-09-29T00:00:00Z",
        sample_size=100,
        universe_ref=universe,
        period_start=period,
        period_end=period,
    )


def _populate(unrelated_pairs: int) -> QEPairwiseSimilarity:
    rng = random.Random(SEED)
    similarity = QEPairwiseSimilarity()
    for i in range(unrelated_pairs):
        similarity.add_result(
            _result(f"COLD{i}", f"OTHER{i}", rng.uniform(-0.2, 0.2))
        )

    # Keep the queried factor's adjacency degree fixed as unrelated cache
    # contents grow. Vary orientation and filter dimensions deterministically.
    for i in range(TARGET_DEGREE):
        other = f"NEAR{i}"
        universe = ("US", "EU")[i % 2]
        period = f"2026-Q{i % 4 + 1}"
        score = rng.choice((-1.0, 1.0)) * rng.uniform(0.71, 0.99)
        if i % 2:
            result = _result(other, TARGET_FACTOR, score, universe=universe, period=period)
        else:
            result = _result(TARGET_FACTOR, other, score, universe=universe, period=period)
        similarity.add_result(result)
    return similarity


def _scan_reference(
    similarity: QEPairwiseSimilarity,
    factor_id: str,
    *,
    threshold: float = 0.7,
    max_results: int = 10,
    method: SimilarityMethod = SimilarityMethod.PEARSON,
    universe_ref: str | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
) -> list[SimilarityResult]:
    """Original full-cache scan, kept as the benchmark's behavior oracle."""
    results = []
    seen_factor_ids = set()
    for key, result in similarity._cache.items():
        fid_a, fid_b, method_val, universe, start, end = key
        if fid_a != factor_id or method_val != method.value:
            continue
        if universe_ref is not None and universe != universe_ref:
            continue
        if period_start is not None and start != period_start:
            continue
        if period_end is not None and end != period_end:
            continue
        if fid_b in seen_factor_ids:
            continue
        if result.is_high_similarity(threshold):
            results.append(result)
            seen_factor_ids.add(fid_b)
    results.sort(key=lambda result: abs(result.similarity_score), reverse=True)
    return results[:max_results]


def _verify_equivalence(similarity: QEPairwiseSimilarity) -> None:
    queries = (
        {},
        {"threshold": 0.9},
        {"max_results": 3},
        {"max_results": 0},
        {"universe_ref": "US"},
        {"period_start": "2026-Q2"},
        {"universe_ref": "EU", "period_start": "2026-Q3", "period_end": "2026-Q3"},
    )
    for kwargs in queries:
        indexed = similarity.find_similar(TARGET_FACTOR, **kwargs)
        scanned = _scan_reference(similarity, TARGET_FACTOR, **kwargs)
        if indexed != scanned:
            raise AssertionError(f"indexed result differs from scan for query {kwargs!r}")


def _parse_sizes(raw: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc
    if not sizes or any(size < 0 or size > MAX_UNRELATED_PAIRS for size in sizes):
        raise argparse.ArgumentTypeError(
            f"each size must be between 0 and {MAX_UNRELATED_PAIRS}"
        )
    return sizes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
        help=f"comma-separated unrelated pair counts, maximum {MAX_UNRELATED_PAIRS}",
    )
    args = parser.parse_args(argv)

    print(
        f"seed={SEED} degree={TARGET_DEGREE} queries_per_sample={QUERY_NUMBER} "
        f"repeats={REPEATS}"
    )
    print("unrelated_pairs,total_cache_keys,scan_us,indexed_us,speedup,equivalent")
    for unrelated_pairs in args.sizes:
        similarity = _populate(unrelated_pairs)
        _verify_equivalence(similarity)

        scan = lambda: _scan_reference(similarity, TARGET_FACTOR)
        indexed = lambda: similarity.find_similar(TARGET_FACTOR)
        # Warm both paths, then interleave A/B/B/A to limit systematic
        # ordering and interpreter/cache warm-up effects.
        scan()
        indexed()
        scan_times = []
        indexed_times = []
        for _ in range(REPEATS):
            scan_times.append(timeit.timeit(scan, number=QUERY_NUMBER))
            indexed_times.append(timeit.timeit(indexed, number=QUERY_NUMBER))
            indexed_times.append(timeit.timeit(indexed, number=QUERY_NUMBER))
            scan_times.append(timeit.timeit(scan, number=QUERY_NUMBER))
        scan_us = statistics.median(scan_times) * 1e6 / QUERY_NUMBER
        indexed_us = statistics.median(indexed_times) * 1e6 / QUERY_NUMBER
        print(
            f"{unrelated_pairs},{len(similarity._cache)},{scan_us:.1f},"
            f"{indexed_us:.1f},{scan_us / indexed_us:.1f}x,yes"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
