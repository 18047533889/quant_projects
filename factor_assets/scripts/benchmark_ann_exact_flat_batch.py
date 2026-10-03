#!/usr/bin/env python3
"""Bounded CPU A/B: one FAISS batch search versus scalar search calls."""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from factor_assets.similarity.ann import FaissANNIndex


def _unit_rows(values: np.ndarray) -> np.ndarray:
    wide = np.asarray(values, dtype=np.float64)
    scaled = wide / np.max(np.abs(wide), axis=1, keepdims=True)
    unit64 = scaled / np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    unit32 = np.ascontiguousarray(unit64, dtype=np.float32)
    norms = np.sqrt(np.sum(unit32 * unit32, axis=1, keepdims=True, dtype=np.float32))
    return np.ascontiguousarray(unit32 / norms, dtype=np.float32)


def _assert_oracle(index, vectors, queries, k):
    expected = _unit_rows(queries) @ _unit_rows(vectors).T
    for q, rows in enumerate(index.search_batch(queries, k=k)):
        scores = {row.factor_id: row.similarity_score for row in rows}
        ranked = np.sort(expected[q])[::-1]
        cutoff = float(ranked[k - 1])
        valid = {
            f"F{i:05d}" for i, score in enumerate(expected[q])
            if float(score) >= cutoff - 3e-6
        }
        if len(scores) != k or not set(scores).issubset(valid):
            raise AssertionError("batch top-k differs from independent cosine oracle")
        for i, score in enumerate(expected[q]):
            factor_id = f"F{i:05d}"
            if factor_id in scores and abs(scores[factor_id] - float(score)) > 3e-6:
                raise AssertionError("batch score differs from independent cosine oracle")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=1024)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if not (1000 <= args.factors <= 10000 and 32 <= args.queries <= 128
            and 4 <= args.dimensions <= 128 and 1 <= args.k <= args.factors
            and 3 <= args.repeats <= 31):
        parser.error("sizes exceed the bounded CPU benchmark limits")

    rng = np.random.default_rng(20261003)
    vectors = rng.normal(size=(args.factors, args.dimensions)).astype(np.float64)
    queries = rng.normal(size=(args.queries, args.dimensions)).astype(np.float64)
    ids = [f"F{i:05d}" for i in range(args.factors)]
    index = FaissANNIndex(args.dimensions)
    index.build(ids, vectors)
    _assert_oracle(index, vectors, queries, args.k)

    batch = lambda: index.search_batch(queries, k=args.k)
    scalar = lambda: [index.search(query, k=args.k) for query in queries]
    batch_rows, scalar_rows = batch(), scalar()
    for left, right in zip(batch_rows, scalar_rows):
        lm = {item.factor_id: item.similarity_score for item in left}
        rm = {item.factor_id: item.similarity_score for item in right}
        if lm.keys() != rm.keys() or any(abs(lm[key] - rm[key]) > 3e-6 for key in lm):
            raise AssertionError("batch and scalar-loop results differ")

    batch_seconds, scalar_seconds = [], []
    for repeat in range(args.repeats):
        paths = (batch, scalar) if repeat % 2 == 0 else (scalar, batch)
        for path in paths:
            start = time.perf_counter()
            path()
            elapsed = time.perf_counter() - start
            (batch_seconds if path is batch else scalar_seconds).append(elapsed)
    batch_median = statistics.median(batch_seconds)
    scalar_median = statistics.median(scalar_seconds)
    print(
        f"factors={args.factors} dimensions={args.dimensions} queries={args.queries} "
        f"k={args.k} repeats={args.repeats} "
        f"batch_ms={batch_median * 1e3:.3f} "
        f"scalar_loop_ms={scalar_median * 1e3:.3f} "
        f"speedup={scalar_median / batch_median:.2f}x oracle=pass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
