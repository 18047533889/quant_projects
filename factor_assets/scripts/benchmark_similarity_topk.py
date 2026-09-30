"""Bounded top-k/full-sort A/B with exact ordered output checks."""
import argparse
import json
import random
import statistics
import time

from factor_assets.similarity.exact import SimilarityMethod, SimilarityResult, _deduplicate_similar_neighbors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbors", type=int, choices=(1000, 10000, 20000), default=10000)
    parser.add_argument("--top-k", type=int, choices=(1, 10, 100), default=10)
    parser.add_argument("--repetitions", type=int, choices=(3, 5, 7), default=5)
    args = parser.parse_args()
    items = []
    for neighbor in range(args.neighbors):
        for scope in (None, "A", "B"):
            fid = f"n{neighbor:06d}"
            result = SimilarityResult(
                factor_id_a="query", factor_id_b=fid,
                similarity_score=((neighbor % 17) - 8) / 8,
                method=SimilarityMethod.PEARSON, timestamp="2026-10-01",
                sample_size=100, universe_ref=scope,
            )
            items.append((fid, result))
    random.Random(17).shuffle(items)
    routes = {
        "full_sort": lambda: _deduplicate_similar_neighbors(items)[:args.top_k],
        "top_k": lambda: _deduplicate_similar_neighbors(items, max_results=args.top_k),
    }
    expected = routes["full_sort"]()
    timings = {name: [] for name in routes}
    for iteration in range(args.repetitions + 1):
        order = ("full_sort", "top_k") if iteration % 2 == 0 else ("top_k", "full_sort")
        for name in order:
            start = time.perf_counter()
            actual = routes[name]()
            elapsed = time.perf_counter() - start
            if actual != expected:
                raise AssertionError("ordered similarity outputs differ")
            if iteration:
                timings[name].append(elapsed)
    print(json.dumps({
        "pass": True, "neighbors": args.neighbors, "scope_measurements": len(items),
        "top_k": args.top_k, "repetitions": args.repetitions, "warmups": 1,
        "median_seconds": {name: statistics.median(values) for name, values in timings.items()},
        "limitations": "Synthetic helper-level dedup/selection; excludes provider, cache filtering and IO.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
