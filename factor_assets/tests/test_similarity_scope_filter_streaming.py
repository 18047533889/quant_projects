from dataclasses import replace

import pytest

from factor_assets.similarity.exact import (
    CorrelationSimilarity,
    QEPairwiseSimilarity,
    SimilarityMeasurementStatus,
    SimilarityMethod,
    SimilarityResult,
)


def _result(a, b, score, *, method=SimilarityMethod.PEARSON,
            universe=None, start=None, end=None, status=None):
    kwargs = {}
    if status is not None:
        kwargs["measurement_status"] = status
    return SimilarityResult(
        factor_id_a=a, factor_id_b=b, similarity_score=score,
        method=method, timestamp="2026-10-01", sample_size=100,
        universe_ref=universe, period_start=start, period_end=end, **kwargs,
    )


def _populate(cls):
    similarity = cls()
    rows = [
        ("Q", "A", .91, SimilarityMethod.PEARSON, "US", "s1", "e1"),
        ("A", "Q", -.95, SimilarityMethod.PEARSON, "US", "s1", "e1"),
        ("Q", "A", .84, SimilarityMethod.PEARSON, "EU", "s1", "e1"),
        ("Q", "A", -.91, SimilarityMethod.PEARSON, "US", "s2", "e2"),
        ("Q", "B", -.91, SimilarityMethod.PEARSON, "US", "s1", "e1"),
        ("Q", "C", .9, SimilarityMethod.SPEARMAN, "US", "s1", "e1"),
        ("Q", "D", .8, SimilarityMethod.PEARSON, "EU", "s2", "e2"),
        ("Q", "E", .69, SimilarityMethod.PEARSON, "US", "s1", "e1"),
        ("X", "Q", .88, SimilarityMethod.PEARSON, "US", "s1", "e1"),
        ("Q", "F", None, SimilarityMethod.PEARSON, "US", "s1", "e1"),
    ]
    for a, b, score, method, universe, start, end in rows:
        status = (SimilarityMeasurementStatus.UNKNOWN
                  if score is None else SimilarityMeasurementStatus.COMPUTED_VALUE)
        similarity.add_result(_result(
            a, b, score, method=method, universe=universe,
            start=start, end=end, status=status,
        ))
    return similarity


def _scan_reference(similarity, factor_id, *, threshold=.7, max_results=10,
                    method=SimilarityMethod.PEARSON, universe_ref=None,
                    period_start=None, period_end=None):
    best = {}
    for key, cached in similarity._cache.items():
        fid_a, fid_b, method_val, universe, start, end = key
        if fid_a != factor_id or method_val != method.value:
            continue
        if universe_ref is not None and universe != universe_ref:
            continue
        if period_start is not None and start != period_start:
            continue
        if period_end is not None and end != period_end:
            continue
        result = cached
        if not result.is_high_similarity(threshold):
            continue
        if (result.factor_id_a, result.factor_id_b) != (fid_a, fid_b):
            result = replace(result, factor_id_a=fid_a, factor_id_b=fid_b)
        scope = tuple((value is not None, value or "")
                      for value in (result.universe_ref, result.period_start,
                                    result.period_end))
        rank = (-abs(result.similarity_score), scope)
        if fid_b not in best or rank < best[fid_b][0]:
            best[fid_b] = rank, result
    ordered = sorted(
        best.items(),
        key=lambda item: (item[1][0][0], item[0], item[1][0][1]),
    )
    return [value[1] for _, value in ordered[:max_results]]


@pytest.mark.parametrize("similarity_class", [QEPairwiseSimilarity, CorrelationSimilarity])
def test_streaming_lookup_matches_full_scan_across_filters_and_replacements(similarity_class):
    similarity = _populate(similarity_class)
    queries = [
        {},
        {"threshold": .9},
        {"method": SimilarityMethod.SPEARMAN},
        {"universe_ref": "US"},
        {"period_start": "s1"},
        {"universe_ref": "US", "period_start": "s1", "period_end": "e1"},
        {"method": SimilarityMethod.PEARSON, "universe_ref": "EU"},
    ]
    for kwargs in queries:
        for limit in (0, 1, 3, 20):
            options = {"max_results": limit, **kwargs}
            assert similarity.find_similar("Q", **options) == _scan_reference(
                similarity, "Q", **options
            )

    # Updating an existing scoped key replaces evidence without duplicating its index entry.
    similarity.add_result(_result(
        "Q", "A", -.99, universe="US", start="s1", end="e1",
    ))
    similarity.add_result(_result("Q", "Z", .93))
    for kwargs in queries:
        options = {"max_results": 2, **kwargs}
        assert similarity.find_similar("Q", **options) == _scan_reference(
            similarity, "Q", **options
        )
    similarity.clear()
    assert similarity.find_similar("Q") == []


@pytest.mark.parametrize("similarity_class", [QEPairwiseSimilarity, CorrelationSimilarity])
def test_filtered_lookup_reads_cache_only_after_direction_and_scope_filters(similarity_class):
    similarity = _populate(similarity_class)

    class CountingDict(dict):
        def __init__(self, source):
            super().__init__(source)
            self.reads = []

        def __getitem__(self, key):
            self.reads.append(key)
            return super().__getitem__(key)

    counted = CountingDict(similarity._cache)
    similarity._cache = counted
    result = similarity.find_similar(
        "Q", method=SimilarityMethod.PEARSON,
        universe_ref="US", period_start="s1", period_end="e1",
    )
    expected_keys = {
        key for key in counted
        if key[0] == "Q" and key[2] == SimilarityMethod.PEARSON.value
        and key[3] == "US" and key[4] == "s1" and key[5] == "e1"
    }
    assert set(counted.reads) == expected_keys
    assert len(counted.reads) == len(expected_keys)
    assert {item.factor_id_b for item in result} == {"A", "B", "X"}


@pytest.mark.parametrize("similarity_class", [QEPairwiseSimilarity, CorrelationSimilarity])
def test_zero_limit_does_not_read_cache(similarity_class):
    similarity = _populate(similarity_class)

    class ForbiddenDict(dict):
        def __getitem__(self, key):
            pytest.fail(f"zero-limit query read cached key {key!r}")

    similarity._cache = ForbiddenDict(similarity._cache)
    assert similarity.find_similar("Q", max_results=0) == []
