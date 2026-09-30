import random

import pytest

from factor_assets.similarity.exact import (
    SimilarityMethod, SimilarityResult, _deduplicate_similar_neighbors,
    _find_similar_in_cache,
)


def candidates():
    result = []
    for neighbor in range(120):
        for scope in (None, "A", "B"):
            score = ((neighbor % 9) - 4) / 4
            item = SimilarityResult(
                factor_id_a="query", factor_id_b=f"n{neighbor:03d}",
                similarity_score=score, method=SimilarityMethod.PEARSON,
                timestamp="2026-10-01", sample_size=100, universe_ref=scope,
            )
            result.append((item.factor_id_b, item))
    random.Random(17).shuffle(result)
    return result


@pytest.mark.parametrize("limit", [0, 1, 2, 10, 119, 120, 200])
def test_topk_matches_full_order_with_scopes_scores_and_ties(limit):
    items = candidates()
    baseline = _deduplicate_similar_neighbors(items)
    assert _deduplicate_similar_neighbors(items, max_results=limit) == baseline[:limit]
    assert _deduplicate_similar_neighbors(list(reversed(items)), max_results=limit) == baseline[:limit]


def test_zero_limit_does_not_visit_cache_index():
    class ForbiddenIndex(dict):
        def get(self, *args):
            pytest.fail("zero top-k must not scan cache keys")

    assert _find_similar_in_cache(
        {}, ForbiddenIndex(), "query", .7, 0, SimilarityMethod.PEARSON,
        None, None, None,
    ) == []


@pytest.mark.parametrize("limit", [0.0, 1.0, 2.5])
def test_noninteger_public_limit_retains_slice_rejection(limit):
    with pytest.raises(TypeError):
        _find_similar_in_cache(
            {}, {}, "query", .7, limit, SimilarityMethod.PEARSON,
            None, None, None,
        )
