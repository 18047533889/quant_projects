"""Contract tests for the incremental batch benchmark's independent result oracle."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from factor_assets.scripts import benchmark_incremental_batch_recall_oct03 as benchmark


def _case():
    query = SimpleNamespace(factor_id="Q0", embedding=(1.0, 0.0))
    mapping = {
        "Q0": query,
        "M0": SimpleNamespace(embedding=(1.0, 0.0)),
        "M1": SimpleNamespace(embedding=(0.8, 0.6)),
        "M2": SimpleNamespace(embedding=(0.6, 0.8)),
    }
    clusters = {
        "C0": SimpleNamespace(member_factor_ids=("M0", "M1")),
        "C1": SimpleNamespace(member_factor_ids=("M2",)),
    }
    assignments = [SimpleNamespace(factor_id="Q0")]
    candidates = [
        SimpleNamespace(cluster_id="C0", factor_id="M0", similarity=1.0),
        SimpleNamespace(cluster_id="C1", factor_id="M2", similarity=0.6),
    ]
    result = SimpleNamespace(assignments=assignments, candidates=candidates)
    return result, [query], mapping, clusters


def test_independent_oracle_accepts_exact_query_cluster_coverage():
    result, queries, mapping, clusters = _case()
    assert benchmark._oracle(result, queries, mapping, clusters) is None


@pytest.mark.parametrize("corruption", ["missing", "extra"])
def test_independent_oracle_rejects_inexact_candidate_count(corruption):
    result, queries, mapping, clusters = _case()
    if corruption == "missing":
        result.candidates.pop()
    else:
        result.candidates.append(
            SimpleNamespace(cluster_id="C0", factor_id="M0", similarity=1.0)
        )
    with pytest.raises(AssertionError, match="coverage"):
        benchmark._oracle(result, queries, mapping, clusters)


def test_independent_oracle_rejects_cluster_order_corruption():
    result, queries, mapping, clusters = _case()
    result.candidates.reverse()
    with pytest.raises(AssertionError, match="order"):
        benchmark._oracle(result, queries, mapping, clusters)


def test_independent_oracle_rejects_wrong_longdouble_candidate_score():
    result, queries, mapping, clusters = _case()
    result.candidates[1] = SimpleNamespace(
        cluster_id="C1", factor_id="M2", similarity=0.5,
    )
    with pytest.raises(AssertionError, match="longdouble cosine"):
        benchmark._oracle(result, queries, mapping, clusters)
