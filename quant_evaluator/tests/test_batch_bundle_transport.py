"""Columnar batch transport must not silently omit metric dimensions."""
import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle


def test_to_dict_preserves_scalar_series_vector_and_counts():
    result = BatchEvaluationBundle(("a", "b"), "ret")
    result.scalar_metrics["rank_ic"] = np.array([0.1, -0.2])
    result.series_metrics["rank_ic_series"] = np.array([[0.1, 0.2], [0.3, 0.4]])
    result.vector_metrics["quantile_returns_full"] = np.array([[1.0, 2.0]])
    result.observation_counts["rank_ic"] = np.array([20, 18], dtype=np.int64)
    result.metadata = {"backend_used": "cuda"}

    payload = result.to_dict()
    assert payload["factor_ids"] == ["a", "b"]
    assert payload["scalar_metrics"]["rank_ic"] == [0.1, -0.2]
    assert payload["series_metrics"]["rank_ic_series"] == [[0.1, 0.2], [0.3, 0.4]]
    assert payload["vector_metrics"]["quantile_returns_full"] == [[1.0, 2.0]]
    assert payload["observation_counts"]["rank_ic"] == [20, 18]
    assert payload["metadata"] == {"backend_used": "cuda"}
