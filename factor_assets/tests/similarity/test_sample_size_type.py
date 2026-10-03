import pytest

from factor_assets.similarity import SimilarityMethod, SimilarityResult


@pytest.mark.parametrize("bad", [True, 1.5, 1.0, "3", None])
def test_similarity_result_rejects_non_integer_sample_size(bad):
    with pytest.raises(TypeError, match="sample_size must be a non-boolean integer"):
        SimilarityResult(
            factor_id_a="F001", factor_id_b="F002", similarity_score=0.85,
            method=SimilarityMethod.PEARSON, timestamp="2024-01-01T00:00:00Z",
            sample_size=bad,
        )


def test_similarity_result_accepts_integral_sample_size_and_normalizes_numpy_integer():
    import numpy as np

    result = SimilarityResult(
        factor_id_a="F001", factor_id_b="F002", similarity_score=0.85,
        method=SimilarityMethod.PEARSON, timestamp="2024-01-01T00:00:00Z",
        sample_size=np.int64(3),
    )
    assert result.sample_size == 3
    assert type(result.sample_size) is int
