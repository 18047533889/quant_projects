import numpy as np
import pytest

from factor_assets.similarity import SimilarityMethod, SimilarityResult


@pytest.mark.parametrize("bad", [False, np.bool_(True), np.bool_(False)])
def test_sample_size_rejects_numpy_and_python_booleans(bad):
    with pytest.raises(TypeError, match="sample_size must be a non-boolean integer"):
        SimilarityResult(
            factor_id_a="A", factor_id_b="B", similarity_score=0.0,
            method=SimilarityMethod.PEARSON, timestamp="2026-10-03", sample_size=bad,
        )


def test_sample_size_accepts_zero():
    result = SimilarityResult(
        factor_id_a="A", factor_id_b="B", similarity_score=0.0,
        method=SimilarityMethod.PEARSON, timestamp="2026-10-03", sample_size=0,
    )
    assert result.sample_size == 0
