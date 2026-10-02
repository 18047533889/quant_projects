"""Cosine search must preserve finite-vector direction across magnitude scales."""
import numpy as np
import pytest

from factor_assets.similarity import ann
from factor_assets.similarity.unit_vectors import _unit_rows


@pytest.mark.parametrize("scale", [1e300, 1e-300, np.nextafter(0., 1.), 1.])
def test_scaled_normalization_preserves_direction_and_input(scale):
    source = np.array([[scale, scale], [scale, 0.], [-scale, scale]])
    original = source.copy()
    actual = _unit_rows(source)
    expected = np.array([[1., 1.], [1., 0.], [-1., 1.]])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.)
    np.testing.assert_array_equal(source, original)
    assert np.isfinite(ann._validate_query(source[0], 2, 1)).all()


@pytest.mark.parametrize("backend", ["faiss", "annoy"])
@pytest.mark.parametrize("scale", [1e300, 1e-300, np.nextafter(0., 1.)])
def test_real_backend_search_is_scale_invariant(backend, scale):
    if backend == "faiss":
        if not ann.FAISS_AVAILABLE:
            pytest.skip("faiss unavailable")
        index = ann.FaissANNIndex(2)
    else:
        if not ann.ANNOY_AVAILABLE:
            pytest.skip("annoy unavailable")
        index = ann.AnnoyANNIndex(2)
    values = np.array([[scale, scale], [scale, 0.], [-scale, -scale]])
    index.build(["diagonal", "axis", "opposite"], values)
    results = index.search(values[0], k=3)
    scores = {result.factor_id: result.similarity_score for result in results}
    assert results[0].factor_id == "diagonal"
    np.testing.assert_allclose(
        [scores["diagonal"], scores["axis"], scores["opposite"]],
        [1., 1. / np.sqrt(2.), -1.], rtol=0., atol=1e-6,
    )
    assert np.isfinite(index._embeddings).all()
    assert not np.any(np.all(index._embeddings == 0, axis=1))


@pytest.mark.parametrize("values", [
    np.zeros((1, 2)), np.array([[np.inf, 1.]]),
    np.array([[1j, 1.]]), np.array([["1", "2"]]),
])
def test_unsupported_or_invalid_vectors_fail_closed(values):
    with pytest.raises(ValueError):
        _unit_rows(values)
    with pytest.raises(ValueError):
        ann._validate_build_inputs(["bad"], values, 2)
    with pytest.raises(ValueError):
        ann._validate_query(values[0], 2, 1)


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, 1.01, -1.01])
def test_invalid_backend_scores_are_rejected(score):
    from factor_assets.similarity.unit_vectors import _canonical_cosine
    with pytest.raises(ValueError):
        _canonical_cosine(score)


@pytest.mark.parametrize("sign", [-1., 1.])
def test_float32_endpoint_roundoff_is_clamped(sign):
    from factor_assets.similarity.unit_vectors import _canonical_cosine
    assert _canonical_cosine(sign * (1. + np.finfo(np.float32).eps)) == sign
