import importlib.util
import numpy as np
import pytest


def run(values, layers, half_lives):
    assert importlib.util.find_spec("factor_preprocess.transforms.layered_decay") is not None
    from factor_preprocess.transforms.layered_decay import layered_decay
    return layered_decay(np.asarray(values, float), np.asarray(layers), half_lives,
                         allow_research=True)


def test_layer_history_keeps_original_decay_after_asset_changes_layer():
    x = [[2.], [6.], [10.], [100.]]
    bins = [[0], [1], [0], [1]]
    actual = run(x, bins, (1., np.log(.5)/np.log(.75)))
    np.testing.assert_allclose(actual[:, 0], [np.nan, 2., 4., 102/13], equal_nan=True)


def test_layered_decay_resets_missing_history_and_never_uses_current_value():
    x = np.array([[2.], [np.nan], [10.], [999.]])
    bins = np.array([[0], [-1], [1], [0]])
    actual = run(x, bins, (1., 2.))
    np.testing.assert_allclose(actual[:, 0], [np.nan, 2., np.nan, 10.], equal_nan=True)
    np.testing.assert_array_equal(actual[:3], run(x[:3], bins[:3], (1., 2.)))


def test_layered_decay_is_asset_permutation_invariant_and_handles_extremes():
    x = np.full((40, 3), np.finfo(float).max)
    bins = np.arange(120).reshape(40, 3) % 2
    actual = run(x, bins, (1., 60.))
    assert np.isfinite(actual[1:]).all()
    np.testing.assert_allclose(actual[1:], x[1:], rtol=1e-14)
    np.testing.assert_array_equal(actual[:, ::-1], run(x[:, ::-1], bins[:, ::-1], (1., 60.)))


@pytest.mark.parametrize("half_lives", [(0., 2.), (1., np.nan), (True, 2.), (1., 61.)])
def test_invalid_layer_half_lives_are_not_silently_clamped(half_lives):
    with pytest.raises(ValueError):
        run([[1.], [2.]], [[0], [1]], half_lives)
