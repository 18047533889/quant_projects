"""V3 QE-29/30/31: input identity and causal clocks, independent fixtures."""
from dataclasses import replace
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import Evaluator, evaluate
from quant_evaluator.contracts.errors import InvalidContractError


def labels(**kwargs):
    fields = dict(target_id='forward', values=np.ones((2, 2)), horizon=1,
                  decision_time=(1, 2), label_start_time=(2, 3), label_end_time=(3, 4))
    return LabelBundle(**(fields | kwargs))


def factors():
    return FactorBatch(('f',), AxisRef('time', 'int64', 2, np.array([1, 2])),
                       AxisRef('asset', 'str', 2, np.array(['A', 'B'])), np.ones((2, 2, 1)))


def test_label_snapshot_isolated_from_external_aliases():
    values = np.ones((2, 2))
    meta = {'nested': {'revision': 1}}
    bundle = labels(values=values, metadata=meta)
    values[:] = 999
    meta['nested']['revision'] = 9
    np.testing.assert_array_equal(bundle.values, np.ones((2, 2)))
    assert bundle.metadata['nested']['revision'] == 1
    with pytest.raises(ValueError):
        bundle.values[0, 0] = 42
    with pytest.raises(TypeError):
        bundle.metadata['nested']['revision'] = 42


def test_signal_must_be_available_before_actual_decision():
    with pytest.raises(ValueError, match='signal_available_time'):
        labels(signal_available_time=(1.5, 2.5))
    labels(signal_available_time=(.5, 1.5))


def test_observation_clock_is_distinct_from_later_decision():
    bundle = labels(observation_time=(1, 2), signal_available_time=(1.2, 2.2),
                    decision_time=(1.5, 2.5), execution_time=(2, 3),
                    asset_axis=factors().asset_axis)
    Evaluator()._validate_inputs(factors(), bundle)


@pytest.mark.parametrize('backend', [None, 'cuda_strict'])
def test_same_shape_wrong_asset_axis_rejected_before_dispatch(backend):
    bundle = labels(asset_axis=AxisRef('asset', 'str', 2, np.array(['B', 'A'])))
    with pytest.raises(InvalidContractError, match='asset|Asset'):
        evaluate(factors(), bundle, metrics=['coverage'], backend=backend)


def test_same_length_wrong_time_axis_rejected_in_low_level():
    bundle = labels(decision_time=(10, 20), label_start_time=(11, 21), label_end_time=(12, 22))
    with pytest.raises(InvalidContractError, match='time|Time'):
        Evaluator()._validate_inputs(factors(), bundle)


@pytest.mark.parametrize('validity', [np.ones((2, 2)), np.array([[0, 2], [1, 1]])])
def test_label_validity_requires_boolean_dtype(validity):
    with pytest.raises(ValueError, match='bool'):
        labels(validity=validity)


def test_label_identity_covers_clocks_basis_and_axes():
    original = labels(asset_axis=factors().asset_axis)
    assert original.content_hash != replace(original, price_convention='close_to_close').content_hash
    assert original.content_hash != replace(original, signal_available_time=(.5, 1.5)).content_hash
    assert original.content_hash != replace(original, asset_axis=AxisRef('asset', 'str', 2, np.array(['B', 'A']))).content_hash


def test_slice_preserves_label_semantics():
    original = labels(asset_axis=factors().asset_axis, signal_available_time=(.5, 1.5), price_convention='close_to_close')
    sliced = original.slice(slice(1, 2), slice(0, 1))
    assert sliced.signal_available_time == (1.5,)
    assert sliced.price_convention == 'close_to_close'
    assert sliced.asset_axis.values.tolist() == ['A']
    np.testing.assert_array_equal(sliced.values, [[1.]])
