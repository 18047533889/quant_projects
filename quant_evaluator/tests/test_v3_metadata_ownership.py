import pickle
import json

import numpy as np
import pytest

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.metric_artifacts import FrozenMapping
from quant_evaluator.contracts._ndarray_codec import encode_value, decode_value
from quant_evaluator.contracts._hashutil import stable_content_hex


def test_metadata_mapping_cannot_replace_or_delete_its_storage():
    metadata = FrozenMapping({'nested': {'array': np.arange(3.)}})
    with pytest.raises(TypeError):
        metadata._data = {}
    with pytest.raises(TypeError):
        del metadata._data
    with pytest.raises(TypeError):
        metadata['nested']._data = {}
    restored = pickle.loads(pickle.dumps(metadata))
    np.testing.assert_array_equal(restored['nested']['array'], np.arange(3.))
    with pytest.raises(ValueError):
        restored['nested']['array'].flags.writeable = True


def test_metadata_numeric_values_detached_from_input_and_immutable():
    values = np.arange(3.)
    metadata = FrozenMapping({'values': values})
    values[:] = 99
    np.testing.assert_array_equal(metadata['values'], np.arange(3.))
    with pytest.raises(ValueError):
        metadata['values'].flags.writeable = True


@pytest.mark.parametrize('values', [np.array(['coordinate'], dtype=object),
                                  np.array([{'mutable': []}], dtype=object)])
def test_metadata_object_array_rejected_instead_of_shallow_frozen(values):
    with pytest.raises(InvalidContractError, match='object arrays'):
        FrozenMapping({'values': values})


def test_structured_object_scalar_rejected():
    value = np.array([([],)], dtype=[('payload', object)])[0]
    with pytest.raises(InvalidContractError, match='scalar schema'):
        FrozenMapping({'value': value})


def test_metadata_mapping_codec_preserves_distinct_integer_and_string_keys():
    value = {1: 'integer', '1': 'string', False: 'boolean'}
    restored = decode_value(json.loads(json.dumps(encode_value(value))))
    assert restored == value
    assert set(type(key) for key in restored) == {int, str, bool}


@pytest.mark.parametrize('tag', ['__ndarray__', '__bytes__', '__mapping__', '__date__'])
def test_user_metadata_codec_tags_are_data_not_interpreted_types(tag):
    value = {tag: 'ordinary metadata', 'nested': {tag: 7}}
    assert decode_value(json.loads(json.dumps(encode_value(value)))) == value


def test_duplicate_decoded_mapping_keys_fail_closed():
    with pytest.raises(ValueError, match='duplicate'):
        decode_value({'__mapping__': [[1, 'first'], [1, 'second']]})


def test_nested_tuple_mapping_keys_roundtrip():
    value = {('factor', (1, '1')): 'window'}
    assert decode_value(json.loads(json.dumps(encode_value(value)))) == value


def test_hash_cannot_confuse_literal_metadata_tag_with_typed_bytes():
    actual = stable_content_hex(tag='test', fields={'value': b'abc'})
    literal = stable_content_hex(tag='test', fields={'value': {'__bytes__': 'YWJj'}})
    assert actual != literal


def test_hash_top_level_non_string_keys_fail_closed():
    with pytest.raises(TypeError, match='keys must be str'):
        stable_content_hex(tag='test', fields={1: 'integer', '1': 'string'})


def test_structured_dtype_hash_cannot_silently_drop_field_schema():
    with pytest.raises(TypeError, match='structured'):
        stable_content_hex(tag='test', fields={'value': np.array([(1,)], dtype=[('field', 'i4')])})
