import datetime
import json
import numpy as np
import pytest
from quant_evaluator.contracts._ndarray_codec import encode_ndarray, decode_ndarray
from quant_evaluator.contracts.factor_batch import AxisRef


def test_object_coordinate_values_roundtrip_without_raw_pointers():
    values = np.array(['600000', datetime.datetime(2026,1,1,tzinfo=datetime.timezone.utc), 7], dtype=object)
    payload = encode_ndarray(values)
    assert 'data_b64' not in payload
    restored = decode_ndarray(json.loads(json.dumps(payload)))
    assert restored.dtype == values.dtype and restored.shape == values.shape
    np.testing.assert_array_equal(restored, values)


def test_scalar_array_shape_is_not_promoted_to_length_one():
    for values in (np.array(3.5), np.array('ticker', dtype=object)):
        restored = decode_ndarray(encode_ndarray(values))
        assert restored.shape == ()
        assert restored.item() == values.item()
    from quant_evaluator.contracts._hashutil import stable_content_hex
    assert stable_content_hex(tag='array', fields={'v':np.array(3.5)}) != stable_content_hex(
        tag='array', fields={'v':np.array([3.5])})


def test_object_axis_public_array_cannot_mutate_owned_coordinates():
    source = np.array(['a', 'b'], dtype=object)
    axis = AxisRef('security', 'object', 2, source)
    exposed = axis.values
    exposed.setflags(write=True)
    exposed[0] = 'corrupted'
    source[1] = 'also-corrupted'
    assert axis.values.tolist() == ['a', 'b']


def test_mutable_coordinate_and_legacy_object_pointer_payload_fail_closed():
    class MutableCoordinate:
        pass
    with pytest.raises(ValueError, match='immutable scalar'):
        AxisRef('security', 'object', 1, np.array([MutableCoordinate()], dtype=object))
    with pytest.raises(ValueError, match='never raw pointer'):
        decode_ndarray({'dtype':'|O', 'shape':[1], 'data_b64':'AAAAAAAAAAA='})
