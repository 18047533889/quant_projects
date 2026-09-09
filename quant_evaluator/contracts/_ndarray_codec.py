"""Lossless ndarray / value codec for QE artifact serialization.

QE-P0-04: replaces the lossy ``tolist()`` / ``np.asarray`` round-trip with a
canonical codec that preserves dtype, shape, endianness, and exact payload
bytes.

- NaN / Inf policy: payload bytes are carried verbatim (base64), so NaN/Inf
  bit patterns round-trip exactly — no coercion, no loss.
- datetime64 arrays round-trip through their dtype string (``<M8[ns]`` etc.).
- Non-contiguous input is made C-contiguous before encoding so ``tobytes()``
  yields the canonical byte layout; decoding reshapes from the raw buffer.
- ``encode_value`` / ``decode_value`` additionally cover datetime/date/
  timedelta, Decimal, Enum, bytes, and numpy scalars so provenance and
  ``time_index`` round-trip losslessly too.
"""

import base64
import datetime
import decimal
import enum
from collections.abc import Mapping

import numpy as np


_RESERVED_TAGS = frozenset({
    '__ndarray__', '__datetime__', '__date__', '__timedelta__',
    '__decimal__', '__enum__', '__bytes__', '__mapping__', '__tuple__',
})


def encode_ndarray(arr: np.ndarray) -> dict:
    """Encode an ndarray to a JSON-friendly dict preserving dtype+shape+bytes."""
    arr = np.asarray(arr)
    if arr.dtype.fields is not None:
        raise TypeError('structured arrays require an explicit schema codec')
    if arr.dtype.hasobject:
        # Object buffers contain process-local PyObject pointers, never portable
        # values. Encode the actual elements, retaining object dtype and shape.
        return {'__ndarray__': True, 'dtype': arr.dtype.str, 'shape': list(arr.shape),
                'object_values': [encode_value(item) for item in arr.flat]}
    return {
        "__ndarray__": True,
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_ndarray(data: dict) -> np.ndarray:
    """Reconstruct a read-only ndarray from :func:`encode_ndarray` output."""
    dtype = np.dtype(data["dtype"])
    shape = tuple(data["shape"])
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape):
        raise ValueError('invalid ndarray shape')
    if dtype.fields is not None:
        raise TypeError('structured arrays require an explicit schema codec')
    if dtype.hasobject:
        import math
        items = data.get('object_values')
        if not isinstance(items, list) or len(items) != math.prod(shape):
            raise ValueError('object array requires element values, never raw pointer bytes')
        arr = np.empty(shape, dtype=object)
        for index, item in enumerate(items):
            arr.flat[index] = decode_value(item)
        arr.flags.writeable = False
        return arr
    raw = base64.b64decode(data["data_b64"])
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    arr.flags.writeable = False
    return arr


def encode_value(value):
    """Encode an arbitrary artifact value to a JSON-friendly canonical form."""
    if isinstance(value, np.ndarray):
        return encode_ndarray(value)
    if isinstance(value, np.generic):
        return encode_value(value.item())
    if isinstance(value, datetime.datetime):
        tz = value.tzinfo.tzname(value) if value.tzinfo else None
        return {"__datetime__": value.isoformat(), "tz": tz}
    if isinstance(value, datetime.date):
        return {"__date__": value.isoformat()}
    if isinstance(value, datetime.timedelta):
        return {"__timedelta__": value.total_seconds()}
    if isinstance(value, decimal.Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, enum.Enum):
        return {
            "__enum__": [type(value).__module__, type(value).__qualname__, value.name],
        }
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        # JSON object keys must be strings. Casting would collapse distinct
        # keys such as 1 and '1'. Escape ordinary maps that resemble codec tags
        # too, otherwise metadata {'__bytes__': ...} decodes as bytes.
        if any(not isinstance(k, str) for k in value) or _RESERVED_TAGS.intersection(value):
            def encode_key(key):
                if isinstance(key, tuple):
                    return {'__tuple__': [encode_key(part) for part in key]}
                return encode_value(key)
            return {'__mapping__': [[encode_key(k), encode_value(v)]
                                    for k, v in value.items()]}
        return {k: encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"encode_value: unsupported type {type(value).__name__}")


def decode_value(value):
    """Inverse of :func:`encode_value`."""
    if isinstance(value, dict):
        if '__tuple__' in value:
            if set(value) != {'__tuple__'} or not isinstance(value['__tuple__'], list):
                raise ValueError('invalid tuple codec payload')
            return tuple(decode_value(part) for part in value['__tuple__'])
        if '__mapping__' in value:
            if set(value) != {'__mapping__'} or not isinstance(value['__mapping__'], list):
                raise ValueError('invalid mapping codec payload')
            result = {}
            for item in value['__mapping__']:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError('mapping codec entries must be key/value pairs')
                key, decoded = decode_value(item[0]), decode_value(item[1])
                try:
                    if key in result:
                        raise ValueError('duplicate decoded mapping key')
                    result[key] = decoded
                except TypeError as exc:
                    raise ValueError('unhashable decoded mapping key') from exc
            return result
        if value.get("__ndarray__"):
            return decode_ndarray(value)
        if "__datetime__" in value:
            return datetime.datetime.fromisoformat(value["__datetime__"])
        if "__date__" in value:
            return datetime.date.fromisoformat(value["__date__"])
        if "__timedelta__" in value:
            return datetime.timedelta(seconds=value["__timedelta__"])
        if "__decimal__" in value:
            return decimal.Decimal(value["__decimal__"])
        if "__enum__" in value:
            import importlib

            mod, qual, name = value["__enum__"]
            cls = importlib.import_module(mod)
            for part in qual.split("."):
                cls = getattr(cls, part)
            return cls[name]
        if "__bytes__" in value:
            return base64.b64decode(value["__bytes__"])
        return {k: decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    return value
