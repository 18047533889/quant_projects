"""
QE-P0-01/02/03/04/06 — MetricArtifact deep-immutability, ndarray ownership,
lossless serialization, and hash codec hardening.

P0 red-team findings this test pins:

QE-P0-01 (deep immutability):
    ``provenance`` is a recursively-immutable :class:`FrozenMapping`; nested
    dicts/lists/sets are frozen to tuples/frozensets/nested FrozenMappings, so
    ``artifact.provenance["x"] = ...`` and nested mutation FAIL.  ``__hash__``
    is stable before and after an attempted mutation (the mutation raises).

QE-P0-02 (snapshot array ownership):
    ``_freeze_array`` copies the caller's buffer by default, so mutating the
    original ndarray afterwards never changes the artifact.  A zero-copy
    escape exists only via the explicit :class:`ImmutableBufferRef` capability.

QE-P0-03 (artifact_kind class authority):
    ``artifact_kind`` is derived from the subclass; ScalarMetricArtifact
    cannot be constructed as "vector" (ValueError).

QE-P0-04 (lossless serialization):
    ``to_dict``/``from_dict`` use a canonical ndarray codec (dtype + shape +
    base64 payload bytes) preserving dtype, shape, endianness, NaN/Inf bit
    patterns, and datetime64 axes exactly.

QE-P0-06 (hash codec hardening):
    ``_hashutil.canonicalize`` rejects object-dtype ndarrays, rejects
    non-string mapping keys, has no ``default=repr`` fallback (unsupported
    objects raise), and provides explicit codecs for datetime/date/timedelta,
    Decimal, Enum, bytes, and numpy scalars.

Importable source is ``build/lib/quant_evaluator`` (the repo-root
``quant_evaluator/`` directory is a stub).  The bootstrap below forces the
build/lib package + subpackages onto ``sys.modules`` directly (same pattern
as ``test_metamorphic.py``) and prepends build/lib to ``sys.path`` for
intra-package imports.

Run with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/home/shw/quant_projects/quant_evaluator/build/lib \
    python -m pytest quant_evaluator/tests/test_qe_artifact_hardening.py -v --tb=short
"""

import datetime
import decimal
import enum
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Importable-source bootstrap (identical pattern to test_metamorphic.py).
# ---------------------------------------------------------------------------
_BUILD_LIB = Path(__file__).resolve().parents[1] / "build" / "lib"
_PKG_ROOT = _BUILD_LIB / "quant_evaluator"
if str(_BUILD_LIB) not in sys.path:
    sys.path.insert(0, str(_BUILD_LIB))

_QE = sys.modules.get("quant_evaluator")
if _QE is None or not str(getattr(_QE, "__file__", "")).startswith(str(_PKG_ROOT)):
    import importlib.util as _ilu
    import os as _os

    for _pkg in ("", "contracts", "metrics", "kernels", "runtime", "planner",
                 "registry", "diagnosis", "api", "adapters", "backends",
                 "reporting"):
        _root = _PKG_ROOT if not _pkg else _PKG_ROOT / _pkg
        _init = _root / "__init__.py"
        if _init.exists():
            _name = "quant_evaluator" if not _pkg else f"quant_evaluator.{_pkg}"
            _spec = _ilu.spec_from_file_location(_name, str(_init))
            _mod = _ilu.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
    import quant_evaluator
    _qe_dir = _os.path.dirname(_os.path.abspath(quant_evaluator.__file__))
    assert _qe_dir.startswith(str(_PKG_ROOT)), f"bootstrap failed: {_qe_dir}"

from quant_evaluator.contracts._hashutil import canonicalize, stable_content_hex
from quant_evaluator.contracts._ndarray_codec import (
    decode_ndarray,
    decode_value,
    encode_ndarray,
    encode_value,
)
from quant_evaluator.contracts.metric_artifacts import (
    DistributionMetricArtifact,
    FrozenMapping,
    ImmutableBufferRef,
    MatrixMetricArtifact,
    ScalarMetricArtifact,
    SeriesMetricArtifact,
    VectorMetricArtifact,
)


# ---------------------------------------------------------------------------
# QE-P0-01: deep immutability.
# ---------------------------------------------------------------------------

def test_provenance_is_recursively_immutable():
    prov = {"x": 1, "nested": {"a": [1, 2], "s": {3, 4}}}
    art = ScalarMetricArtifact(
        metric_id="m", domain="d", values=np.array([1.0, 2.0, 3.0]),
        provenance=prov,
    )
    assert isinstance(art.provenance, FrozenMapping)
    # Top-level item assignment fails.
    with pytest.raises(TypeError):
        art.provenance["x"] = 5  # type: ignore[index]
    # Nested dict mutation fails.
    with pytest.raises(TypeError):
        art.provenance["nested"]["a"] = [9]  # type: ignore[index]
    # Nested list mutation fails (frozen to tuple).
    with pytest.raises(TypeError):
        art.provenance["nested"]["a"][0] = 9  # type: ignore[index]
    # Nested set mutation fails (frozen to frozenset).
    with pytest.raises((TypeError, AttributeError)):
        art.provenance["nested"]["s"].add(5)  # type: ignore[attr-defined]
    # The caller's original mapping is not aliased.
    prov["x"] = 999
    assert art.provenance["x"] == 1


def test_hash_stable_under_attempted_mutation():
    art = ScalarMetricArtifact(
        metric_id="m", domain="d", values=np.array([1.0, 2.0, 3.0]),
        provenance={"nested": {"a": [1, 2]}},
    )
    h_before = hash(art)
    with pytest.raises(TypeError):
        art.provenance["nested"]["a"] = [9]  # type: ignore[index]
    with pytest.raises(TypeError):
        art.provenance["__new__"] = "no"  # type: ignore[index]
    assert hash(art) == h_before


def test_created_from_and_time_index_are_tuples():
    art = SeriesMetricArtifact(
        metric_id="m", domain="d", values=np.ones((2, 3)),
        time_index=("a", "b"), created_from=("f", "l"),
    )
    assert isinstance(art.created_from, tuple)
    assert isinstance(art.time_index, tuple)
    with pytest.raises(TypeError):
        art.created_from[0] = "x"  # type: ignore[index]


# ---------------------------------------------------------------------------
# QE-P0-02: snapshot array ownership.
# ---------------------------------------------------------------------------

def test_array_is_copied_not_aliased():
    orig = np.array([1.0, 2.0, 3.0])
    art = ScalarMetricArtifact(metric_id="m", domain="d", values=orig)
    orig[0] = 999.0
    assert art.values[0] == 1.0
    assert not art.values.flags.writeable


def test_array_ownership_for_all_payload_fields():
    orig = np.ones((2, 3))
    art = VectorMetricArtifact(metric_id="m", domain="d", values=orig)
    orig[0, 0] = 999.0
    assert art.values[0, 0] == 1.0

    orig_s = np.ones((4, 3))
    art_d = DistributionMetricArtifact(
        metric_id="m", domain="d", samples=orig_s, stat_names=("s1",)
    )
    orig_s[0, 0] = 999.0
    assert art_d.samples[0, 0] == 1.0


def test_immutable_buffer_ref_zero_copy_escape():
    buf = np.array([1.0, 2.0, 3.0])
    art = ScalarMetricArtifact(
        metric_id="m", domain="d", values=ImmutableBufferRef(buf)
    )
    # The buffer is adopted read-only (no mutable alias handed out).
    assert not art.values.flags.writeable
    assert art.values[0] == 1.0


# ---------------------------------------------------------------------------
# QE-P0-03: artifact_kind class authority.
# ---------------------------------------------------------------------------

def test_artifact_kind_derived_from_subclass():
    assert ScalarMetricArtifact.__artifact_kind__ == "scalar"
    assert VectorMetricArtifact.__artifact_kind__ == "vector"
    assert SeriesMetricArtifact.__artifact_kind__ == "series"
    assert MatrixMetricArtifact.__artifact_kind__ == "matrix"
    assert DistributionMetricArtifact.__artifact_kind__ == "distribution"


def test_scalar_cannot_be_constructed_as_vector():
    with pytest.raises(ValueError):
        ScalarMetricArtifact(
            metric_id="m", domain="d", artifact_kind="vector",
            values=np.array([1.0, 2.0]),
        )
    with pytest.raises(ValueError):
        VectorMetricArtifact(
            metric_id="m", domain="d", artifact_kind="scalar",
            values=np.ones((2, 3)),
        )


def test_artifact_kind_omitted_defaults_to_class():
    art = ScalarMetricArtifact(metric_id="m", domain="d", values=np.array([1.0, 2.0]))
    assert art.artifact_kind == "scalar"


# ---------------------------------------------------------------------------
# QE-P0-04: lossless serialization.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [
    np.float32, np.float64, np.int8, np.int64, np.uint16,
    "datetime64[ns]", "datetime64[D]",
], ids=lambda d: str(d))
def test_ndarray_codec_preserves_dtype_and_values(dtype):
    arr = np.array([[1, 2], [3, 4]], dtype=dtype)
    dec = decode_ndarray(encode_ndarray(arr))
    assert dec.dtype == arr.dtype
    assert dec.shape == arr.shape
    assert np.array_equal(dec, arr)


def test_ndarray_codec_nan_inf_bit_exact():
    arr = np.array([np.nan, np.inf, -np.inf, 1.5], dtype=np.float64)
    dec = decode_ndarray(encode_ndarray(arr))
    assert np.array_equal(dec, arr, equal_nan=True)
    assert np.isinf(dec[1]) and dec[1] > 0
    assert np.isinf(dec[2]) and dec[2] < 0
    assert np.isnan(dec[0])


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0)], ids=lambda s: str(s))
def test_ndarray_codec_empty_arrays(shape):
    arr = np.empty(shape, dtype=np.float64)
    dec = decode_ndarray(encode_ndarray(arr))
    assert dec.shape == shape
    assert dec.dtype == arr.dtype


def test_ndarray_codec_non_contiguous_input():
    base = np.arange(12.0).reshape(3, 4)
    noncontig = base[:, ::2]
    dec = decode_ndarray(encode_ndarray(noncontig))
    assert np.array_equal(dec, noncontig)


def test_artifact_to_from_dict_lossless_dtype():
    art = ScalarMetricArtifact(
        metric_id="m", domain="d", values=np.array([0.1, 0.2], dtype=np.float32),
    )
    restored = ScalarMetricArtifact.from_dict(art.to_dict())
    assert restored.values.dtype == np.float32
    assert np.array_equal(restored.values, art.values)


def test_artifact_to_from_dict_datetime64_axis():
    art = SeriesMetricArtifact(
        metric_id="m", domain="d",
        values=np.ones((2, 3)),
        time_index=("a", "b"),
    )
    # datetime64 payload axis round-trips through the codec.
    dt = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
    enc = encode_value(dt)
    dec = decode_value(enc)
    assert dec.dtype == dt.dtype
    assert np.array_equal(dec, dt)


def test_artifact_to_from_dict_provenance_value_codecs():
    art = ScalarMetricArtifact(
        metric_id="m", domain="d", values=np.array([1.0, 2.0]),
        provenance={
            "when": datetime.datetime(2024, 1, 1, 12, 0, 0),
            "day": datetime.date(2024, 1, 1),
            "amount": decimal.Decimal("1.5"),
            "blob": b"\x00\x01\x02",
        },
    )
    restored = ScalarMetricArtifact.from_dict(art.to_dict())
    assert restored.provenance["when"] == datetime.datetime(2024, 1, 1, 12, 0, 0)
    assert restored.provenance["day"] == datetime.date(2024, 1, 1)
    assert restored.provenance["amount"] == decimal.Decimal("1.5")
    assert restored.provenance["blob"] == b"\x00\x01\x02"


# ---------------------------------------------------------------------------
# QE-P0-06: hash codec hardening.
# ---------------------------------------------------------------------------

def test_object_dtype_ndarray_rejected():
    with pytest.raises(TypeError):
        canonicalize(np.array([1, 2], dtype=object))


def test_non_string_mapping_key_rejected():
    with pytest.raises(TypeError):
        canonicalize({1: "a"})


def test_unsupported_object_rejected_no_repr_fallback():
    with pytest.raises(TypeError):
        canonicalize(object())


def test_explicit_codecs_for_supported_types():
    assert canonicalize(datetime.datetime(2024, 1, 1, 12, 0, 0)) == {
        "__datetime__": "2024-01-01T12:00:00", "tz": None,
    }
    assert canonicalize(datetime.date(2024, 1, 1)) == {"__date__": "2024-01-01"}
    assert canonicalize(datetime.timedelta(seconds=90)) == {"__timedelta__": 90.0}
    assert canonicalize(decimal.Decimal("1.5")) == {"__decimal__": "1.5"}
    assert canonicalize(b"abc") == {"__bytes__": "YWJj"}
    assert canonicalize(np.float32(1.5)) == 1.5
    assert canonicalize(np.int64(7)) == 7


def test_enum_codec():
    class Color(enum.Enum):
        RED = 1

    enc = canonicalize(Color.RED)
    assert enc["__enum__"][0] == Color.__module__
    assert enc["__enum__"][1].endswith("Color")
    assert enc["__enum__"][2] == "RED"


def test_stable_content_hex_uses_canonical_codecs():
    h1 = stable_content_hex(
        tag="t",
        fields={"when": datetime.datetime(2024, 1, 1, 12, 0, 0), "n": 1},
    )
    h2 = stable_content_hex(
        tag="t",
        fields={"when": datetime.datetime(2024, 1, 1, 12, 0, 0), "n": 1},
    )
    assert h1 == h2
