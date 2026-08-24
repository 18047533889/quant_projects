"""
R46 P0-T + P1-U — production MetricArtifact serialization + zero-copy default off.

P0-T (production serialization):
    A production artifact must survive ``to_dict`` / ``from_dict`` round-trip
    with its ``production`` flag, ``contract_schema_version``, and
    ``producer_version`` intact.  Previously ``production`` was NOT serialized,
    so a production artifact silently downgraded to ``production=False`` on
    deserialization, and the restored artifact did not re-enforce the
    FactorAxisRef production contract.  These are now part of ``to_dict`` /
    ``from_dict`` / ``__eq__`` / ``__hash__`` / stable content hash.

P1-U (zero-copy default off in production):
    A numpy ndarray can have an original / a view / an artifact view aliasing
    the same memory; ``writeable=False`` on one alias does not stop a caller
    that still holds another writable view from mutating the shared buffer.  A
    PRODUCTION artifact must therefore COPY + freeze even when handed an
    :class:`ImmutableBufferRef`; zero-copy adoption remains available only for
    research (non-production) use.

Importable source is the repo-root ``quant_evaluator/`` package (the pinned
source tree, not build/lib).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quant_evaluator.contracts.axis_refs import FactorAxisRef
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.metric_artifacts import (
   
        ImmutableBufferRef,
    MetricArtifact,
    ScalarMetricArtifact,
)

_F2 = FactorAxisRef(factor_ids=("f1", "f2"))


def _prod(**overrides):
    """A canonical production ScalarMetricArtifact."""
    kwargs = dict(
        metric_id="m1", domain="d", values=np.array([1.0, 2.0]),
        factor_axis=_F2, production=True,
        contract_schema_version="2.0", producer_version="7",
    )
    kwargs.update(overrides)
    return ScalarMetricArtifact(**kwargs)


# ---------------------------------------------------------------------------
# P0-T: production serialization round-trip.
# ---------------------------------------------------------------------------

def test_production_fields_serialized_into_payload():
    art = _prod()
    payload = art.to_dict()
    assert payload["production"] is True
    assert payload["contract_schema_version"] == "2.0"
    assert payload["producer_version"] == "7"


def test_production_artifact_roundtrip_keeps_contract_and_identity():
    art = _prod()
    restored = MetricArtifact.from_dict(art.to_dict())
    # Production flag and version ride through the round-trip.
    assert restored.production is True
    assert restored.contract_schema_version == "2.0"
    assert restored.producer_version == "7"
    # Factor axis still enforced and identity unchanged.
    assert isinstance(restored.factor_axis, FactorAxisRef)
    assert restored.factor_axis == _F2
    assert restored == art
    assert hash(restored) == hash(art)


def test_production_restored_artifact_re_enforces_factor_axis():
    # A production artifact cannot deserialize with a missing factor_axis:
    # __post_init__ must fire on the restored object.
    payload = _prod().to_dict()
    payload["factor_axis"] = None
    with pytest.raises(InvalidContractError):
        MetricArtifact.from_dict(payload)


def test_production_flag_distinguishes_eq_and_hash():
    research = _prod(production=False)
    prod = _prod()
    assert research != prod
    assert hash(research) != hash(prod)


def test_producer_version_distinguishes_identity():
    a = _prod(contract_schema_version="1.0", producer_version="1")
    b = _prod(contract_schema_version="1.0", producer_version="2")
    assert a != b
    assert hash(a) != hash(b)


# ---------------------------------------------------------------------------
# P1-U: zero-copy default off in production.
# ---------------------------------------------------------------------------

def test_production_immutable_buffer_ref_copies_not_adopts():
    buf = np.array([1.0, 2.0])
    ref = ImmutableBufferRef(buf)
    art = _prod(values=ref)
    # Copy + freeze: mutating the original (writable alias) must not change
    # the artifact's stored data (ownership transferred by COPY).
    buf.flags.writeable = True
    buf[0] = 999.0
    assert art.values[0] == 1.0
    assert not art.values.flags.writeable


def test_production_zero_copy_view_alias_cannot_mutate_artifact():
    # Even a second writable VIEW aliasing the original buffer cannot corrupt
    # a production artifact, because production copied the data.
    base = np.array([1.0, 2.0, 3.0, 4.0])
    buf = base[0:2]  # view aliasing base's memory
    art = _prod(values=ImmutableBufferRef(buf))
    # Overwrite through the still-writable original buffer (writable alias).
    base[0] = 999.0
    assert art.values[0] == 1.0


def test_research_zero_copy_still_allowed():
    # Research (non-production) keeps the zero-copy escape: adopting the
    # caller buffer read-only is still permitted.
    buf = np.array([1.0, 2.0, 3.0])
    art = ScalarMetricArtifact(
        metric_id="m", domain="d", values=ImmutableBufferRef(buf),
    )
    assert not art.values.flags.writeable
    assert art.values[0] == 1.0
