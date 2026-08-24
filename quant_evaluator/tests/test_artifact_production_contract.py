"""
P1-14 — production metric-artifact contract + stable-hash / fail-closed freeze
/ buffer-ownership hardening.

This test imports ``quant_evaluator`` from the SOURCE tree at the repo root
(no build/lib bootstrap):

    PYTHONPATH=/home/sunhaiwei/quant_projects \
    /tmp/fe2/bin/python -m pytest quant_evaluator/tests/test_artifact_production_contract.py -q \
        -p no:cacheprovider --tb=short

Covers:

1. ``MetricArtifact(production=True)`` REQUIRES a real :class:`FactorAxisRef`;
   ``factor_axis=None`` raises :class:`InvalidContractError`.  The default
   (``production=False``) keeps the research back-compat of allowing ``None``.
2. ``FrozenMapping.__hash__`` is a canonical SHA-256 digest (stable across
   processes under different ``PYTHONHASHSEED``) and stays correct for
   ndarray values (which the salted builtin ``hash(frozenset(items))`` broke).
3. ``_freeze_value`` is fail-closed: an unsupported mutable object raises
   :class:`InvalidContractError` instead of silently passing through.
4. ``ImmutableBufferRef`` enforces real ownership: it refuses a read-only
   buffer, captures a content SHA-256 at construction, and detects caller
   mutation before (via ``_freeze_array``) and after (via
   ``verify_untouched()``) adoption.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quant_evaluator.contracts.axis_refs import FactorAxisRef
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.metric_artifacts import (
    FrozenMapping,
    ImmutableBufferRef,
    ScalarMetricArtifact,
    _freeze_value,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_F2 = FactorAxisRef(factor_ids=("f1", "f2"))


# ---------------------------------------------------------------------------
# 1. Production artifact requires a real FactorAxisRef.
# ---------------------------------------------------------------------------

def test_production_requires_factor_axis():
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(
            metric_id="m", domain="d", values=np.array([1.0, 2.0]),
            production=True,
        )


def test_production_with_valid_factor_axis_works():
    art = ScalarMetricArtifact(
        metric_id="m", domain="d", values=np.array([1.0, 2.0]),
        factor_axis=_F2, production=True,
    )
    assert art.production is True
    assert art.factor_axis is _F2


def test_default_research_factor_axis_none_still_works():
    art = ScalarMetricArtifact(metric_id="m", domain="d", values=np.array([1.0, 2.0]))
    assert art.production is False
    assert art.factor_axis is None


def test_production_mismatched_factor_count_still_raises():
    # P1-14 keeps QE-P0-05: even with a real FactorAxisRef, a production
    # artifact whose payload F does not match num_factors is rejected.
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(
            metric_id="m", domain="d", values=np.array([1.0, 2.0]),
            factor_axis=FactorAxisRef(factor_ids=("f1", "f2", "f3")),
            production=True,
        )


# ---------------------------------------------------------------------------
# 2. FrozenMapping stable hash.
# ---------------------------------------------------------------------------

def test_frozen_mapping_equal_mappings_hash_equal():
    fm1 = FrozenMapping({"a": 1, "b": (1, 2), "c": "x"})
    fm2 = FrozenMapping({"a": 1, "b": [1, 2], "c": "x"})
    assert fm1 == fm2
    assert hash(fm1) == hash(fm2)


def test_frozen_mapping_with_ndarray_values_is_hashable():
    # The old builtin hash(frozenset(items)) raised on ndarray values; the
    # canonical SHA-256 digest hashes them by raw bytes instead.
    fm = FrozenMapping({"a": 1, "b": np.array([1.0, 2.0]), "c": [1, 2]})
    assert isinstance(hash(fm), int)
    # Same raw bytes, same digest.
    fm_same = FrozenMapping({"a": 1, "b": np.array([1.0, 2.0]), "c": (1, 2)})
    assert hash(fm) == hash(fm_same)


_SUBPROCESS_SNIPPET = r"""
import sys

sys.path.insert(0, "@REPO_ROOT@")

import numpy as np

from quant_evaluator.contracts.metric_artifacts import FrozenMapping

fm = FrozenMapping({"a": 1, "b": np.array([1.0, 2.0]), "c": [1, 2], "d": "x"})
print(hash(fm))
"""


def _run_frozen_mapping_hash(seed: str) -> int:
    script = _SUBPROCESS_SNIPPET.replace("@REPO_ROOT@", str(_REPO_ROOT))
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, (
        f"subprocess failed (seed={seed}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return int(proc.stdout.strip().splitlines()[-1])


def test_frozen_mapping_hash_stable_across_processes():
    """hash(FrozenMapping) must be byte-identical under different seeds."""
    assert _run_frozen_mapping_hash("0") == _run_frozen_mapping_hash("12345")


# ---------------------------------------------------------------------------
# 3. _freeze_value is fail-closed.
# ---------------------------------------------------------------------------

class _Mutable:
    """A plain mutable object (has __dict__) unsupported by the freeze codec."""

    def __init__(self, x=1):
        self.x = x


def test_freeze_value_rejects_unsupported_mutable_object():
    with pytest.raises(InvalidContractError):
        _freeze_value(_Mutable())
    # Also nested inside a mapping / list.
    with pytest.raises(InvalidContractError):
        _freeze_value({"bad": _Mutable()})
    with pytest.raises(InvalidContractError):
        _freeze_value([_Mutable()])


def test_freeze_value_keeps_immutable_scalars():
    assert _freeze_value(1) == 1
    assert _freeze_value("x") == "x"
    assert _freeze_value(None) is None
    assert _freeze_value((1, 2)) == (1, 2)


# ---------------------------------------------------------------------------
# 4. ImmutableBufferRef real ownership.
# ---------------------------------------------------------------------------

def test_immutable_buffer_ref_readonly_buffer_rejected():
    ro = np.array([1.0, 2.0, 3.0])
    ro.flags.writeable = False
    with pytest.raises(InvalidContractError):
        ImmutableBufferRef(ro)


def test_immutable_buffer_ref_adoption_still_works():
    # Back-compat (existing hardening test): construct ImmutableBufferRef(buf)
    # then reading art.values must work and be read-only.
    buf = np.array([1.0, 2.0, 3.0])
    art = ScalarMetricArtifact(
        metric_id="m", domain="d", values=ImmutableBufferRef(buf)
    )
    assert not art.values.flags.writeable
    assert art.values[0] == 1.0


def test_immutable_buffer_ref_detects_pre_adoption_mutation():
    buf = np.array([1.0, 2.0, 3.0])
    ref = ImmutableBufferRef(buf)
    buf[0] = 999.0
    # The caller mutated the buffer after handing it over but before the
    # artifact adopted it -- _freeze_array must catch this.
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(metric_id="m", domain="d", values=ref)


def test_immutable_buffer_ref_verify_untouched_detects_post_adoption_mutation():
    buf = np.array([1.0, 2.0, 3.0])
    ref = ImmutableBufferRef(buf)
    art = ScalarMetricArtifact(metric_id="m", domain="d", values=ref)
    assert not art.values.flags.writeable

    # Unmutated buffer verifies clean.
    ref.verify_untouched()

    # The artifact adopted the caller's buffer zero-copy; re-enabling write on
    # the original (same memory) lets the caller corrupt the artifact.  The
    # ref must detect it.
    buf.flags.writeable = True
    buf[0] = 42.0
    with pytest.raises(InvalidContractError):
        ref.verify_untouched()
    # _freeze_array re-adoption of the same (mutated) ref also fails.
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(metric_id="m", domain="d", values=ref)
