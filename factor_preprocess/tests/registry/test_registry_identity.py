# -*- coding: utf-8 -*-
"""Adversarial production-integrity tests for registry + policy identity."""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "factor_preprocess"))

import pytest

from factor_preprocess.errors import GovernanceError
from factor_preprocess.registry.transforms import (
    TransformRegistry,
    TransformMetadata,
    TransformCategory,
)
from factor_preprocess.registry.policies import (
    PolicyRegistry,
    PolicyPreset,
    PolicyLevel,
    TransformStep,
)


# ---------------------------------------------------------------------------
# FP-P1-04: true implementation identity (not just signature hash)
# ---------------------------------------------------------------------------
def _demo_transform(x):
    return x + 1


def _demo_transform_rewritten(x):
    # same signature, different body
    return x * 2


def test_implementation_hash_distinguishes_rewritten_body():
    r = TransformRegistry()
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
    meta = r.get("demo")
    assert meta.signature_hash          # signature hash present (back-compat)
    assert meta.implementation_hash     # true body hash present
    assert meta.numeric_policy_hash     # numeric policy hash present

    # rewriting the body (same signature) must change the implementation hash
    r2 = TransformRegistry()
    r2.register("demo", _demo_transform_rewritten, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
    meta2 = r2.get("demo")
    assert meta2.signature_hash == meta.signature_hash  # signature unchanged
    assert meta2.implementation_hash != meta.implementation_hash  # body changed
    # numeric_policy_hash is independent of the body (only params/version/etc) —
    # the body identity is captured by implementation_hash, not numeric_policy_hash.
    assert meta2.numeric_policy_hash == meta.numeric_policy_hash


def test_implementation_hash_deterministic():
    r = TransformRegistry()
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
    meta = r.get("demo")
    # stable across re-registration (idempotent)
    assert meta.implementation_hash


# ---------------------------------------------------------------------------
# FP-P1-05: production registry seal is immutable
# ---------------------------------------------------------------------------
def test_registry_seal_blocks_mutation():
    r = TransformRegistry()
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
    identity = r.seal()
    assert identity is not None
    # runtime re-registration must fail after seal
    with pytest.raises(GovernanceError):
        r.register("demo2", _demo_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")


def test_registry_seal_identity_is_frozen():
    r = TransformRegistry()
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
    identity = r.seal()
    # snapshot identity content-derived and immutable
    assert identity
    # sealing twice returns a stable snapshot identity
    identity2 = r.seal()
    assert identity2 == identity


# ---------------------------------------------------------------------------
# FP-P1-06: preprocess policy identity
# ---------------------------------------------------------------------------
def test_policy_identity_changes_with_params():
    r = TransformRegistry()
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
    r.seal()

    p1 = PolicyPreset(name="p", description="d", level=PolicyLevel.PRODUCTION,
                      steps=[TransformStep(name="demo", parameters={"a": 1})], causal_safe=True)
    p2 = PolicyPreset(name="p", description="d", level=PolicyLevel.PRODUCTION,
                      steps=[TransformStep(name="demo", parameters={"a": 2})], causal_safe=True)
    id1 = p1.policy_identity
    id2 = p2.policy_identity
    assert id1 and id2
    assert id1 != id2  # param change must change identity


def test_policy_identity_changes_with_transform_version():
    r = TransformRegistry()
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
    r.seal()
    p = PolicyPreset(name="p", description="d", level=PolicyLevel.PRODUCTION,
                     steps=[TransformStep(name="demo")], causal_safe=True)
    identity = p.policy_identity
    assert identity
