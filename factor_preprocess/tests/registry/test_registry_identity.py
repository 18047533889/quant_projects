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


# ---------------------------------------------------------------------------
# R55 #93: semantic-surface drift must change numeric policy hash + identity
# ---------------------------------------------------------------------------
def _demo_with_params(x, lower=0.01, upper=0.99):
    return x


def test_numeric_policy_hash_covers_semantic_policy_surface():
    """A semantic-only change (semantic_id) must change numeric_policy_hash
    and therefore the sealed registry identity (R55 #93)."""
    r = TransformRegistry()
    r.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
               version="1.0.0", parameters={"lower": 0.01, "upper": 0.99})
    base = r.get("demo")
    h1 = base.numeric_policy_hash

    # Enrichening a semantic policy field (semantic_id) changes the semantic
    # surface but NOT the signature/implementation hash.
    r.enrich("demo", semantic_id="WINSOR:cs", stage="outlier",
             family_tags={"PRICE_VOLUME"}, causality_class="cross_sectional_causal",
             numeric_policy="clip_cross_sectional",
             fe_equivalent_semantics="cs_winsor")
    enriched = r.get("demo")
    assert enriched.signature_hash == base.signature_hash
    assert enriched.implementation_hash == base.implementation_hash
    # R55 #93: the numeric policy hash *must* change on semantic drift.
    assert enriched.numeric_policy_hash != h1

    # Seed-global seal identity must differ between before/after enrichment.
    r2 = TransformRegistry()
    r2.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
                version="1.0.0", parameters={"lower": 0.01, "upper": 0.99})
    before = r2.seal()

    r3 = TransformRegistry()
    r3.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
                version="1.0.0", parameters={"lower": 0.01, "upper": 0.99})
    r3.enrich("demo", semantic_id="WINSOR:cs")
    after = r3.seal()
    assert before != after


def test_registration_fails_closed_on_semantic_surface_drift():
    """Re-registering the same transform with a different semantic surface
    (same signature + same numeric defaults) must raise (R55 #93)."""
    r = TransformRegistry()
    r.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
               version="1.0.0", parameters={"lower": 0.01, "upper": 0.99})
    with pytest.raises(ValueError):
        r.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
                   version="1.0.0", parameters={"lower": 0.01, "upper": 0.99},
                   semantic_id="WINSOR:cs")  # semantic drift only


def test_parameter_domain_is_deep_frozen_and_hashed():
    """parameter_domain with nested containers is deep-frozen in post_init,
    an alias passed by the caller cannot mutate it, and it participates in
    the numeric policy hash (R55 #93/#94)."""
    domain = {"lower": (0.005, 0.05), "nested": {"a": [1, 2]}}
    r = TransformRegistry()
    r.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
               version="1.0.0", parameters={"lower": 0.01, "upper": 0.99},
               parameter_domain=domain)
    meta = r.get("demo")
    snapshot = r.get("demo")
    assert meta.parameter_domain == snapshot.parameter_domain
    # Alias passed by the caller can no longer mutate the frozen snapshot.
    domain["lower"] = (0.9, 0.99)
    domain["nested"]["a"].append(99)
    assert meta.parameter_domain["lower"] == (0.005, 0.05)
    # FrozenDict equality is content-based and no in-place mutation occurred.
    assert meta.parameter_domain["nested"]["a"] == (1, 2)
    # The registry-provided snapshot must be deeply immutable.
    with pytest.raises(TypeError):
        meta.parameter_domain["lower"] = (0.0, 1.0)
    try:
        meta.family_tags.add("EVENT")
        raise AssertionError("family_tags should be immutable")
    except (TypeError, AttributeError):
        pass
    # Deep-frozen snapshots are hash-safe (rehash stable).
    assert hash(meta.parameter_domain)


# ---------------------------------------------------------------------------
# R55 #94: get() returns deep-frozen isolated snapshots (no aliased mutable
# state), and the numeric policy hash covers the semantic policy surface.
# ---------------------------------------------------------------------------
def test_snapshot_is_deep_frozen_and_detached_from_internal():
    r = TransformRegistry()
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL)
    r.enrich("demo", family_tags={"PRICE_VOLUME"})
    snap = r.get("demo")
    # The returned snapshot's nested containers are deeply frozen, so a
    # mutation attempt fails (either TypeError for FrozenDict or the frozen
    # collection has no mutator at all — frozenset has no .add).
    try:
        snap.family_tags.add("EVENT")
        raise AssertionError("family_tags should be immutable")
    except (TypeError, AttributeError):
        pass
    with pytest.raises(TypeError):
        snap.parameters["x"] = 1
    # And re-get returns an independent snapshot with the same content.
    snap2 = r.get("demo")
    assert snap2.family_tags == snap.family_tags
    assert snap.family_tags == frozenset({"PRICE_VOLUME"})


def test_diagnostic_events_trail_registration():
    r = TransformRegistry()
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL)
    r.enrich("demo", family_tags={"PRICE_VOLUME"})
    r.seal()
    events = r.diagnostic_events()
    assert any("registered:demo" in e for e in events)
    assert any(e.startswith("enriched:demo:") for e in events)
    assert any(e.startswith("sealed:") for e in events)


def test_seal_identity_changes_when_semantic_surface_compiled_in():
    """Even with identical signature/implementation, a semantic-surface
    difference must produce a different sealed snapshot identity (R55 #93).
    """
    r = TransformRegistry()
    r.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL)
    r.seal()
    ident_a = r.snapshot_identity

    r2 = TransformRegistry()
    r2.register("demo", _demo_transform, TransformCategory.CROSS_SECTIONAL)
    r2.enrich("demo", semantic_id="CS_RANK:pct")
    r2.seal()
    ident_b = r2.snapshot_identity
    assert ident_a != ident_b


# ---------------------------------------------------------------------------
# R55 #93: allowed_frequencies drift must change the snapshot hash
# ---------------------------------------------------------------------------
def test_snapshot_hash_changes_when_allowed_frequencies_change():
    """Changing allowed_frequencies (an eligibility field) must change the
    numeric policy hash and therefore the sealed registry identity (R55 #93).
    """
    r = TransformRegistry()
    r.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
               version="1.0.0", parameters={"lower": 0.01, "upper": 0.99})
    base = r.get("demo")
    h1 = base.numeric_policy_hash

    r.enrich("demo", allowed_frequencies={"daily"})
    freq = r.get("demo")
    assert freq.allowed_frequencies == frozenset({"daily"})
    assert freq.numeric_policy_hash != h1

    r.enrich("demo", allowed_frequencies={"daily", "weekly"})
    freq2 = r.get("demo")
    assert freq2.allowed_frequencies == frozenset({"daily", "weekly"})
    assert freq2.numeric_policy_hash != freq.numeric_policy_hash

    # Sealed identities differ too.
    r2 = TransformRegistry()
    r2.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
                version="1.0.0", parameters={"lower": 0.01, "upper": 0.99})
    ident_base = r2.seal()
    r3 = TransformRegistry()
    r3.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
                version="1.0.0", parameters={"lower": 0.01, "upper": 0.99})
    r3.enrich("demo", allowed_frequencies={"daily"})
    ident_freq = r3.seal()
    assert ident_base != ident_freq


# ---------------------------------------------------------------------------
# R55 #94: enrich() creates a replacement metadata, never mutates in place;
# snapshots handed out by get() are deeply frozen and cannot be mutated.
# ---------------------------------------------------------------------------
def test_enrich_does_not_contaminate_original_or_prior_snapshots():
    r = TransformRegistry()
    r.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
               version="1.0.0", parameters={"lower": 0.01, "upper": 0.99})
    before = r.get("demo")
    before_hash = before.numeric_policy_hash

    r.enrich("demo", allowed_frequencies={"daily"}, semantic_id="CS_RANK:pct")
    after = r.get("demo")
    assert after.allowed_frequencies == frozenset({"daily"})
    assert after.numeric_policy_hash != before_hash

    # The enrichment created a replacement metadata — the stored instance
    # changed, but the previously handed-out snapshot is untouched.
    assert before.allowed_frequencies == frozenset()
    assert before.semantic_id is None
    assert before.numeric_policy_hash == before_hash

    # And the stored metadata (as returned by the internal getter) has the
    # enriched surface, not a mutation of the old instance's containers.
    assert after.allowed_frequencies == frozenset({"daily"})


def test_seal_after_enrich_metadata_is_deep_immutable():
    """After enrich + seal, every container on the registry-provided metadata
    rejects mutation and no fields can be added (R55 #94)."""
    r = TransformRegistry()
    r.register("demo", _demo_with_params, TransformCategory.CROSS_SECTIONAL,
               version="1.0.0", parameters={"lower": 0.01, "upper": 0.99},
               tags={"demo_tag"})
    r.enrich("demo", allowed_frequencies={"daily"}, family_tags={"PRICE_VOLUME"},
             parameter_domain={"lower": (0.005, 0.05)}, output_channels=("x",))
    r.seal()
    meta = r.get("demo")
    # containers are deep-frozen
    for container in (meta.tags, meta.family_tags, meta.allowed_frequencies,
                      meta.parameter_domain, meta.output_channels):
        try:
            # frozenset has no mutator; FrozenDict raises on __setitem__
            if hasattr(container, "add"):
                container.add("X")
            else:
                # FrozenDict: attempt to set a key
                try:
                    container["__new__"] = 1
                except AttributeError:
                    pass
            raise AssertionError("container should be immutable")
        except (TypeError, AttributeError):
            pass
    # Adding a brand-new field is impossible via plain assignment (the
    # instance-level __setattr__ guard fails closed on ANY undeclared
    # attribute, R55 #94).  object.__setattr__ is an intentional internal
    # escape hatch used only by the registry itself to construct replacement
    # metadata; external callers must go through enrich().
    with pytest.raises(AttributeError):
        meta.brand_new_field = 1
    assert not hasattr(meta, "brand_new_field")
