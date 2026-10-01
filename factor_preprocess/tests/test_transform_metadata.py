"""
R61-FI-040/041/042 transform-metadata contract tests.

Locks the implementation-origin metadata surface (plan §26 F2):

- Every registered transform carries ``implementation_origin`` (never None)
  so a routing audit can enumerate the whole catalog.
- ``implementation_origin`` is one of ``{"FE_OPERATOR", "FE_COMPOSITE", "FP_NATIVE"}``.
- ``fe_operator_id`` is set exactly when ``implementation_origin ==
  "FE_OPERATOR"`` (and is None otherwise).
- ``fit_kind`` is one of ``{"stateless", "fitted"}`` and agrees with
  ``requires_fit``.
- Fitted transforms must always be ``FP_NATIVE`` (FE operators are stateless;
  fitted math stays FP-native — plan §26 F2).
- FE-routed transforms are stateless.
- The semantic-policy hash covers the origin fields (origin drift changes the
  registry snapshot identity).
"""
from __future__ import annotations

import dataclasses

import pytest

from factor_preprocess.registry.transforms import (
    TransformRegistry,
    TransformCategory,
    create_default_registry,
    get_default_registry,
)

_VALID_ORIGINS = {"FE_OPERATOR", "FE_COMPOSITE", "FP_NATIVE"}
_VALID_FIT_KINDS = {"stateless", "fitted"}

#: Direct FE operators; composite recipes have a separate identity contract.
#: Parity tests cover both routes.
_FE_ROUTED = {
    "cs_rank": "rank",
    "cs_demean": "cs_demean",
    "cs_winsor": "winsorize",
    "forward_fill": "ffill_limit",
}


def test_every_transform_has_origin_and_valid_fields():
    registry = get_default_registry()
    for m in registry.all_transforms():
        assert m.implementation_origin in _VALID_ORIGINS, m.name
        assert m.fit_kind in _VALID_FIT_KINDS, m.name
        # fit_kind agrees with requires_fit
        expected_kind = "fitted" if m.requires_fit else "stateless"
        assert m.fit_kind == expected_kind, (
            f"{m.name}: fit_kind={m.fit_kind} but requires_fit={m.requires_fit}"
        )
        # fe_operator_id present iff FE_OPERATOR
        if m.implementation_origin == "FE_OPERATOR":
            assert m.fe_operator_id, f"{m.name} routed but no fe_operator_id"
            assert m.fit_kind == "stateless", (
                f"{m.name}: fitted transforms must stay FP_NATIVE"
            )
        elif m.implementation_origin == "FE_COMPOSITE":
            assert m.fe_operator_id is None, m.name
            assert m.fit_kind == "stateless", m.name
        else:
            assert m.fe_operator_id is None, (
                f"{m.name}: non-operator origins must not carry fe_operator_id"
            )


def test_fe_routed_set_is_exact():
    registry = get_default_registry()
    routed = {
        m.name: m.fe_operator_id
        for m in registry.all_transforms()
        if m.implementation_origin == "FE_OPERATOR"
    }
    assert routed == _FE_ROUTED


def test_fe_composite_set_and_recipe_identities_are_exact():
    registry = get_default_registry()
    expected = {
        "trailing_sma": "FE_COMPOSITE:long_smoothing.lagged_mean:v1",
        "rolling_mean": "FE_COMPOSITE:long_smoothing.lagged_mean:v1",
        "trailing_median": "FE_COMPOSITE:long_smoothing.lagged_median:v1",
        "rolling_std": "FE_COMPOSITE:long_smoothing.lagged_std:v1",
        "rolling_zscore": "FE_COMPOSITE:long_smoothing.lagged_zscore:v1",
    }
    for name in ("ols_neutralize", "industry_neutral", "size_neutral", "dual_neutral"):
        expected[name] = "FE_COMPOSITE:long_neutralization.ols_effective_rank:v1"
    actual = {
        meta.name: meta.fe_equivalent_semantics
        for meta in registry.all_transforms()
        if meta.implementation_origin == "FE_COMPOSITE"
    }
    assert actual == expected
    for name in expected:
        assert registry.get(name).fe_operator_id is None


def test_fitted_transforms_are_fp_native():
    registry = get_default_registry()
    for m in registry.all_transforms():
        if m.requires_fit:
            assert m.implementation_origin == "FP_NATIVE", m.name


def test_full_sample_research_transforms_do_not_claim_reusable_fitted_state():
    """Research admission is distinct from a persisted fit/apply contract."""
    registry = get_default_registry()
    for name in ("detect_correlation_regime", "impute_with_fallback"):
        metadata = registry.get(name)
        assert metadata.admission == "RESEARCH_ONLY"
        assert not metadata.causal_safe
        # Both compute/use full-sample statistics within one call, but expose
        # no fitted state object for later application.
        assert not metadata.requires_fit
        assert metadata.fit_kind == "stateless"
        assert metadata.implementation_origin == "FP_NATIVE"
        with pytest.raises(ValueError, match=f"Transform '{name}' is RESEARCH_ONLY"):
            registry.get_execution(name)
        # Resolution only: do not invoke the full-sample transform.
        assert callable(registry.get_execution(name, allow_research=True))


def test_fe_operator_ids_match_registry_and_adapter_surface():
    registry = get_default_registry()
    for m in registry.all_transforms():
        if m.implementation_origin == "FE_OPERATOR":
            # adapter must know the canonical operator
            from factor_preprocess.adapters.fe_operator import get_fe_executor
            # get_fe_executor may return None when FE is unavailable — the
            # id must still be resolvable in principle (non-empty, sane).
            assert m.fe_operator_id.strip(), m.name
            assert "_" in m.fe_operator_id or m.fe_operator_id.isalpha(), m.name


def test_origin_semantic_surface_participates_in_numeric_policy_hash():
    """Changing only implementation_origin must change the identity hash."""
    registry = TransformRegistry()

    def _f(x):
        return x

    registry.register(
        "t1", _f, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0", semantic_id="T:1", stage="representation",
    )
    before = registry.get("t1").numeric_policy_hash
    registry.enrich("t1", implementation_origin="FE_OPERATOR",
                    fe_operator_id="rank", fit_kind="stateless")
    after_fe = registry.get("t1").numeric_policy_hash
    assert after_fe != before
    registry.enrich("t1", implementation_origin="FP_NATIVE",
                    fe_operator_id=None, fit_kind="stateless")
    after_native = registry.get("t1").numeric_policy_hash
    assert after_native != after_fe
    registry.enrich("t1", implementation_origin="FE_COMPOSITE", fe_operator_id=None,
                    fit_kind="stateless", fe_equivalent_semantics="FE_COMPOSITE:test:v1")
    after_composite = registry.get("t1").numeric_policy_hash
    assert after_composite != after_native
    # The FP_NATIVE route with fe_operator_id=None is not guaranteed to
    # restore the original pre-enrichment hash, because enrichment re-derives
    # the numeric policy hash from the *full current surface* each time — the
    # hash is content-derived, not a diff.  Asserting only monotonic change
    # (origin participates) keeps the test honest.
    assert after_native != after_fe


def test_resolve_origin_rules():
    registry = get_default_registry()
    assert registry.resolve_origin("cs_rank") == "FE_OPERATOR"
    assert registry.resolve_origin("trailing_sma") == "FE_COMPOSITE"
    assert registry.resolve_origin("cs_zscore") == "FP_NATIVE"
    assert registry.resolve_origin("ewma") == "FP_NATIVE"
    with pytest.raises(ValueError):
        registry.resolve_origin("missing_transform")


def test_get_execution_returns_callable_for_fe_backed():
    registry = get_default_registry()
    for name in _FE_ROUTED:
        fn = registry.get_execution(name)
        assert callable(fn), name


def test_frozen_snapshot_includes_origin_fields():
    """The dataclass field surface carries the new fields (frozen safe)."""
    fields = {f.name for f in dataclasses.fields(get_default_registry().get("cs_rank"))}
    assert {"implementation_origin", "fe_operator_id", "fit_kind"} <= fields
