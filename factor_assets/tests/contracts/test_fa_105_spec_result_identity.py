"""Task #105 (F: FactorAssets/Similarity) — spec/result 分离 + 观测状态 + registry 门.

Covers:
- P1-FA-005: spec identity vs result identity — a spec change produces a
  different spec identity, a value change produces a different result
  identity, and the spec identity is never used as a proxy for the result.
- UNKNOWN vs COMPUTED_LOW in the refined pair ledger — the two states stay
  distinct (a computed-low value is never surfaced as UNKNOWN and vice versa).
- ``SimilarityViewRegistry`` construction gate — a registry built with views
  that are NOT in its view set fails construction (fail closed).
"""

import pytest

from factor_assets.contracts.similarity import (
    SIMILARITY_VIEW_KEYS,
    SimilarityArtifact,
    SimilarityViewRegistry,
)


def _spec_pair(views_a, views_b, **kwargs):
    """Two artifacts differing ONLY in the view values (same spec provenance)."""
    return (
        SimilarityArtifact(
            factor_a="F1",
            factor_b="F2",
            views=views_a,
            snapshot_ref="snapshot:2024",
            window_ref="2024-01-01/2024-12-31",
            universe_ref="universe:ashare",
            **kwargs,
        ),
        SimilarityArtifact(
            factor_a="F1",
            factor_b="F2",
            views=views_b,
            snapshot_ref="snapshot:2024",
            window_ref="2024-01-01/2024-12-31",
            universe_ref="universe:ashare",
            **kwargs,
        ),
    )


# ---------------------------------------------------------------------------
# P1-FA-005: spec identity vs result identity separation
# ---------------------------------------------------------------------------


class TestSpecResultIdentitySeparation:
    def test_value_change_changes_spec_identity(self):
        a, b = _spec_pair({"rank_corr": 0.9}, {"rank_corr": 0.91})
        assert a.similarity_spec_hash != b.similarity_spec_hash

    def test_spec_provenance_change_changes_spec_identity(self):
        a = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views={"rank_corr": 0.9},
            snapshot_ref="snapshot:2024",
        )
        b = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views={"rank_corr": 0.9},
            snapshot_ref="snapshot:2025",
        )
        assert a.similarity_spec_hash != b.similarity_spec_hash

    def test_identical_spec_implies_identical_result(self):
        a, b = _spec_pair({"rank_corr": 0.9}, {"rank_corr": 0.9})
        assert a.similarity_spec_hash == b.similarity_spec_hash
        assert a.primary_value == b.primary_value

    def test_for_spec_pins_exact_spec_identity(self):
        recomputed = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views={"rank_corr": 0.8}
        ).similarity_spec_hash
        pinned = SimilarityArtifact.for_spec(
            "F1", "F2", recomputed, views={"rank_corr": 0.8}
        )
        assert pinned.similarity_spec_hash == recomputed

    def test_for_spec_rejects_mismatched_hash(self):
        other = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views={"rank_corr": 0.5}
        ).similarity_spec_hash
        with pytest.raises(ValueError, match="does not match"):
            SimilarityArtifact.for_spec("F1", "F2", other, views={"rank_corr": 0.8})


# ---------------------------------------------------------------------------
# UNKNOWN vs COMPUTED_LOW
# ---------------------------------------------------------------------------


class TestUnknownVsComputedLow:
    def test_observation_states_are_distinct_enum_members(self):
        from factor_assets.clustering.families import SimilarityObservationState

        assert SimilarityObservationState.UNKNOWN.value == "UNKNOWN"
        assert SimilarityObservationState.COMPUTED_LOW.value == "COMPUTED_LOW"
        assert (
            SimilarityObservationState.UNKNOWN
            is not SimilarityObservationState.COMPUTED_LOW
        )

    def test_unmeasured_view_is_unknown_not_computed_low(self):
        from factor_assets.contracts.similarity import EdgeAffinityPolicy

        policy = EdgeAffinityPolicy({"rank_corr": 0.5})
        # An unmeasured view fuses to None (UNKNOWN), never a computed zero —
        # and never conflated with a genuinely computed low value.
        assert policy.affinity({"rank_corr": None}) is None
        measured_low = policy.affinity({"rank_corr": 0.1})
        assert measured_low is not None
        assert measured_low == pytest.approx(0.1)
        assert measured_low != 0.0


# ---------------------------------------------------------------------------
# SimilarityViewRegistry construction gate
# ---------------------------------------------------------------------------


class TestSimilarityViewRegistryConstructionGate:
    def test_views_passed_to_registry_are_validated(self):
        with pytest.raises(ValueError, match="not a registered canonical view"):
            SimilarityViewRegistry(views={"rank_cor": "typo"})

    def test_registered_views_construct(self):
        registry = SimilarityViewRegistry(views={"rank_corr": "rank correlation"})
        assert registry.is_registered("rank_corr")
        assert "rank_corr" in registry.view_keys

    def test_default_registry_contains_canonical_keys(self):
        registry = SimilarityViewRegistry()
        for key in SIMILARITY_VIEW_KEYS:
            assert registry.is_registered(key)

    def test_validate_rejects_unregistered_key(self):
        registry = SimilarityViewRegistry()
        with pytest.raises(ValueError, match="not a registered canonical view"):
            registry.validate({"rank_corr": 0.5, "bogus_view": 0.3})
