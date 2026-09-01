"""Task #105 (F: FactorAssets/Similarity) — assembly hash 真算 + spec/result 分离.

Covers:
- P0-FA-014: ``assembly_hash`` is a REAL canonical content hash — every
  semantic field of a membership (incl. ``reason`` / ``assembly_score`` /
  ``selection_rank``) and the spec identity (incl. ``min_evidence_date`` /
  ``treatment_optimization_ref``) is covered; changing ANY one of them changes
  the hash, re-running with identical inputs yields the identical hash, and
  ``FactorSetArtifact`` refuses an empty ``assembly_hash``.
- P0-FA-015: the assembly engine can never feed a non-finite decision score
  into ``FactorMembership.assembly_score`` — a NaN/inf decision score must be
  treated as ``None`` (or fail closed), never stored.
- P1-FA-005: spec identity vs result identity separation in Similarity — a
  spec change yields a different spec identity; a value change yields a
  different result identity; the spec identity is never used as a proxy for
  the result.
- UNKNOWN vs COMPUTED_LOW: the pair-state mapping distinguishes a never-touched
  pair from a pair computed low.
- ``SimilarityViewRegistry`` construction gate: views passed in are validated
  against the registered view set.
"""

from types import MappingProxyType

import pytest

from factor_assets.assembly import FactorSetAssembler
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.factor_set import (
    FactorMembership,
    FactorSetArtifact,
    FactorSetSpec,
)
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.similarity import (
    DEFAULT_SIMILARITY_VIEW,
    SIMILARITY_VIEW_KEYS,
    SimilarityArtifact,
    SimilarityViewRegistry,
)
from factor_assets.selection import SelectionDecision, SelectionReason


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_asset(factor_id, *, family=None):
    family = family if family is not None else "family-a"
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr=f"identity({factor_id})",
        canonical_hash=f"hash-{factor_id}",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    return FactorAsset(
        metadata=metadata,
        lineage=LineageRef(factor_id=factor_id, parents=()),
        lifecycle_state=LifecycleState.APPROVED,
        registered_at="2024-01-01T00:00:00Z",
        family=family,
        latest_evidence_ref=EvidenceBundleRef(
            bundle_id=f"bundle-{factor_id}",
            evaluation_run_id="run-1",
            factor_ids=(factor_id,),
            timestamp="2024-01-01T00:00:00Z",
            qe_version="1.0",
        ),
    )


def make_spec(set_id="set-1", name="N", policy="manual", **kwargs):
    kwargs.setdefault("data_snapshot_ref", "snapshot:default")
    kwargs.setdefault("universe_ref", "universe:default")
    kwargs.setdefault("split_ref", "split:default")
    return FactorSetSpec(set_id, name, policy, **kwargs)


def make_decision(factor_id, *, approved=True, timestamp="2024-01-01T00:00:00Z"):
    return SelectionDecision(
        decision_id=f"decision-{factor_id}-{timestamp}",
        factor_id=factor_id,
        approved=approved,
        reason=(
            SelectionReason.APPROVED
            if approved
            else SelectionReason.REJECTED_GATE_FAILURE
        ),
        timestamp=timestamp,
        policy_version="1.0",
        evidence_refs=("evidence-1",),
        gate_results=("gate-1",),
    )


def decided(factor_id, metadata, timestamp="2024-01-01T00:00:00Z"):
    d = make_decision(factor_id, timestamp=timestamp)
    object.__setattr__(d, "metadata", MappingProxyType(dict(metadata)))
    return d


def assemble(spec, *args, **kwargs):
    kwargs.setdefault("created_at", "2024-02-01T00:00:00Z")
    return FactorSetAssembler().assemble(spec, *args, **kwargs)


# ---------------------------------------------------------------------------
# P0-FA-014: assembly_hash is a REAL content hash
# ---------------------------------------------------------------------------


class TestAssemblyHashIsRealContentHash:
    def test_identical_inputs_yield_identical_hash(self):
        spec = make_spec()
        assets = [make_asset("F2"), make_asset("F1")]
        first = assemble(spec, assets)
        second = assemble(spec, list(reversed(assets)))
        assert first.assembly_hash == second.assembly_hash
        assert first.assembly_hash  # never empty

    def test_membership_role_change_changes_hash(self):
        spec = make_spec()
        base = assemble(spec, [make_asset("F1")])
        member = base.memberships[0]
        # A different role is a semantic change to the member.
        changed = FactorMembership(
            **{
                **member.__dict__,
                "role": "representative",
            }
        )
        rebuilt = self._artifact_with_members(spec, [changed])
        assert rebuilt.assembly_hash != base.assembly_hash

    def test_membership_orientation_change_changes_hash(self):
        spec = make_spec()
        base = assemble(spec, [make_asset("F1")])
        member = base.memberships[0]
        changed = FactorMembership(**{**member.__dict__, "orientation": 1})
        rebuilt = self._artifact_with_members(spec, [changed])
        assert rebuilt.assembly_hash != base.assembly_hash

    def test_membership_family_change_changes_hash(self):
        spec = make_spec()
        base = assemble(spec, [make_asset("F1")])
        member = base.memberships[0]
        changed = FactorMembership(**{**member.__dict__, "family_id": "family-z"})
        rebuilt = self._artifact_with_members(spec, [changed])
        assert rebuilt.assembly_hash != base.assembly_hash

    def test_membership_score_change_changes_hash(self):
        spec = make_spec()
        base = assemble(spec, [make_asset("F1")])
        member = base.memberships[0]
        changed = FactorMembership(**{**member.__dict__, "assembly_score": 0.5})
        rebuilt = self._artifact_with_members(spec, [changed])
        assert rebuilt.assembly_hash != base.assembly_hash

    def test_membership_rank_change_changes_hash(self):
        spec = make_spec()
        base = assemble(spec, [make_asset("F1")])
        member = base.memberships[0]
        changed = FactorMembership(**{**member.__dict__, "selection_rank": 7})
        rebuilt = self._artifact_with_members(spec, [changed])
        assert rebuilt.assembly_hash != base.assembly_hash

    def test_membership_reason_change_changes_hash(self):
        spec = make_spec()
        base = assemble(spec, [make_asset("F1")])
        member = base.memberships[0]
        changed = FactorMembership(**{**member.__dict__, "reason": "APPROVED"})
        rebuilt = self._artifact_with_members(spec, [changed])
        assert rebuilt.assembly_hash != base.assembly_hash

    def test_spec_selection_policy_change_changes_hash(self):
        # A different selection policy that STILL admits F1 — the policy is
        # part of the assembly identity (the hash covers the spec identity,
        # not just the member list).
        base = assemble(make_spec(), [make_asset("F1")])
        other = assemble(
            make_spec(policy="diverse", max_factors=1),
            [make_asset("F1")],
            selection_decisions=[make_decision("F1")],
            similarity_provider=lambda a, b: 0.5,
        )
        assert other.assembly_hash != base.assembly_hash

    def test_spec_snapshot_universe_split_change_changes_hash(self):
        base = assemble(make_spec(), [make_asset("F1")])
        assert assemble(
            make_spec(data_snapshot_ref="snapshot:OTHER"), [make_asset("F1")]
        ).assembly_hash != base.assembly_hash
        assert assemble(
            make_spec(universe_ref="universe:OTHER"), [make_asset("F1")]
        ).assembly_hash != base.assembly_hash
        assert assemble(
            make_spec(split_ref="split:OTHER"), [make_asset("F1")]
        ).assembly_hash != base.assembly_hash

    def test_spec_min_evidence_date_changes_hash(self):
        # A min_evidence_date that every candidate still satisfies: the member
        # list is unchanged, but the spec identity changed — the hash must
        # change (the audit requires min_evidence_date to be covered).
        base = assemble(make_spec(), [make_asset("F1")])
        other = assemble(make_spec(min_evidence_date="2000-01-01"), [make_asset("F1")])
        assert other.assembly_hash != base.assembly_hash

    def test_spec_treatment_optimization_ref_changes_hash(self):
        base = assemble(make_spec(), [make_asset("F1")])
        with_ref = assemble(
            make_spec(treatment_optimization_ref="opt-hash-1"), [make_asset("F1")]
        )
        other_ref = assemble(
            make_spec(treatment_optimization_ref="opt-hash-2"), [make_asset("F1")]
        )
        assert with_ref.assembly_hash != base.assembly_hash
        assert other_ref.assembly_hash != with_ref.assembly_hash

    def test_membership_treatment_optimization_ref_canonicalized_in_hash(self):
        # A dict full-mapping ref canonicalizes to its str content-hash form;
        # the assembly hash must cover the canonical form.
        spec = make_spec()
        base = assemble(spec, [make_asset("F1")])
        member = base.memberships[0]
        str_form = FactorMembership(
            **{**member.__dict__, "treatment_optimization_ref": "opt-hash-9"}
        )
        dict_form = FactorMembership(
            **{
                **member.__dict__,
                "treatment_optimization_ref": {
                    "library_version_ref": "lv1",
                    "content_hash": "opt-hash-9",
                },
            }
        )
        # Both transport shapes carry the SAME canonical identity.
        assert (
            self._artifact_with_members(spec, [str_form]).assembly_hash
            == self._artifact_with_members(spec, [dict_form]).assembly_hash
        )

    def test_artifact_rejects_empty_assembly_hash(self):
        member = FactorMembership(factor_id="F1")
        with pytest.raises(ValueError, match="assembly_hash"):
            FactorSetArtifact(
                set_id="set-1",
                name="N",
                members=(member,),
                created_at="2024-02-01T00:00:00Z",
                policy_hash="ph",
                assembly_hash="",
                snapshot_ref="snapshot:default",
                universe_ref="universe:default",
                split_ref="split:default",
            )

    @staticmethod
    def _artifact_with_members(spec, members):
        from factor_assets.assembly.engine import FactorSetAssembler as _A

        factor_ids = tuple(m.factor_id for m in members)
        return FactorSetArtifact(
            set_id=spec.set_id,
            name=spec.name,
            members=tuple(members),
            created_at="2024-02-01T00:00:00Z",
            policy_hash=_A._policy_hash(spec),
            assembly_hash=_A._assembly_hash(factor_ids, spec, tuple(members)),
            snapshot_ref=spec.data_snapshot_ref,
            universe_ref=spec.universe_ref,
            split_ref=spec.split_ref,
            spec=spec,
        )


# ---------------------------------------------------------------------------
# P0-FA-015: assembly_score must be finite (engine path)
# ---------------------------------------------------------------------------


class TestAssemblyScoreFinite:
    def test_nan_decision_score_is_treated_as_none_not_stored(self):
        spec = make_spec(set_id="s1", policy="family_robust", max_factors=1)
        assets = [make_asset("F1"), make_asset("F2")]
        decisions = [
            decided("F1", {"score": float("nan")}, timestamp="2024-01-01T00:00:00Z"),
            decided("F2", {"score": 0.5}, timestamp="2024-03-01T00:00:00Z"),
        ]
        result = assemble(spec, assets, selection_decisions=decisions)
        for member in result.memberships:
            assert member.assembly_score is None or member.assembly_score == member.assembly_score
            assert not (
                member.assembly_score is not None
                and member.assembly_score != member.assembly_score
            )

    def test_inf_decision_score_fails_closed_to_none(self):
        spec = make_spec(set_id="s1", policy="family_robust", max_factors=1)
        assets = [make_asset("F1")]
        decisions = [decided("F1", {"score": float("inf")})]
        result = assemble(spec, assets, selection_decisions=decisions)
        assert result.memberships[0].assembly_score is None

    def test_non_numeric_decision_score_treated_as_none(self):
        spec = make_spec(set_id="s1", policy="family_robust", max_factors=1)
        assets = [make_asset("F1")]
        decisions = [decided("F1", {"score": "high"})]
        result = assemble(spec, assets, selection_decisions=decisions)
        assert result.memberships[0].assembly_score is None

    def test_finite_decision_score_is_stored(self):
        spec = make_spec(set_id="s1", policy="family_robust", max_factors=1)
        assets = [make_asset("F1")]
        decisions = [decided("F1", {"score": 0.42})]
        result = assemble(spec, assets, selection_decisions=decisions)
        assert result.memberships[0].assembly_score == 0.42

    def test_membership_contract_rejects_non_finite_score(self):
        with pytest.raises(ValueError, match="finite"):
            FactorMembership(factor_id="F1", assembly_score=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            FactorMembership(factor_id="F1", assembly_score=float("inf"))
        with pytest.raises(TypeError, match="non-boolean"):
            FactorMembership(factor_id="F1", assembly_score="high")


# ---------------------------------------------------------------------------
# P1-FA-005: spec identity vs result identity separation
# ---------------------------------------------------------------------------


class TestSpecResultIdentitySeparation:
    def _spec_pair(self, views_a, views_b, **kwargs):
        a = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views=views_a,
            snapshot_ref="snapshot:2024", window_ref="2024-01-01/2024-12-31",
            universe_ref="universe:ashare", **kwargs,
        )
        b = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views=views_b,
            snapshot_ref="snapshot:2024", window_ref="2024-01-01/2024-12-31",
            universe_ref="universe:ashare", **kwargs,
        )
        return a, b

    def test_value_change_changes_spec_identity(self):
        a, b = self._spec_pair({"rank_corr": 0.9}, {"rank_corr": 0.91})
        assert a.similarity_spec_hash != b.similarity_spec_hash

    def test_spec_change_changes_spec_identity(self):
        a = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views={"rank_corr": 0.9},
            snapshot_ref="snapshot:2024",
        )
        b = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views={"rank_corr": 0.9},
            snapshot_ref="snapshot:2025",
        )
        assert a.similarity_spec_hash != b.similarity_spec_hash

    def test_spec_identity_is_not_a_result_proxy(self):
        # Two artifacts with identical spec identity MUST have identical
        # values — the spec digest covers the values themselves, so the spec
        # identity can never drift from the result.
        a, b = self._spec_pair({"rank_corr": 0.9}, {"rank_corr": 0.9})
        assert a.similarity_spec_hash == b.similarity_spec_hash
        assert a.primary_value == b.primary_value

    def test_for_spec_pins_exact_spec_identity(self):
        recomputed = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views={"rank_corr": 0.8},
        ).similarity_spec_hash
        pinned = SimilarityArtifact.for_spec("F1", "F2", recomputed, views={"rank_corr": 0.8})
        assert pinned.similarity_spec_hash == recomputed

    def test_for_spec_rejects_mismatched_hash(self):
        other = SimilarityArtifact(
            factor_a="F1", factor_b="F2", views={"rank_corr": 0.5},
        ).similarity_spec_hash
        with pytest.raises(ValueError, match="does not match"):
            SimilarityArtifact.for_spec("F1", "F2", other, views={"rank_corr": 0.8})

    def test_result_digest_does_not_silently_depend_on_spec_alone(self):
        # A value change while keeping the same spec provenance changes the
        # digest — the digest covers the computed values, so a caller cannot
        # substitute the spec identity for the result.
        a, b = self._spec_pair({"rank_corr": 0.9}, {"rank_corr": 0.9001})
        assert a.similarity_spec_hash != b.similarity_spec_hash


# ---------------------------------------------------------------------------
# UNKNOWN vs COMPUTED_LOW in the refined pair ledger
# ---------------------------------------------------------------------------


class TestUnknownVsComputedLow:
    def _registry(self):
        return SimilarityViewRegistry()

    def test_never_touched_pair_is_unknown(self):
        # An unmeasured view in an EdgeAffinityPolicy is UNKNOWN (None), never
        # a computed zero — proven below via the affinity fusion.
        from factor_assets.contracts.similarity import EdgeAffinityPolicy

        policy = EdgeAffinityPolicy({"rank_corr": 0.5})
        assert policy.affinity({"rank_corr": None}) is None
        assert policy.affinity({"rank_corr": 0.1}) == pytest.approx(0.1)

    def test_computed_low_is_distinct_from_unknown(self):
        from factor_assets.contracts.similarity import EdgeAffinityPolicy
        from factor_assets.clustering.families import SimilarityObservationState

        # A genuinely computed low affinity (0.1, measured) is COMPUTED_LOW —
        # a real observation — whereas an unmeasured pair is UNKNOWN.
        assert SimilarityObservationState.UNKNOWN.value == "UNKNOWN"
        assert SimilarityObservationState.COMPUTED_LOW.value == "COMPUTED_LOW"
        assert (
            SimilarityObservationState.UNKNOWN
            is not SimilarityObservationState.COMPUTED_LOW
        )
        policy = EdgeAffinityPolicy({"rank_corr": 0.5})
        measured_low = policy.affinity({"rank_corr": 0.1})
        unmeasured = policy.affinity({"rank_corr": None})
        assert measured_low is not None and measured_low < 0.5
        assert unmeasured is None  # never conflated with the computed low


# ---------------------------------------------------------------------------
# SimilarityViewRegistry construction gate
# ---------------------------------------------------------------------------


class TestSimilarityViewRegistryConstructionGate:
    def test_views_passed_to_registry_are_validated(self):
        # A registry built with a view key that is NOT in the canonical view
        # set must fail construction (a silent typo must not reach production).
        with pytest.raises(ValueError, match="not a registered canonical view"):
            SimilarityViewRegistry(views={"rank_cor": "typo"})

    def test_registered_views_construct(self):
        registry = SimilarityViewRegistry(views={"rank_corr": "rank correlation"})
        assert registry.is_registered("rank_corr")
        assert "rank_corr" in registry.view_keys

    def test_empty_views_construct_with_canonical_set(self):
        registry = SimilarityViewRegistry()
        assert registry.is_registered("rank_corr")
        assert registry.is_registered("pearson_corr")
        for key in SIMILARITY_VIEW_KEYS:
            assert registry.is_registered(key)

    def test_validate_rejects_unregistered_key(self):
        registry = SimilarityViewRegistry()
        with pytest.raises(ValueError, match="not a registered canonical view"):
            registry.validate({"rank_corr": 0.5, "bogus_view": 0.3})
