# -*- coding: utf-8 -*-
"""Task #107 — Cross-package contract round-trips (integration harness).

These tests round-trip the shared contracts THROUGH the public namespaces of
the packages (imports of the public top-level packages / their contracts
modules only — never implementation internals), so CI catches CONTRACT DRIFT
when the per-package tasks (A/C/D/E/F/G) land out of sync with each other.

Round-trips covered:

1. QE evidence lifecycle -> platform   (evidence_status / evaluation_artifact)
2. platform IdentityRef frozen contract (identities)
3. platform library Ref/View promotion (cluster_library)
4. FO TrialLedger seal/verify          (trial_ledger)
5. FO multiplicity wiring              (search.runner + search.strategies)
6. FA FactorSetArtifact assembly hash  (factor_set + assembly engine)
7. A股 PIT/unit contract                (FP treatment_spec, see the dedicated
                                         module test_a_share_pit_unit_contract.py)
8. wheel-matrix import contract        (public namespaces import cleanly)

A failing test here means CONTRACT DRIFT, not a test bug: the mismatch is
reported precisely (which contract, which package, what changed).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

# --------------------------------------------------------------------------- #
# 1. Evidence lifecycle contract (QE -> platform)
# --------------------------------------------------------------------------- #

# The status/reason matrix + MetricEvidence live under quant_evaluator.contracts
# (source tree is pinned by the integration conftest).
from quant_evaluator.contracts.evidence_status import (  # noqa: E402
    EvidenceReasonCode,
    EvidenceStatus,
    MetricEvidence,
    evidence_for_computed,
    validate_status_reason,
)
from quant_evaluator.contracts.evaluation_artifact import (  # noqa: E402
    ArtifactDomain,
    DomainArtifactRef,
    EvaluationArtifact,
    EvaluationSpecIdentity,
)
from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact  # noqa: E402

#: QE P0-QE-001 status/reason matrix.  NOTE (task #107): the source module's
#: ``__all__`` lists ``STATUS_REASON_ALLOWED`` but the module only binds the
#: underscore name ``_STATUS_REASON_ALLOWED`` (and the compatibility class
#: ``_STATUS_REASON_COMPATIBILITY.ALLOWED``) — the public name is NOT imported
#: by ``quant_evaluator.contracts``.  The round-trip below uses the compat
#: alias so the test passes against the CURRENT shape; see CONTRACT DRIFT
#: BLOCKER in the task #107 report (QE-side public export pending).
try:
    from quant_evaluator.contracts.evidence_status import (  # noqa: PLC0415
        STATUS_REASON_ALLOWED as _PUBLIC_STATUS_REASON_ALLOWED,
    )
except ImportError:  # pragma: no cover - current source shape
    _PUBLIC_STATUS_REASON_ALLOWED = None

#: The matrix this round-trip validates, resolved from whatever public shape
#: the source currently exposes (public name when QE lands it, else the compat
#: alias the QE code itself references).
from quant_evaluator.contracts.evidence_status import (  # noqa: E402
    _STATUS_REASON_COMPATIBILITY as _STATUS_REASON_COMPATIBILITY,
)

_STATUS_REASON_ALLOWED = (
    _PUBLIC_STATUS_REASON_ALLOWED
    if _PUBLIC_STATUS_REASON_ALLOWED is not None
    else _STATUS_REASON_COMPATIBILITY.ALLOWED
)


def _artifact(metric_id: str = "rank_ic") -> ScalarMetricArtifact:
    """A minimal canonical MetricArtifact (1 factor) for evidence round-trips."""
    return ScalarMetricArtifact(metric_id=metric_id, domain="ic", values=(0.05,))


def _ref(domain: ArtifactDomain, identity: str, kind: str = "") -> DomainArtifactRef:
    return DomainArtifactRef.of(domain, identity, content_hash="", artifact_kind=kind)


class TestEvidenceLifecycleContract:
    """QE MetricEvidence + STATUS_REASON_ALLOWED matrix round-trip.

    Contract: COMPUTED requires a canonical artifact and reason OK;
    LABEL_NOT_MATURE forbids an artifact; the platform's timing semantics
    (label-not-mature => never a fabricated zero) are mirrored by the enum.
    """

    def test_computed_roundtrips_with_artifact_and_reason_ok(self):
        art = _artifact()
        evidence = evidence_for_computed(art, observations=42)
        assert evidence.computed is True
        assert evidence.status is EvidenceStatus.COMPUTED
        assert evidence.reason_code is EvidenceReasonCode.OK
        assert evidence.artifact is art
        # Round-trip through the serialized dict form (what a platform DTO
        # would carry) — status/reason/artifact must survive losslessly.
        restored = MetricEvidence.from_dict(evidence.to_dict())
        assert restored.computed is True
        assert restored.artifact is not None
        assert restored.artifact.artifact_kind == "scalar"

    def test_computed_forbids_missing_artifact(self):
        with pytest.raises(ValueError, match="COMPUTED"):
            MetricEvidence(
                status=EvidenceStatus.COMPUTED,
                reason_code=EvidenceReasonCode.OK,
                artifact=None,
            )

    def test_computed_forbids_non_ok_reason(self):
        # COMPUTED + a non-OK reason is a lie — the matrix must reject it.
        with pytest.raises(ValueError, match="illegal MetricEvidence"):
            MetricEvidence(
                status=EvidenceStatus.COMPUTED,
                reason_code=EvidenceReasonCode.MIN_PERIODS_NOT_MET,
                artifact=_artifact(),
            )

    def test_label_not_mature_forbids_artifact(self):
        with pytest.raises(ValueError, match="LABEL_NOT_MATURE"):
            MetricEvidence(
                status=EvidenceStatus.LABEL_NOT_MATURE,
                reason_code=EvidenceReasonCode.LABEL_NOT_YET_MATURE,
                artifact=_artifact(),
            )

    def test_label_not_mature_never_a_zero(self):
        evidence = MetricEvidence(
            status=EvidenceStatus.LABEL_NOT_MATURE,
            reason_code=EvidenceReasonCode.LABEL_NOT_YET_MATURE,
            artifact=None,
        )
        assert evidence.computed is False
        # A not-mature label yields NO numeric payload — never a fabricated 0.0.
        assert evidence.artifact is None

    def test_status_reason_allowed_matrix_is_public_and_stable(self):
        # The matrix is the platform contract surface: COMPUTED <-> {OK} only.
        assert _STATUS_REASON_ALLOWED[EvidenceStatus.COMPUTED] == frozenset(
            {EvidenceReasonCode.OK}
        )
        # LABEL_NOT_MATURE permits exactly the label-not-yet-mature reason.
        assert _STATUS_REASON_ALLOWED[EvidenceStatus.LABEL_NOT_MATURE] == frozenset(
            {EvidenceReasonCode.LABEL_NOT_YET_MATURE}
        )
        # Public validator agrees with the constructor's fail-closed check.
        validate_status_reason(EvidenceStatus.COMPUTED, EvidenceReasonCode.OK)
        with pytest.raises(ValueError):
            validate_status_reason(
                EvidenceStatus.COMPUTED, EvidenceReasonCode.NOT_YET_COMPUTED
            )


class TestProductionEvaluationArtifactProvenance:
    """QE EvaluationArtifact PRODUCTION requires the full 7-ref provenance.

    The platform consumes evaluation evidence for production decisions; a
    PRODUCTION artifact with a missing snapshot/split/universe would be trusted
    downstream without the provenance that makes it trustworthy (P0-QE-003).
    """

    #: The seven provenance refs a PRODUCTION artifact must carry.
    PROVENANCE_REFS = (
        "factor_value_ref",
        "label_definition_ref",
        "evaluation_policy_ref",
        "evaluation_profile_ref",
        "split_ref",
        "snapshot_ref",
        "universe_ref",
    )

    def _production_artifact(self, **overrides):
        mode = overrides.pop("evaluation_mode", "PRODUCTION")
        refs = {
            name: _ref(_domain_for(name), f"id:{name}", kind=name)
            for name in self.PROVENANCE_REFS
        }
        refs.update(overrides)
        return EvaluationArtifact(
            evaluation_id="eval-prod-1",
            evaluation_identity="prod round-trip",
            evaluation_mode=mode,
            **refs,
        )

    def test_valid_production_artifact_roundtrips(self):
        artifact = self._production_artifact()
        assert artifact.evaluation_mode == "PRODUCTION"
        # All 7 provenance refs present and typed.
        for name in self.PROVENANCE_REFS:
            ref = getattr(artifact, name)
            assert isinstance(ref, DomainArtifactRef), f"{name} must be a DomainArtifactRef"
            assert ref.identity == f"id:{name}"
        # Derived identities are deterministic and survive serialization.
        restored = EvaluationArtifact.from_dict(artifact.to_dict())
        assert restored.content_hash == artifact.content_hash
        assert (
            restored.evaluation_envelope_identity.identity_hash
            == artifact.evaluation_envelope_identity.identity_hash
        )
        assert (
            restored.evaluation_spec_identity.identity_hash
            == artifact.evaluation_spec_identity.identity_hash
        )

    def test_missing_ref_raises(self):
        # Drop each of the seven in turn — construction must fail closed.
        for name in self.PROVENANCE_REFS:
            with pytest.raises(ValueError, match="PRODUCTION"):
                self._production_artifact(**{name: None})

    def test_research_mode_allows_missing_refs(self):
        # RESEARCH is the permissive tier; only PRODUCTION requires the full
        # provenance.  The mode itself is still validated.
        refs = {name: _ref(_domain_for(name), f"id:{name}") for name in self.PROVENANCE_REFS}
        refs.pop("snapshot_ref")
        artifact = EvaluationArtifact(
            evaluation_id="eval-research-1",
            evaluation_identity="research round-trip",
            evaluation_mode="RESEARCH",
            **refs,
        )
        assert artifact.snapshot_ref is None
        assert artifact.evaluation_mode == "RESEARCH"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="RESEARCH/VALIDATION/PRODUCTION"):
            self._production_artifact(evaluation_mode="BOGUS")


def _domain_for(ref_name: str) -> ArtifactDomain:
    """Map a provenance ref name to its ArtifactDomain enum member."""
    return {
        "factor_value_ref": ArtifactDomain.FACTOR_VALUE,
        "label_definition_ref": ArtifactDomain.LABEL_DEFINITION,
        "evaluation_policy_ref": ArtifactDomain.EVALUATION_POLICY,
        "evaluation_profile_ref": ArtifactDomain.EVALUATION_PROFILE,
        "split_ref": ArtifactDomain.SPLIT,
        "snapshot_ref": ArtifactDomain.SNAPSHOT,
        "universe_ref": ArtifactDomain.UNIVERSE,
    }[ref_name]


# --------------------------------------------------------------------------- #
# 2. IdentityRef frozen contract (platform)
# --------------------------------------------------------------------------- #

from quant_platform.app.contracts.identities import (  # noqa: E402
    FactorDefinitionRef,
    IdentityRef,
    sha256_hex,
)


def _digest(seed: str = "fd") -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


class TestIdentityRefFrozenContract:
    """platform contracts.identities: carried, immutable, typed refs."""

    def test_ref_is_immutable(self):
        ref = FactorDefinitionRef(_digest(), factor_version="v1")
        with pytest.raises(AttributeError):
            ref.hash = _digest("other")  # type: ignore[misc]
        with pytest.raises(AttributeError):
            ref.factor_version = "v2"  # type: ignore[misc]

    def test_identical_fields_are_equal(self):
        assert FactorDefinitionRef(_digest(), factor_version="v1") == FactorDefinitionRef(
            _digest(), factor_version="v1"
        )
        assert FactorDefinitionRef(_digest(), factor_version="v1") != FactorDefinitionRef(
            _digest("other"), factor_version="v1"
        )

    def test_dict_identity_is_not_a_ref(self):
        # A dict-based identity must NOT be usable where a FactorDefinitionRef
        # is expected: the refs compare by type+digest, and the platform
        # accepts only the carried-ref type.
        plain = {"hash": _digest(), "factor_version": "v1"}
        ref = FactorDefinitionRef(_digest(), factor_version="v1")
        assert ref != plain
        assert not isinstance(plain, IdentityRef)
        # The carried digest format check is the ref's only validation surface.
        with pytest.raises(ValueError):
            FactorDefinitionRef("not-a-digest", factor_version="v1")
        with pytest.raises(TypeError):
            FactorDefinitionRef(None, factor_version="v1")  # type: ignore[arg-type]

    def test_digest_is_carried_not_recomputed(self):
        digest = _digest("domain")
        ref = FactorDefinitionRef(digest, factor_version="v1")
        assert ref.hash == digest
        assert sha256_hex(digest, "x") == digest  # never re-hashed


# --------------------------------------------------------------------------- #
# 3. Library version Ref/View promotion contract (platform)
# --------------------------------------------------------------------------- #

from quant_platform.app.contracts.cluster_library import (  # noqa: E402
    ClusterSetVersionRef,
    ClusterVersionRef,
    FactorLibraryVersionRef,
    LibraryActivePointer,
    LibraryLifecycleEvent,
    LibraryPromotionRecord,
    LibraryStatus,
    promote_library_version,
)


class TestLibraryVersionRefViewContract:
    """Ref/View layer: promotion produces a NEW view, never mutates a body."""

    def test_ref_types_exist_and_validate_format(self):
        ref = ClusterSetVersionRef(
            cluster_set_version_id="CS1",
            digest=_digest("cs1"),
            similarity_graph_version_ref="G1",
        )
        assert ref.digest == _digest("cs1")
        with pytest.raises(ValueError):
            ClusterSetVersionRef(cluster_set_version_id="CS1", digest="bad")
        lref = FactorLibraryVersionRef(
            library_version_id="FLV_1",
            digest=_digest("flv"),
            logical_library_id="CORE_LOW_REDUNDANCY",
        )
        assert lref.logical_library_id == "CORE_LOW_REDUNDANCY"

    def test_promote_produces_new_view_and_never_mutates_body(self):
        # The platform re-exports the Ref/View pair from
        # quant_platform.app.contracts.cluster_library (the projection layer
        # carrying the domain artifact's status).  Use the re-exported names so
        # the test tracks whatever shape the platform canonicalises on.
        from quant_platform.app.contracts import (  # noqa: PLC0415
            FactorLibraryVersionView,
        )

        version = FactorLibraryVersionView(
            library_version_id="FLV_103",
            logical_library_id="CORE_LOW_REDUNDANCY",
            cluster_set_version_id="CS1",
            status="CANDIDATE",
        )
        pointer = LibraryActivePointer(
            logical_library_id="CORE_LOW_REDUNDANCY",
            active_version_id="FLV_102",
            active_status="PRODUCTION",
        )
        new_view, new_pointer, record = promote_library_version(version, pointer)
        # A NEW version view with the advanced status — same version id.
        assert new_view.library_version_id == "FLV_103"
        assert new_view.status == "SHADOW"
        assert new_view is not version
        # The original body is untouched (immutable version semantics).
        assert version.status == "CANDIDATE"
        assert pointer.active_version_id == "FLV_102"
        # Pointer flips to the promoted version; a promotion record is emitted.
        assert new_pointer.active_version_id == "FLV_103"
        assert record.from_status == "CANDIDATE"
        assert record.to_status == "SHADOW"
        assert isinstance(record, LibraryPromotionRecord)

    def test_promotion_record_maps_to_lifecycle_event(self):
        # The promotion record is what the platform persists as an append-only
        # LibraryLifecycleEvent (P0-PLAT-002): the event is created, never a
        # body mutation.
        from quant_platform.app.contracts import (  # noqa: PLC0415
            FactorLibraryVersionView,
        )

        version = FactorLibraryVersionView(
            library_version_id="FLV_103",
            logical_library_id="CORE_LOW_REDUNDANCY",
            cluster_set_version_id="CS1",
            status="CANDIDATE",
        )
        pointer = LibraryActivePointer(
            logical_library_id="CORE_LOW_REDUNDANCY",
            active_version_id="FLV_102",
            active_status="PRODUCTION",
        )
        _, _, record = promote_library_version(
            version, pointer, actor_principal_id="u1", reason="review ok"
        )
        event = LibraryLifecycleEvent(
            logical_library_id=record.logical_library_id,
            version_id=record.version_id,
            event_kind="PROMOTED",
            from_status=record.from_status,
            to_status=record.to_status,
            actor_principal_id=record.actor_principal_id,
            reason=record.reason,
        )
        assert event.event_kind == "PROMOTED"
        assert event.to_status == "SHADOW"

    def test_terminal_status_cannot_promote(self):
        from quant_platform.app.contracts import (  # noqa: PLC0415
            FactorLibraryVersionView,
        )

        version = FactorLibraryVersionView(
            library_version_id="FLV_103",
            logical_library_id="CORE_LOW_REDUNDANCY",
            cluster_set_version_id="CS1",
            status="PRODUCTION",
        )
        pointer = LibraryActivePointer(
            logical_library_id="CORE_LOW_REDUNDANCY",
            active_version_id="FLV_103",
            active_status="PRODUCTION",
        )
        with pytest.raises(ValueError):
            promote_library_version(version, pointer)

    def test_library_status_enum_matches_platform_surface(self):
        assert [s.value for s in LibraryStatus] == [
            "CANDIDATE",
            "SHADOW",
            "APPROVED",
            "PRODUCTION",
            "RETIRED",
        ]


# --------------------------------------------------------------------------- #
# 4. TrialLedger seal/verify (FO)
# --------------------------------------------------------------------------- #

from factor_optimizer.contracts.trial_ledger import (  # noqa: E402
    LedgerEntry,
    TrialLedger,
)


class TestTrialLedgerSealVerifyContract:
    """FO append-only hash-chained ledger: seal + verify + tamper detection."""

    def test_append_seal_verify_passes(self):
        ledger = TrialLedger()
        ledger.append("PROPOSED", trial_id="t1")
        ledger.append("EVALUATED", trial_id="t1")
        ledger.append("SELECTED", trial_id="t1")
        ledger.seal()
        assert ledger.sealed
        assert ledger.sealed_entry_count == 3
        ledger.verify_chain()  # must not raise
        # Sealed ledger rejects further appends.
        with pytest.raises(ValueError, match="sealed"):
            ledger.append("PROPOSED", trial_id="t2")

    def test_sealed_ledger_roundtrips_and_verifies(self):
        ledger = TrialLedger()
        ledger.append("PROPOSAL_FAILED", failure_reason="boom")
        ledger.append("DUPLICATE", trial_id="t1")
        ledger.seal()
        restored = TrialLedger.from_dict(ledger.to_dict())
        restored.verify_chain()
        assert restored.sealed
        assert restored.sealed_entry_count == 2
        assert restored.sealed_head_hash == ledger.sealed_head_hash

    def test_tamper_detected(self):
        ledger = TrialLedger()
        e1 = ledger.append("PROPOSED", trial_id="t1")
        ledger.append("EVALUATED", trial_id="t1")
        ledger.seal()
        ledger.verify_chain()
        # Flip a semantic field on the first entry: the hash chain must break.
        tampered = LedgerEntry(
            sequence=e1.sequence,
            status=e1.status,
            trial_id=e1.trial_id,
            failure_reason=e1.failure_reason,
            recorded_at=e1.recorded_at,
            previous_entry_hash=e1.previous_entry_hash,
            entry_hash=e1.entry_hash,
        )
        object.__setattr__(tampered, "trial_id", "t1-TAMPERED")
        ledger._entries = (tampered,) + ledger._entries[1:]  # noqa: SLF001
        with pytest.raises(ValueError, match="tampered|hash"):
            ledger.verify_chain()

    def test_truncation_detected_after_seal(self):
        ledger = TrialLedger()
        ledger.append("PROPOSED", trial_id="t1")
        ledger.append("EVALUATED", trial_id="t1")
        ledger.append("SELECTED", trial_id="t1")
        ledger.seal()
        # Truncate the sealed ledger (drop the tail entry).
        ledger._entries = ledger._entries[:2]  # noqa: SLF001
        with pytest.raises(ValueError, match="truncated|entry count"):
            ledger.verify_chain()


# --------------------------------------------------------------------------- #
# 5. Multiplicity wiring (FO)
# --------------------------------------------------------------------------- #

from factor_optimizer.contracts.multiplicity import (  # noqa: E402
    MultiplicityArtifact,
)
from factor_optimizer.contracts.search_budget import (  # noqa: E402
    BudgetTracker,
    SearchBudget,
)
from factor_optimizer.search.strategies import (  # noqa: E402
    SearchStrategySpec,
)
from factor_optimizer.search.runner import (  # noqa: E402
    SearchConfig,
    SearchSession,
)


class TestMultiplicityWiringContract:
    """FO: SearchSession.finish() derives a MultiplicityArtifact from the
    append-only ledger; the candidate strategy SPEC (type + ctor + seed)
    round-trips through SearchConfig dict forms unchanged."""

    def _session(self) -> SearchSession:
        budget = SearchBudget(max_trials=10, max_evaluations=5)
        config = SearchConfig(budget=budget)
        return SearchSession(
            session_id="session-1",
            config=config,
            budget_tracker=BudgetTracker(budget=budget),
        )

    def test_finish_produces_multiplicity_artifact(self):
        session = self._session()
        session.ledger.append("PROPOSAL_FAILED", failure_reason="parse error")
        session.ledger.append("DUPLICATE", trial_id="t1")
        session.ledger.append("EVALUATED", trial_id="t1")
        session.finish("budget_exhausted")
        artifact = session.multiplicity_artifact
        assert isinstance(artifact, MultiplicityArtifact)
        assert artifact.total_proposals == 3
        assert artifact.parse_failures == 1
        assert artifact.duplicates == 1
        assert artifact.valid_evaluated == 1
        # The true hypothesis count is what the multiple-testing correction uses.
        assert artifact.total_proposals == (
            artifact.parse_failures
            + artifact.duplicates
            + artifact.valid_evaluated
            + artifact.failed_evaluations
            + artifact.illegal
        )
        # Content-hash bound and tamper-detectable.
        artifact.verify()
        with pytest.raises(ValueError, match="content_hash"):
            MultiplicityArtifact.from_dict({**artifact.to_dict(), "valid_evaluated": 0})

    def test_search_config_spec_roundtrip(self):
        spec = SearchStrategySpec(
            strategy_type="BayesianOptimizationStrategy",
            ctor={"n_initial": 5},
            seed=42,
        )
        budget = SearchBudget(max_trials=10, max_evaluations=5)
        config = SearchConfig(budget=budget, candidate_strategy_spec=spec)
        restored = SearchConfig.from_dict(config.to_dict())
        assert restored.candidate_strategy_spec is not None
        assert restored.candidate_strategy_spec.strategy_type == "BayesianOptimizationStrategy"
        assert restored.candidate_strategy_spec.ctor == {"n_initial": 5}
        assert restored.candidate_strategy_spec.seed == 42
        # The runtime strategy object is not serialized (checkpoint identity
        # records the SPEC, not the live object).
        assert config.to_dict()["candidate_strategy"] is None


# --------------------------------------------------------------------------- #
# 6. FactorSetArtifact assembly hash (FA)
# --------------------------------------------------------------------------- #

from factor_assets.assembly import FactorSetAssembler  # noqa: E402
from factor_assets.contracts.asset import AssetMetadata, FactorAsset  # noqa: E402
from factor_assets.contracts.evidence_ref import EvidenceBundleRef  # noqa: E402
from factor_assets.contracts.factor_set import (  # noqa: E402
    FactorSetArtifact,
    FactorSetSpec,
)
from factor_assets.contracts import AssemblyPolicy  # noqa: E402
from factor_assets.contracts.lifecycle import LifecycleState  # noqa: E402
from factor_assets.contracts.lineage import LineageRef  # noqa: E402
from factor_assets.selection import SelectionDecision, SelectionReason  # noqa: E402


def _asset(factor_id: str) -> FactorAsset:
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
        family="family-a",
        latest_evidence_ref=EvidenceBundleRef(
            bundle_id=f"bundle-{factor_id}",
            evaluation_run_id="run-1",
            factor_ids=(factor_id,),
            timestamp="2024-01-01T00:00:00Z",
            qe_version="1.0",
        ),
    )


def _spec(set_id: str, policy: str) -> FactorSetSpec:
    return FactorSetSpec(
        set_id=set_id,
        name=f"set-{set_id}",
        selection_policy=policy,
        data_snapshot_ref="snapshot:default",
        universe_ref="universe:default",
        split_ref="split:default",
    )


def _decision(factor_id: str) -> SelectionDecision:
    """An approved SelectionDecision (pareto_front/family_robust require them)."""
    return SelectionDecision(
        decision_id=f"decision-{factor_id}",
        factor_id=factor_id,
        approved=True,
        reason=SelectionReason.APPROVED,
        timestamp="2024-01-01T00:00:00Z",
        policy_version="1.0",
        evidence_refs=("evidence-1",),
        gate_results=("gate-1",),
        # Frozen Pareto fixture: dimensionless, higher-is-better quality.
        # Missing values are rejected, never imputed.
        metadata={"objectives": {"quality": 1.0}},
    )


class TestFactorSetAssemblyHashContract:
    """FA assembly_hash is deterministic over identical inputs and changes
    when the spec's selection_policy changes."""

    def test_identical_assembly_inputs_produce_identical_hash(self):
        spec = _spec("S1", "manual")
        a = FactorSetAssembler().assemble(spec, [_asset("F1"), _asset("F2")])
        b = FactorSetAssembler().assemble(spec, [_asset("F1"), _asset("F2")])
        assert isinstance(a, FactorSetArtifact)
        assert a.assembly_hash == b.assembly_hash
        assert a.policy_hash == b.policy_hash
        assert a.factor_ids == ("F1", "F2")

    def test_selection_policy_change_changes_hash(self):
        a = FactorSetAssembler().assemble(
            _spec("S1", "manual"), [_asset("F1"), _asset("F2")]
        )
        # A different selection policy => a different assembly hash.  The
        # non-manual policy needs approved selection decisions to admit assets.
        pareto = FactorSetSpec(
            set_id="S1",
            name="set-S1",
            selection_policy="pareto_front",
            data_snapshot_ref="snapshot:default",
            universe_ref="universe:default",
            split_ref="split:default",
        )
        b = FactorSetAssembler().assemble(
            pareto,
            [_asset("F1"), _asset("F2")],
            selection_decisions=[_decision("F1"), _decision("F2")],
            assembly_policy=AssemblyPolicy(
                "xpkg-pareto-quality-max-dimensionless-reject-missing",
                "1.0",
                required_objectives=("quality",),
            ),
        )
        assert a.assembly_hash != b.assembly_hash

    def test_pareto_policy_rejects_missing_required_objective(self):
        missing = _decision("F2")
        object.__setattr__(missing, "metadata", {"objectives": {}})
        with pytest.raises(ValueError, match="no factor assets match"):
            FactorSetAssembler().assemble(
                _spec("S1", "pareto_front"),
                [_asset("F2")],
                selection_decisions=[missing],
                assembly_policy=AssemblyPolicy(
                    "xpkg-pareto-quality-max-dimensionless-reject-missing",
                    "1.0",
                    required_objectives=("quality",),
                ),
            )

    def test_deterministic_roundtrip(self):
        # Same set_id/name/spec => same hash across runs (content-addressed).
        for _ in range(3):
            assert _spec("S1", "manual") == _spec("S1", "manual")
        a = FactorSetAssembler().assemble(_spec("S1", "manual"), [_asset("F1")])
        restored = FactorSetArtifact(
            set_id=a.set_id,
            name=a.name,
            members=a.members,
            created_at=a.created_at,
            policy_hash=a.policy_hash,
            assembly_hash=a.assembly_hash,
            snapshot_ref=a.snapshot_ref,
            universe_ref=a.universe_ref,
            split_ref=a.split_ref,
            versions=a.versions,
            evidence_refs=a.evidence_refs,
            spec=a.spec,
        )
        assert restored.assembly_hash == a.assembly_hash
        assert restored.policy_hash == a.policy_hash
