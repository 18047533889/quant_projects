"""Tests for the SimilarityArtifact and FactorAdmissionArtifact contracts."""

import dataclasses

import pytest

from factor_assets.contracts.admission import (
    ADMISSION_DECISION_APPROVED,
    ADMISSION_DECISION_REJECTED,
    ADMISSION_DECISION_SHADOWED,
    AdmissionDecision,
    FactorAdmissionArtifact,
)
from factor_assets.contracts.similarity import (
    CORRELATION_METRIC_VIEWS,
    DEFAULT_SIMILARITY_VIEW,
    SIMILARITY_VIEW_KEYS,
    SimilarityArtifact,
    SimilarityMetricSpecError,
    SimilarityView,
    is_unknown_identity,
    metric_for_view,
    resolve_metric_view,
)
from factor_assets.similarity import SimilarityMethod, SimilarityResult
from factor_assets.selection import SelectionDecision, SelectionReason


def _artifact(
    a="F001", b="F002", views=None, snapshot="snapshot:2024-08", window="2024-01-01/2024-12-31",
    universe="universe:ashare", producer="qe", created_at="2024-08-01T00:00:00Z",
    primary_view=DEFAULT_SIMILARITY_VIEW,
):
    return SimilarityArtifact(
        factor_a=a,
        factor_b=b,
        views=views if views is not None else {"rank_corr": 0.9, "pnl_corr": 0.7},
        snapshot_ref=snapshot,
        window_ref=window,
        universe_ref=universe,
        producer=producer,
        created_at=created_at,
        primary_view=primary_view,
    )


def _admission(factor_id="F001", decision=AdmissionDecision.APPROVED, **overrides):
    defaults = dict(
        factor_id=factor_id,
        decision=decision,
        quality=0.8,
        factor_version="v1",
        health_state_ref="lifecycle:APPROVED",
        similarity_ref="sim-hash-abc",
        novelty_ref="novelty-1",
        cluster_id=3,
        orientation=1,
        reason="APPROVED",
        evidence_refs=("bundle-F001",),
        gate_results=("gate-1",),
        policy_ref="policy:1.0",
        created_at="2024-08-01T00:00:00Z",
    )
    defaults.update(overrides)
    return FactorAdmissionArtifact(**defaults)


class TestSimilarityArtifactConstruction:
    def test_basic_construction(self):
        artifact = _artifact()
        assert artifact.factor_a == "F001"
        assert artifact.factor_b == "F002"
        assert artifact.views["rank_corr"] == 0.9
        assert artifact.snapshot_ref == "snapshot:2024-08"
        assert artifact.window_ref == "2024-01-01/2024-12-31"
        assert artifact.universe_ref == "universe:ashare"
        assert artifact.producer == "qe"
        assert artifact.primary_view == DEFAULT_SIMILARITY_VIEW == "rank_corr"

    def test_requires_factor_a(self):
        with pytest.raises(ValueError, match="factor_a"):
            _artifact(a="")

    def test_requires_factor_b(self):
        with pytest.raises(ValueError, match="factor_b"):
            _artifact(b="")

    def test_rejects_same_factor_pair(self):
        with pytest.raises(ValueError, match="must differ"):
            _artifact(a="F1", b="F1")

    def test_requires_at_least_one_non_none_view(self):
        with pytest.raises(ValueError, match="at least one non-None"):
            _artifact(views={"rank_corr": None, "pnl_corr": None})

    def test_rejects_non_finite_view(self):
        with pytest.raises(ValueError, match="finite"):
            _artifact(views={"rank_corr": float("nan")})
        with pytest.raises(ValueError, match="finite"):
            _artifact(views={"rank_corr": float("inf")})

    def test_rejects_bool_view(self):
        with pytest.raises(TypeError, match="non-boolean"):
            _artifact(views={"rank_corr": True})

    def test_mixed_none_views_allowed(self):
        artifact = _artifact(views={"rank_corr": 0.9, "pnl_corr": None})
        assert artifact.views["pnl_corr"] is None
        assert artifact.views["rank_corr"] == 0.9

    def test_allows_legacy_view_keys_beyond_canonical_set(self):
        artifact = _artifact(views={"rank_corr": 0.9, "custom_view": 0.4})
        assert artifact.views["custom_view"] == 0.4

    def test_primary_view_must_exist(self):
        with pytest.raises(ValueError, match="primary_view"):
            _artifact(views={"pnl_corr": 0.9}, primary_view="rank_corr")

    def test_primary_view_must_be_non_none(self):
        with pytest.raises(ValueError, match="primary_view"):
            _artifact(views={"rank_corr": None, "pnl_corr": 0.9})

    def test_primary_view_none_requires_single_view(self):
        artifact = _artifact(views={"rank_corr": 0.9}, primary_view=None)
        assert artifact.primary_value == 0.9
        with pytest.raises(ValueError, match="exactly one view"):
            _artifact(views={"rank_corr": 0.9, "pnl_corr": 0.8}, primary_view=None)

    def test_frozen(self):
        artifact = _artifact()
        with pytest.raises(dataclasses.FrozenInstanceError):
            artifact.factor_a = "X"  # type: ignore

    def test_default_snapshot_window_universe_are_none(self):
        artifact = SimilarityArtifact(factor_a="A", factor_b="B", views={"rank_corr": 0.5})
        assert artifact.snapshot_ref is None
        assert artifact.window_ref is None
        assert artifact.universe_ref is None

    def test_view_keys_constant(self):
        assert "rank_corr" in SIMILARITY_VIEW_KEYS
        assert "pnl_corr" in SIMILARITY_VIEW_KEYS
        assert "top_overlap" in SIMILARITY_VIEW_KEYS
        assert "bottom_overlap" in SIMILARITY_VIEW_KEYS
        assert "residual_similarity" in SIMILARITY_VIEW_KEYS
        assert "horizon_similarity" in SIMILARITY_VIEW_KEYS


class TestSimilarityArtifactHash:
    def test_hash_is_deterministic_and_content_sensitive(self):
        first = _artifact()
        second = _artifact()
        assert first.similarity_spec_hash == second.similarity_spec_hash

        # Different view value -> different hash.
        different_view = _artifact(views={"rank_corr": 0.91, "pnl_corr": 0.7})
        assert different_view.similarity_spec_hash != first.similarity_spec_hash

        # Different snapshot -> different hash.
        different_snapshot = _artifact(snapshot="snapshot:2024-09")
        assert different_snapshot.similarity_spec_hash != first.similarity_spec_hash

        # Different factor pair -> different hash.
        different_pair = _artifact(a="F001", b="F003")
        assert different_pair.similarity_spec_hash != first.similarity_spec_hash

    def test_hash_is_stable_across_producer_and_created_at(self):
        first = _artifact(producer="qe", created_at="2024-08-01T00:00:00Z")
        second = _artifact(producer="manual", created_at="2025-01-01T00:00:00Z")
        assert first.similarity_spec_hash == second.similarity_spec_hash

    def test_hash_is_stable_across_view_insertion_order(self):
        ordered = _artifact(views={"rank_corr": 0.9, "pnl_corr": 0.7})
        unordered = _artifact(views={"pnl_corr": 0.7, "rank_corr": 0.9})
        assert ordered.similarity_spec_hash == unordered.similarity_spec_hash

    def test_supplied_hash_is_preserved_when_it_matches(self):
        # A caller-supplied hash is only accepted when it equals the hash
        # recomputed from the views/provenance (identity is derived, not
        # self-reported).  Passing the recomputed hash round-trips exactly.
        recomputed = SimilarityArtifact(
            factor_a="F001", factor_b="F002", views={"rank_corr": 0.8}
        ).similarity_spec_hash
        artifact = SimilarityArtifact.for_spec(
            "F001", "F002", recomputed, views={"rank_corr": 0.8}
        )
        assert artifact.similarity_spec_hash == recomputed
        assert artifact.primary_value == 0.8

    def test_for_spec_rejects_mismatched_supplied_hash(self):
        # A caller must not self-report an arbitrary hash: deserialize with a
        # stored hash that does not match the recomputed spec hash fails closed.
        with pytest.raises(ValueError, match="does not match"):
            SimilarityArtifact.for_spec(
                "F001", "F002", "not-the-real-hash", views={"rank_corr": 0.8}
            )

    def test_for_spec_accepts_recomputed_hash(self):
        recomputed = SimilarityArtifact(
            factor_a="F001", factor_b="F002", views={"rank_corr": 0.8}
        ).similarity_spec_hash
        artifact = SimilarityArtifact.for_spec(
            "F001", "F002", recomputed, views={"rank_corr": 0.8}
        )
        assert artifact.similarity_spec_hash == recomputed

    def test_for_spec_rejects_empty_hash(self):
        # P0-12: ``for_spec`` exists to PIN a spec identity.  The empty string
        # is an unknown-identity token, so it must not silently fall back to
        # auto-computation — a caller who cannot supply the hash must call the
        # plain constructor (which derives it) instead of pinning nothing.
        with pytest.raises(Exception, match="UNKNOWN|similarity_spec_hash"):
            SimilarityArtifact.for_spec("F001", "F002", "", views={"rank_corr": 0.8})

    def test_primary_value_and_view_accessor(self):
        from factor_assets.contracts._frozen import FrozenMapping

        artifact = _artifact()
        assert artifact.primary_value == 0.9
        assert artifact.view("pnl_corr") == 0.7
        assert artifact.view("missing") is None
        assert isinstance(artifact.views, FrozenMapping)

    def test_to_dict_roundtrip(self):
        artifact = _artifact()
        data = artifact.to_dict()
        assert data["factor_a"] == "F001"
        assert data["views"] == {"rank_corr": 0.9, "pnl_corr": 0.7}
        assert data["similarity_spec_hash"] == artifact.similarity_spec_hash
        restored = SimilarityArtifact(
            factor_a=data["factor_a"],
            factor_b=data["factor_b"],
            views=data["views"],
            snapshot_ref=data["snapshot_ref"],
            window_ref=data["window_ref"],
            universe_ref=data["universe_ref"],
            metric=data["metric"],
            similarity_spec_hash=data["similarity_spec_hash"],
            primary_view=data["primary_view"],
            created_at=artifact.created_at,
            producer=artifact.producer,
        )
        assert restored == artifact

    def test_from_similarity_result_adaptation(self):
        result = SimilarityResult(
            factor_id_a="F001",
            factor_id_b="F002",
            similarity_score=0.85,
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
            universe_ref="US_500",
            period_start="2024-01-01",
            period_end="2024-12-31",
        )
        artifact = SimilarityArtifact.from_similarity_result(
            result, snapshot_ref="snapshot:2024-08"
        )
        # P0-12: the declared metric drives which view the score is stored
        # under — PEARSON maps to the ``pearson_corr`` view, not the Spearman
        # ``rank_corr`` view the legacy adapter silently used.
        assert artifact.metric == "pearson"
        assert artifact.views["pearson_corr"] == 0.85
        assert artifact.views.get("rank_corr") is None
        assert artifact.primary_value == 0.85
        assert artifact.universe_ref == "US_500"
        assert artifact.window_ref == "2024-01-01/2024-12-31"
        assert artifact.snapshot_ref == "snapshot:2024-08"
        assert artifact.producer == "legacy:SimilarityResult"

    def test_from_similarity_result_spearman_maps_to_rank_corr(self):
        result = SimilarityResult(
            factor_id_a="F001",
            factor_id_b="F002",
            similarity_score=0.8,
            method=SimilarityMethod.SPEARMAN,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
        )
        artifact = SimilarityArtifact.from_similarity_result(result)
        assert artifact.metric == "spearman"
        assert artifact.views["rank_corr"] == 0.8

    def test_from_similarity_result_kendall_maps_to_kendall_tau(self):
        result = SimilarityResult(
            factor_id_a="F001",
            factor_id_b="F002",
            similarity_score=0.6,
            method=SimilarityMethod.KENDALL,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
        )
        artifact = SimilarityArtifact.from_similarity_result(result)
        assert artifact.metric == "kendall"
        assert artifact.views["kendall_tau"] == 0.6

    def test_with_provenance_recomputes_hash(self):
        artifact = _artifact()
        augmented = artifact.with_provenance(
            snapshot_ref="snapshot:2025", universe_ref="universe:hs300"
        )
        assert augmented.snapshot_ref == "snapshot:2025"
        assert augmented.universe_ref == "universe:hs300"
        assert augmented.window_ref == artifact.window_ref
        assert augmented.factor_a == artifact.factor_a
        # Provenance change is a change to measurement identity: the spec hash
        # must be recomputed, not carried over stale.
        assert augmented.similarity_spec_hash != artifact.similarity_spec_hash
        recomputed = SimilarityArtifact(
            factor_a="F001",
            factor_b="F002",
            views={"rank_corr": 0.9, "pnl_corr": 0.7},
            snapshot_ref="snapshot:2025",
            window_ref="2024-01-01/2024-12-31",
            universe_ref="universe:hs300",
        )
        assert augmented.similarity_spec_hash == recomputed.similarity_spec_hash
        assert augmented.created_at == artifact.created_at

    def test_views_are_immutable_deep_frozen(self):
        from factor_assets.contracts._frozen import FrozenMapping

        artifact = _artifact()
        assert isinstance(artifact.views, FrozenMapping)
        with pytest.raises(TypeError):
            artifact.views["rank_corr"] = 0.5  # type: ignore[misc]

    def test_views_snapshot_isolated_from_caller_dict(self):
        source = {"rank_corr": 0.9}
        artifact = SimilarityArtifact(
            factor_a="A", factor_b="B", views=source
        )
        # Mutating the caller's dict must not mutate the artifact's snapshot.
        source["rank_corr"] = 0.0
        assert artifact.views["rank_corr"] == 0.9
        assert artifact.primary_value == 0.9

    def test_views_are_hashable(self):
        import copy
        import pickle

        artifact = _artifact()
        # DLIB-XPKG: mapping fields are hashable + deepcopy-safe (matching the
        # QE FrozenMapping convention), so artifact deepcopy/hash no longer
        # raise.  Hash is content-stable across equal artifacts.
        assert isinstance(hash(artifact), int)
        assert hash(artifact) == hash(_artifact())
        # Content-sensitive: a different view value yields a different hash.
        assert hash(artifact) != hash(_artifact(views={"rank_corr": 0.8, "pnl_corr": 0.7}))
        # deepcopy + pickle round-trips preserve content and hash.
        assert copy.deepcopy(artifact) == artifact
        assert pickle.loads(pickle.dumps(artifact)) == artifact


class TestSimilarityView:
    def test_similarity_view_construction(self):
        view = SimilarityView(key="rank_corr", value=0.9)
        assert view.key == "rank_corr"
        assert view.value == 0.9
        assert view.is_absolute is True

    def test_similarity_view_requires_key(self):
        with pytest.raises(ValueError, match="key"):
            SimilarityView(key="", value=0.5)

    def test_similarity_view_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            SimilarityView(key="rank_corr", value=float("nan"))


class TestSimilarityMetricMappingP0_12:
    """R55 audit P0-12: authoritative metric mapping + UNKNOWN identity fail-closed."""

    # --- (c) the mapping round-trips for pearson / spearman / kendall -------

    def test_metric_mapping_table(self):
        # The single authoritative mapping: metric -> view key.
        assert dict(CORRELATION_METRIC_VIEWS) == {
            "pearson": "pearson_corr",
            "spearman": "rank_corr",
            "kendall": "kendall_tau",
        }

    @pytest.mark.parametrize("metric,view", sorted(CORRELATION_METRIC_VIEWS.items()))
    def test_metric_mapping_round_trip(self, metric, view):
        # metric -> view and back must be an involution.
        assert resolve_metric_view(metric) == view
        assert metric_for_view(view) == metric

    def test_metric_mapping_view_aliases_are_accepted(self):
        # A caller that names the *view* where a metric is declared still
        # canonicalizes to the metric (no silent mislabel, no silent default).
        artifact = SimilarityArtifact(
            factor_a="A", factor_b="B",
            views={"pearson_corr": 0.4}, metric="pearson_corr",
            primary_view="pearson_corr",
        )
        assert artifact.metric == "pearson"

    # --- (e) an unknown metric name raises ----------------------------------

    def test_unknown_metric_name_raises(self):
        with pytest.raises(ValueError, match="unknown similarity metric"):
            resolve_metric_view("spearman2")
        with pytest.raises(ValueError, match="unknown similarity metric"):
            SimilarityArtifact(
                factor_a="A", factor_b="B",
                views={"rank_corr": 0.5}, metric="pearsonr",
            )

    def test_metric_none_is_not_silently_mapped(self):
        # metric=None stays legal (legacy artifacts) but declares nothing.
        artifact = SimilarityArtifact(factor_a="A", factor_b="B", views={"rank_corr": 0.5})
        assert artifact.metric is None

    # --- (d) declared != computed raises ------------------------------------

    def test_declared_metric_mismatching_view_raises(self):
        # "pearson" declared but the score lives in the Spearman view — the
        # artifact would record a metric the computation did not use.
        with pytest.raises(SimilarityMetricSpecError, match="FAIL CLOSED"):
            SimilarityArtifact(
                factor_a="A", factor_b="B",
                views={"rank_corr": 0.9}, metric="pearson",
            )
        # Symmetric: "spearman" declared but only the Pearson view is present.
        with pytest.raises(SimilarityMetricSpecError, match="FAIL CLOSED"):
            SimilarityArtifact(
                factor_a="A", factor_b="B",
                views={"pearson_corr": 0.9}, metric="spearman",
            )

    def test_declared_metric_matching_view_constructs(self):
        artifact = SimilarityArtifact(
            factor_a="A", factor_b="B",
            views={"pearson_corr": 0.9, "rank_corr": None}, metric="pearson",
            primary_view="pearson_corr",
        )
        assert artifact.metric == "pearson"
        # The spec hash is metric-sensitive: a Pearson measurement and a
        # Spearman measurement of the same pair are NOT the same spec.
        spearman = SimilarityArtifact(
            factor_a="A", factor_b="B",
            views={"rank_corr": 0.9, "pearson_corr": None}, metric="spearman",
        )
        assert spearman.similarity_spec_hash != artifact.similarity_spec_hash

    def test_declared_metric_with_unmeasured_view_raises(self):
        # The mapped view is present but None (never measured): the declared
        # metric would point at a value that does not exist.
        with pytest.raises(SimilarityMetricSpecError, match="absent or unmeasured"):
            SimilarityArtifact(
                factor_a="A", factor_b="B",
                views={"pearson_corr": None, "rank_corr": 0.9}, metric="pearson",
            )

    # --- (a) UNKNOWN for_spec is not constructible --------------------------

    @pytest.mark.parametrize("unknown", ["UNKNOWN", "unknown", " Unknown ", ""])
    def test_for_spec_unknown_hash_raises(self, unknown):
        with pytest.raises(SimilarityMetricSpecError, match="UNKNOWN"):
            SimilarityArtifact.for_spec(
                "F001", "F002", unknown, views={"rank_corr": 0.8}
            )

    @pytest.mark.parametrize(
        "field", ["snapshot_ref", "window_ref", "universe_ref"]
    )
    def test_unknown_provenance_identity_raises(self, field):
        with pytest.raises(SimilarityMetricSpecError, match="UNKNOWN"):
            SimilarityArtifact(
                factor_a="A", factor_b="B",
                views={"rank_corr": 0.5}, **{field: "UNKNOWN"},
            )

    def test_unknown_declared_metric_raises(self):
        with pytest.raises(SimilarityMetricSpecError, match="metric is UNKNOWN"):
            SimilarityArtifact(
                factor_a="A", factor_b="B",
                views={"rank_corr": 0.5}, metric="UNKNOWN",
            )

    def test_none_provenance_is_absent_not_unknown(self):
        # ``None`` means "this provenance dimension is absent" (encoded as an
        # empty field in the spec digest) — it is NOT the UNKNOWN token, so it
        # stays constructible for legacy/research artifacts.
        artifact = SimilarityArtifact(factor_a="A", factor_b="B", views={"rank_corr": 0.5})
        assert artifact.snapshot_ref is None

    # --- (b) a resolvable spec resolves to the concrete identity ------------

    def test_for_spec_resolved_identity_round_trips(self):
        recomputed = SimilarityArtifact(
            factor_a="F001", factor_b="F002", views={"rank_corr": 0.8}
        ).similarity_spec_hash
        artifact = SimilarityArtifact.for_spec(
            "F001", "F002", recomputed, views={"rank_corr": 0.8}
        )
        assert artifact.similarity_spec_hash == recomputed
        assert artifact.similarity_spec_key == recomputed[:16]
        assert not is_unknown_identity(artifact.similarity_spec_hash)

    def test_unknown_identity_helper(self):
        for token in ("UNKNOWN", "UNKNOWN_SPEC_HASH", "TBD", "PLACEHOLDER", "n/a", ""):
            assert is_unknown_identity(token) is True
        assert is_unknown_identity(None) is True
        for concrete in ("sha256:abcd", "snapshot:2024-08", "universe:ashare"):
            assert is_unknown_identity(concrete) is False
        assert is_unknown_identity(12345) is False

    def test_from_similarity_result_requires_a_metric(self):
        # A legacy result with no method cannot honestly declare any metric —
        # fail closed instead of defaulting to rank_corr/Spearman.
        class _Bare:
            factor_id_a = "F001"
            factor_id_b = "F002"
            similarity_score = 0.5

        with pytest.raises(SimilarityMetricSpecError, match="method"):
            SimilarityArtifact.from_similarity_result(_Bare())


class TestFactorAdmissionArtifact:
    def test_basic_construction(self):
        artifact = _admission()
        assert artifact.factor_id == "F001"
        assert artifact.decision is ADMISSION_DECISION_APPROVED
        assert artifact.quality == 0.8
        assert artifact.factor_version == "v1"
        assert artifact.health_state_ref == "lifecycle:APPROVED"
        assert artifact.similarity_ref == "sim-hash-abc"
        assert artifact.novelty_ref == "novelty-1"
        assert artifact.cluster_id == 3
        assert artifact.orientation == 1
        assert artifact.reason == "APPROVED"
        assert artifact.evidence_refs == ("bundle-F001",)
        assert artifact.gate_results == ("gate-1",)
        assert artifact.policy_ref == "policy:1.0"

    def test_frozen(self):
        artifact = _admission()
        with pytest.raises(dataclasses.FrozenInstanceError):
            artifact.factor_id = "X"  # type: ignore

    def test_requires_factor_id(self):
        with pytest.raises(ValueError, match="factor_id"):
            _admission(factor_id="")

    def test_requires_decision_enum(self):
        with pytest.raises(TypeError, match="AdmissionDecision"):
            _admission(decision="APPROVED")

    def test_approved_requires_evidence_refs(self):
        with pytest.raises(ValueError, match="evidence_refs"):
            _admission(evidence_refs=())

    def test_approved_requires_gate_results(self):
        with pytest.raises(ValueError, match="gate_results"):
            _admission(gate_results=())

    def test_rejected_does_not_require_evidence(self):
        artifact = _admission(decision=AdmissionDecision.REJECTED, evidence_refs=(), gate_results=())
        assert artifact.is_rejected
        assert not artifact.is_approved

    def test_shadowed_status(self):
        artifact = _admission(decision=AdmissionDecision.SHADOWED, evidence_refs=(), gate_results=())
        assert artifact.is_shadowed
        assert artifact.decision is ADMISSION_DECISION_SHADOWED

    def test_orientation_validation(self):
        with pytest.raises(ValueError, match="orientation"):
            _admission(orientation=0)
        with pytest.raises(ValueError, match="orientation"):
            _admission(orientation=2)

    def test_quality_must_be_finite(self):
        with pytest.raises(ValueError, match="finite"):
            _admission(quality=float("nan"))

    def test_decision_properties(self):
        assert _admission().is_approved
        assert not _admission().is_rejected
        assert ADMISSION_DECISION_APPROVED.value == "APPROVED"
        assert ADMISSION_DECISION_REJECTED.value == "REJECTED"
        assert ADMISSION_DECISION_SHADOWED.value == "SHADOWED"


class TestFactorAdmissionArtifactHash:
    def test_hash_is_deterministic_and_content_sensitive(self):
        first = _admission()
        second = _admission()
        assert first.content_hash == second.content_hash

        changed_cluster = _admission(cluster_id=4)
        assert changed_cluster.content_hash != first.content_hash

        changed_health = _admission(health_state_ref="lifecycle:PRODUCTION_READY")
        assert changed_health.content_hash != first.content_hash

        changed_quality = _admission(quality=0.9)
        assert changed_quality.content_hash != first.content_hash

        changed_reason = _admission(reason="SHADOWED_COEXIST")
        assert changed_reason.content_hash != first.content_hash

    def test_hash_is_stable_across_created_at(self):
        first = _admission(created_at="2024-08-01T00:00:00Z")
        second = _admission(created_at="2025-01-01T00:00:00Z")
        assert first.content_hash == second.content_hash

    def test_hash_changes_with_gate_results(self):
        first = _admission()
        second = _admission(gate_results=("gate-1", "gate-2"))
        assert second.content_hash != first.content_hash

    def test_from_selection_decision_approved(self):
        decision = SelectionDecision(
            decision_id="SD_F001",
            factor_id="F001",
            approved=True,
            reason=SelectionReason.APPROVED,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="1.0",
            evidence_refs=("e1",),
            gate_results=("g1",),
        )
        artifact = FactorAdmissionArtifact.from_selection_decision(
            decision,
            factor_version="v1",
            quality=0.8,
            health_state_ref="lifecycle:APPROVED",
            cluster_id=2,
            orientation=1,
            policy_ref="policy:1.0",
        )
        assert artifact.decision is AdmissionDecision.APPROVED
        assert artifact.factor_version == "v1"
        assert artifact.cluster_id == 2
        assert artifact.orientation == 1
        assert artifact.evidence_refs == ("e1",)
        assert artifact.gate_results == ("g1",)
        assert artifact.policy_ref == "policy:1.0"
        # content_hash is derived and deterministic: an artifact built from the
        # same decision content recomputes the identical hash.
        rebuilt = FactorAdmissionArtifact.from_selection_decision(
            decision,
            factor_version="v1",
            quality=0.8,
            health_state_ref="lifecycle:APPROVED",
            cluster_id=2,
            orientation=1,
            policy_ref="policy:1.0",
        )
        assert artifact.content_hash == rebuilt.content_hash

    def test_from_selection_decision_shadowed(self):
        decision = SelectionDecision(
            decision_id="SD_F001",
            factor_id="F001",
            approved=False,
            reason=SelectionReason.SHADOW,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="1.0",
            evidence_refs=(),
            gate_results=(),
        )
        artifact = FactorAdmissionArtifact.from_selection_decision(decision)
        assert artifact.decision is AdmissionDecision.SHADOWED

    def test_from_selection_decision_rejected(self):
        decision = SelectionDecision(
            decision_id="SD_F001",
            factor_id="F001",
            approved=False,
            reason=SelectionReason.REJECTED_GATE_FAILURE,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="1.0",
            evidence_refs=(),
            gate_results=(),
        )
        artifact = FactorAdmissionArtifact.from_selection_decision(decision)
        assert artifact.decision is AdmissionDecision.REJECTED
        assert artifact.reason == "REJECTED_GATE_FAILURE"

    def test_to_dict(self):
        artifact = _admission()
        data = artifact.to_dict()
        assert data["decision"] == "APPROVED"
        assert data["content_hash"] == artifact.content_hash
        assert data["cluster_id"] == 3

    def test_content_hash_is_derived_and_fails_closed_on_mismatch(self):
        # A caller must not self-report an arbitrary content hash: passing a
        # stored hash that does not equal the recomputed content hash FAILS
        # CLOSED, so content and hash can never diverge.
        with pytest.raises(ValueError, match="does not match"):
            _admission(content_hash="forged-hash")

    def test_evidence_refs_and_gate_results_are_construction_time_snapshot(self):
        # Tuples passed by the caller are snapshotted at construction time; a
        # caller mutating a list cannot change the artifact's frozen evidence.
        source_evidence = ["bundle-X"]
        source_gates = ["gate-X"]
        artifact = FactorAdmissionArtifact(
            factor_id="F001",
            decision=AdmissionDecision.APPROVED,
            quality=0.8,
            reason="APPROVED",
            evidence_refs=source_evidence,
            gate_results=source_gates,
        )
        source_evidence.append("tampered")
        source_gates.append("tampered")
        assert artifact.evidence_refs == ("bundle-X",)
        assert artifact.gate_results == ("gate-X",)
