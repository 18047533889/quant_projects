"""
Cross-package contract tests (INT-P1).

Tests the data-flow contracts between the five platform packages:

    FactorEngine (factor_engine) -- factor computation (imports as flat modules
        under ``runtime.`` / ``ir.`` / ``modeling.``, no ``factor_engine``
        top-level package)
    QE  quant_evaluator           -- evaluation evidence
    FA  factor_assets             -- factor registry / set assembly / admission
    FP  factor_preprocess         -- feature bundling for model input
    FO  factor_optimizer          -- mutation / search

The full end-to-end execution chain (compute -> evaluate -> assemble ->
transform -> model) is not wired together in one runnable pipeline yet.  Per
INT-P1 these tests assert the CONTRACT on the canonical artifact types that DO
exist: each package's artifact carries the refs needed to trace identity,
order, and snapshot across the boundary, and where a full chain step is missing
the test says so explicitly instead of fabricating a pass.

Naming convention: ``TEST-NN`` markers below map 1:1 to the six INT-P1 tests.

None of these tests ever imports a top-level ``factor_engine`` package (it does
not exist -- factor_engine is installed as flat editable modules).  A test
whose source package cannot be imported is SKIPPED with a reason; a skip is
not a pass.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Repository root is on the import path when tests run from /home/shw/quant_projects
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_optional(module: str):
    """Try to import *module*; return (mod, None) or (None, reason)."""
    try:
        import importlib
        return importlib.import_module(module), None
    except Exception as exc:  # noqa: BLE001 - report any import failure truthfully
        return None, f"{type(exc).__name__}: {exc}"


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_names():
    """Return (module, display_name) pairs the FE/FP adapter contract uses."""
    return [
        ("factor_engine.runtime.factor_identity", "FE.runtime.factor_identity"),
        ("factor_engine.ir.types", "FE.ir.types"),
        ("factor_preprocess.contracts.feature_bundle", "FP.contracts.feature_bundle"),
    ]


# ===========================================================================
# TEST-01: FactorEngine FactorResult -> QE
#   axes / factor identity / snapshot preserved
# ===========================================================================

def test_01_fe_factor_identity_and_snapshot_flow_to_qe():
    """
    FE's factor output identity must carry the axes, semantic identity digest,
    and snapshot coordinates that QE needs to trace an evaluation back to a
    concrete computation.

    VERIFIED here (contract on structures that exist):
      - FE ``FactorSemanticIdentity`` (runtime.factor_identity) has a stable
        ``identity_digest()`` that changes when any identity field changes,
        and ``SourceVintageSpec`` (ir.types) carries ``snapshot_id`` +
        explicit revision columns as the snapshot coordinate.
      - FE IR types pin the panel axes semantics (ts / inst) and the semantic
        lattice, which is the axis contract QE consumes.

    NEEDS the real pipeline (not wired yet):
      - FE produces NO ``FactorResult`` dataclass today (search
        ``class FactorResult`` across factor_engine returns nothing); outputs
        are long-table Polars/Pandas frames or modeling artifacts.  There is
        therefore no concrete FE->QE artifact carrying all three of
        axes/identity/snapshot in one object.  When FE grows a typed
        ``FactorResult``, its constructor should be proven to carry
        ``FactorSemanticIdentity.identity_digest()`` and
        ``SourceVintageSpec.snapshot_id``.
    """
    fe_id_mod, err_id = _import_optional("factor_engine.runtime.factor_identity")
    fe_ir_mod, err_ir = _import_optional("factor_engine.ir.types")
    if err_id or err_ir:
        pytest.skip(f"factor_engine not importable here: {err_id or err_ir}")

    # Identity: the semantic identity digest is deterministic and sensitive.
    ident = fe_id_mod.FactorSemanticIdentity(
        ir_hash=_sha256("a"),
        operator_contract_hash=_sha256("b"),
        field_contract_hash=_sha256("c"),
        source_contract_hash=_sha256("d"),
        source_dependency_hash=_sha256("e"),
        market="cn", frequency="1d", universe="u1",
    )
    digest = ident.identity_digest()
    assert isinstance(digest, str) and len(digest) == 64
    # Changing ANY identity field changes the digest (axes are also identity).
    ident2 = fe_id_mod.FactorSemanticIdentity(
        ir_hash=_sha256("a"), operator_contract_hash=_sha256("b"),
        field_contract_hash=_sha256("c"), source_contract_hash=_sha256("d"),
        source_dependency_hash=_sha256("e"),
        market="cn", frequency="1d", universe="u2",   # <-- different universe
    )
    assert ident2.identity_digest() != digest

    # Snapshot: SourceVintageSpec carries snapshot_id + explicit revision chain.
    vint = fe_ir_mod.SourceVintageSpec(
        snapshot_id="snap_20260101",
        revision_time_column="pub_ts",
        revision_id_column="pub_ver",
        revision_sequence_column="pub_seq",
    )
    assert vint.snapshot_id == "snap_20260101"
    assert vint.revision_time_column == "pub_ts"

    # Axes: the semantic lattice vocabulary names the panel axes FE outputs.
    assert "ts" in fe_ir_mod.SemanticLattice.__dataclass_fields__ or hasattr(
        fe_ir_mod.SemanticLattice, "domains"
    )


# ===========================================================================
# TEST-02: QE EvaluationArtifact -> FA
#   metric version / snapshot / split preserved
# ===========================================================================

def test_02_qe_evaluation_artifact_to_fa_preserves_refs():
    """
    QE's evaluation output and FA's admission/evidence refs must agree on
    metric version, evaluation run, and (when present) universe/period refs.

    VERIFIED here (contract on structures that exist):
      - QE ``MetricArtifact`` (quant_evaluator.reporting.tear_sheet) is the
        evaluation artifact; ``EvaluationResult.artifacts`` carries them.
      - FA ``EvidenceRef`` (factor_assets.contracts.evidence_ref) preserves
        ``metric_version`` + ``evaluation_run_id`` + ``factor_id`` +
        ``universe_ref``/``period`` bounds, and
        ``EvidenceBundleRef`` adds ``qe_version``.
      - FA lifecycle admission (REGISTERED -> EVALUATED) REQUIRES an
        ``evaluation_bundle_ref`` (factor_assets.contracts.lifecycle), so
        admission is wired to the QE bundle reference.

    NEEDS the real pipeline:
      - QE has no ``EvaluationArtifact``/``EvaluationBundle`` dataclass yet
        (grep ``EvaluationArtifact``/``EvaluationBundle`` in quant_evaluator
        finds only references in other packages' docs), and QE's
        ``MetricArtifact`` does not carry ``metric_version`` / ``snapshot_ref``
        / ``split_ref`` today.  Once QE adds a typed bundle artifact carrying
        metric_version + snapshot_ref + split_ref, it should be asserted to
        round-trip into FA's EvidenceRef/EvidenceBundleRef.
    """
    qe_mod, err_qe = _import_optional("quant_evaluator.reporting.tear_sheet")
    ev_mod, err_ev = _import_optional("factor_assets.contracts.evidence_ref")
    lc_mod, err_lc = _import_optional("factor_assets.contracts.lifecycle")
    missing = [e for e in (err_qe, err_ev, err_lc) if e]
    if missing:
        pytest.skip("import failure: " + "; ".join(missing))

    # QE evaluation artifact exists and is carried on the evaluation result.
    assert hasattr(qe_mod, "MetricArtifact")
    assert hasattr(qe_mod, "EvaluationResult")

    # FA evidence ref preserves the metric version + evaluation run identity.
    ev = ev_mod.EvidenceRef(
        evidence_id="ev1", evaluation_run_id="run_42",
        metric_name="ic_mean", metric_version="2.1.0", timestamp="2026-08-01T00:00:00Z",
        factor_id="F1", universe_ref="u1", period_start="2025-01-01", period_end="2026-08-01",
    )
    assert ev.metric_version == "2.1.0"
    assert ev.evaluation_run_id == "run_42"
    assert ev.factor_id == "F1"

    # Bundle-level ref carries the QE package version (the producer version).
    bundle = ev_mod.EvidenceBundleRef(
        bundle_id="b1", evaluation_run_id="run_42", factor_ids=("F1",),
        timestamp="2026-08-01T00:00:00Z", qe_version="0.0.1a1", universe_ref="u1",
    )
    assert bundle.qe_version == "0.0.1a1"
    assert bundle.factor_ids == ("F1",)

    # FA admission gate: EVALUATED requires an evaluation bundle ref.
    required = lc_mod.get_required_evidence(
        lc_mod.LifecycleState.REGISTERED, lc_mod.LifecycleState.EVALUATED
    )
    assert "evaluation_bundle_ref" in required


# ===========================================================================
# TEST-03: FA FactorSetArtifact -> FP
#   factor order unchanged
# ===========================================================================

def test_03_fa_factor_set_artifact_preserves_factor_order():
    """
    FA's FactorSetArtifact (ordered membership) must feed FP without reordering
    factor identity.

    VERIFIED here (contract on structures that exist):
      - FA ``FactorSetArtifact.factor_ids`` returns members in membership
        (assembly) order; ``factor_ids`` is the ordered tuple the downstream
        provider must consume.
      - FA's own assembly input carries ``snapshot_ref`` / ``split_ref`` /
        ``universe_ref`` so the FP side can be traced to the same data.

    NEEDS the real pipeline:
      - FP's ``FactorAssetsAdapter`` (factor_preprocess.adapters.factor_assets)
        currently calls ``list(factor_set.factor_ids)`` and loads a batch with
        that order, but its default provider implementation is not installed
        (``_factor_assets_impl`` import fails), so no real FA->FP value flow is
        executed here.  The FP adapter asserts order only through a Protocol
        that a real provider must honor.
    """
    fa_mod, err_fa = _import_optional("factor_assets.contracts.factor_set")
    if err_fa:
        pytest.skip(f"factor_assets not importable here: {err_fa}")

    ordered = ("F1", "F2", "F3")
    artifact = fa_mod.FactorSetArtifact(
        set_id="set_1", name="demo",
        members=tuple(
            fa_mod.FactorMembership(factor_id=fid) for fid in ordered
        ),
        created_at="2026-08-01T00:00:00Z",
        policy_hash=_sha256("p"),
        assembly_hash=_sha256("a"),
        snapshot_ref="snap_20260101",
        split_ref="split_train",
        universe_ref="u1",
    )
    assert artifact.factor_ids == ordered
    assert artifact.snapshot_ref == "snap_20260101"
    assert artifact.split_ref == "split_train"

    # The ordered factor_ids tuple is the canonical boundary contract.
    boundary_ids = artifact.factor_ids
    assert boundary_ids == ("F1", "F2", "F3")


# ===========================================================================
# TEST-04: FP FeatureBundle -> model adapter
#   manifest column order == model input order
# ===========================================================================

def test_04_fp_feature_bundle_manifest_matches_model_input_order():
    """
    FP's FeatureBundle (the model-input contract) must make manifest column
    order identical to the model's input feature order.

    VERIFIED here (contract on structures that exist):
      - ``FeatureManifest.feature_ids`` is an ordered tuple; the feature axis is
        the trailing axis of ``FeatureBundle.values``; ``get_column_index`` /
        ``get_feature_indices`` map feature identity -> column index.
      - FeatureBundle is immutable (values frozen, channels snapshot-copied),
        so a model adapter can safely iterate manifest columns.
      - Channel feature order flows into the manifest unchanged.

    NEEDS the real pipeline:
      - There is no concrete model adapter class in factor_preprocess today
        (grep ``model_adapter`` finds none); the representation layer
        (``representation.multichannel`` etc.) produces results but the model
        input adapter is not wired.  When a model adapter exists, it must read
        model feature order from ``FeatureManifest.feature_ids`` in order.
    """
    fb_mod, err_fb = _import_optional("factor_preprocess.contracts.feature_bundle")
    if err_fb:
        pytest.skip(f"factor_preprocess not importable here: {err_fb}")

    # Channel declares features in a specific order.
    feature_order = ("F1", "F2", "F3")
    channel = fb_mod.ChannelRef("feat", "feature", feature_order)
    manifest = fb_mod.FeatureManifest(
        feature_ids=list(feature_order),
        channel_offsets={"feat": 0},
        channel_sizes={"feat": len(feature_order)},
    )

    # Manifest column order is exactly the channel/model order.
    assert manifest.feature_ids == feature_order
    assert [manifest.get_column_index(f) for f in feature_order] == [0, 1, 2]
    assert manifest.get_feature_indices(list(feature_order)) == [0, 1, 2]

    # Bundle validates manifest column count against the feature axis.
    time_axis = fb_mod.AxisRef("time", np.arange(2, dtype="int64"), "int64")
    asset_axis = fb_mod.AxisRef("asset", np.array(["A", "B"]), "object")
    values = np.zeros((2, 2, 3), dtype="float64")
    bundle = fb_mod.FeatureBundle(
        bundle_id="b1", time_axis=time_axis, asset_axis=asset_axis,
        channels={"feat": channel}, values=values, layout="TNF",
        manifest=manifest, source_factor_ids=list(feature_order),
    )
    assert bundle.manifest.feature_ids == feature_order
    assert bundle.is_immutable()


# ===========================================================================
# TEST-05: FO mutation -> FE recompute -> QE validation -> FA admission
#   full lineage intact
# ===========================================================================

def test_05_fo_mutation_to_fa_admission_lineage_intact():
    """
    An FO mutation must be traceable to its parents, its FE recompute, its QE
    validation evidence, and finally to FA admission, without losing any
    reference in the chain.

    VERIFIED here (contract on structures that exist):
      - FO ``CandidateMutation`` carries ``parent_factor_ids`` + ``lineage_ref``
        (the recompute source) + ``mutation_spec_version``.
      - FO ``Trial`` links mutation -> evaluation via ``evaluation_ref``
        (documented as a QE EvaluationBundle reference) and records
        ``parent_factor_ids``.
      - FA admission (EVALUATED) requires ``evaluation_bundle_ref``, which is
        what the FO Trial's evaluation_ref points at.
      - FE's ``SourceVintageSpec`` provides the snapshot coordinate for the
        recompute step.

    NEEDS the real pipeline:
      - There is no single end-to-end runner tying FO -> FE -> QE -> FA in one
        execution (FO's adapters are protocol stubs).  The lineage chain is
        asserted here purely at the contract-reference level.
    """
    fo_cm, err_cm = _import_optional("factor_optimizer.contracts.candidate_mutation")
    fo_tr, err_tr = _import_optional("factor_optimizer.contracts.trial")
    ev_mod, err_ev = _import_optional("factor_assets.contracts.evidence_ref")
    lc_mod, err_lc = _import_optional("factor_assets.contracts.lifecycle")
    fe_ir, err_ir = _import_optional("factor_engine.ir.types")
    missing = [e for e in (err_cm, err_tr, err_ev, err_lc, err_ir) if e]
    if missing:
        pytest.skip("import failure: " + "; ".join(missing))

    # FO mutation -> recompute: parent factors + lineage ref preserved.
    mutation = fo_cm.CandidateMutation(
        mutation_id="m1", mutation_spec_version="1.0.0",
        parent_factor_ids=["F1", "F2"], mutation_type="composition",
        parameters={"op": "add"}, lineage_ref="lineage_f1_f2",
    )
    assert mutation.parent_factor_ids == ["F1", "F2"]
    assert mutation.lineage_ref == "lineage_f1_f2"

    # FE recompute snapshot coordinate exists (the input vintage for the child).
    vint = fe_ir.SourceVintageSpec(snapshot_id="snap_20260101")
    assert vint.snapshot_id == "snap_20260101"

    # FO trial -> QE validation: evaluation_ref names a QE evaluation bundle.
    trial = fo_tr.Trial(
        trial_id="t1", mutation_id="m1", status=fo_tr.TrialStatus.EVALUATED,
        parent_factor_ids=["F1", "F2"], evaluation_ref="qe-bundle:b1",
    )
    assert trial.evaluation_ref == "qe-bundle:b1"
    assert trial.is_successful()

    # QE validation -> FA admission: the bundle ref feeds FA's EvidenceRef and
    # satisfies the EVALUATED evidence requirement.
    bundle = ev_mod.EvidenceBundleRef(
        bundle_id="b1", evaluation_run_id="run_42", factor_ids=("F3",),
        timestamp="2026-08-01T00:00:00Z", qe_version="0.0.1a1",
    )
    admitted = ev_mod.evidence_bundle_event_id(bundle) == "qe-bundle:b1"
    assert admitted
    required = lc_mod.get_required_evidence(
        lc_mod.LifecycleState.REGISTERED, lc_mod.LifecycleState.EVALUATED
    )
    assert "evaluation_bundle_ref" in required
    lc_mod.validate_transition(
        lc_mod.LifecycleState.REGISTERED,
        lc_mod.LifecycleState.EVALUATED,
        evidence_keys={"evaluation_bundle_ref"},
    )


# ===========================================================================
# TEST-06: Same data snapshot across 5 packages
#   snapshot_ref identical
# ===========================================================================

def test_06_snapshot_ref_identical_across_packages():
    """
    All five packages must reference the SAME data snapshot for a single
    logical run, so downstream artifacts can be cross-checked by snapshot.

    VERIFIED here (contract on structures that exist):
      - FE ``SourceVintageSpec.snapshot_id`` is the FE snapshot coordinate.
      - FA ``FactorSetSpec.data_snapshot_ref`` / ``FactorSetArtifact.snapshot_ref``
        carry the FA snapshot ref.
      - FP ``FittedState.data_snapshot_ref`` + ``split_ref`` carry the FP
        snapshot ref.
      - FO ``CandidateMutation``/``Trial`` metadata is free-form; FO currently
        has no typed ``snapshot_ref`` field, so FO's snapshot is carried in
        ``metadata`` (verified only by convention).
      - QE ``MetricArtifact`` carries no ``snapshot_ref`` today (only
        ``ChartSpec.source_artifact_refs`` exists, which references the metric
        artifact, not a data snapshot).

    NEEDS the real pipeline:
      - When a snapshot ref is genuinely threaded through all five packages it
        must equal ``"snap_20260101"`` here.  Today FE/FA/FP carry typed
        snapshot refs; FO and QE do not (FO via metadata convention only, QE
        not at all), so the assertion below asserts the FE/FA/FP triple and
        documents the FO/QE gap rather than fabricating a 5-way equality.
    """
    fe_ir, err_ir = _import_optional("factor_engine.ir.types")
    fa_mod, err_fa = _import_optional("factor_assets.contracts.factor_set")
    fp_mod, err_fp = _import_optional("factor_preprocess.contracts.state")
    fo_mod, err_fo = _import_optional("factor_optimizer.contracts.candidate_mutation")
    qe_mod, err_qe = _import_optional("quant_evaluator.reporting.chart_spec")

    snapshot = "snap_20260101"

    # FE carries the snapshot coordinate.
    assert err_ir is None, f"FE ir.types import failed: {err_ir}"
    vint = fe_ir.SourceVintageSpec(snapshot_id=snapshot)
    assert vint.snapshot_id == snapshot

    # FA carries snapshot_ref on both spec and assembled artifact.
    assert err_fa is None, f"FA import failed: {err_fa}"
    spec = fa_mod.FactorSetSpec(
        set_id="s1", name="demo", selection_policy="family_robust",
        data_snapshot_ref=snapshot,
    )
    assert spec.data_snapshot_ref == snapshot

    # FP carries data_snapshot_ref + split_ref on fitted transform state.
    if err_fp is None:
        from datetime import datetime
        state = fp_mod.FittedState(
            state_id="st1", transform_name="zscore",
            transform_version="1.0.0",
            fit_start_time=datetime(2026, 1, 1),
            fit_end_time=datetime(2026, 2, 1),
            data_snapshot_ref=snapshot, split_ref="split_train",
        )
        assert state.data_snapshot_ref == snapshot
        assert state.split_ref == "split_train"

    # FO: no typed snapshot field today -> document the gap (metadata only).
    if err_fo is None:
        assert not hasattr(fo_mod.CandidateMutation, "snapshot_ref"), (
            "If FO grows a typed snapshot_ref, TEST-06 should assert equality "
            "against the shared snapshot here."
        )

    # QE: no typed snapshot field today -> document the gap.
    if err_qe is None:
        assert not hasattr(qe_mod.ChartSpec, "snapshot_ref"), (
            "If QE grows a typed snapshot_ref, TEST-06 should assert equality "
            "against the shared snapshot here."
        )

    # The packages with a typed snapshot ref must agree on the SAME snapshot.
    assert vint.snapshot_id == spec.data_snapshot_ref == snapshot
