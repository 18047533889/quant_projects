import json
import hashlib
from datetime import datetime, timezone
import pytest
import numpy as np

from factor_assets.contracts.evidence_ref import EvidenceRef
from modeling.ledger import ArtifactDependencyDAG
from quant_platform.app.ops08_migration import (
    ShadowReplayArtifact, plan_ops08_migration, execute_ops08_shadow,
)
from jobs.ops08_replay_fixture import (
    RealParameterBridgeReplayAuthority, RealRiskReplayAuthority, UnsupportedReplayKind,
)
import pandas as pd
from factor_assets.adapters.factor_engine import FEIdentityProvider
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from jobs.e2e_a_fe_qe_fa_spine import materialized_factor_value_ref
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.storage.generation_coordinator import DurableGenerationCoordinator
from quant_platform.app.contracts import ArtifactRef, ARTIFACT_TYPE_FACTOR_CANDIDATE


class _Publisher:
    def __init__(self): self.blobs = {}
    def publish(self, artifact, data): self.blobs[artifact.artifact_id] = data; return artifact


def _artifact(artifact_id, payload):
    digest = hashlib.sha256(payload).hexdigest()
    return ArtifactRef(artifact_id=artifact_id, artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE,
        schema_version="1.0", content_hash=digest, storage_uri=f"test://ops08/{artifact_id}/{digest}",
        size_bytes=len(payload), created_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
        producer_type="ops08-isolation-test", producer_version="1")


def _risk_authority(version="risk-v2"):
    times = np.arange(40); assets = np.arange(3)
    values = np.arange(120, dtype=float).reshape(40, 3)
    definition = "factor-definition:" + FEIdentityProvider().get_full_identity("rank(close)").canonical_hash
    value_ref = materialized_factor_value_ref(
        values=values, time_values=times, asset_values=assets,
        snapshot_ref="snapshot:frozen", factor_definition_ref=definition,
    )
    batch = FactorBatch(
        ("f",), AxisRef("time", "int64", 40, times), AxisRef("asset", "int64", 3, assets),
        values[..., None], context_refs={"snapshot_ref": "snapshot:frozen", "catalog_ref": "catalog:frozen",
            "factor_definition_refs": {"f": definition}, "factor_value_ref": value_ref},
    )
    labels = LabelBundle("forward", values, 1, decision_time=tuple(times),
        label_start_time=tuple(times + 1), label_end_time=tuple(times + 2), asset_axis=batch.asset_axis,
        source_ref="labels:frozen-axes")
    returns = np.zeros((40, 1)); returns[10, 0] = -.2; returns[20, 0] = .05
    portfolio = ProbePortfolioArtifact(returns, time_index=tuple(times), factor_ids=("f",))
    return RealRiskReplayAuthority(frozen_batch=batch, frozen_labels=labels,
        frozen_portfolio=portfolio, algorithm_version=version)


def _parameter_authority(version="bridge-v2"):
    times = np.arange(6); assets = np.array(["A", "B"], dtype=object)
    raw = np.array([[1., 10.], [np.nan, 11.], [np.nan, 12.], [4., 13.], [5., 14.], [6., 15.]])
    definition = "factor-definition:" + FEIdentityProvider().get_full_identity("rank(close)").canonical_hash
    value_ref = materialized_factor_value_ref(values=raw, time_values=times, asset_values=assets,
        snapshot_ref="snapshot:frozen-raw", factor_definition_ref=definition)
    batch = FactorBatch(("f",), AxisRef("time", "int64", 6, times),
        AxisRef("asset", "object", 2, assets), raw[..., None], validity=np.isfinite(raw[..., None]),
        context_refs={"snapshot_ref": "snapshot:frozen-raw", "catalog_ref": "catalog:frozen",
            "factor_definition_refs": {"f": definition}, "factor_value_ref": value_ref})
    labels = LabelBundle("forward", np.array([[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.]]), 1,
        decision_time=tuple(times), label_start_time=tuple(times+1), label_end_time=tuple(times+2),
        asset_axis=batch.asset_axis, source_ref="labels:frozen-axes")
    rows = [(t, a, raw[i,j]) for i,t in enumerate(times) for j,a in enumerate(assets)]
    long = pd.DataFrame(rows, columns=["date", "asset_id", "value"])
    return RealParameterBridgeReplayAuthority(frozen_raw_long=long,
        frozen_batch_template=batch, frozen_labels=labels, source_definition_ref=definition,
        source_value_ref=value_ref, new_max_lag=2, algorithm_version=version), raw.copy()


def evidence(eid, metric, version, factor):
    return EvidenceRef(
        evidence_id=eid, evaluation_run_id=f"run:{eid}", metric_name=metric,
        metric_version=version, timestamp="2026-01-01T00:00:00Z", factor_id=factor,
    )


def test_metric_fix_traverses_affected_chain_and_retains_unaffected_branch():
    dag = ArtifactDependencyDAG()
    dag.add("health:F1", "e:risk-old")
    dag.add("cluster:C1", "health:F1")
    dag.add("features:V2", "cluster:C1")
    dag.add("model:M2", "features:V2")
    dag.add("health:F2", "e:ic-good")
    dag.add("cluster:C2", "health:F2")
    dag.add("raw-value:F1", "recipe:F1")
    kinds = {key: key.split(":", 1)[0] for key in dag.edges}
    kinds.update({"e:risk-old": "evaluation", "e:ic-good": "evaluation",
                  "recipe:F1": "recipe", "raw-value:F1": "value"})
    plan = plan_ops08_migration(
        evidence_refs=[evidence("e:risk-old", "max_drawdown", "1", "F1"),
                       evidence("e:ic-good", "rank_ic", "2", "F2")],
        required_metric_versions={"max_drawdown": "2", "rank_ic": "2"},
        dependency_dag=dag, artifact_kinds=kinds,
    )
    affected = set(plan.affected_ids)
    assert {"e:risk-old", "health:F1", "cluster:C1", "features:V2", "model:M2"} <= affected
    assert {"e:ic-good", "health:F2", "cluster:C2", "raw-value:F1", "recipe:F1"}.isdisjoint(affected)
    assert plan.production_writes == ()
    assert all(i.action == "RETAIN_IMMUTABLE" for i in plan.impacts if i.artifact_id in {"raw-value:F1", "cluster:C2"})


def test_non_dry_run_is_not_a_public_mutation_entry():
    with pytest.raises(ValueError, match="dry-run only"):
        plan_ops08_migration(
            evidence_refs=[], required_metric_versions={},
            dependency_dag=ArtifactDependencyDAG(), artifact_kinds={}, dry_run=False,
        )


def test_missing_kind_metadata_cannot_hide_graph_nodes_or_orphan_evidence():
    dag = ArtifactDependencyDAG()
    dag.add('health', 'old')
    dag.add('model', 'health')
    plan = plan_ops08_migration(
        evidence_refs=[evidence('old', 'rank_ic', '1', 'F'),
                       evidence('orphan', 'rank_ic', '1', 'G'),
                       evidence('unchanged', 'rank_ic', '2', 'H')],
        required_metric_versions={'rank_ic': '2'}, dependency_dag=dag,
        artifact_kinds={},
    )
    assert set(plan.affected_ids) == {'old', 'health', 'model', 'orphan'}
    assert {item.artifact_id for item in plan.impacts} == {
        'old', 'health', 'model', 'orphan', 'unchanged'}
    assert all(item.artifact_kind == 'unknown' for item in plan.impacts)


def test_conflicting_immutable_evidence_versions_rejected():
    with pytest.raises(ValueError, match='conflicting evidence'):
        plan_ops08_migration(
            evidence_refs=[evidence('same', 'rank_ic', '1', 'F'),
                           evidence('same', 'rank_ic', '2', 'F')],
            required_metric_versions={'rank_ic': '2'}, dependency_dag=ArtifactDependencyDAG(),
            artifact_kinds={},
        )


def _recipe(periods=1, implementation="forward_fill@1"):
    from factor_preprocess.contracts.treatment_recipe import TreatmentRecipe, RecipeStep
    return TreatmentRecipe(
        recipe_id="fill-recipe", source_factor_definition_ref="definition",
        source_factor_value_ref="raw-value",
        ordered_steps=(RecipeStep(
            step_id="fill", semantic_transform_id="forward_fill",
            implementation_ref=implementation, stage="missingness",
            parameters={"max_periods": periods},
        ),),
    )


@pytest.mark.parametrize("cause", ["parameters", "implementation"])
def test_recipe_bridge_change_invalidates_values_and_every_downstream(cause):
    original = _recipe()
    original_hash = original.content_hash
    dag = ArtifactDependencyDAG()
    dag.add("recipe:old", "raw-value")
    dag.add("treated-value", "recipe:old")
    dag.add("evaluation", "treated-value")
    dag.add("health", "evaluation")
    dag.add("cluster", "health")
    dag.add("feature", "cluster")
    dag.add("model", "feature")
    dag.add("unaffected-value", "unaffected-recipe")
    kwargs = ({"required_recipe_hashes": {"recipe:old": _recipe(2).content_hash}}
              if cause == "parameters" else
              {"superseded_implementation_refs": ("forward_fill@1",)})
    plan = plan_ops08_migration(
        evidence_refs=[], required_metric_versions={}, dependency_dag=dag,
        artifact_kinds={}, recipe_refs={"recipe:old": original}, **kwargs,
    )
    assert set(plan.affected_ids) == {
        "recipe:old", "treated-value", "evaluation", "health", "cluster", "feature", "model"}
    assert plan.superseded_recipe_ids == ("recipe:old",)
    assert original.content_hash == original_hash
    assert plan.production_writes == ()


def test_recipe_change_without_resolved_history_fails_closed():
    with pytest.raises(ValueError, match="resolved historical recipe"):
        plan_ops08_migration(
            evidence_refs=[], required_metric_versions={},
            dependency_dag=ArtifactDependencyDAG(), artifact_kinds={},
            required_recipe_hashes={"missing": "replacement"},
        )


def test_unchanged_resolved_recipe_is_retained_without_kind_metadata():
    recipe = _recipe()
    plan = plan_ops08_migration(
        evidence_refs=[], required_metric_versions={},
        dependency_dag=ArtifactDependencyDAG(), artifact_kinds={},
        recipe_refs={"recipe": recipe}, required_recipe_hashes={"recipe": recipe.content_hash},
    )
    assert plan.affected_ids == ()
    assert [impact.artifact_id for impact in plan.impacts] == ["recipe"]


def test_risk_fix_executes_selective_shadow_and_preserves_raw_and_production(tmp_path):
    dag = ArtifactDependencyDAG()
    dag.add("health:F1", "e:risk-old")
    dag.add("e:risk-old", "raw-value:F1")
    dag.add("health:F2", "e:ic-good")
    kinds = {"raw-value:F1": "value", "e:risk-old": "evaluation",
             "health:F1": "health", "e:ic-good": "evaluation", "health:F2": "health"}
    old_risk = evidence("e:risk-old", "max_drawdown", "1", "F1")
    old_good = evidence("e:ic-good", "rank_ic", "2", "F2")
    plan = plan_ops08_migration(
        evidence_refs=[old_risk, old_good],
        required_metric_versions={"max_drawdown": "2", "rank_ic": "2"},
        dependency_dag=dag, artifact_kinds=kinds,
    )
    publisher = _Publisher()
    coordinator = DurableGenerationCoordinator(SqliteDb(str(tmp_path / "registry.db")), publisher)
    production_old = _artifact("production:pointer", b"old-production")
    production_old_gen = coordinator.stage(production_old, b"old-production")
    assert coordinator.outbox.publish_pending(
        idempotency_key=f"publish:{production_old_gen}"
    ) == 1
    production_new_gen = coordinator.stage(
        _artifact("production:pointer", b"pending-production"), b"pending-production"
    )
    unrelated_gen = coordinator.stage(_artifact("unrelated:claimed", b"claimed"), b"claimed")
    unrelated_event = coordinator.db.query(
        "SELECT id FROM outbox_events WHERE idempotency_key=?",
        (f"publish:{unrelated_gen}",),
    )[0]
    assert coordinator.outbox.claim(
        unrelated_event["id"], worker_id="expired-worker", claim_token="expired-token", now=0.0
    )
    result = execute_ops08_shadow(
        plan=plan, replay_authority=_risk_authority(),
        generation_coordinator=coordinator,
    )
    assert result.reconciled
    assert set(result.actual_recomputed_ids) == {"e:risk-old", "health:F1"}
    assert "raw-value:F1" not in result.actual_recomputed_ids
    assert result.production_writes == ()
    assert old_risk.metric_version == "1" and old_good.metric_version == "2"
    assert all(new_id.startswith("shadow:") for new_id, _ in result.shadow_generation_ids)
    assert coordinator.resolve_active("e:risk-old") is None
    assert coordinator.resolve_active("production:pointer")["generation_id"] == production_old_gen
    statuses = {row["idempotency_key"]: row["status"] for row in coordinator.db.query(
        "SELECT idempotency_key,status FROM outbox_events"
    )}
    assert statuses[f"publish:{production_new_gen}"] == "pending"
    assert statuses[f"publish:{unrelated_gen}"] == "claimed"
    domain_payloads = {}
    for artifact_id, payload in publisher.blobs.items():
        if not artifact_id.startswith("shadow:"):
            continue
        envelope = json.loads(payload)
        assert envelope["evidence_class"] == "AUDIT_REPLAY_NOT_FRESH_HOLDOUT"
        assert envelope["old_artifact_id"] in result.actual_recomputed_ids
        assert "actual_dependencies" in envelope
        domain_payloads[envelope["old_artifact_id"]] = json.loads(
            bytes.fromhex(envelope["domain_payload_hex"])
        )
    assert domain_payloads["e:risk-old"]["metric_id"] == "max_drawdown"
    assert domain_payloads["e:risk-old"]["value"] == pytest.approx(.2)
    assert domain_payloads["health:F1"]["evaluation_ref"] == (
        domain_payloads["e:risk-old"]["evaluation_ref"]
    )


def test_shadow_retry_is_idempotent_and_dependency_mismatch_fails(tmp_path):
    dag = ArtifactDependencyDAG(); dag.add("health", "evaluation")
    plan = plan_ops08_migration(
        evidence_refs=[evidence("evaluation", "max_drawdown", "1", "F")],
        required_metric_versions={"max_drawdown": "2"}, dependency_dag=dag,
        artifact_kinds={"evaluation": "evaluation", "health": "health"},
    )
    coordinator = DurableGenerationCoordinator(SqliteDb(str(tmp_path / "registry.db")), _Publisher())
    authority = _risk_authority()
    first = execute_ops08_shadow(plan=plan, replay_authority=authority,
                                 generation_coordinator=coordinator)
    second = execute_ops08_shadow(plan=plan, replay_authority=authority,
                                  generation_coordinator=coordinator)
    assert first.shadow_generation_ids == second.shadow_generation_ids

    class BadDependencies:
        def __init__(self): self.real = _risk_authority("bad")
        def replay(self, impact, refs):
            value = self.real.replay(impact, refs)
            return ShadowReplayArtifact(value.old_artifact_id, value.new_artifact,
                value.payload, value.evidence_class, ("not-a-direct-dependency",))
    bad = DurableGenerationCoordinator(SqliteDb(str(tmp_path / "bad.db")), _Publisher())
    with pytest.raises(ValueError, match="dependencies"):
        execute_ops08_shadow(plan=plan, replay_authority=BadDependencies(),
                             generation_coordinator=bad)


def test_parameter_bridge_recomputes_value_qe_and_health_from_direct_bytes(tmp_path):
    old_recipe = _recipe()
    dag = ArtifactDependencyDAG()
    dag.add("recipe:old", "raw-value")
    dag.add("treated-value", "recipe:old")
    dag.add("evaluation", "treated-value")
    dag.add("health", "evaluation")
    plan = plan_ops08_migration(
        evidence_refs=[], required_metric_versions={}, dependency_dag=dag,
        artifact_kinds={"raw-value": "value", "recipe:old": "recipe",
                        "treated-value": "value", "evaluation": "evaluation", "health": "health"},
        recipe_refs={"recipe:old": old_recipe},
        required_recipe_hashes={"recipe:old": _recipe(2).content_hash},
    )
    publisher = _Publisher()
    coordinator = DurableGenerationCoordinator(SqliteDb(str(tmp_path / "registry.db")), publisher)
    authority, old_raw = _parameter_authority()
    old_bytes = old_raw.tobytes()
    result = execute_ops08_shadow(plan=plan, replay_authority=authority,
                                  generation_coordinator=coordinator)
    retry = execute_ops08_shadow(plan=plan, replay_authority=authority,
                                 generation_coordinator=coordinator)
    assert retry.shadow_generation_ids == result.shadow_generation_ids
    assert set(result.actual_recomputed_ids) == {"recipe:old", "treated-value", "evaluation", "health"}
    docs = {}
    for persisted in publisher.blobs.values():
        env = json.loads(persisted); docs[env["old_artifact_id"]] = json.loads(bytes.fromhex(env["domain_payload_hex"]))
    assert docs["recipe:old"]["effective_max_periods"] == 2
    treated = np.asarray(docs["treated-value"]["values"])
    assert treated[2, 0] == 1.0 and not np.array_equal(treated, old_raw, equal_nan=True)
    assert set(docs["evaluation"]["metrics"]) == {"rank_ic", "coverage"}
    assert docs["health"]["evaluation_ref"] == docs["evaluation"]["evaluation_ref"]
    assert old_raw.tobytes() == old_bytes
    assert old_recipe.content_hash != _recipe(2).content_hash


def test_unimplemented_cluster_authority_fails_explicitly():
    authority, _ = _parameter_authority()
    from quant_platform.app.ops08_migration import MigrationImpact
    with pytest.raises(UnsupportedReplayKind):
        authority.replay(MigrationImpact("cluster", "cluster", "REQUIRES_REEVALUATION", "x"), {})
