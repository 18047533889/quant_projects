"""OPS-08 dry-run orchestration for semantic-version invalidation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence
import importlib
import hashlib


@dataclass(frozen=True)
class MigrationImpact:
    artifact_id: str
    artifact_kind: str
    disposition: str
    action: str


@dataclass(frozen=True)
class Ops08MigrationPlan:
    dry_run: bool
    required_metric_versions: tuple[tuple[str, str], ...]
    impacts: tuple[MigrationImpact, ...]
    production_writes: tuple[str, ...] = ()
    superseded_recipe_ids: tuple[str, ...] = ()
    required_recipe_hashes: tuple[tuple[str, str], ...] = ()
    superseded_implementation_refs: tuple[str, ...] = ()
    dependency_edges: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def affected_ids(self) -> tuple[str, ...]:
        return tuple(i.artifact_id for i in self.impacts if i.disposition != "VALID")


def plan_ops08_migration(
    *,
    evidence_refs: Sequence[Any],
    required_metric_versions: Mapping[str, str],
    dependency_dag: Any,
    artifact_kinds: Mapping[str, str],
    dry_run: bool = True,
    recipe_refs: Mapping[str, Any] | None = None,
    required_recipe_hashes: Mapping[str, str] | None = None,
    superseded_implementation_refs: Sequence[str] = (),
) -> Ops08MigrationPlan:
    """Plan selective recomputation without mutating history or pointers."""
    if dry_run is not True:
        raise ValueError("OPS-08 public orchestrator is dry-run only")
    recipes = dict(recipe_refs or {})
    required_recipes = dict(required_recipe_hashes or {})
    if isinstance(superseded_implementation_refs, (str, bytes)):
        raise TypeError("superseded implementation refs must be a sequence, not one string")
    implementation_refs = tuple(superseded_implementation_refs)
    if any(not isinstance(ref, str) or not ref.strip() for ref in implementation_refs):
        raise ValueError("superseded implementation refs must be non-empty strings")
    superseded_impls = tuple(sorted(set(implementation_refs)))
    if any(not isinstance(key, str) or not key.strip() for key in recipes):
        raise ValueError("recipe artifact IDs must be non-empty strings")
    if any(not isinstance(key, str) or not key.strip()
           or not isinstance(value, str) or not value.strip()
           for key, value in required_recipes.items()):
        raise ValueError("required recipe IDs and hashes must be non-empty strings")
    if set(required_recipes) - set(recipes):
        raise ValueError("required recipe change has no resolved historical recipe")
    stale_recipes = set()
    if recipes:
        recipe_type = importlib.import_module(
            "factor_preprocess.contracts.treatment_recipe"
        ).TreatmentRecipe
        for artifact_id, recipe in recipes.items():
            if not isinstance(recipe, recipe_type):
                raise TypeError("recipe_refs must resolve to immutable TreatmentRecipe objects")
            if ((artifact_id in required_recipes
                 and recipe.content_hash != required_recipes[artifact_id])
                    or any(step.implementation_ref in superseded_impls
                           for step in recipe.ordered_steps)):
                stale_recipes.add(artifact_id)
    # Classification metadata must not determine inventory. An omitted kind
    # cannot hide an affected root, leaf, or orphan evaluation from the plan.
    evidence_by_id = {}
    for ref in evidence_refs:
        previous = evidence_by_id.get(ref.evidence_id)
        if previous is not None and previous != ref:
            raise ValueError("conflicting evidence records for one immutable evidence ID")
        evidence_by_id[ref.evidence_id] = ref
    # Runtime composition keeps the platform DTO layer free of domain imports;
    # these are the existing FA/modeling authorities, resolved only at the
    # application orchestration boundary.
    evidence_plan = importlib.import_module(
        "factor_assets.contracts.evidence_ref"
    ).plan_metric_version_invalidation(
        tuple(evidence_by_id.values()), required_metric_versions
    )
    stale = set(evidence_plan.stale_evidence_ids)
    if set(recipes) & set(evidence_by_id):
        raise ValueError("one artifact ID cannot identify both recipe and evaluation")
    affected = dependency_dag.invalidate(stale | stale_recipes)
    all_ids = (set(artifact_kinds) | set(dependency_dag.edges)
               | set().union(*dependency_dag.edges.values()) | set(evidence_by_id)
               | set(recipes))
    impacts = []
    for artifact_id in sorted(all_ids):
        kind = artifact_kinds.get(artifact_id, "unknown")
        if artifact_id in stale_recipes:
            disposition, action = "SUPERSEDED", "REBUILD_RECIPE_AND_VALUES_SHADOW"
        elif artifact_id in stale:
            disposition, action = "SUPERSEDED", "RECOMPUTE_EVALUATION_SHADOW"
        elif artifact_id in affected:
            disposition, action = "REQUIRES_REEVALUATION", "RECOMPUTE_SHADOW_AND_REQUIRE_APPROVAL"
        else:
            disposition, action = "VALID", "RETAIN_IMMUTABLE"
        impacts.append(MigrationImpact(artifact_id, kind, disposition, action))
    return Ops08MigrationPlan(
        dry_run=True,
        required_metric_versions=evidence_plan.required_metric_versions,
        impacts=tuple(impacts),
        production_writes=(),
        superseded_recipe_ids=tuple(sorted(stale_recipes)),
        required_recipe_hashes=tuple(sorted(required_recipes.items())),
        superseded_implementation_refs=superseded_impls,
        dependency_edges=tuple(sorted(
            (str(child), tuple(sorted(map(str, parents))))
            for child, parents in dependency_dag.edges.items()
        )),
    )


AUDIT_REPLAY_EVIDENCE_CLASS = "AUDIT_REPLAY_NOT_FRESH_HOLDOUT"


@dataclass(frozen=True)
class ShadowReplayArtifact:
    old_artifact_id: str
    new_artifact: Any
    payload: bytes
    evidence_class: str
    reused_input_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShadowDependency:
    artifact: Any
    persisted_payload: bytes
    domain_payload: bytes


class Ops08ReplayAuthority(Protocol):
    def replay(self, impact: MigrationImpact,
               available_shadow_refs: Mapping[str, Any]) -> ShadowReplayArtifact: ...


@dataclass(frozen=True)
class Ops08ShadowExecution:
    dry_run_affected_ids: tuple[str, ...]
    actual_recomputed_ids: tuple[str, ...]
    shadow_generation_ids: tuple[tuple[str, str], ...]
    production_writes: tuple[str, ...]
    reconciled: bool


def execute_ops08_shadow(*, plan: Ops08MigrationPlan, replay_authority: Ops08ReplayAuthority,
                         generation_coordinator: Any) -> Ops08ShadowExecution:
    """Execute every affected node into a new audit-only shadow identity."""
    if not isinstance(plan, Ops08MigrationPlan) or not plan.dry_run:
        raise TypeError("execution requires an immutable OPS-08 dry-run plan")
    if plan.production_writes:
        raise ValueError("OPS-08 shadow execution forbids production writes")
    pending = {i.artifact_id: i for i in plan.impacts if i.disposition != "VALID"}
    dependencies = dict(plan.dependency_edges)
    completed: dict[str, Any] = {}
    generations = []
    while pending:
        ready = sorted(old_id for old_id in pending
                       if not (set(dependencies.get(old_id, ())) & set(pending)))
        if not ready:
            raise ValueError("affected dependency graph contains a cycle")
        for old_id in ready:
            direct_ids = tuple(sorted(set(dependencies.get(old_id, ())) & set(completed)))
            direct_refs = {key: completed[key] for key in direct_ids}
            produced = replay_authority.replay(pending.pop(old_id), direct_refs)
            if not isinstance(produced, ShadowReplayArtifact):
                raise TypeError("replay authority must return ShadowReplayArtifact")
            if produced.old_artifact_id != old_id:
                raise ValueError("replay result does not match planned old artifact")
            artifact = produced.new_artifact
            if artifact.artifact_id == old_id or not artifact.artifact_id.startswith("shadow:"):
                raise ValueError("shadow replay requires a new shadow artifact identity")
            if not isinstance(produced.payload, bytes):
                raise TypeError("shadow payload must be immutable bytes")
            if hashlib.sha256(produced.payload).hexdigest() != artifact.content_hash:
                raise ValueError("shadow payload content hash mismatch")
            if produced.evidence_class != AUDIT_REPLAY_EVIDENCE_CLASS:
                raise ValueError("historical test replay must remain audit-only")
            if tuple(sorted(produced.reused_input_ids)) != direct_ids:
                raise ValueError("replay result dependencies do not match direct planned dependencies")
            envelope = {
                "evidence_class": AUDIT_REPLAY_EVIDENCE_CLASS,
                "old_artifact_id": old_id,
                "actual_dependencies": list(direct_ids),
                "domain_payload_sha256": hashlib.sha256(produced.payload).hexdigest(),
                "domain_payload_hex": produced.payload.hex(),
            }
            persisted = __import__("json").dumps(
                envelope, sort_keys=True, separators=(",", ":")
            ).encode()
            artifact = replace(
                artifact, content_hash=hashlib.sha256(persisted).hexdigest(),
                size_bytes=len(persisted),
            )
            generation = generation_coordinator.stage(artifact, persisted)
            generation_coordinator.outbox.publish_pending(
                idempotency_key=f"publish:{generation}"
            )
            active = generation_coordinator.resolve_active(artifact.artifact_id)
            if active is None or active["generation_id"] != generation:
                raise RuntimeError("exact shadow generation is not active")
            completed[old_id] = ShadowDependency(artifact, persisted, produced.payload)
            generations.append((artifact.artifact_id, generation))
    expected, actual = tuple(sorted(plan.affected_ids)), tuple(sorted(completed))
    if actual != expected:
        raise RuntimeError("actual shadow replay does not reconcile with dry-run plan")
    return Ops08ShadowExecution(expected, actual, tuple(generations), (), True)
