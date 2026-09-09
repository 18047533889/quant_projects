"""Hermetic risk replay authority over frozen QE portfolio inputs."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd

from factor_assets.adapters.quant_evaluator import QEEvidenceProvider
from factor_assets.profiling.dimensions import build_dimension_grades
from factor_assets.profiling.health_card import build_health_card
from factor_assets.profiling.policies import INTEGRITY_GATE_IDS, get_health_policy
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.contracts.factor_batch import FactorBatch
from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe
from factor_preprocess.registry.transforms import create_default_registry
from quant_platform.app.contracts import ArtifactRef, ARTIFACT_TYPE_FACTOR_CANDIDATE
from quant_platform.app.ops08_migration import AUDIT_REPLAY_EVIDENCE_CLASS, ShadowReplayArtifact


class UnsupportedReplayKind(RuntimeError): pass


class RealRiskReplayAuthority:
    """Supports only real QE max-drawdown -> FA health replay."""
    def __init__(self, *, frozen_batch, frozen_labels, frozen_portfolio,
                 algorithm_version: str):
        self.batch = frozen_batch
        self.labels = frozen_labels
        self.portfolio = frozen_portfolio
        self.algorithm_version = algorithm_version
        self._evaluations = {}
        self._results = {}

    def replay(self, impact, available_shadow_refs):
        if impact.artifact_id in self._results:
            return self._results[impact.artifact_id]
        if impact.artifact_kind == "evaluation":
            bundle = evaluate(
                self.batch, self.labels, metrics=["max_drawdown"],
                portfolio_returns=self.portfolio,
            )
            grades = QEEvidenceProvider().grade_typed_metrics(
                bundle, self.batch, expected_config_hash=bundle.config_hash,
                expected_evaluation_ref=bundle.request_id,
                # Grade and aggregate under the same frozen replay policy.
                # A later default policy must not reinterpret historical units.
                policy=get_health_policy(policy_version="1.0.0"),
            )
            grade = next(g for g in grades if g.metric_id == "max_drawdown")
            payload = json.dumps({
                "metric_id": grade.metric_id, "metric_version": grade.metric_version,
                "value": grade.value, "config_hash": bundle.config_hash,
                "evaluation_ref": bundle.request_id,
            }, sort_keys=True).encode()
            self._evaluations[impact.artifact_id] = (grades, bundle)
        elif impact.artifact_kind == "health":
            if len(available_shadow_refs) != 1:
                raise ValueError("health replay requires exactly its shadow evaluation dependency")
            evaluation_old_id = next(iter(available_shadow_refs))
            if evaluation_old_id not in self._evaluations:
                raise ValueError("health dependency was not produced by this risk replay")
            grades, bundle = self._evaluations[evaluation_old_id]
            # OPS-08 replays the frozen pre-V5 max-drawdown health contract.
            # Selecting that policy explicitly keeps the real QE metric bound
            # to its matching typed dimension instead of silently interpreting
            # an absolute drawdown as V5 budget utilization.
            policy = get_health_policy(policy_version="1.0.0")
            dims = build_dimension_grades(
                factor_definition_id=grades[0].factor_definition_id,
                evaluation_ref=bundle.request_id,
                metric_grade_refs={g.metric_id: (g,) for g in grades}, policy=policy,
            )
            card = build_health_card(
                factor_definition_id=grades[0].factor_definition_id,
                evaluation_ref=bundle.request_id, dimension_grades=dims,
                integrity_gates={gate: None for gate in INTEGRITY_GATE_IDS}, policy=policy,
                production=True,
            )
            payload = json.dumps(card.to_dict(), sort_keys=True, default=str).encode()
        else:
            raise UnsupportedReplayKind(f"no real OPS-08 authority for {impact.artifact_kind!r}")
        digest = hashlib.sha256(payload).hexdigest()
        artifact_id = f"shadow:{impact.artifact_id}:{self.algorithm_version}:{digest[:12]}"
        ref = ArtifactRef(
            artifact_id=artifact_id, artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE,
            schema_version="1.0", content_hash=digest,
            storage_uri=f"test://ops08/{artifact_id}", size_bytes=len(payload),
            created_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
            producer_type="ops08-risk-replay", producer_version=self.algorithm_version,
        )
        result = ShadowReplayArtifact(
            impact.artifact_id, ref, payload, AUDIT_REPLAY_EVIDENCE_CLASS,
            tuple(sorted(available_shadow_refs)),
        )
        self._results[impact.artifact_id] = result
        return result


class RealParameterBridgeReplayAuthority:
    """Lazy recipe -> FP value -> QE IC/coverage -> FA health replay."""
    def __init__(self, *, frozen_raw_long: pd.DataFrame, frozen_batch_template,
                 frozen_labels, source_definition_ref: str, source_value_ref: str,
                 new_max_lag: int, algorithm_version: str):
        self.raw = frozen_raw_long.copy(deep=True)
        self.raw_digest = hashlib.sha256(self.raw.to_json().encode()).hexdigest()
        self.template = frozen_batch_template
        self.labels = frozen_labels
        self.definition_ref = source_definition_ref
        self.source_value_ref = source_value_ref
        self.max_lag = int(new_max_lag)
        self.version = algorithm_version
        self._evaluations = {}
        self._results = {}

    def _assert_raw(self):
        if hashlib.sha256(self.raw.to_json().encode()).hexdigest() != self.raw_digest:
            raise ValueError("frozen raw input mutated")

    def replay(self, impact, available_shadow_refs):
        if impact.artifact_id in self._results:
            return self._results[impact.artifact_id]
        self._assert_raw()
        if impact.artifact_kind == "recipe":
            recipe = TreatmentRecipe(
                recipe_id="fill-recipe-v2", source_factor_definition_ref=self.definition_ref,
                source_factor_value_ref=self.source_value_ref,
                ordered_steps=(RecipeStep(
                    step_id="fill", semantic_transform_id="forward_fill",
                    implementation_ref="forward_fill", stage="missingness",
                    parameters={"max_lag": self.max_lag},
                ),),
            )
            payload = json.dumps({"recipe_hash": recipe.content_hash,
                "max_lag": self.max_lag, "effective_max_periods": self.max_lag}, sort_keys=True).encode()
            self._recipe = recipe
        elif impact.artifact_kind == "value":
            if tuple(available_shadow_refs) != ("recipe:old",):
                raise ValueError("value requires direct shadow recipe bytes")
            recipe_doc = json.loads(available_shadow_refs["recipe:old"].domain_payload)
            if recipe_doc["effective_max_periods"] != self.max_lag:
                raise ValueError("recipe parameter bridge payload mismatch")
            executor = create_default_registry().get_recipe_execution(self._recipe)
            treated = executor(self.raw, value_col="value", time_col="date", asset_col="asset_id")
            times = list(self.template.time_axis.values); assets = list(self.template.asset_axis.values)
            frame = self.raw.assign(value=treated).pivot(index="date", columns="asset_id", values="value")
            array = frame.reindex(index=times, columns=assets).to_numpy(float)
            payload = json.dumps({"shape": list(array.shape), "values": array.tolist(),
                                  "recipe_hash": self._recipe.content_hash}, sort_keys=True).encode()
        elif impact.artifact_kind == "evaluation":
            if len(available_shadow_refs) != 1:
                raise ValueError("evaluation requires direct shadow value bytes")
            value_doc = json.loads(next(iter(available_shadow_refs.values())).domain_payload)
            array = np.asarray(value_doc["values"], dtype=float)
            batch = FactorBatch(
                self.template.factor_ids, self.template.time_axis, self.template.asset_axis,
                array[..., None], validity=np.isfinite(array[..., None]),
                context_refs=dict(self.template.context_refs) | {"factor_value_ref":
                    "shadow-value:" + hashlib.sha256(array.tobytes()).hexdigest()},
            )
            bundle = evaluate(batch, self.labels, metrics=["rank_ic", "coverage"])
            grades = QEEvidenceProvider().grade_typed_metrics(
                bundle, batch, expected_config_hash=bundle.config_hash,
                expected_evaluation_ref=bundle.request_id,
            )
            self._evaluations[impact.artifact_id] = (grades, bundle)
            payload = json.dumps({"evaluation_ref": bundle.request_id,
                "metrics": {g.metric_id: g.value for g in grades},
                "input_value_sha256": hashlib.sha256(array.tobytes()).hexdigest()}, sort_keys=True).encode()
        elif impact.artifact_kind == "health":
            if len(available_shadow_refs) != 1:
                raise ValueError("health requires direct shadow evaluation bytes")
            evaluation_old = next(iter(available_shadow_refs))
            grades, bundle = self._evaluations[evaluation_old]
            policy = get_health_policy()
            dims = build_dimension_grades(factor_definition_id=self.definition_ref,
                evaluation_ref=bundle.request_id,
                metric_grade_refs={g.metric_id: (g,) for g in grades}, policy=policy)
            card = build_health_card(factor_definition_id=self.definition_ref,
                evaluation_ref=bundle.request_id, dimension_grades=dims,
                integrity_gates={gate: None for gate in INTEGRITY_GATE_IDS}, policy=policy,
                production=True)
            payload = json.dumps(card.to_dict(), sort_keys=True, default=str).encode()
        else:
            raise UnsupportedReplayKind(f"no real parameter replay authority for {impact.artifact_kind!r}")
        self._assert_raw()
        digest = hashlib.sha256(payload).hexdigest()
        ref = ArtifactRef(artifact_id=f"shadow:{impact.artifact_id}:{self.version}:{digest[:12]}",
            artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE, schema_version="1.0",
            content_hash=digest, storage_uri=f"test://ops08/{impact.artifact_id}/{digest}",
            size_bytes=len(payload), created_at=datetime(2026,9,7,tzinfo=timezone.utc),
            producer_type="ops08-parameter-replay", producer_version=self.version)
        result = ShadowReplayArtifact(impact.artifact_id, ref, payload,
            AUDIT_REPLAY_EVIDENCE_CLASS, tuple(sorted(available_shadow_refs)))
        self._results[impact.artifact_id] = result
        return result
