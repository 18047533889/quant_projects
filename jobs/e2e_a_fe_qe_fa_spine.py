"""Executable upstream spine for V3 E2E-A using public FE/QE/FA adapters."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from data_access.read.semantic_catalog import (
    SemanticField,
    SemanticFieldCatalog,
    SemanticFieldTaxonomyProvider,
)
from factor_assets.adapters.factor_engine import FEIdentityProvider
from factor_assets.adapters.production_taxonomy import classify_definition_taxonomy
from factor_assets.adapters.quant_evaluator import QEEvidenceProvider
from factor_assets.profiling.dimensions import build_dimension_grades
from factor_assets.profiling.health_card import IntegrityGateResult, build_health_card
from factor_assets.profiling.policies import INTEGRITY_GATE_IDS, get_health_policy
from factor_engine.api.dsl_parser import parse_expr
from factor_engine.api.factor import Factor
from factor_engine.api.static_analysis import analyze_factor_definition
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.sources.datasource import DataSource, TemporalContract
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import authoritative_array_hash, evaluate


@dataclass(frozen=True)
class SyntheticSnapshotSource(DataSource):
    _data: Mapping[str, pd.Series]
    snapshot_ref: str

    @staticmethod
    def _snapshot_ref_for(data: Mapping[str, pd.Series]) -> str:
        parts = []
        for name in sorted(data):
            series = data[name]
            parts.extend((name, authoritative_array_hash(series.index.get_level_values(0).to_numpy()),
                          authoritative_array_hash(series.index.get_level_values(1).to_numpy()),
                          authoritative_array_hash(series.to_numpy())))
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return f"synthetic-snapshot:{digest}"

    @classmethod
    def from_data(cls, data: dict[str, pd.Series]):
        isolated = {name: series.copy(deep=True) for name, series in data.items()}
        return cls(MappingProxyType(isolated), cls._snapshot_ref_for(isolated))

    def _assert_snapshot(self):
        if self._snapshot_ref_for(self._data) != self.snapshot_ref:
            raise ValueError("synthetic source mutated after snapshot creation")

    def load_column(self, name):
        self._assert_snapshot()
        return self._data[name].copy(deep=True)

    def load_columns(self, names):
        self._assert_snapshot()
        return {name: self._data[name].copy(deep=True) for name in names}

    def temporal_contract(self):
        return TemporalContract("none", "token", "generic_asof")

    def execution_spec(self):
        return {"kind": "synthetic_snapshot", "snapshot_ref": self.snapshot_ref}


@dataclass(frozen=True)
class E2EAUpstreamResult:
    dsl: str
    snapshot_ref: str
    catalog_ref: str
    factor_definition_ref: str
    factor_value_ref: str
    taxonomy: object
    factor_batch: FactorBatch
    evaluation_bundle: object
    metric_grades: tuple
    label_fixture_kind: str


@dataclass(frozen=True)
class PITValidationArtifact:
    """Content-addressed result of checking the source's PIT contract."""
    source_snapshot_ref: str
    observed_knowledge_time: str
    required_knowledge_time: str
    passed: bool
    evidence_ref: str


@dataclass(frozen=True)
class E2EATrace:
    request_id: str
    parent_trial_ref: str | None
    factor_definition_ref: str
    recipe_ref: str | None
    factor_value_ref: str
    evaluation_ref: str
    health_policy_ref: str
    verdict_ref: str
    library_version_ref: str
    feature_version_ref: str | None
    failure_stage: str


@dataclass(frozen=True)
class E2EARejectionResult:
    frozen_direction: int
    training_rank_ic: float
    oos_rank_ic: float
    label_generation_rule: str
    oos_labels_ref: str
    catalog_ref: str
    evaluation_bundle: object
    pit_validation: PITValidationArtifact
    health_card: object
    admission_decision: object
    promotion_attempted: bool
    promotion_decision: object | None
    trace: E2EATrace


@dataclass(frozen=True)
class HealthAdmissionDecision:
    decision: str
    reason_codes: tuple[str, ...]
    factor_definition_ref: str
    evaluation_ref: str
    health_card_ref: str
    content_hash: str


def materialized_factor_value_ref(*, values, time_values, asset_values,
                                  snapshot_ref: str, factor_definition_ref: str) -> str:
    """Identity of executed values bound to definition, source, and axes."""
    payload = {
        "values_hash": authoritative_array_hash(values),
        "time_axis_hash": authoritative_array_hash(time_values),
        "asset_axis_hash": authoritative_array_hash(asset_values),
        "snapshot_ref": snapshot_ref,
        "factor_definition_ref": factor_definition_ref,
    }
    return "factor-value:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_e2e_a_fe_qe_fa_spine(*, include_factor_provenance: bool = True) -> E2EAUpstreamResult:
    """Run DSL/catalog/snapshot -> FE values -> QE metrics -> FA grades."""
    dsl = "rank(close)"
    dates = pd.bdate_range("2026-01-02", periods=30)
    assets = np.asarray([f"A{i:03d}" for i in range(40)], dtype=object)
    index = pd.MultiIndex.from_product([dates, assets], names=("timestamp", "instrument"))
    # Deterministic monotone cross section with a time-varying scale.
    close = pd.Series(
        np.concatenate([np.arange(40, dtype=float) * (1 + day / 100) + day
                        for day in range(30)]),
        index=index, name="close",
    )
    source = SyntheticSnapshotSource.from_data({"close": close})

    analysis = analyze_factor_definition(dsl)
    (field_id,) = tuple(usage.canonical_field_id for usage in analysis.field_usages)
    catalog = SemanticFieldCatalog({
        "market.close": SemanticField(
            logical_name="market.close", dataset="synthetic_daily",
            physical_name="close", aliases=(field_id, "close"),
            data_domains=("PRICE",), market="ashare", frequency="daily",
            grain="instrument", availability="same_day",
        )
    })
    catalog_ref = f"semantic-catalog:{catalog.get_identity(strict=True).digest}"
    taxonomy = classify_definition_taxonomy(
        dsl, taxonomy_provider=SemanticFieldTaxonomyProvider(catalog),
        field_roles={field_id: "alpha"},
    )
    identity = FEIdentityProvider().get_full_identity(dsl)
    factor_definition_ref = f"factor-definition:{identity.canonical_hash}"

    factor = Factor("e2e_a_monotone", parse_expr(dsl), source_expr=dsl)
    executed = FactorEngine(backend=PandasBackend(), data_source=source,
                            run_mode="research").run(factor)["result"]
    panel = executed.unstack("instrument").reindex(index=dates, columns=assets)
    values = panel.to_numpy(dtype=float)
    factor_value_ref = materialized_factor_value_ref(
        values=values, time_values=dates.to_numpy(), asset_values=assets,
        snapshot_ref=source.snapshot_ref,
        factor_definition_ref=factor_definition_ref,
    )
    context = {
        "snapshot_ref": source.snapshot_ref,
        "catalog_ref": catalog_ref,
    }
    if include_factor_provenance:
        context.update({
            "factor_definition_refs": {"e2e_a_monotone": factor_definition_ref},
            "factor_value_ref": factor_value_ref,
        })
    batch = FactorBatch(
        ("e2e_a_monotone",),
        AxisRef("time", str(dates.to_numpy().dtype), len(dates), dates.to_numpy()),
        AxisRef("asset", str(assets.dtype), len(assets), assets),
        values[..., None], validity=np.isfinite(values[..., None]),
        context_refs=context,
    )
    # Controlled self-correlated synthetic labels exercise QE mathematics.
    # They are not independent OOS evidence and cannot certify health/admission.
    label_fixture_kind = "CONTROLLED_SELF_CORRELATED_SYNTHETIC_NOT_OOS"
    labels = LabelBundle(
        "forward", values * .01, 1, decision_time=tuple(dates.to_numpy()),
        label_start_time=tuple((dates + pd.Timedelta(days=1)).to_numpy()),
        label_end_time=tuple((dates + pd.Timedelta(days=2)).to_numpy()),
        asset_axis=batch.asset_axis, source_ref=f"labels:{source.snapshot_ref}",
    )
    request_ref = "evaluation:" + hashlib.sha256(
        f"{factor_value_ref}|{source.snapshot_ref}|rank_ic|coverage|daily-q10".encode()
    ).hexdigest()
    bundle = evaluate(EvaluationRequest(
        batch, labels, metric_ids=("rank_ic", "coverage", "quantile_returns_daily"),
        tier="research",
        metric_parameters={
            "quantile_returns_daily": {"n_quantiles": 10, "min_assets": 20}
        },
        metadata={"request_id": request_ref, "label_fixture_kind": label_fixture_kind},
    ))
    grades = QEEvidenceProvider().grade_typed_metrics(
        bundle, batch, expected_config_hash=bundle.config_hash,
        expected_evaluation_ref=request_ref,
    )
    return E2EAUpstreamResult(
        dsl, source.snapshot_ref, catalog_ref, factor_definition_ref,
        factor_value_ref, taxonomy, batch, bundle, grades, label_fixture_kind,
    )


def _digest_ref(kind: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"{kind}:" + hashlib.sha256(encoded).hexdigest()


def run_e2e_a_oos_pit_rejection() -> E2EARejectionResult:
    """Run a real QE->FA health path whose high OOS IC is rejected for PIT.

    The label rule is declared before either sample is generated.  Training
    freezes the direction; OOS observations are then generated from a distinct
    deterministic sample and cannot alter it.
    """
    label_rule = "forward_return=0.02*asset_latent+0.0001*((day+asset)%7-3)"
    assets = np.asarray([f"A{i:03d}" for i in range(40)], dtype=object)

    def sample(start: str, periods: int, phase: int):
        dates = pd.bdate_range(start, periods=periods)
        index = pd.MultiIndex.from_product([dates, assets], names=("timestamp", "instrument"))
        latent = np.arange(40, dtype=float)
        close = pd.Series(np.concatenate([
            100.0 + latent * (1.0 + (day + phase) / 200.0) + day * .1
            for day in range(periods)
        ]), index=index, name="close")
        # This predetermined rule consumes latent generator inputs, not FE output.
        labels = np.stack([
            .02 * latent + .0001 * (((day + phase) + np.arange(40)) % 7 - 3)
            for day in range(periods)
        ])
        return dates, close, labels

    def execute(dates, close, raw_labels, phase_name: str, direction: int):
        source = SyntheticSnapshotSource.from_data({"close": close})
        dsl = "rank(close)"
        analysis = analyze_factor_definition(dsl)
        (field_id,) = tuple(usage.canonical_field_id for usage in analysis.field_usages)
        catalog = SemanticFieldCatalog({
            "market.close": SemanticField(
                logical_name="market.close", dataset="synthetic_daily",
                physical_name="close", aliases=(field_id, "close"),
                data_domains=("PRICE",), market="ashare", frequency="daily",
                grain="instrument", availability="same_day",
            )
        })
        catalog_ref = f"semantic-catalog:{catalog.get_identity(strict=True).digest}"
        definition_ref = f"factor-definition:{FEIdentityProvider().get_full_identity(dsl).canonical_hash}"
        factor = Factor("e2e_a_oos", parse_expr(dsl), source_expr=dsl)
        executed = FactorEngine(backend=PandasBackend(), data_source=source,
                                run_mode="research").run(factor)["result"]
        values = executed.unstack("instrument").reindex(index=dates, columns=assets).to_numpy(float)
        signed_values = values * direction
        value_ref = materialized_factor_value_ref(
            values=signed_values, time_values=dates.to_numpy(), asset_values=assets,
            snapshot_ref=source.snapshot_ref, factor_definition_ref=definition_ref,
        )
        batch = FactorBatch(
            ("e2e_a_oos",),
            AxisRef("time", str(dates.to_numpy().dtype), len(dates), dates.to_numpy()),
            AxisRef("asset", str(assets.dtype), len(assets), assets),
            signed_values[..., None], validity=np.isfinite(signed_values[..., None]),
            context_refs={
                "snapshot_ref": source.snapshot_ref,
                "catalog_ref": catalog_ref,
                "factor_definition_refs": {"e2e_a_oos": definition_ref},
                "factor_value_ref": value_ref,
            },
        )
        labels_ref = _digest_ref("labels", {
            "rule": label_rule, "phase": phase_name,
            "time": authoritative_array_hash(dates.to_numpy()),
            "values": authoritative_array_hash(raw_labels),
        })
        labels = LabelBundle(
            "forward", raw_labels, 1, decision_time=tuple(dates.to_numpy()),
            label_start_time=tuple((dates + pd.Timedelta(days=1)).to_numpy()),
            label_end_time=tuple((dates + pd.Timedelta(days=2)).to_numpy()),
            asset_axis=batch.asset_axis, source_ref=labels_ref,
        )
        request_ref = _digest_ref("evaluation", {
            "phase": phase_name, "factor_value_ref": value_ref,
            "labels_ref": labels_ref,
            "metrics": ["rank_ic", "coverage", "quantile_returns_daily"],
        })
        bundle = evaluate(EvaluationRequest(
            batch, labels,
            metric_ids=("rank_ic", "coverage", "quantile_returns_daily"),
            metric_parameters={
                "quantile_returns_daily": {"n_quantiles": 10, "min_assets": 20}
            },
            tier="research",
            metadata={"request_id": request_ref, "label_generation_rule": label_rule,
                      "direction_frozen_before_oos": phase_name == "oos"},
        ))
        return source, definition_ref, value_ref, labels_ref, request_ref, batch, bundle, catalog_ref

    train_dates, train_close, train_labels = sample("2026-01-02", 20, 0)
    train = execute(train_dates, train_close, train_labels, "train", 1)
    train_grades = QEEvidenceProvider().grade_typed_metrics(
        train[6], train[5], expected_config_hash=train[6].config_hash,
        expected_evaluation_ref=train[4],
    )
    train_ic = next(g.value for g in train_grades if g.metric_id == "rank_ic")
    frozen_direction = 1 if train_ic >= 0 else -1

    oos_dates, oos_close, oos_labels = sample("2026-05-04", 30, 53)
    oos = execute(oos_dates, oos_close, oos_labels, "oos", frozen_direction)
    grades = QEEvidenceProvider().grade_typed_metrics(
        oos[6], oos[5], expected_config_hash=oos[6].config_hash,
        expected_evaluation_ref=oos[4],
    )
    oos_ic = next(g.value for g in grades if g.metric_id == "rank_ic")

    policy = get_health_policy()
    dimensions = build_dimension_grades(
        factor_definition_id=oos[1], evaluation_ref=oos[4],
        metric_grade_refs={g.metric_id: (g,) for g in grades}, policy=policy,
    )
    temporal = oos[0].temporal_contract()
    pit_payload = {
        "snapshot_ref": oos[0].snapshot_ref,
        "observed_knowledge_time": temporal.temporal_sensitivity,
        "required_knowledge_time": "point_in_time",
        "result": "FAIL_MISSING_KNOWLEDGE_TIME",
    }
    pit = PITValidationArtifact(
        oos[0].snapshot_ref, temporal.temporal_sensitivity, "point_in_time", False,
        _digest_ref("pit-validation", pit_payload),
    )
    gates = tuple(
        IntegrityGateResult(
            gate_id=gate_id,
            passed=False if gate_id == "pit_valid" else None,
            evidence_ref=pit.evidence_ref if gate_id == "pit_valid" else "",
            detail=("source declares no knowledge-time contract"
                    if gate_id == "pit_valid" else "not evaluated after PIT failure"),
        ) for gate_id in INTEGRITY_GATE_IDS
    )
    health = build_health_card(
        factor_definition_id=oos[1], evaluation_ref=oos[4],
        dimension_grades=dimensions, integrity_gates=gates,
        policy=policy, production=True,
    )
    health_card_ref = _digest_ref("health-card-content", health.to_dict())
    admission_payload = {
        "decision": "REJECT", "reason_codes": ["pit_invalid"],
        "factor_definition_ref": oos[1], "evaluation_ref": oos[4],
        "health_card_ref": health_card_ref,
    }
    admission = HealthAdmissionDecision(
        "REJECT", ("pit_invalid",), oos[1], oos[4], health_card_ref,
        hashlib.sha256(json.dumps(
            admission_payload, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
    )
    # PromotionGate is deliberately not called: health admission is a strict
    # predecessor and has already rejected the candidate.  No library mutation
    # or feature version may exist after this terminal decision.
    library_ref = _digest_ref("library-version", {"members": [], "parent": None})
    trace = E2EATrace(
        request_id=oos[4], parent_trial_ref=None,
        factor_definition_ref=oos[1], recipe_ref=None,
        factor_value_ref=oos[2], evaluation_ref=oos[4],
        health_policy_ref=f"{policy.policy_id}:{policy.policy_version}",
        verdict_ref=f"health-admission-decision:{admission.content_hash}",
        library_version_ref=library_ref, feature_version_ref=None,
        failure_stage="PIT_INTEGRITY_GATE",
    )
    return E2EARejectionResult(
        frozen_direction, train_ic, oos_ic, label_rule, oos[3], oos[7], oos[6], pit,
        health, admission, False, None, trace,
    )
