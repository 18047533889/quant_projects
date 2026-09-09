"""Executable V3 E2E-C: Q10 trigger -> Q20 tail evidence -> bounded repair."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from data_access.read.semantic_catalog import SemanticField, SemanticFieldCatalog
from factor_assets.adapters.factor_engine import FEIdentityProvider
from factor_engine.api.dsl_parser import parse_expr
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_optimizer.search.supervised_parameter import (
    FrozenSupervisedParameter, TAIL_CUTOFF_GRID, fit_supervised_parameter,
)
from jobs.e2e_a_fe_qe_fa_spine import (
    SyntheticSnapshotSource, materialized_factor_value_ref,
)
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.adaptive_bins_policy import AdaptiveBinsPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.exposure_evidence import ExposurePanel
from quant_evaluator.runtime.evaluator import authoritative_array_hash, evaluate


@dataclass(frozen=True)
class E2ECResult:
    fixture_kind: str
    policy_branch: str
    snapshot_ref: str
    catalog_ref: str
    q10_evaluation: object
    adaptive_evaluation: Optional[object]
    selected_q: Optional[int]
    frozen_cutoff: Optional[FrozenSupervisedParameter]
    selected_family: Optional[str]
    train_scores: Mapping[str, float]
    candidate_dsls: Mapping[str, str]
    train_risk_evidence: Mapping[str, float]
    oos_evaluations: Mapping[str, object]
    trace: Mapping[str, object]


def _hash(prefix, payload):
    return prefix + hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _valid_public_value(bundle, metric_id, factor_id):
    """Return finite public scalar evidence, otherwise explicit insufficiency."""
    metric = bundle.get_metric(metric_id, factor_id)
    if not metric.valid or metric.observation_count <= 0 or metric.value is None:
        return None
    value = float(metric.value)
    return value if np.isfinite(value) else None


def _execute(source, dsl, dates, assets):
    result = FactorEngine(PandasBackend(), source, run_mode="research").run(
        Factor("e2e_c_factor", parse_expr(dsl), source_expr=dsl)
    )["result"]
    return result.unstack("instrument").reindex(index=dates, columns=assets).to_numpy(float)


def _evaluate(values, labels, dates, assets, *, factor_id, dsl, snapshot_ref,
              catalog_ref, label_ref, liquidity, size, metrics,
              quantile_parameters=None, adaptive_policy=None):
    definition_ref = "factor-definition:" + FEIdentityProvider().get_full_identity(
        dsl
    ).canonical_hash
    value_ref = materialized_factor_value_ref(
        values=values, time_values=dates, asset_values=assets,
        snapshot_ref=snapshot_ref, factor_definition_ref=definition_ref,
    )
    batch = FactorBatch(
        (factor_id,), AxisRef("time", str(dates.dtype), len(dates), dates),
        AxisRef("asset", str(assets.dtype), len(assets), assets),
        values[..., None], validity=np.isfinite(values[..., None]),
        context_refs={"snapshot_ref": snapshot_ref, "catalog_ref": catalog_ref,
                      "factor_definition_refs": {factor_id: definition_ref},
                      "factor_value_ref": value_ref},
    )
    bound_label_ref = _hash("label:", {
        "labels": authoritative_array_hash(labels),
        "time_axis": authoritative_array_hash(dates),
        "asset_axis": authoritative_array_hash(assets),
        "snapshot_ref": snapshot_ref,
    })
    labels_bundle = LabelBundle(
        "forward", labels, 1, decision_time=tuple(dates),
        label_start_time=tuple(dates + np.timedelta64(1, "D")),
        label_end_time=tuple(dates + np.timedelta64(2, "D")),
        asset_axis=batch.asset_axis, source_ref=bound_label_ref,
    )
    panel = ExposurePanel(
        np.stack((liquidity, size), axis=-1),
        style_names=("liquidity", "size"), source_ref="risk:" + snapshot_ref,
        provider="synthetic_data_access_fixture", date_index=tuple(dates),
        security_ids=tuple(assets), factor_ids=(factor_id,),
        universe_snapshot_ref=snapshot_ref,
        validity=np.isfinite(np.stack((liquidity, size), axis=-1)),
    )
    metric_parameters = {}
    builder_parameters = dict(quantile_parameters or {})
    if adaptive_policy is not None:
        metric_parameters["adaptive_quantile_count"] = {
            "policy": adaptive_policy.to_dict()
        }
    request_id = _hash("evaluation:", {
        "factor_value_ref": value_ref, "label_ref": bound_label_ref,
        "metrics": tuple(metrics), "quantile_parameters": builder_parameters,
        "adaptive_policy": None if adaptive_policy is None else adaptive_policy.to_dict(),
    })
    bundle = evaluate(EvaluationRequest(
        batch, labels_bundle, metric_ids=tuple(metrics), tier="research",
        cost_budget=20, metric_parameters=metric_parameters,
        quantile_builder_parameters=builder_parameters,
        exposure_panel=panel,
        metadata={"request_id": request_id, "label_ref": bound_label_ref},
    ))
    return bundle, definition_ref, value_ref


def run_e2e_c_tail_rescue(fixture_kind="q20", *, poison_oos=False):
    if fixture_kind not in {"q20", "small", "ties", "missing_risk"}:
        raise ValueError("fixture_kind must be q20, small, ties, or missing_risk")
    rng = np.random.default_rng(904)
    t, train_t = 48, 28
    n = 400 if fixture_kind != "small" else 120
    dates = pd.bdate_range("2026-01-02", periods=t).to_numpy()
    assets = np.asarray([f"A{i:04d}" for i in range(n)], dtype=object)
    raw = np.tile(np.linspace(-3, 3, n), (t, 1)) + rng.normal(0, .015, (t, n))
    if fixture_kind == "ties":
        # Ten genuine cross-sectional levels: Q10 remains feasible, while
        # tie-aware Q20 must not fabricate twenty buckets.
        raw = np.tile(np.repeat(np.arange(10, dtype=float), n // 10), (t, 1))
    rank = np.argsort(np.argsort(raw, axis=1), axis=1) / (n - 1)
    labels = rank + rng.normal(0, .035, (t, n))
    # The full top decile collapses, so the cheap Q10 screen sees a repeatable
    # anomaly and Q20 can subsequently localise it without changing labels.
    labels[rank >= .90] -= .80
    if poison_oos:
        labels[train_t:] = -labels[train_t:] + 3.0
    liquidity = np.log1p(np.tile(np.linspace(1e5, 5e7, n), (t, 1)))
    liquidity += rng.normal(0, .05, (t, n))
    size = np.tile(np.linspace(-1, 1, n)[::-1], (t, 1)) + rng.normal(0, .05, (t, n))
    if fixture_kind == "missing_risk":
        liquidity[:train_t] = np.nan

    index = pd.MultiIndex.from_product([dates, assets], names=("timestamp", "instrument"))
    source = SyntheticSnapshotSource.from_data({
        "close": pd.Series(raw.reshape(-1), index=index),
        "turnover_value": pd.Series(np.expm1(liquidity).reshape(-1), index=index),
        "market_cap": pd.Series(size.reshape(-1), index=index),
    })
    fields = (
        ("market.close", "close", "PRICE"),
        ("market.turnover_value", "turnover_value", "LIQUIDITY"),
        ("valuation.market_cap", "market_cap", "SIZE"),
    )
    catalog = SemanticFieldCatalog({name: SemanticField(
        logical_name=name, dataset="synthetic_e2e_c", physical_name=physical,
        aliases=(physical,), data_domains=(domain,), market="ashare",
        frequency="daily", grain="instrument", availability="same_day",
    ) for name, physical, domain in fields})
    catalog_ref = "semantic-catalog:" + catalog.get_identity(strict=True).digest
    parent_dsl = "rank(close)"
    parent = _execute(source, parent_dsl, dates, assets)
    label_ref = _hash("label:", {
        "labels": authoritative_array_hash(labels),
        "time_axis": authoritative_array_hash(dates),
        "asset_axis": authoritative_array_hash(assets),
        "snapshot_ref": source.snapshot_ref,
    })
    q10, parent_definition_ref, parent_value_ref = _evaluate(
        parent[:train_t], labels[:train_t], dates[:train_t], assets,
        factor_id="parent_q10", dsl=parent_dsl, snapshot_ref=source.snapshot_ref,
        catalog_ref=catalog_ref, label_ref=label_ref, liquidity=liquidity[:train_t],
        size=size[:train_t], metrics=("rank_ic", "quantile_returns_daily",
            "top_quantile_cliff_robust", "liquidity_exposure", "size_exposure"),
        quantile_parameters={"n_quantiles": 10, "min_assets": 5},
    )
    q10_cliff = _valid_public_value(q10, "top_quantile_cliff_robust", "parent_q10")
    if q10_cliff is None:
        return E2ECResult(
            fixture_kind, "Q10_EVIDENCE_INSUFFICIENT_NO_REPAIR", source.snapshot_ref,
            catalog_ref, q10, None, None, None, None, {}, {}, {}, {},
            {"request_id": q10.request_id, "parent_trial": None,
             "factor_definition": parent_definition_ref, "recipe": None,
             "value": parent_value_ref, "evaluation": q10.request_id,
             "health_policy": None, "verdict": "Q10_INSUFFICIENT_NO_REPAIR",
             "library": None, "feature_version": None,
             "pipeline_status": "PARTIAL_NOT_PUBLISHED"},
        )
    if q10_cliff >= 0:
        raise AssertionError("fixture did not produce the public Q10 tail trigger")

    policy = AdaptiveBinsPolicy(preferred_bins=20, fallback_bins=(10, 5),
                                min_effective_names_per_bin=10,
                                policy_id="E2E_C_TIE_AWARE_BINS",
                                policy_version="1.0.0")
    adaptive, _, _ = _evaluate(
        parent[:train_t], labels[:train_t], dates[:train_t], assets,
        factor_id="parent_adaptive", dsl=parent_dsl, snapshot_ref=source.snapshot_ref,
        catalog_ref=catalog_ref, label_ref=label_ref, liquidity=liquidity[:train_t],
        size=size[:train_t], metrics=("adaptive_quantile_count", "quantile_returns_daily"),
        adaptive_policy=policy,
    )
    selected_q_value = _valid_public_value(
        adaptive, "adaptive_quantile_count", "parent_adaptive"
    )
    selected_q = int(selected_q_value) if selected_q_value is not None else None
    if selected_q != 20:
        return E2ECResult(
            fixture_kind, "DEGRADED_NO_Q20_REPAIR", source.snapshot_ref, catalog_ref,
            q10, adaptive, selected_q, None, None, {}, {}, {
                "q10_top_cliff": q10_cliff,
                "liquidity_exposure": q10.get_metric("liquidity_exposure", "parent_q10").value,
                "size_exposure": q10.get_metric("size_exposure", "parent_q10").value,
            }, {}, {"request_id": adaptive.request_id, "parent_trial": None,
                "factor_definition": parent_definition_ref, "recipe": None,
                "value": parent_value_ref, "evaluation": adaptive.request_id,
                "health_policy": None, "verdict": "Q20_INFEASIBLE_NO_REPAIR",
                "library": None, "feature_version": None,
                "pipeline_status": "PARTIAL_NOT_PUBLISHED"},
        )

    q20, _, _ = _evaluate(
        parent[:train_t], labels[:train_t], dates[:train_t], assets,
        factor_id="parent_q20", dsl=parent_dsl, snapshot_ref=source.snapshot_ref,
        catalog_ref=catalog_ref, label_ref=label_ref, liquidity=liquidity[:train_t],
        size=size[:train_t], metrics=("rank_ic", "quantile_returns_daily",
            "top_quantile_cliff_robust", "liquidity_exposure", "size_exposure"),
        quantile_parameters={"n_quantiles": 20, "min_assets": 10},
    )
    q20_cliff = _valid_public_value(q20, "top_quantile_cliff_robust", "parent_q20")
    liquidity_exposure = _valid_public_value(q20, "liquidity_exposure", "parent_q20")
    size_exposure = _valid_public_value(q20, "size_exposure", "parent_q20")
    if q20_cliff is None or liquidity_exposure is None or size_exposure is None:
        return E2ECResult(
            fixture_kind, "Q20_RISK_EVIDENCE_INSUFFICIENT_NO_REPAIR",
            source.snapshot_ref, catalog_ref, q10, adaptive, selected_q,
            None, None, {}, {}, {"q10_top_cliff": q10_cliff,
                "q20_top_cliff": q20_cliff,
                "liquidity_exposure": liquidity_exposure,
                "size_exposure": size_exposure}, {},
            {"request_id": q20.request_id, "parent_trial": None,
             "factor_definition": parent_definition_ref, "recipe": None,
             "value": parent_value_ref, "evaluation": q20.request_id,
             "health_policy": None,
             "verdict": "Q20_OR_RISK_EVIDENCE_INSUFFICIENT_NO_REPAIR",
             "library": None, "feature_version": None,
             "pipeline_status": "PARTIAL_NOT_PUBLISHED"},
        )
    if q20_cliff >= 0:
        raise AssertionError("Q20 public evidence did not confirm top-tail collapse")

    candidate_dsls = {}
    scores = {}
    family_best = {}
    train_split_ref = _hash("split:", {
        "role": "TRAIN", "dates": authoritative_array_hash(dates[:train_t]),
        "snapshot_ref": source.snapshot_ref,
    })
    frozen_by_family = {}
    for family in ("TAIL_SATURATION", "TAIL_HINGE"):
        family_scores = {}
        for cutoff in TAIL_CUTOFF_GRID:
            if family == "TAIL_SATURATION":
                dsl = f"minimum(rank(close), {cutoff:.2f})"
            else:
                dsl = (f"subtract(rank(close), multiply(2, maximum("
                       f"subtract(rank(close), {cutoff:.2f}), 0)))")
            key = f"{family}:{cutoff:.2f}"
            candidate_dsls[key] = dsl
            values = _execute(source, dsl, dates, assets)
            ev = _evaluate(
                values[:train_t], labels[:train_t], dates[:train_t], assets,
                factor_id=key, dsl=dsl, snapshot_ref=source.snapshot_ref,
                catalog_ref=catalog_ref, label_ref=label_ref,
                liquidity=liquidity[:train_t], size=size[:train_t],
                metrics=("rank_ic",),
            )[0]
            score = ev.get_metric("rank_ic", key).value
            family_scores[cutoff] = score
            scores[key] = score
        frozen = fit_supervised_parameter(
            parent_factor_id=parent_definition_ref, repair_family=family,
            parameter_name="cutoff", candidate_grid=TAIL_CUTOFF_GRID,
            train_scores=family_scores, split_role="TRAIN",
            train_split_ref=train_split_ref,
            training_evidence_ref=_hash("evaluation-grid:", family_scores),
            objective_id="rank_ic",
        )
        frozen_by_family[family] = frozen
        family_best[family] = family_scores[frozen.value]
    selected_family = max(family_best, key=lambda name: (family_best[name], name))
    frozen = frozen_by_family[selected_family]
    selected_key = f"{selected_family}:{frozen.value:.2f}"
    selected_dsl = candidate_dsls[selected_key]
    selected_values = _execute(source, selected_dsl, dates, assets)
    oos = {}
    for factor_id, dsl, values in (
        ("parent_oos", parent_dsl, parent),
        ("selected_oos", selected_dsl, selected_values),
    ):
        oos[factor_id] = _evaluate(
            values[train_t:], labels[train_t:], dates[train_t:], assets,
            factor_id=factor_id, dsl=dsl, snapshot_ref=source.snapshot_ref,
            catalog_ref=catalog_ref, label_ref=label_ref,
            liquidity=liquidity[train_t:], size=size[train_t:],
            metrics=("rank_ic", "quantile_returns_daily", "top_quantile_cliff_robust",
                     "liquidity_exposure", "size_exposure"),
            quantile_parameters={"n_quantiles": 20, "min_assets": 10},
        )[0]
    selected_definition = "factor-definition:" + FEIdentityProvider().get_full_identity(
        selected_dsl
    ).canonical_hash
    selected_value_ref = materialized_factor_value_ref(
        values=selected_values[train_t:], time_values=dates[train_t:], asset_values=assets,
        snapshot_ref=source.snapshot_ref, factor_definition_ref=selected_definition,
    )
    return E2ECResult(
        fixture_kind, "Q20_CONFIRMED_BOUNDED_REPAIR", source.snapshot_ref, catalog_ref,
        q10, adaptive, selected_q, frozen, selected_family, scores, candidate_dsls,
        {"q10_top_cliff": q10_cliff, "q20_top_cliff": q20_cliff,
         "liquidity_exposure": liquidity_exposure,
         "size_exposure": size_exposure},
        oos, {"request_id": oos["selected_oos"].request_id, "parent_trial": None,
              "factor_definition": selected_definition, "recipe": None,
              "value": selected_value_ref,
              "evaluation": oos["selected_oos"].request_id, "health_policy": None,
              "verdict": "BOUNDED_TAIL_REPAIR_REEVALUATED", "library": None,
              "feature_version": None, "selection_split": "TRAIN",
              "frozen_parameter_state": frozen.state_hash,
              "oos_role": "EVALUATION_ONLY",
              "pipeline_status": "PARTIAL_NOT_PUBLISHED"},
    )
