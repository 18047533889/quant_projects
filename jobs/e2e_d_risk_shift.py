"""Executable V3 E2E-D composition: risk neutralization with RAW protection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from data_access.read.semantic_catalog import SemanticField, SemanticFieldCatalog
from factor_engine.api.dsl_parser import parse_expr
from factor_engine.api.factor import Factor
from factor_engine.runtime.engine import FactorEngine
from factor_engine.backend.pandas_backend import PandasBackend
from factor_assets.adapters.factor_engine import FEIdentityProvider
from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe
from factor_preprocess.neutralization.ols import ols_neutralize
from factor_preprocess.representation.policy import (
    SignalDestructionConflict, non_inferior,
)
from factor_optimizer.search.pareto import ParetoFrontier, ParetoPoint
from factor_optimizer.search.winner_selector import (
    WinnerPolicy, WinnerSetPolicy, select_complementary_winners,
)
from jobs.e2e_a_fe_qe_fa_spine import SyntheticSnapshotSource
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.exposure_evidence import ExposurePanel
from quant_evaluator.runtime.evaluator import authoritative_array_hash, evaluate


@dataclass(frozen=True)
class E2EDResult:
    scenario: str
    raw_dsl: str
    snapshot_ref: str
    catalog_ref: str
    candidate_dsls: tuple[str, ...]
    evaluations: Mapping[str, object]
    winner_id: str
    raw_preserved: bool
    noninferiority_conflicts: tuple[str, ...]
    trace: Mapping[str, object]


def _long_frame(dates, assets, values, name):
    return pd.DataFrame({
        "date": np.repeat(dates, len(assets)),
        "asset_id": np.tile(assets, len(dates)),
        name: np.asarray(values).reshape(-1),
    })


def _neutralize(raw, exposure, dates, assets, exposure_name):
    values = _long_frame(dates, assets, raw, "value")
    risks = _long_frame(dates, assets, exposure, exposure_name)
    return ols_neutralize(
        values, risks, date_col="date", asset_col="asset_id",
        value_col="value", min_observations=10,
    ).to_numpy().reshape(raw.shape)


def _evaluate_candidate(candidate_id, values, labels, risk, dates, assets, snapshot_ref,
                        catalog_ref, definition_ref, value_ref, label_ref, scenario):
    batch = FactorBatch(
        (candidate_id,),
        AxisRef("time", str(dates.dtype), len(dates), dates),
        AxisRef("asset", str(assets.dtype), len(assets), assets),
        values[..., None], validity=np.isfinite(values[..., None]),
        context_refs={"snapshot_ref": snapshot_ref, "catalog_ref": catalog_ref,
                      "factor_definition_refs": {candidate_id: definition_ref},
                      "factor_value_ref": value_ref},
    )
    label_bundle = LabelBundle(
        "forward", labels, 1, decision_time=tuple(dates),
        label_start_time=tuple(dates + np.timedelta64(1, "D")),
        label_end_time=tuple(dates + np.timedelta64(2, "D")),
        asset_axis=batch.asset_axis, source_ref=label_ref,
    )
    panel = ExposurePanel(
        risk, style_names=("industry", "size"), source_ref=f"risk:{snapshot_ref}",
        provider="synthetic_data_access_fixture", date_index=tuple(dates),
        security_ids=tuple(assets), factor_ids=(candidate_id,),
        universe_snapshot_ref=snapshot_ref, validity=np.isfinite(risk),
    )
    request_id = "evaluation:" + hashlib.sha256(
        f"{scenario}|{value_ref}|{label_ref}|rank_ic|coverage|exposure".encode()
    ).hexdigest()
    return evaluate(EvaluationRequest(
        batch, label_bundle,
        metric_ids=("rank_ic", "coverage", "industry_exposure", "size_exposure"),
        tier="extended", cost_budget=6, exposure_panel=panel,
        metadata={"request_id": request_id, "label_ref": label_ref,
                  "scenario": scenario},
    ))


def run_e2e_d_risk_shift(*, utility_damage: bool = False) -> E2EDResult:
    """Run bounded synthetic E2E-D success or signal-destruction scenario."""
    scenario = "utility_damage" if utility_damage else "exposure_improvement"
    rng = np.random.default_rng(630 if utility_damage else 629)
    t, n = 40, 60
    dates = pd.bdate_range("2026-01-02", periods=t).to_numpy()
    assets = np.asarray([f"A{i:03d}" for i in range(n)], dtype=object)
    industry = np.tile(np.repeat(np.asarray([-1.0, 0.0, 1.0]), n // 3), (t, 1))
    size = rng.normal(size=(t, n))
    alpha = rng.normal(size=(t, n))
    if utility_damage:
        raw = 1.3 * industry + 1.3 * size + .15 * alpha
        forward = raw + rng.normal(scale=.05, size=(t, n))
    else:
        raw = alpha + .25 * industry + .25 * size
        forward = alpha + rng.normal(scale=.10, size=(t, n))

    index = pd.MultiIndex.from_product([dates, assets], names=("timestamp", "instrument"))
    source = SyntheticSnapshotSource.from_data({
        "close": pd.Series(raw.reshape(-1), index=index, name="close"),
        "industry_code": pd.Series(industry.reshape(-1), index=index, name="industry_code"),
        "market_cap": pd.Series(size.reshape(-1), index=index, name="market_cap"),
    })
    catalog = SemanticFieldCatalog({
        name: SemanticField(
            logical_name=name, dataset="synthetic_e2e_d", physical_name=physical,
            aliases=(physical,), data_domains=(domain,), market="ashare",
            frequency="daily", grain="instrument", availability="same_day",
        ) for name, physical, domain in (
            ("market.close", "close", "PRICE"),
            ("classification.industry", "industry_code", "INDUSTRY"),
            ("valuation.market_cap", "market_cap", "SIZE"),
        )
    })
    catalog_ref = f"semantic-catalog:{catalog.get_identity(strict=True).digest}"
    raw_dsl = "close"
    definition_ref = "factor-definition:" + FEIdentityProvider().get_full_identity(
        raw_dsl
    ).canonical_hash
    executed = FactorEngine(PandasBackend(), source, run_mode="research").run(
        Factor("raw", parse_expr(raw_dsl), source_expr=raw_dsl)
    )["result"].unstack("instrument").reindex(index=dates, columns=assets)
    raw_values = executed.to_numpy(dtype=float)

    candidates = {
        "raw": raw_values,
        "industry_neutral": _neutralize(raw_values, industry, dates, assets, "industry"),
        "size_neutral": _neutralize(raw_values, size, dates, assets, "size"),
    }
    value_refs = {
        cid: "factor-value:" + hashlib.sha256(
            f"{definition_ref}|{source.snapshot_ref}|{cid}|"
            f"{authoritative_array_hash(values)}|"
            f"{authoritative_array_hash(dates)}|{authoritative_array_hash(assets)}".encode()
        ).hexdigest()
        for cid, values in candidates.items()
    }
    recipes: dict[str, Optional[TreatmentRecipe]] = {"raw": None}
    for cid, semantic in (("industry_neutral", "industry_neutral"),
                          ("size_neutral", "size_neutral")):
        recipes[cid] = TreatmentRecipe(
            recipe_id=f"e2e-d:{cid}", source_factor_definition_ref=definition_ref,
            source_factor_value_ref=value_refs["raw"],
            ordered_steps=(RecipeStep(
                step_id=f"step:{cid}", semantic_transform_id=semantic,
                implementation_ref="factor_preprocess.neutralization.ols.ols_neutralize",
                stage="neutralization", parameters={"min_observations": 10},
            ),),
            policy_identity="E2E_D_PREDECLARED_NEUTRALIZATION_V1",
        )
    label_ref = "label-bundle:" + hashlib.sha256(
        f"{scenario}|{source.snapshot_ref}|{authoritative_array_hash(forward)}|"
        f"{authoritative_array_hash(dates)}|{authoritative_array_hash(assets)}".encode()
    ).hexdigest()
    risk = np.stack([industry, size], axis=-1)
    evaluations = {
        cid: _evaluate_candidate(
            cid, values, forward, risk, dates, assets, source.snapshot_ref,
            catalog_ref, definition_ref, value_refs[cid], label_ref, scenario,
        )
        for cid, values in candidates.items()
    }
    raw_ic = evaluations["raw"].get_metric("rank_ic", "raw").value
    conflicts = []
    points = []
    robustness = {}
    complexity = {}
    raw_exposure = sum(abs(evaluations["raw"].get_metric(mid, "raw").value)
                       for mid in ("industry_exposure", "size_exposure"))
    for cid, bundle in evaluations.items():
        rank_ic = bundle.get_metric("rank_ic", cid).value
        exposure = sum(abs(bundle.get_metric(mid, cid).value)
                       for mid in ("industry_exposure", "size_exposure"))
        eligible = True
        if cid != "raw":
            try:
                non_inferior(raw_ic, rank_ic)
            except SignalDestructionConflict:
                eligible = False
                conflicts.append(cid)
        point = ParetoPoint(cid, ((rank_ic + 1.0) / 2.0, 1.0 / (1.0 + exposure)), {
            "eligible": eligible,
            "family_id": "raw" if cid == "raw" else "neutralization",
            "purpose": "E2E_D_RISK_SHIFT",
            "incremental_value": 0.0 if cid == "raw" else raw_exposure - exposure,
        })
        points.append(point)
        robustness[cid] = bundle.get_metric("coverage", cid).value
        complexity[cid] = 0.0 if cid == "raw" else 0.25
    frontier = ParetoFrontier(["rank_ic", "exposure_cleanliness"])
    for point in points:
        frontier.add(point)
    selected = select_complementary_winners(
        frontier.frontier(), robustness, complexity,
        WinnerPolicy(.5, .4, .1, .05, "E2E_D_BALANCED", "1.0.0"),
        WinnerSetPolicy(max_winners=1, max_per_family=1, minimum_incremental_value=0.0),
    )
    winner_id = selected[0].trial_id if selected else "raw"
    if winner_id != "raw" and winner_id in conflicts:
        raise AssertionError("non-inferior authority admitted a destructive treatment")
    trace_seed = "|".join((scenario, source.snapshot_ref, catalog_ref, winner_id))
    trace_id = hashlib.sha256(trace_seed.encode()).hexdigest()
    trace = {
        "request_id": f"e2e-d:{trace_id}",
        "parent_trial": None,
        "factor_definition": definition_ref,
        "recipe": None if recipes[winner_id] is None else recipes[winner_id].content_hash,
        "value": value_refs[winner_id],
        "evaluation": evaluations[winner_id].request_id,
        "health_policy": None,
        "noninferiority_policy": "NON_INFERIORITY_POLICY:2026-09-05.1",
        "verdict": "RAW_PRESERVED" if winner_id == "raw" else "NEUTRALIZATION_SELECTED",
        "library": None,
        "feature_version": None,
        "pipeline_status": "PARTIAL_NOT_PUBLISHED_NOT_HEALTH_GRADED",
    }
    return E2EDResult(
        scenario, raw_dsl, source.snapshot_ref, catalog_ref,
        ("group_neutralize(close, industry_code)", "size_neutralize(close, market_cap)"),
        evaluations, winner_id, winner_id == "raw", tuple(conflicts), trace,
    )
