"""Executable V3 E2E-B U/inverted-U rescue with train-frozen centre."""
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
    FrozenSupervisedParameter, U_CENTER_GRID, fit_supervised_parameter,
)
from jobs.e2e_a_fe_qe_fa_spine import (
    SyntheticSnapshotSource, materialized_factor_value_ref,
)
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import authoritative_array_hash, evaluate


@dataclass(frozen=True)
class E2EBResult:
    shape_kind: str
    policy_branch: str
    snapshot_ref: str
    catalog_ref: str
    parent_dsl: str
    frozen_center: Optional[FrozenSupervisedParameter]
    train_scores: Mapping[float, float]
    child_dsls: Mapping[str, str]
    parent_train_evaluation: object
    oos_evaluations: Mapping[str, object]
    trace: Mapping[str, object]


def _execute(source, dsl, name, dates, assets):
    result = FactorEngine(PandasBackend(), source, run_mode="research").run(
        Factor(name, parse_expr(dsl), source_expr=dsl)
    )["result"]
    return result.unstack("instrument").reindex(index=dates, columns=assets).to_numpy(float)


def _qe(values, labels, dates, assets, factor_id, snapshot_ref, catalog_ref,
        *, shape=False):
    dsl = factor_id.split("|", 1)[-1]
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
        context_refs={
            "snapshot_ref": snapshot_ref,
            "catalog_ref": catalog_ref,
            "factor_definition_refs": {factor_id: definition_ref},
            "factor_value_ref": value_ref,
        },
    )
    bundle = LabelBundle(
        "forward", labels, 1, decision_time=tuple(dates),
        label_start_time=tuple(dates + np.timedelta64(1, "D")),
        label_end_time=tuple(dates + np.timedelta64(2, "D")),
        asset_axis=batch.asset_axis,
        source_ref="label:" + hashlib.sha256(json.dumps({
            "labels_hash": authoritative_array_hash(labels),
            "time_axis_hash": authoritative_array_hash(dates),
            "asset_axis_hash": authoritative_array_hash(assets),
            "snapshot_ref": snapshot_ref,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )
    metrics = (("rank_ic", "quantile_returns_daily", "u_shape_score",
                "inverted_u_score", "quantile_monotonicity",
                "quantile_curvature") if shape else ("rank_ic",))
    return evaluate(EvaluationRequest(
        batch, bundle, metric_ids=metrics, tier="research",
        cost_budget=8 if shape else 1,
        metric_parameters={
            "quantile_returns_daily": {"n_quantiles": 10, "min_assets": 5}
        } if shape else {},
        quantile_builder_parameters={"n_quantiles": 10, "min_assets": 5}
        if shape else {},
        metadata={"request_id": "evaluation:" + hashlib.sha256(
            f"{factor_id}|{catalog_ref}|{value_ref}|{bundle.source_ref}".encode()
        ).hexdigest()},
    ))


def run_e2e_b_shape_rescue(shape_kind: str, *, poison_oos: bool = False) -> E2EBResult:
    if shape_kind not in {"u", "inverted_u", "monotonic"}:
        raise ValueError("shape_kind must be u, inverted_u, or monotonic")
    rng = np.random.default_rng(771)
    t, n, train_t = 50, 60, 30
    dates = pd.bdate_range("2026-01-02", periods=t).to_numpy()
    assets = np.asarray([f"A{i:03d}" for i in range(n)], dtype=object)
    base = np.tile(np.linspace(-2, 2, n), (t, 1)) + rng.normal(scale=.04, size=(t, n))
    ranks = np.argsort(np.argsort(base, axis=1), axis=1) / (n - 1)
    centre = .50
    if shape_kind == "u":
        labels = np.abs(ranks - centre) + rng.normal(scale=.02, size=(t, n))
    elif shape_kind == "inverted_u":
        labels = -np.abs(ranks - centre) + rng.normal(scale=.02, size=(t, n))
    else:
        labels = ranks + rng.normal(scale=.02, size=(t, n))
    if poison_oos:
        labels[train_t:] = -labels[train_t:] + 7.0

    index = pd.MultiIndex.from_product([dates, assets], names=("timestamp", "instrument"))
    source = SyntheticSnapshotSource.from_data({
        "close": pd.Series(base.reshape(-1), index=index, name="close")
    })
    catalog = SemanticFieldCatalog({
        "market.close": SemanticField(
            logical_name="market.close", dataset="synthetic_e2e_b",
            physical_name="close", aliases=("close",), data_domains=("PRICE",),
            market="ashare", frequency="daily", grain="instrument",
            availability="same_day",
        )
    })
    catalog_ref = f"semantic-catalog:{catalog.get_identity(strict=True).digest}"
    parent_dsl = "rank(close)"
    parent = _execute(source, parent_dsl, parent_dsl, dates, assets)
    parent_train = _qe(parent[:train_t], labels[:train_t], dates[:train_t], assets,
                       f"parent|{parent_dsl}", source.snapshot_ref, catalog_ref,
                       shape=True)
    parent_id = f"parent|{parent_dsl}"
    mono = parent_train.get_metric("quantile_monotonicity", parent_id).value
    u_score = parent_train.get_metric("u_shape_score", parent_id).value
    inv_score = parent_train.get_metric("inverted_u_score", parent_id).value
    curvature = parent_train.get_metric("quantile_curvature", parent_id).value
    if not np.all(np.isfinite((mono, u_score, inv_score, curvature))):
        raise ValueError("TRAIN public shape evidence is insufficient")
    if mono >= max(u_score, inv_score):
        parent_oos = _qe(parent[train_t:], labels[train_t:], dates[train_t:], assets,
                         f"parent|{parent_dsl}", source.snapshot_ref, catalog_ref,
                         shape=True)
        return E2EBResult(
            shape_kind, "MONOTONIC_NO_SHAPE_REPAIR", source.snapshot_ref, catalog_ref,
            parent_dsl, None, {}, {}, parent_train, {"parent": parent_oos},
            {"center_source": None, "selection_split": "TRAIN",
             "oos_role": "EVALUATION_ONLY", "pipeline_status": "PARTIAL_NOT_PUBLISHED"},
        )

    # Direction is supplied by QE's signed TRAIN curvature evidence.  The two
    # template scores establish non-monotone shape strength but are not a
    # reliable orientation tie-break because their free-scale fits can tie.
    orientation = 1.0 if curvature > 0.0 else -1.0
    train_scores = {}
    for c in U_CENTER_GRID:
        core = f"abs(subtract(rank(close), {c:.2f}))"
        dsl = core if orientation > 0 else f"subtract(0, {core})"
        values = _execute(source, dsl, dsl, dates, assets)
        ev = _qe(values[:train_t], labels[:train_t], dates[:train_t], assets,
                 f"candidate|{dsl}", source.snapshot_ref, catalog_ref)
        train_scores[c] = ev.get_metric("rank_ic", f"candidate|{dsl}").value
    frozen = fit_supervised_parameter(
        parent_factor_id="parent:" + FEIdentityProvider().get_full_identity(parent_dsl).canonical_hash,
        repair_family="U_SHAPE_REPAIR", parameter_name="center",
        candidate_grid=U_CENTER_GRID, train_scores=train_scores, split_role="TRAIN",
        train_split_ref="split:" + hashlib.sha256(json.dumps({
            "role": "TRAIN", "dates_hash": authoritative_array_hash(dates[:train_t]),
            "snapshot_ref": source.snapshot_ref,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        training_evidence_ref="evaluation-grid:" + hashlib.sha256(
            repr(sorted(train_scores.items())).encode()
        ).hexdigest(), objective_id="rank_ic",
    )
    c = frozen.recipe_parameters(split_role="TEST")["center"]
    left = f"maximum(subtract({c:.2f}, rank(close)), 0)"
    right = f"maximum(subtract(rank(close), {c:.2f}), 0)"
    recombined_core = f"abs(subtract(rank(close), {c:.2f}))"
    recombined = recombined_core if orientation > 0 else f"subtract(0, {recombined_core})"
    child_dsls = {"left": left, "right": right, "recombined": recombined}
    oos = {"parent": _qe(parent[train_t:], labels[train_t:], dates[train_t:], assets,
                          f"parent|{parent_dsl}", source.snapshot_ref, catalog_ref,
                          shape=True)}
    for name, dsl in child_dsls.items():
        values = _execute(source, dsl, dsl, dates, assets)
        oos[name] = _qe(values[train_t:], labels[train_t:], dates[train_t:], assets,
                        f"{name}|{dsl}", source.snapshot_ref, catalog_ref,
                        shape=True)
    return E2EBResult(
        shape_kind, "U_SHAPE_RESCUE" if orientation > 0 else "INVERTED_U_RESCUE",
        source.snapshot_ref, catalog_ref, parent_dsl, frozen, train_scores,
        child_dsls, parent_train, oos,
        {"center_source": frozen.training_evidence_ref, "selection_split": "TRAIN",
         "oos_role": "EVALUATION_ONLY", "parent_evaluation": oos["parent"].request_id,
         "child_evaluations": {k: v.request_id for k, v in oos.items() if k != "parent"},
         "pipeline_status": "PARTIAL_NOT_HEALTH_GRADED_OR_PUBLISHED"},
    )
