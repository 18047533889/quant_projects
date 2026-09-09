"""Executable E2E-F campaign spine: bounded noise evaluations into GC inputs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_access.read.semantic_catalog import SemanticField, SemanticFieldCatalog
from factor_engine.api.dsl_parser import parse_expr
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_assets.adapters.factor_engine import FEIdentityProvider
from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore
from factor_optimizer.contracts.trial_ledger import TrialLedger
from factor_assets.adapters.quant_evaluator import QEEvidenceProvider
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import authoritative_array_hash, evaluate

from jobs.e2e_a_fe_qe_fa_spine import SyntheticSnapshotSource, materialized_factor_value_ref


@dataclass(frozen=True)
class E2EFResult:
    campaign_id: str
    attempted_proposals: int
    completed_evaluations: int
    budget_exhausted: bool
    budget_reservation_denied: bool
    ledger_path: str
    ledger_head_hash: str
    formula_history: tuple[str, ...]
    trial_history: tuple[str, ...]
    evaluation_history: tuple[str, ...]
    observed_rank_ics: tuple[float, ...]
    rejected_value_bytes: tuple[tuple[str, bytes], ...]


def run_e2e_f_campaign(work_dir: str | Path) -> E2EFResult:
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    campaign_id = "e2e-f-noise"
    campaign = SQLiteCampaignStore(root / "campaign.sqlite3")
    campaign.create_campaign(campaign_id, max_evaluations=2, max_cost=2.0)
    ledger = TrialLedger()
    formulas, trials, evaluations, rank_ics = [], [], [], []
    rejected_values = []
    reservation_denied = False
    reject_abs_rank_ic_below = 0.10

    dates = pd.bdate_range("2026-02-02", periods=24)
    assets = np.asarray([f"N{i:03d}" for i in range(40)], dtype=object)
    index = pd.MultiIndex.from_product([dates, assets], names=("timestamp", "instrument"))
    catalog = SemanticFieldCatalog({
        "market.close": SemanticField(
            logical_name="market.close", dataset="e2e_f_synthetic", physical_name="close",
            aliases=("close",), data_domains=("PRICE",), market="ashare",
            frequency="daily", grain="instrument", availability="same_day",
        )
    })
    catalog_ref = f"semantic-catalog:{catalog.get_identity(strict=True).digest}"

    attempted = 0
    for proposal_no in range(5):
        attempt_id = f"noise-{proposal_no}"
        if not campaign.reserve(campaign_id, attempt_id, 1.0):
            reservation_denied = True
            break
        attempted += 1
        campaign.start(campaign_id, attempt_id)
        trial_id = f"trial:{campaign_id}:{proposal_no}"
        dsl = "rank(close)"
        formula_ref = "factor-definition:" + FEIdentityProvider().get_full_identity(dsl).canonical_hash
        ledger.append("PROPOSED", trial_id=trial_id)

        rng_x = np.random.default_rng(1000 + proposal_no)
        rng_y = np.random.default_rng(9000 + proposal_no)
        close = pd.Series(rng_x.normal(size=len(index)), index=index, name="close")
        source = SyntheticSnapshotSource.from_data({"close": close})
        values = FactorEngine(backend=PandasBackend(), data_source=source, run_mode="research").run(
            Factor("noise", parse_expr(dsl), source_expr=dsl)
        )["result"].unstack("instrument").reindex(index=dates, columns=assets).to_numpy(float)
        value_ref = materialized_factor_value_ref(
            values=values, time_values=dates.to_numpy(), asset_values=assets,
            snapshot_ref=source.snapshot_ref, factor_definition_ref=formula_ref,
        )
        batch = FactorBatch(
            ("noise",), AxisRef("time", str(dates.to_numpy().dtype), len(dates), dates.to_numpy()),
            AxisRef("asset", str(assets.dtype), len(assets), assets), values[..., None],
            validity=np.isfinite(values[..., None]), context_refs={
                "snapshot_ref": source.snapshot_ref, "catalog_ref": catalog_ref,
                "factor_definition_refs": {"noise": formula_ref}, "factor_value_ref": value_ref,
            },
        )
        independent_noise = rng_y.normal(size=values.shape)
        labels_ref = "labels:" + hashlib.sha256("|".join((
            authoritative_array_hash(independent_noise),
            authoritative_array_hash(dates.to_numpy()),
            authoritative_array_hash(assets),
        )).encode()).hexdigest()
        labels = LabelBundle(
            "forward", independent_noise, 1, decision_time=tuple(dates.to_numpy()),
            label_start_time=tuple((dates + pd.Timedelta(days=1)).to_numpy()),
            label_end_time=tuple((dates + pd.Timedelta(days=2)).to_numpy()),
            asset_axis=batch.asset_axis, source_ref=labels_ref,
        )
        evaluation_ref = "evaluation:" + hashlib.sha256(
            f"{value_ref}|{labels_ref}|rank_ic".encode()
        ).hexdigest()
        bundle = evaluate(EvaluationRequest(
            batch, labels, metric_ids=("rank_ic",), tier="research",
            metadata={"request_id": evaluation_ref, "fixture_kind": "INDEPENDENT_PURE_NOISE"},
        ))
        grades = QEEvidenceProvider().grade_typed_metrics(
            bundle, batch, expected_config_hash=bundle.config_hash,
            expected_evaluation_ref=evaluation_ref,
        )
        rank_ics.append(float(next(g.value for g in grades if g.metric_id == "rank_ic")))
        campaign.settle(campaign_id, attempt_id, 1.0)
        ledger.append("EVALUATED", trial_id=trial_id)
        observed_ic = rank_ics[-1]
        if abs(observed_ic) < reject_abs_rank_ic_below:
            ledger.append("PRUNED", trial_id=trial_id,
                          failure_reason=f"RANK_IC_BELOW_POLICY:{reject_abs_rank_ic_below}")
        else:
            ledger.append("SELECTED", trial_id=trial_id)
        formulas.append(formula_ref); trials.append(trial_id); evaluations.append(evaluation_ref)
        blob = values.astype("<f8").tobytes(order="C")
        if abs(observed_ic) < reject_abs_rank_ic_below:
            rejected_values.append((value_ref, blob))

    ledger.seal(); ledger.verify_chain()
    ledger_path = root / "trial-ledger.json"
    ledger_path.write_text(json.dumps(ledger.to_dict(), sort_keys=True), encoding="utf-8")

    state = campaign.budget_state(campaign_id)
    return E2EFResult(
        campaign_id, attempted, int(state["evaluations_used"]),
        bool(reservation_denied and state["evaluations_used"] == state["max_evaluations"]),
        reservation_denied, str(ledger_path), ledger.sealed_head_hash or "",
        tuple(formulas), tuple(trials), tuple(evaluations), tuple(rank_ics),
        tuple(rejected_values),
    )
