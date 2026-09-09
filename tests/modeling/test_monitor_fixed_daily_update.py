import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from factor_engine.api import ts_ema
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.ir.analyzer import Analyzer
from factor_engine.runtime.incremental_parity import default_panel_factory
from factor_engine.runtime.stateful_checkpoint_store import StatefulCheckpointStore
from factor_engine.storage.catalog import compute_ir_hash
from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe
from jobs.e2e_h_fixed_update import run_fixed_daily_update
from jobs.monitor_fixed_daily_update import FixedMonitorPolicy, monitor_fixed_daily_update
from modeling.monitoring import DRIFT_DIMENSIONS, DriftContract, DriftThreshold
from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.contracts import ArtifactRef
from quant_platform.app.storage.generation_coordinator import DurableGenerationCoordinator


class _Publisher:
    def __init__(self):
        self.blobs = {}

    def publish(self, artifact, data):
        self.blobs[artifact.content_hash] = bytes(data)
        return artifact

    def open(self, artifact):
        data = self.blobs.get(artifact.content_hash)
        if data is None:
            raise FileNotFoundError(artifact.storage_uri)
        if hashlib.sha256(data).hexdigest() != artifact.content_hash:
            raise ValueError("stored bytes are corrupt")
        return bytes(data)


def _setup(tmp_path):
    factor = Factor(
        name="monitor_ema3", expr=ts_ema(col("close"), 3),
        source_expr="ts_ema(close, 3)", universe="synthetic:A,B",
    )
    definition = compute_ir_hash(Analyzer().lower(factor.expr).ir)
    recipe = TreatmentRecipe(
        recipe_id="monitor-rank-v1", source_factor_definition_ref=definition,
        source_factor_value_ref="snapshot:monitor",
        ordered_steps=(RecipeStep(
            "rank", "CS_RANK:pct", "rank", "representation", parameters={},
        ),), policy_identity="fixed-monitor:no-search",
    )
    panel = default_panel_factory(25)
    long = panel["close"].stack().rename("close").reset_index()
    long.columns = ["timestamp", "instrument", "close"]
    path = tmp_path / "source.parquet"
    long.to_parquet(path, index=False)
    source = {"type": "parquet", "root": str(path), "timestamp_col": "timestamp",
              "instrument_col": "instrument", "fields": {"close": "close"}}
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, _Publisher())
    store = StatefulCheckpointStore(tmp_path / "cp")
    artifact_id = "factor-value:monitor-history"
    first = run_fixed_daily_update(
        factor=factor, recipe=recipe, source_spec=source,
        start=panel.index[0], end=panel.index[10], bootstrap=True,
        checkpoint_store=store, generation_coordinator=coordinator,
        artifact_id=artifact_id,
    )
    second = run_fixed_daily_update(
        factor=factor, recipe=recipe, source_spec=source,
        start=panel.index[10], end=panel.index[11], bootstrap=False,
        checkpoint_store=store, generation_coordinator=coordinator,
        artifact_id=artifact_id,
    )
    times = tuple(panel.index[:12])
    assets = ("A", "B")
    generations = (first.generation_id, second.generation_id)
    return coordinator, artifact_id, generations, times, assets


def _labels(times, assets, values, ends):
    return LabelBundle(
        target_id="forward-return:h1", values=np.asarray(values, dtype=float), horizon=1,
        decision_time=times, signal_available_time=times, label_start_time=times,
        label_end_time=tuple(ends), observation_time=times,
        validity=np.ones((len(times), len(assets)), dtype=bool),
        source_ref="label-source:" + hashlib.sha256(b"labels").hexdigest(),
        calendar_ref="calendar:" + hashlib.sha256(b"sessions").hexdigest(),
        asset_axis=AxisRef("asset", "str", len(assets), values=np.asarray(assets)),
    )


def _policy():
    thresholds = {name: DriftThreshold(10.0) for name in DRIFT_DIMENSIONS}
    non_qe = {name: 0.0 for name in DRIFT_DIMENSIONS if name not in {"ic_decay", "coverage"}}
    return FixedMonitorPolicy(
        "fixed-monitor-policy-v1", min_mature_dates=5, min_assets=2,
        reference_ic=1.0, reference_coverage=1.0,
        external_drift_evidence=non_qe, drift_contract=DriftContract(thresholds),
    )


def test_only_mature_labels_reach_qe_and_future_label_changes_do_not_affect_monitor(tmp_path):
    coordinator, artifact_id, generations, times, assets = _setup(tmp_path)
    active_before = dict(coordinator.resolve_active(artifact_id))
    asof = times[5] + pd.Timedelta(hours=2)
    ends = [
        time + pd.Timedelta(hours=1) if i <= 5 else times[-1] + pd.Timedelta(days=i + 1)
        for i, time in enumerate(times)
    ]
    values = np.tile(np.asarray([0.0, 1.0]), (len(times), 1))
    first_labels = _labels(times, assets, values, ends)
    changed = values.copy()
    changed[6:] = np.asarray([9999.0, -9999.0])
    second_labels = _labels(times, assets, changed, ends)
    policy = _policy()
    first = monitor_fixed_daily_update(
        artifact_id=artifact_id, generation_coordinator=coordinator,
        artifact_resolver=coordinator.artifact_publisher,
        segment_generation_ids=generations, label_bundle=first_labels,
        asof=asof, monitor_policy=policy,
    )
    second = monitor_fixed_daily_update(
        artifact_id=artifact_id, generation_coordinator=coordinator,
        artifact_resolver=coordinator.artifact_publisher,
        segment_generation_ids=generations, label_bundle=second_labels,
        asof=asof, monitor_policy=policy,
    )
    assert first.status == second.status == "MONITORED_PARTIAL_EXTERNAL_DIMENSIONS"
    assert first.mature_time_count == second.mature_time_count == 6
    assert first.rank_ic == second.rank_ic
    assert first.coverage == second.coverage
    assert first.actions == second.actions
    assert first.monitor_policy_hash == policy.content_hash
    assert first.label_bundle_ref != second.label_bundle_ref
    assert first.valid_ic_date_count == second.valid_ic_date_count == 6
    assert first.evaluated_window_hash == second.evaluated_window_hash
    assert dict(coordinator.resolve_active(artifact_id)) == active_before
    reordered = monitor_fixed_daily_update(
        artifact_id=artifact_id, generation_coordinator=coordinator,
        artifact_resolver=coordinator.artifact_publisher,
        segment_generation_ids=tuple(reversed(generations)),
        label_bundle=first_labels, asof=asof, monitor_policy=policy,
    )
    assert reordered.rank_ic == first.rank_ic
    assert reordered.evaluated_window_hash != first.evaluated_window_hash


def test_no_mature_labels_returns_wait_without_calling_qe(tmp_path, monkeypatch):
    coordinator, artifact_id, generations, times, assets = _setup(tmp_path)
    labels = _labels(
        times, assets, np.ones((len(times), len(assets))),
        [time + pd.Timedelta(days=1) for time in times],
    )
    monkeypatch.setattr(
        "jobs.monitor_fixed_daily_update.evaluate",
        lambda *_args, **_kwargs: pytest.fail("immature labels must not reach QE"),
    )
    result = monitor_fixed_daily_update(
        artifact_id=artifact_id, generation_coordinator=coordinator,
        artifact_resolver=coordinator.artifact_publisher,
        segment_generation_ids=generations, label_bundle=labels,
        asof=times[0], monitor_policy=_policy(),
    )
    assert result.status == "WAIT_FOR_MATURITY"
    assert result.mature_time_count == 0


def test_too_few_mature_history_rows_is_insufficient_without_calling_qe(tmp_path, monkeypatch):
    coordinator, artifact_id, generations, times, assets = _setup(tmp_path)
    asof = times[2] + pd.Timedelta(hours=2)
    ends = [
        time + pd.Timedelta(hours=1) if i <= 2 else times[-1] + pd.Timedelta(days=i + 1)
        for i, time in enumerate(times)
    ]
    labels = _labels(times, assets, np.ones((len(times), len(assets))), ends)
    monkeypatch.setattr(
        "jobs.monitor_fixed_daily_update.evaluate",
        lambda *_args, **_kwargs: pytest.fail("insufficient samples must not reach QE"),
    )
    result = monitor_fixed_daily_update(
        artifact_id=artifact_id, generation_coordinator=coordinator,
        artifact_resolver=coordinator.artifact_publisher,
        segment_generation_ids=generations, label_bundle=labels,
        asof=asof, monitor_policy=_policy(),
    )
    assert result.status == "INSUFFICIENT_MATURE_SAMPLE"
    assert result.mature_time_count == 3


def test_many_mature_rows_but_too_few_valid_daily_ics_is_insufficient(tmp_path):
    coordinator, artifact_id, generations, times, assets = _setup(tmp_path)
    # Only one date has cross-sectional label variation. All rows are mature,
    # but mature cardinality cannot stand in for actual finite daily IC count.
    values = np.ones((len(times), len(assets)))
    values[0] = [0.0, 1.0]
    labels = _labels(
        times, assets, values,
        [time + pd.Timedelta(hours=1) for time in times],
    )
    result = monitor_fixed_daily_update(
        artifact_id=artifact_id, generation_coordinator=coordinator,
        artifact_resolver=coordinator.artifact_publisher,
        segment_generation_ids=generations, label_bundle=labels,
        asof=times[-1] + pd.Timedelta(days=1), monitor_policy=_policy(),
    )
    assert result.mature_time_count == len(times)
    assert result.valid_ic_date_count == 1
    assert result.status == "INSUFFICIENT_MATURE_SAMPLE"
    assert result.actions is None


def test_missing_label_refs_and_unregistered_segment_refs_fail_closed(tmp_path):
    coordinator, artifact_id, generations, times, assets = _setup(tmp_path)
    ends = [time + pd.Timedelta(hours=1) for time in times]
    labels = _labels(times, assets, np.ones((len(times), len(assets))), ends)
    broken = LabelBundle(
        target_id=labels.target_id, values=labels.values, horizon=labels.horizon,
        decision_time=labels.decision_time, signal_available_time=labels.signal_available_time,
        label_start_time=labels.label_start_time, label_end_time=labels.label_end_time,
        observation_time=labels.observation_time, validity=labels.validity,
        source_ref=None, calendar_ref=labels.calendar_ref, asset_axis=labels.asset_axis,
    )
    with pytest.raises(ValueError, match="source_ref and calendar_ref"):
        monitor_fixed_daily_update(
            artifact_id=artifact_id, generation_coordinator=coordinator,
            artifact_resolver=coordinator.artifact_publisher,
            segment_generation_ids=generations, label_bundle=broken,
            asof=times[-1] + pd.Timedelta(days=1), monitor_policy=_policy(),
        )
    with pytest.raises(ValueError, match="not a COMPLETE generation"):
        monitor_fixed_daily_update(
            artifact_id=artifact_id, generation_coordinator=coordinator,
            artifact_resolver=coordinator.artifact_publisher,
            segment_generation_ids=(generations[-1], "gen:v2:" + "0" * 64),
            label_bundle=labels, asof=times[-1] + pd.Timedelta(days=1),
            monitor_policy=_policy(),
        )
    active_hash = coordinator.resolve_active(artifact_id)["content_hash"]
    coordinator.artifact_publisher.blobs.pop(active_hash)
    with pytest.raises(ValueError, match="missing or unverifiable"):
        monitor_fixed_daily_update(
            artifact_id=artifact_id, generation_coordinator=coordinator,
            artifact_resolver=coordinator.artifact_publisher,
            segment_generation_ids=generations, label_bundle=labels,
            asof=times[-1] + pd.Timedelta(days=1), monitor_policy=_policy(),
        )


def test_overlapping_complete_segments_with_conflicting_values_are_rejected(tmp_path):
    coordinator, artifact_id, generations, times, assets = _setup(tmp_path)
    original = coordinator.db.query(
        "SELECT * FROM artifact_generations WHERE generation_id=?", (generations[0],)
    )[0]
    document = json.loads(bytes.fromhex(original["payload_hex"]))
    document["values"][0][0] += 0.25
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode()
    content_hash = hashlib.sha256(payload).hexdigest()
    artifact_data = json.loads(original["artifact_json"])
    artifact_data.update(
        content_hash=content_hash, size_bytes=len(payload),
        storage_uri=f"memory://{artifact_id}/{content_hash}",
    )
    artifact_data["created_at"] = datetime.fromisoformat(artifact_data["created_at"])
    conflict_id = coordinator.stage(ArtifactRef(**artifact_data), payload)
    coordinator.outbox.publish_pending()
    labels = _labels(
        times, assets, np.ones((len(times), len(assets))),
        [time + pd.Timedelta(hours=1) for time in times],
    )
    with pytest.raises(ValueError, match="overlapping segment coordinates"):
        monitor_fixed_daily_update(
            artifact_id=artifact_id, generation_coordinator=coordinator,
            artifact_resolver=coordinator.artifact_publisher,
            segment_generation_ids=(generations[0], conflict_id),
            label_bundle=labels, asof=times[-1] + pd.Timedelta(days=1),
            monitor_policy=_policy(),
        )
