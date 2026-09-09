import pytest
import fcntl
import json
import multiprocessing
import os
import hashlib
import pandas as pd

from factor_engine.api import ts_ema, ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.ir.analyzer import Analyzer
from factor_engine.runtime.incremental_parity import default_panel_factory
from factor_engine.runtime.stateful_checkpoint_store import StatefulCheckpointStore
from factor_engine.stateful_runtime import execute_stateful_segment
from factor_engine.backend.cleaned_bridge import execute_operator_recipe
from factor_engine.storage.catalog import compute_ir_hash
from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe
from factor_preprocess.contracts.state import FittedState, StateKind
from factor_preprocess.adapters.fitted_recipe import (
    FITTED_STANDARDIZE_NAME,
    FITTED_STANDARDIZE_VERSION,
    fitted_standardize_implementation_hash,
)
from jobs.e2e_h_fixed_update import recover_fixed_daily_update, run_fixed_daily_update
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.storage.generation_coordinator import DurableGenerationCoordinator


class _Publisher:
    def __init__(self):
        self.blobs = {}

    def publish(self, artifact, data):
        assert artifact.content_hash == __import__("hashlib").sha256(data).hexdigest()
        self.blobs[artifact.content_hash] = bytes(data)
        return artifact


def _die_during_stage(db_path, checkpoint_root, source_spec, start, end, artifact_id):
    factor, recipe = _factor_and_recipe()

    def die(stage, _generation_id):
        if stage == "transaction_before_commit":
            os._exit(91)

    coordinator = DurableGenerationCoordinator(
        SqliteDb(str(db_path)), _Publisher(), failure_injector=die
    )
    run_fixed_daily_update(
        factor=factor, recipe=recipe, source_spec=source_spec,
        start=start, end=end, bootstrap=False,
        checkpoint_store=StatefulCheckpointStore(checkpoint_root),
        generation_coordinator=coordinator, artifact_id=artifact_id,
    )


def _attempt_locked_recovery(db_path, checkpoint_root, artifact_id, result_queue):
    try:
        recover_fixed_daily_update(
            checkpoint_store=StatefulCheckpointStore(checkpoint_root),
            generation_coordinator=DurableGenerationCoordinator(
                SqliteDb(str(db_path)), _Publisher()
            ),
            artifact_id=artifact_id,
        )
    except Exception as exc:
        result_queue.put(f"{type(exc).__name__}: {exc}")


def _factor_and_recipe():
    factor = Factor(
        name="fixed_ema3", expr=ts_ema(col("close"), 3),
        source_expr="ts_ema(close, 3)", universe="synthetic:A,B",
    )
    definition_hash = compute_ir_hash(Analyzer().lower(factor.expr).ir)
    recipe = TreatmentRecipe(
        recipe_id="fixed-ema-rank-v1",
        source_factor_definition_ref=definition_hash,
        source_factor_value_ref="snapshot:fixed-update-v1",
        ordered_steps=(RecipeStep(
            "rank", "CS_RANK:pct", "rank", "representation", parameters={},
        ),),
        policy_identity="fixed-update:no-search-v1",
    )
    return factor, recipe


def _source_spec(panel, path):
    long = panel["close"].stack().rename("close").reset_index()
    long.columns = ["timestamp", "instrument", "close"]
    long.to_parquet(path, index=False)
    return {
        "type": "parquet", "root": str(path),
        "timestamp_col": "timestamp", "instrument_col": "instrument",
        "fields": {"close": "close"},
    }


def test_fixed_daily_checkpoint_restart_matches_full_and_failed_publish_keeps_active(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("quant_platform.app.outbox.time.time", lambda: 1000.0)
    factor, recipe = _factor_and_recipe()
    publisher = _Publisher()
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, publisher)
    panel = default_panel_factory(40)
    source_spec = _source_spec(panel, tmp_path / "daily40.parquet")
    checkpoint_root = tmp_path / "checkpoints"
    first = run_fixed_daily_update(
        factor=factor, recipe=recipe, source_spec=source_spec,
        start=panel.index[0], end=panel.index[20], bootstrap=True,
        checkpoint_store=StatefulCheckpointStore(checkpoint_root),
        generation_coordinator=coordinator, artifact_id="factor-value:fixed-ema-rank",
    )
    assert first.bootstrap and first.row_count == 42
    second = run_fixed_daily_update(
        factor=factor, recipe=recipe, source_spec=source_spec,
        start=panel.index[20], end=panel.index[21], bootstrap=False,
        checkpoint_store=StatefulCheckpointStore(checkpoint_root),
        generation_coordinator=coordinator, artifact_id="factor-value:fixed-ema-rank",
    )
    assert not second.bootstrap and second.row_count == 4
    active_before = dict(coordinator.resolve_active("factor-value:fixed-ema-rank"))
    assert active_before["content_hash"] == second.content_hash

    full = {}
    for instrument in ("A", "B"):
        full[instrument] = execute_stateful_segment(
            "ts_ema", {"x": panel["close"][instrument].to_numpy()},
            timestamps=panel.index, instrument=instrument,
            input_identity={"oracle": "full"}, params={"span": 3},
            starts_at_dataset_origin=True,
        ).values
    full_rank = execute_operator_recipe(pd.DataFrame(full, index=panel.index), (("rank", {}),))
    published = json.loads(publisher.blobs[second.content_hash])
    assert published["dates"] == [value.isoformat() for value in panel.index[20:22]]
    assert published["values"] == full_rank.iloc[20:22].to_numpy().tolist()

    armed = {"value": True}

    def fail(stage, _generation_id):
        if stage == "complete_before_commit" and armed["value"]:
            armed["value"] = False
            raise RuntimeError("injected publish failure")

    failing = DurableGenerationCoordinator(db, publisher, failure_injector=fail)
    with pytest.raises(RuntimeError, match="not atomically activated"):
        run_fixed_daily_update(
            factor=factor, recipe=recipe, source_spec=source_spec,
            start=panel.index[21], end=panel.index[22], bootstrap=False,
            checkpoint_store=StatefulCheckpointStore(checkpoint_root),
            generation_coordinator=failing, artifact_id="factor-value:fixed-ema-rank",
        )
    active_after = dict(failing.resolve_active("factor-value:fixed-ema-rank"))
    assert active_after["generation_id"] == active_before["generation_id"]
    assert active_after["content_hash"] == active_before["content_hash"]
    assert len(publisher.blobs) == 3  # failed generation bytes may exist but remain invisible

    # The checkpoint already advanced, so recovery must publish the exact
    # persisted outbox payload rather than recomputing (and potentially losing)
    # the unpublished segment.
    assert list(checkpoint_root.glob("**/*.json"))
    reopened_store = StatefulCheckpointStore(checkpoint_root)
    assert reopened_store is not None
    monkeypatch.setattr("quant_platform.app.outbox.time.time", lambda: 1002.0)
    retry = DurableGenerationCoordinator(db, publisher)
    assert retry.outbox.publish_pending() == 1
    recovered = dict(retry.resolve_active("factor-value:fixed-ema-rank"))
    assert recovered["generation_id"] != active_before["generation_id"]
    recovered_payload = json.loads(publisher.blobs[recovered["content_hash"]])
    assert recovered_payload["dates"] == [value.isoformat() for value in panel.index[21:23]]
    assert recovered_payload["values"] == full_rank.iloc[21:23].to_numpy().tolist()


def test_fixed_update_rejects_recipe_for_other_executable_definition(tmp_path):
    factor, recipe = _factor_and_recipe()
    wrong = Factor(name="other", expr=ts_ema(col("close"), 5), source_expr="ts_ema(close, 5)")
    db = SqliteDb(str(tmp_path / "metadata.db"))
    with pytest.raises(ValueError, match="recipe does not match"):
        run_fixed_daily_update(
            factor=wrong, recipe=recipe,
            source_spec={"type": "parquet", "root": str(tmp_path / "unused")},
            start="2024-01-01", end="2024-01-02", bootstrap=True,
            checkpoint_store=StatefulCheckpointStore(tmp_path / "cp"),
            generation_coordinator=DurableGenerationCoordinator(db, _Publisher()),
            artifact_id="factor-value:wrong",
        )


def test_fixed_update_rejects_unsupported_ir_before_checkpoint_or_publication(tmp_path):
    factor = Factor(name="mean3", expr=ts_mean(col("close"), 3), source_expr="ts_mean(close, 3)")
    definition_hash = compute_ir_hash(Analyzer().lower(factor.expr).ir)
    recipe = TreatmentRecipe(
        recipe_id="unsupported-mean-rank", source_factor_definition_ref=definition_hash,
        source_factor_value_ref="source:close",
        ordered_steps=(RecipeStep(
            "rank", "CS_RANK:pct", "rank", "representation", parameters={},
        ),),
    )
    checkpoint_root = tmp_path / "cp"
    db = SqliteDb(str(tmp_path / "metadata.db"))
    with pytest.raises(ValueError, match="not supported"):
        run_fixed_daily_update(
            factor=factor, recipe=recipe,
            source_spec={"type": "parquet", "root": str(tmp_path / "unused")},
            start="2024-01-01", end="2024-01-02", bootstrap=True,
            checkpoint_store=StatefulCheckpointStore(checkpoint_root),
            generation_coordinator=DurableGenerationCoordinator(db, _Publisher()),
            artifact_id="factor-value:unsupported",
        )
    assert not list(checkpoint_root.glob("**/*.json"))
    assert db.query("SELECT * FROM artifact_generations") == []


def test_in_place_source_revision_cannot_resume_old_checkpoint(tmp_path):
    factor, recipe = _factor_and_recipe()
    panel = default_panel_factory(30)
    path = tmp_path / "source.parquet"
    spec = _source_spec(panel, path)
    db = SqliteDb(str(tmp_path / "metadata.db"))
    publisher = _Publisher()
    coordinator = DurableGenerationCoordinator(db, publisher)
    checkpoint_root = tmp_path / "cp"
    first = run_fixed_daily_update(
        factor=factor, recipe=recipe, source_spec=spec,
        start=panel.index[0], end=panel.index[20], bootstrap=True,
        checkpoint_store=StatefulCheckpointStore(checkpoint_root),
        generation_coordinator=coordinator, artifact_id="factor-value:revision-guard",
    )
    active = dict(coordinator.resolve_active("factor-value:revision-guard"))
    assert active["content_hash"] == first.content_hash

    revised = panel.copy()
    revised.loc[revised.index[19], ("close", "A")] = -999.0
    _source_spec(revised, path)  # same path, changed source bytes
    with pytest.raises(RuntimeError, match="checkpoint input fingerprint mismatch"):
        run_fixed_daily_update(
            factor=factor, recipe=recipe, source_spec=spec,
            start=panel.index[20], end=panel.index[21], bootstrap=False,
            checkpoint_store=StatefulCheckpointStore(checkpoint_root),
            generation_coordinator=coordinator, artifact_id="factor-value:revision-guard",
        )
    assert dict(coordinator.resolve_active("factor-value:revision-guard"))["content_hash"] == first.content_hash


def test_stage_process_death_recovers_saved_segment_without_reexecution(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("quant_platform.app.outbox.time.time", lambda: 2000.0)
    factor, recipe = _factor_and_recipe()
    panel = default_panel_factory(30)
    source_spec = _source_spec(panel, tmp_path / "source.parquet")
    checkpoint_root = tmp_path / "cp"
    store = StatefulCheckpointStore(checkpoint_root)
    db = SqliteDb(str(tmp_path / "metadata.db"))
    publisher = _Publisher()
    artifact_id = "factor-value:stage-crash-recovery"
    coordinator = DurableGenerationCoordinator(db, publisher)
    first = run_fixed_daily_update(
        factor=factor, recipe=recipe, source_spec=source_spec,
        start=panel.index[0], end=panel.index[20], bootstrap=True,
        checkpoint_store=store, generation_coordinator=coordinator,
        artifact_id=artifact_id,
    )
    active_before = dict(coordinator.resolve_active(artifact_id))

    process = multiprocessing.get_context("spawn").Process(
        target=_die_during_stage,
        args=(tmp_path / "metadata.db", checkpoint_root, source_spec,
              panel.index[20], panel.index[21], artifact_id),
    )
    process.start()
    process.join(30)
    assert process.exitcode == 91
    assert dict(coordinator.resolve_active(artifact_id)) == active_before
    assert pd.Timestamp(store.latest_as_of(factor.name, "ts_ema", "A")) == pd.Timestamp(
        panel.index[20], tz="UTC"
    )
    assert len(db.query(
        "SELECT * FROM outbox_events WHERE correlation_id=?", (artifact_id,)
    )) == 1  # only the already-published bootstrap intent exists
    assert len(db.query(
        "SELECT * FROM artifact_generations WHERE artifact_id=?", (artifact_id,)
    )) == 1

    # Recovery consumes the durable segment journal. FE/FP execution is not
    # entered again, and the already-advanced checkpoint is harmless.
    monkeypatch.setattr(
        "jobs.e2e_h_fixed_update.try_stateful_segmented_incremental",
        lambda **_kwargs: pytest.fail("recovery must not reexecute FE"),
    )
    monkeypatch.setattr(
        "jobs.e2e_h_fixed_update.execute_operator_recipe",
        lambda *_args, **_kwargs: pytest.fail("recovery must not reexecute FP"),
    )
    recovered = recover_fixed_daily_update(
        checkpoint_store=StatefulCheckpointStore(checkpoint_root),
        generation_coordinator=DurableGenerationCoordinator(db, publisher),
        artifact_id=artifact_id,
    )
    assert recovered.content_hash != first.content_hash
    published = json.loads(publisher.blobs[recovered.content_hash])
    assert published["dates"] == [value.isoformat() for value in panel.index[20:22]]


def test_same_checkpoint_root_is_cross_process_single_writer(tmp_path):
    checkpoint_root = tmp_path / "cp"
    checkpoint_root.mkdir()
    lock_path = checkpoint_root / ".fixed_update.lock"
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    with lock_path.open("a+b") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = context.Process(
            target=_attempt_locked_recovery,
            args=(tmp_path / "metadata.db", checkpoint_root,
                  "factor-value:locked", result_queue),
        )
        process.start()
        process.join(30)
        assert process.exitcode == 0
        assert "another fixed update or recovery owns" in result_queue.get(timeout=5)
    # Releasing the owner lock exposes the normal no-journal result, proving
    # process death/release does not leave a stale logical lock behind.
    with pytest.raises(RuntimeError, match="no persisted fixed-update segment"):
        recover_fixed_daily_update(
            checkpoint_store=StatefulCheckpointStore(checkpoint_root),
            generation_coordinator=DurableGenerationCoordinator(
                SqliteDb(str(tmp_path / "metadata.db")), _Publisher()
            ),
            artifact_id="factor-value:locked",
        )


def test_source_change_during_execution_commits_neither_checkpoint_nor_publication(
    tmp_path, monkeypatch
):
    factor, recipe = _factor_and_recipe()
    panel = default_panel_factory(30)
    path = tmp_path / "source.parquet"
    source_spec = _source_spec(panel, path)
    checkpoint_root = tmp_path / "cp"
    db = SqliteDb(str(tmp_path / "metadata.db"))
    real_execute = execute_operator_recipe

    def mutate_after_read(panel_value, steps):
        result = real_execute(panel_value, steps)
        revised = panel.copy()
        revised.loc[revised.index[3], ("close", "B")] = -1234.0
        _source_spec(revised, path)
        return result

    monkeypatch.setattr(
        "jobs.e2e_h_fixed_update.execute_operator_recipe", mutate_after_read
    )
    with pytest.raises(RuntimeError, match="source changed during fixed update"):
        run_fixed_daily_update(
            factor=factor, recipe=recipe, source_spec=source_spec,
            start=panel.index[0], end=panel.index[20], bootstrap=True,
            checkpoint_store=StatefulCheckpointStore(checkpoint_root),
            generation_coordinator=DurableGenerationCoordinator(db, _Publisher()),
            artifact_id="factor-value:toctou-guard",
        )
    assert not list(checkpoint_root.glob("**/*.json"))
    assert db.query("SELECT * FROM artifact_generations") == []


def test_frozen_fitted_state_is_required_and_matches_full_oracle_despite_environment_defaults(
    tmp_path, monkeypatch
):
    factor, rank_recipe = _factor_and_recipe()
    panel = default_panel_factory(35)
    source_spec = _source_spec(panel, tmp_path / "source.parquet")
    checkpoint_root = tmp_path / "cp"
    store = StatefulCheckpointStore(checkpoint_root)
    db = SqliteDb(str(tmp_path / "metadata.db"))
    publisher = _Publisher()
    coordinator = DurableGenerationCoordinator(db, publisher)
    seed = run_fixed_daily_update(
        factor=factor, recipe=rank_recipe, source_spec=source_spec,
        start=panel.index[0], end=panel.index[20], bootstrap=True,
        checkpoint_store=store, generation_coordinator=coordinator,
        artifact_id="factor-value:fitted-seed",
    )

    full = {}
    for instrument in ("A", "B"):
        full[instrument] = execute_stateful_segment(
            "ts_ema", {"x": panel["close"][instrument].to_numpy()},
            timestamps=panel.index, instrument=instrument,
            input_identity={"oracle": "full-fitted"}, params={"span": 3},
            starts_at_dataset_origin=True,
        ).values
    full_panel = pd.DataFrame(full, index=panel.index)
    train_values = full_panel.iloc[:20].to_numpy()
    learned_mean = float(train_values.mean())
    learned_scale = float(train_values.std())
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    state = FittedState(
        transform_name=FITTED_STANDARDIZE_NAME,
        transform_version=FITTED_STANDARDIZE_VERSION,
        fit_start_time=panel.index[0].to_pydatetime(),
        fit_end_time=panel.index[19].to_pydatetime(),
        state_kind=StateKind.FITTED,
        feature_ids=[rank_recipe.source_factor_definition_ref],
        feature_order=[rank_recipe.source_factor_definition_ref],
        learned_params={"mean": learned_mean, "scale": learned_scale},
        implementation_hash=fitted_standardize_implementation_hash(),
        data_snapshot_ref=seed.snapshot_ref,
        split_ref=digest("train:0:20"), universe_ref=digest(factor.universe),
        calendar_ref=digest("|".join(map(str, panel.index[:20]))),
        fit_coordinate_hash=digest("|".join(
            [*(map(str, panel.index[:20])), "A", "B"]
        )),
        policy_hash=digest("train-standardize:population-v1"),
        production=True,
    )
    fitted_recipe = TreatmentRecipe(
        recipe_id="fixed-ema-fitted-standardize-v1",
        source_factor_definition_ref=rank_recipe.source_factor_definition_ref,
        source_factor_value_ref=seed.content_hash,
        ordered_steps=(RecipeStep(
            "fitted-standardize", "TRAIN_STANDARDIZE:population-v1",
            FITTED_STANDARDIZE_NAME, "representation", requires_fit=True,
            state_ref=state.state_id, parameters={},
        ),),
        policy_identity="fixed-update:no-refit-v1",
    )
    checkpoint_before = store.latest_as_of(factor.name, "ts_ema", "A")
    with pytest.raises(ValueError, match="missing fitted state"):
        run_fixed_daily_update(
            factor=factor, recipe=fitted_recipe, source_spec=source_spec,
            start=panel.index[20], end=panel.index[21], bootstrap=False,
            checkpoint_store=store, generation_coordinator=coordinator,
            artifact_id="factor-value:fitted",
        )
    assert store.latest_as_of(factor.name, "ts_ema", "A") == checkpoint_before
    assert coordinator.resolve_active("factor-value:fitted") is None

    monkeypatch.setenv("FP_STANDARDIZE_DDOF", "99")
    monkeypatch.setenv("FP_STANDARDIZE_SCALE", "999999")
    result = run_fixed_daily_update(
        factor=factor, recipe=fitted_recipe, source_spec=source_spec,
        start=panel.index[20], end=panel.index[21], bootstrap=False,
        checkpoint_store=store, generation_coordinator=coordinator,
        artifact_id="factor-value:fitted", fitted_states={state.state_id: state},
    )
    assert result.fitted_state_refs == (state.state_id,)
    published = json.loads(publisher.blobs[result.content_hash])
    expected = (full_panel.iloc[20:22] - learned_mean) / learned_scale
    assert published["values"] == expected.to_numpy().tolist()
    assert published["fitted_state_refs"] == [state.state_id]


def test_same_factor_name_cannot_graft_state_from_different_dsl(tmp_path):
    original, _ = _factor_and_recipe()
    original_hash = compute_ir_hash(Analyzer().lower(original.expr).ir)
    changed = Factor(
        name=original.name, expr=ts_ema(col("close"), 5),
        source_expr="ts_ema(close, 5)", universe=original.universe,
    )
    changed_hash = compute_ir_hash(Analyzer().lower(changed.expr).ir)
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    state = FittedState(
        transform_name=FITTED_STANDARDIZE_NAME,
        transform_version=FITTED_STANDARDIZE_VERSION,
        fit_start_time=pd.Timestamp("2023-01-01").to_pydatetime(),
        fit_end_time=pd.Timestamp("2023-12-31").to_pydatetime(),
        state_kind=StateKind.FITTED,
        feature_ids=[original_hash], feature_order=[original_hash],
        learned_params={"mean": 1.0, "scale": 2.0},
        implementation_hash=fitted_standardize_implementation_hash(),
        data_snapshot_ref=digest("fit-snapshot"), split_ref=digest("train"),
        universe_ref=digest(original.universe), calendar_ref=digest("calendar"),
        fit_coordinate_hash=digest("coordinates"), policy_hash=digest("policy"),
        production=True,
    )
    recipe = TreatmentRecipe(
        recipe_id="same-name-changed-dsl", source_factor_definition_ref=changed_hash,
        source_factor_value_ref=digest("changed-value"),
        ordered_steps=(RecipeStep(
            "standardize", "TRAIN_STANDARDIZE:population-v1",
            FITTED_STANDARDIZE_NAME, "representation", requires_fit=True,
            state_ref=state.state_id,
        ),),
    )
    checkpoint_root = tmp_path / "cp"
    db = SqliteDb(str(tmp_path / "metadata.db"))
    with pytest.raises(ValueError, match="factor-definition contract mismatch"):
        run_fixed_daily_update(
            factor=changed, recipe=recipe,
            source_spec={"type": "parquet", "root": str(tmp_path / "unused")},
            start="2024-01-02", end="2024-01-03", bootstrap=False,
            checkpoint_store=StatefulCheckpointStore(checkpoint_root),
            generation_coordinator=DurableGenerationCoordinator(db, _Publisher()),
            artifact_id="factor-value:same-name-graft",
            fitted_states={state.state_id: state},
        )
    assert not list(checkpoint_root.glob("**/*.json"))
    assert db.query("SELECT * FROM artifact_generations") == []


def test_fixed_update_publishes_only_its_exact_outbox_intent(tmp_path):
    factor, recipe = _factor_and_recipe()
    panel = default_panel_factory(12)
    source_spec = _source_spec(panel, tmp_path / "source.parquet")
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, _Publisher())
    with db.transaction():
        coordinator.outbox.emit(
            event_type="UnrelatedProductionWork", aggregate_type="unrelated",
            aggregate_id="other", correlation_id="other",
            idempotency_key="unrelated:must-remain-pending", payload={"other": True},
        )
    run_fixed_daily_update(
        factor=factor, recipe=recipe, source_spec=source_spec,
        start=panel.index[0], end=panel.index[5], bootstrap=True,
        checkpoint_store=StatefulCheckpointStore(tmp_path / "cp"),
        generation_coordinator=coordinator,
        artifact_id="factor-value:scoped-outbox",
    )
    unrelated = db.query(
        "SELECT * FROM outbox_events WHERE idempotency_key=?",
        ("unrelated:must-remain-pending",),
    )
    assert len(unrelated) == 1 and unrelated[0]["status"] == "pending"
