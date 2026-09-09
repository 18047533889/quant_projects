from datetime import datetime, timedelta, timezone
import json
import numpy as np
import pytest

from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_instance import MetricInstance
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.label_maturation import LabelMaturationQueue
from quant_evaluator.runtime.online_moments import OnlineMoments
from quant_platform.app.db.sqlite_backend import SqliteDb

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def sem(window):
    return dict(factor_value_semantics="raw.v1", universe="u", clock="daily",
                window=window, policy_version="v7")


def req(indices, reverse=False, missing=None, factor_count=1):
    indices = tuple(indices)
    rng = np.random.default_rng(812)
    x = rng.normal(size=(12, 40))
    y = .2*x + rng.normal(size=x.shape)
    if reverse:
        y = -y
    values = x[list(indices)].copy()
    if missing in indices:
        values[indices.index(missing)] = 1.0
    dates = tuple(START + timedelta(days=i) for i in indices)
    factor_ids = tuple(f"f{i}" for i in range(factor_count))
    factor_values = np.stack(
        [values + i * 1e-6 for i in range(factor_count)], axis=-1)
    batch = FactorBatch(factor_ids, AxisRef("time", "datetime", len(indices)),
                        AxisRef("asset", "int", 40), factor_values)
    labels = LabelBundle("h1", y[list(indices)], 1, decision_time=dates,
                         label_start_time=dates,
                         label_end_time=tuple(d + timedelta(days=1) for d in dates))
    ref = "r:" + ",".join(map(str, indices)) + (":rev" if reverse else "")
    return EvaluationRequest(
        batch, labels, metric_ids=(), tier="research",
        metric_instances=(MetricInstance("rank_ic_series", horizon=1),),
        factor_value_ref=FactorValueRef(ref, batch.factor_ids),
        label_bundle_ref=LabelBundleRef(ref, "h1", 1),
    )


def resolver(items):
    by_ref = {x.factor_value_ref.factor_value_id: x for x in items}
    return lambda ref: by_ref[ref.factor_value_ref.factor_value_id]


def expected(db, stream, width):
    rows = db.query(
        "SELECT value_json FROM qe_maturity_observations WHERE stream_id=? "
        "ORDER BY decision_at DESC LIMIT ?", (stream, width))
    state = OnlineMoments()
    state.update([np.nan if row["value_json"] == "null"
                  else json.loads(row["value_json"]) for row in reversed(rows)])
    return state.summary()


def only_summary(queue, stream):
    values = queue.summaries(stream)
    assert len(values) == 1
    return next(iter(values.values()))


def test_capabilities_and_unknown_windows(tmp_path):
    caps = LabelMaturationQueue.CAPABILITIES
    assert caps["metric_ids"] == ("pearson_ic_series", "rank_ic_series")
    assert caps["remove"] is False
    assert caps["rolling_algorithm"] == "exact_bounded_database_recomputation"
    assert caps["multi_day_shard_maturity"] == "max_label_end_time"
    assert caps["revision_completeness_validation"] is False
    queue = LabelMaturationQueue(SqliteDb(str(tmp_path / "bad.db")))
    assert queue.capabilities["max_factors_per_shard"] == 256
    bad_windows = (
        "30d",
        {"kind": "rolling_observations", "size": 0},
        {"kind": "rolling_days", "size": 3},
        {"kind": "rolling_observations", "size": 3, "unit": "days"},
    )
    for number, bad in enumerate(bad_windows):
        with pytest.raises(ValueError, match="window|positive integer"):
            queue.enqueue(str(number), req((0,)), stream_semantics=sem(bad),
                          available_at=START)
    with pytest.raises(ValueError, match="max_factors_per_shard"):
        LabelMaturationQueue(SqliteDb(str(tmp_path / "zero.db")),
                             max_factors_per_shard=0)
    bounded = LabelMaturationQueue(SqliteDb(str(tmp_path / "bounded.db")),
                                   max_factors_per_shard=2)
    with pytest.raises(ValueError, match="factor shard"):
        bounded.enqueue("wide", req((0,), factor_count=3),
                        stream_semantics=sem("all_history"), available_at=START)


def test_all_history_keeps_v2_identity_and_rolling_uses_new_schema(tmp_path):
    db = SqliteDb(str(tmp_path / "identity.db"))
    queue = LabelMaturationQueue(db)
    item = req((0,))
    old_semantics = sem("all_history")
    old_stream = queue.enqueue("old", item, stream_semantics=old_semantics,
                               available_at=START)
    expected_old = stable_content_hex(
        tag="MaturityStream.v1",
        fields={**old_semantics, "mode": "AS_KNOWN", "revision_id": None,
                "state_schema": "centered_moments.v2"},
    )
    assert old_stream == expected_old
    old_payload = json.loads(db.query(
        "SELECT payload FROM qe_maturity_events WHERE event_id='old'")[0]["payload"])
    assert "window" not in old_payload
    rolling = queue.enqueue(
        "rolling", item,
        stream_semantics=sem({"kind": "rolling_observations", "size": 3}),
        available_at=START,
    )
    assert rolling != old_stream


@pytest.mark.parametrize("tile", (12, 3, 1))
def test_rolling_full_tile_bar_and_restart(tmp_path, tile):
    db_path = str(tmp_path / f"tile-{tile}.db")
    db = SqliteDb(db_path)
    queue = LabelMaturationQueue(db)
    items = [req(range(lo, min(lo + tile, 12)), missing=10)
             for lo in range(0, 12, tile)]
    stream = None
    for number, item in enumerate(items):
        stream = queue.enqueue(
            str(number), item,
            stream_semantics=sem({"kind": "rolling_observations", "size": 4}),
            available_at=START,
        )
    queue.drain(START + timedelta(days=30), resolver(items),
                limit=max(1, len(items)//2))
    db.close()
    db = SqliteDb(db_path)
    queue = LabelMaturationQueue(db)
    queue.drain(START + timedelta(days=30), resolver(items), limit=256)
    actual = only_summary(queue, stream)
    assert actual == pytest.approx(expected(db, stream, 4), nan_ok=True)
    assert actual["count"] == 3


def test_late_duplicate_and_revision_are_bounded(tmp_path):
    db = SqliteDb(str(tmp_path / "late.db"))
    queue = LabelMaturationQueue(db)
    window = sem({"kind": "rolling_observations", "size": 3})
    newer, older = req((8, 9, 10)), req((1, 2))
    stream = queue.enqueue("new", newer, stream_semantics=window,
                           available_at=START)
    queue.drain(START + timedelta(days=30), resolver((newer,)))
    queue.enqueue("old", older, stream_semantics=window, available_at=START)
    queue.enqueue("duplicate-envelope", newer, stream_semantics=window,
                  available_at=START)
    queue.drain(START + timedelta(days=30), resolver((newer, older)), limit=256)
    actual = only_summary(queue, stream)
    assert actual == pytest.approx(expected(db, stream, 3), nan_ok=True)
    assert actual["count"] == 3

    revised = req((8, 9, 10), reverse=True)
    revised_stream = queue.enqueue(
        "revision", revised, stream_semantics=window, available_at=START,
        snapshot_mode="RESTATED_RESEARCH", revision_id="vendor-v2",
    )
    queue.drain(START + timedelta(days=30), resolver((revised,)), limit=256)
    assert revised_stream != stream
    assert queue.summaries(revised_stream) != queue.summaries(stream)


def test_restart_reapplies_tighter_factor_bound_before_resolver(tmp_path):
    db_path = str(tmp_path / "restart-bound.db")
    db = SqliteDb(db_path)
    item = req((0,), factor_count=3)
    LabelMaturationQueue(db, max_factors_per_shard=3).enqueue(
        "pending", item, stream_semantics=sem("all_history"), available_at=START)
    db.close()
    queue = LabelMaturationQueue(SqliteDb(db_path), max_factors_per_shard=2)
    called = False

    def should_not_resolve(_):
        nonlocal called
        called = True
        return item

    with pytest.raises(ValueError, match="max_factors_per_shard"):
        queue.drain(START + timedelta(days=30), should_not_resolve)
    assert called is False


def test_resolved_batch_rechecks_current_factor_bound(tmp_path):
    db = SqliteDb(str(tmp_path / "resolved-bound.db"))
    admitted = req((0,), factor_count=2)
    queue = LabelMaturationQueue(db, max_factors_per_shard=2)
    queue.enqueue("pending", admitted, stream_semantics=sem("all_history"),
                  available_at=START)
    with pytest.raises(ValueError, match="factor shard"):
        queue.drain(START + timedelta(days=30),
                    lambda _: req((0,), factor_count=3))
