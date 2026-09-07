from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from textwrap import dedent

from factor_engine.runtime import materialize_batch as mb
from factor_engine.storage.materialize import lake_publish as lp


def _item(name, options):
    return SimpleNamespace(
        factor=SimpleNamespace(name=name), factor_id=name, options=options
    )


def test_typed_receipt_does_not_equate_commit_with_publish():
    inventory = [{"path": "p.parquet", "rows": 1, "bytes": 1, "sha256": "0" * 64}]
    inventory_digest = lp._inventory_digest(inventory)
    assert hasattr(mb, "WriteReceipt")
    receipt = mb.WriteReceipt(
        generation_id="g1", expected_items=("a",),
        items={"a": mb.WriteItemReceipt(
            "a", state=mb.WriteState.COMMITTED, run_id="r1",
            inventory_digest=inventory_digest, inventory=inventory,
        )},
        manifest_digest="d", idempotency_key="g1",
    )
    assert receipt.state is mb.WriteState.COMMITTED
    assert receipt.to_dict()["items"]["a"]["state"] == "COMMITTED"
    assert mb.WriteReceipt.from_dict(
        receipt.to_dict(), expected_items=["a"]
    ).state is mb.WriteState.COMMITTED


@pytest.mark.parametrize("mutation", ["missing_item", "unknown_state", "negative_rows", "missing_manifest"])
def test_receipt_deserialization_fails_closed(mutation):
    payload = {
        "generation_id": "g1", "idempotency_key": "g1",
        "manifest_digest": "d", "expected_items": ["a"],
        "items": {"a": {"name": "a", "state": "COMMITTED", "rows": 1,
                         "error": None, "run_id": "r1", "inventory_digest": lp._inventory_digest([{"path": "p", "rows": 1, "bytes": 1, "sha256": "0" * 64}]),
                         "inventory": [{"path": "p", "rows": 1, "bytes": 1, "sha256": "0" * 64}]}},
    }
    if mutation == "missing_item":
        payload["items"] = {}
    elif mutation == "unknown_state":
        payload["items"]["a"]["state"] = "MAYBE"
    elif mutation == "negative_rows":
        payload["items"]["a"]["rows"] = -1
    else:
        payload["manifest_digest"] = None
    with pytest.raises(ValueError):
        mb.WriteReceipt.from_dict(payload, expected_items=["a"])


@pytest.mark.parametrize("payload_change", [
    lambda p: p.update(expected_items="a"),
    lambda p: p.update(generation_id=1),
    lambda p: p.update(idempotency_key={"bad": True}),
    lambda p: p.update(manifest_digest=7),
    lambda p: p.update(items={1: p["items"]["f"]}),
])
def test_receipt_rejects_coercible_but_wrong_types(payload_change):
    receipt = _receipt()
    payload_change(receipt)
    with pytest.raises(ValueError):
        mb.WriteReceipt.from_dict(receipt, expected_items=["f"])


def test_typed_receipt_rejects_string_expected_items():
    receipt = mb.WriteReceipt(
        generation_id="g", expected_items="ab", items={}, idempotency_key="g"
    )
    with pytest.raises(ValueError, match="list/tuple"):
        receipt.validate()


@pytest.mark.parametrize("inventory,digest", [
    ([{}], "0" * 64),
    ([{"path": "p", "rows": 1, "bytes": 1, "sha256": "0" * 64}], "f" * 64),
])
def test_receipt_rejects_untyped_or_misdigested_inventory(inventory, digest):
    receipt = _receipt()
    receipt["items"]["f"]["inventory"] = inventory
    receipt["items"]["f"]["inventory_digest"] = digest
    with pytest.raises(ValueError, match="inventory"):
        mb.WriteReceipt.from_dict(receipt, expected_items=["f"])


def test_heterogeneous_write_context_rejected_before_any_writer():
    assert hasattr(mb, "_resolve_batch_contexts")
    items = [_item("a", {"lake_root": "/a"}), _item("b", {"lake_root": "/b"})]
    with pytest.raises(ValueError, match="heterogeneous WriteContext"):
        mb._resolve_batch_contexts(items, None)


def test_first_item_context_is_not_lost_to_defaults():
    items = [_item("a", {"target": "staging", "lake_root": "/expected"})]
    _shared, _resolved, context = mb._resolve_batch_contexts(items, None)
    assert context.target == "staging"
    assert context.lake_root == "/expected"


def test_write_target_alias_overrides_default_local_target():
    items = [_item("a", {"write_target": "staging"})]
    _shared, _resolved, context = mb._resolve_batch_contexts(items, None)
    assert context.target == "staging"


def test_context_encoder_preserves_key_types_and_rejects_opaque_values():
    assert mb._canonical_context_value({1: "x"}) != mb._canonical_context_value({"1": "x"})
    with pytest.raises(ValueError, match="unsupported"):
        mb._canonical_context_value({"x": object()})


def test_shared_axis_observation_never_builds_value_block(monkeypatch):
    import factor_engine.runtime.factor_block_ref as refs

    monkeypatch.setattr(refs, "build_factor_block", lambda *_a, **_k: pytest.fail("allocated value block"))
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-01"]), ["A"]], names=["timestamp", "instrument"]
    )
    preps = [
        {"factor_id": "a", "output": {"result": pd.Series([1.0], index=index)}, "opts": {}},
        {"factor_id": "b", "output": {"result": pd.Series([2.0], index=index)}, "opts": {}},
    ]
    assert mb._record_shared_axes(preps) == 1


def _write_parquet(root: Path, value: float = 1.0):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "datetime": pd.to_datetime(["2025-01-01 09:30Z"]),
        "asset": ["A"], "value": [value],
    }).to_parquet(root / "part.parquet", index=False)


def _receipt(factor_id="f", generation="g-old", root: Path | None = None,
             *, coverage_proof=None, run_id="r1"):
    inventory = lp._factor_inventory(root) if root is not None else [
        {"path": "p.parquet", "rows": 1, "bytes": 1, "sha256": "0" * 64}
    ]
    digest = lp._inventory_digest(inventory)
    return mb.WriteReceipt(
        generation_id=generation, expected_items=(factor_id,),
        items={factor_id: mb.WriteItemReceipt(
            factor_id, state=mb.WriteState.COMMITTED, rows=1, run_id=run_id,
            inventory_digest=digest, inventory=inventory,
            coverage_proof=coverage_proof,
        )},
        manifest_digest="dependency-manifest", idempotency_key=generation,
    ).to_dict()


def _coverage_proof(root: Path, *, universe="u1", frequency="1min"):
    start = pd.Timestamp("2025-01-01 09:30Z").isoformat()
    intervals = [{
        "start": start, "end": start, "expected_rows": 1, "observed_rows": 1,
    }]
    return {
        "proof_version": "sqlite-axis-v1",
        "actual_axis_digest": "axis-1", "expected_axis_digest": "axis-1",
        "actual_key_count": 1, "expected_key_count": 1,
        "universe_snapshot": universe, "calendar_id": "XNYS-v1",
        "market_timezone": "UTC",
        "frequency": frequency,
        "inventory_digest": lp._inventory_digest(lp._factor_inventory(root)),
        "run_id": "r1",
        "actual_start": start, "actual_end": start,
        "coverage_intervals": intervals,
    }


def test_staging_identity_binds_generation_run_and_bytes(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    _write_parquet(staging)
    monkeypatch.setattr(lp, "_resolve_staging_factor_dir", lambda _fid: staging)
    assert hasattr(lp, "write_staging_identity")
    identity = lp.write_staging_identity(
        factor_id="f", materialization_receipt=_receipt(root=staging)
    )
    assert identity["manifest_digest"]
    pd.DataFrame({
        "datetime": pd.to_datetime(["2025-01-01 09:30Z"]),
        "asset": ["A"], "value": [2.0],
    }).to_parquet(staging / "part.parquet", index=False)
    with pytest.raises(ValueError, match="does not match"):
        lp._read_verified_staging_identity("f")


def test_sync_rejects_source_bytes_not_bound_to_receipt_before_staging_mutation(
    tmp_path, monkeypatch
):
    source = tmp_path / "lake" / "factors" / "f"
    staging = tmp_path / "staging"
    _write_parquet(source, 1.0)
    receipt = _receipt(root=source)
    _write_parquet(source, 2.0)
    _write_parquet(staging, 9.0)
    before = lp._factor_inventory(staging)
    monkeypatch.setattr(lp, "_resolve_staging_factor_dir", lambda _fid: staging)
    with pytest.raises(ValueError, match="local source bytes"):
        lp.sync_local_factor_to_staging(
            factor_id="f", lake_root=tmp_path / "lake",
            materialization_receipt=receipt,
        )
    assert lp._factor_inventory(staging) == before


def test_sync_rejects_factor_id_traversal_before_resolving_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lp, "_resolve_staging_factor_dir",
        lambda _fid: pytest.fail("staging resolution must not run for unsafe factor_id"),
    )
    with pytest.raises(ValueError, match="forbidden"):
        lp.sync_local_factor_to_staging(
            factor_id="../victim", lake_root=tmp_path,
        )


def test_sync_keeps_staging_lock_effective_during_child_replacement(tmp_path, monkeypatch):
    from data_access.write.mutation_lock import mutation_lock

    source = tmp_path / "lake" / "factors" / "f"
    staging = tmp_path / "staging"
    _write_parquet(source, 1.0)
    _write_parquet(staging, 9.0)
    receipt = _receipt(root=source)
    monkeypatch.setattr(lp, "_resolve_staging_factor_dir", lambda _fid: staging)
    original_inventory = lp._factor_inventory
    competing = []

    def inventory_with_competitor(root):
        result = original_inventory(root)
        if Path(root) == staging and not competing:
            def contend():
                try:
                    with mutation_lock(staging, timeout=0.1, poll=0.01):
                        competing.append("acquired")
                except Exception:
                    competing.append("blocked")
            thread = threading.Thread(target=contend)
            thread.start()
            thread.join(timeout=2)
        return result

    monkeypatch.setattr(lp, "_factor_inventory", inventory_with_competitor)
    lp.sync_local_factor_to_staging(
        factor_id="f", lake_root=tmp_path / "lake",
        materialization_receipt=receipt,
    )
    assert competing == ["blocked"]


def test_sync_writes_identity_before_releasing_staging_lock(tmp_path, monkeypatch):
    from data_access.core import atomic
    from data_access.core.exceptions import ValidationError
    from data_access.write.mutation_lock import mutation_lock

    source = tmp_path / "lake" / "factors" / "f"
    staging = tmp_path / "staging"
    _write_parquet(source, 1.0)
    receipt = _receipt(root=source)
    monkeypatch.setattr(lp, "_resolve_staging_factor_dir", lambda _fid: staging)
    original_atomic_write = atomic.atomic_write_text
    competing = []

    def atomic_write_with_competitor(path, content, *args, **kwargs):
        def contend():
            try:
                with mutation_lock(staging, timeout=0.1, poll=0.01):
                    competing.append("acquired")
            except ValidationError:
                competing.append("blocked")

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        return original_atomic_write(path, content, *args, **kwargs)

    monkeypatch.setattr(atomic, "atomic_write_text", atomic_write_with_competitor)
    result = lp.sync_local_factor_to_staging(
        factor_id="f", lake_root=tmp_path / "lake",
        materialization_receipt=receipt,
    )

    assert competing == ["blocked"]
    assert result["identity"] == lp._read_verified_staging_identity("f")


def test_sync_restores_previous_generation_when_identity_write_fails(tmp_path, monkeypatch):
    from data_access.core import atomic

    source = tmp_path / "lake" / "factors" / "f"
    staging = tmp_path / "staging"
    _write_parquet(source, 1.0)
    _write_parquet(staging, 9.0)
    old_identity = '{"generation_id": "old"}'
    (staging / lp._STAGING_IDENTITY).write_text(old_identity, encoding="utf-8")
    before = lp._factor_inventory(staging)
    monkeypatch.setattr(lp, "_resolve_staging_factor_dir", lambda _fid: staging)
    monkeypatch.setattr(
        atomic, "atomic_write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("identity write failed")),
    )

    with pytest.raises(OSError, match="identity write failed"):
        lp.sync_local_factor_to_staging(
            factor_id="f", lake_root=tmp_path / "lake",
            materialization_receipt=_receipt(root=source),
        )

    assert lp._factor_inventory(staging) == before
    assert (staging / lp._STAGING_IDENTITY).read_text(encoding="utf-8") == old_identity


@pytest.mark.parametrize("operation", ["identity", "sync"])
def test_local_helpers_reject_cos_before_path_or_lock(operation, tmp_path, monkeypatch):
    import importlib
    import data_access
    lock_module = importlib.import_module("data_access.write.mutation_lock")

    dataset = SimpleNamespace(
        name="factor_lake_staging",
        storage={"type": "cos", "uri": "cos://bucket/staging"},
    )
    store = SimpleNamespace(
        get_dataset=lambda _name: dataset,
        resolve_dataset_path=lambda *_a, **_k: pytest.fail("path resolution must not run"),
    )
    monkeypatch.setattr(data_access, "get_store", lambda: store)
    monkeypatch.setattr(
        lock_module, "mutation_lock", lambda *_a, **_k: pytest.fail("lock must not run")
    )
    monkeypatch.setattr(
        lp, "_resolve_staging_factor_dir",
        lambda _fid: pytest.fail("staging path resolution must not run"),
    )

    with pytest.raises(ValueError, match="cos publication"):
        if operation == "identity":
            lp.write_staging_identity(
                factor_id="f", materialization_receipt=_receipt()
            )
        else:
            lp.sync_local_factor_to_staging(
                factor_id="f", lake_root=tmp_path,
                materialization_receipt=_receipt(),
            )


@pytest.mark.parametrize("expected, message", [
    ([], "differs from expected"),
    ([(pd.Timestamp("2025-01-01 09:30Z"), "A"),
      (pd.Timestamp("2025-01-01 09:31Z"), "A")], "differs from expected"),
])
def test_coverage_producer_rejects_extra_or_missing_keys(
    tmp_path, monkeypatch, expected, message
):
    staging = tmp_path / "staging"
    _write_parquet(staging)
    monkeypatch.setattr(lp, "_resolve_staging_factor_dir", lambda _fid: staging)
    with pytest.raises(ValueError, match=message):
        lp.produce_coverage_receipt(
            factor_id="f", materialization_receipt=_receipt(root=staging),
            expected_keys=iter(expected), universe_snapshot="u",
            calendar_id="XNYS-v1", market_timezone="UTC", frequency="1min",
        )


def test_coverage_producer_rejects_duplicate_actual_keys(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    _write_parquet(staging)
    duplicate = staging / "duplicate.parquet"
    pd.read_parquet(staging / "part.parquet").to_parquet(duplicate, index=False)
    monkeypatch.setattr(lp, "_resolve_staging_factor_dir", lambda _fid: staging)
    with pytest.raises(ValueError, match="duplicate actual"):
        lp.produce_coverage_receipt(
            factor_id="f", materialization_receipt=_receipt(root=staging),
            expected_keys=iter([(pd.Timestamp("2025-01-01 09:30Z"), "A")]),
            universe_snapshot="u", calendar_id="XNYS-v1",
            market_timezone="UTC", frequency="1min",
        )


def test_coverage_producer_rejects_interval_extrapolation(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    _write_parquet(staging)
    monkeypatch.setattr(lp, "_resolve_staging_factor_dir", lambda _fid: staging)
    with pytest.raises(ValueError, match="actual keys|exactly bound"):
        lp.produce_coverage_receipt(
            factor_id="f", materialization_receipt=_receipt(root=staging),
            expected_keys=iter([(pd.Timestamp("2025-01-01 09:30Z"), "A")]),
            universe_snapshot="u", calendar_id="XNYS-v1",
            market_timezone="UTC", frequency="1min",
            coverage_intervals=[{
                "start": "2030-01-01T09:30:00Z", "end": "2030-01-01T09:30:00Z",
                "expected_rows": 1, "observed_rows": 1,
            }],
        )


def test_public_publish_rejects_unbound_staging_before_catalog_or_store_access():
    with pytest.raises(ValueError, match="expected_staging_generation"):
        lp.publish_factor_lake(
            factor_id="f", approve=True, sync_from_local=False, reconcile=False
        )


def test_public_complete_publish_rejects_missing_expected_axis_contract():
    with pytest.raises(ValueError, match="expected_axis_keys"):
        lp.publish_factor_lake(
            factor_id="f", approve=True, sync_from_local=False, reconcile=False,
            expected_staging_generation="g", expected_manifest_digest="d",
            expected_run_id="r", frequency="1min", universe_snapshot="u",
            materialization_receipt=_receipt(), coverage_complete=True,
            calendar_id="XNYS-v1",
            market_timezone="UTC",
        )


def test_cos_backend_rejected_before_sync_even_when_path_resolver_returns_path(
    tmp_path, monkeypatch
):
    import data_access

    called = {"sync": False, "publish": False, "resolve": False}
    dataset = SimpleNamespace(
        name="factor_lake",
        storage={"type": "cos", "uri": "cos://bucket/factors"},
    )

    class FakeStore:
        def get_dataset(self, _name):
            return dataset
        def resolve_dataset_path(self, *_args, **_kwargs):
            called["resolve"] = True
            return tmp_path / "misleading-path"
        def publish_from_staging(self, *_args, **_kwargs):
            called["publish"] = True

    monkeypatch.setattr(data_access, "get_store", lambda: FakeStore())
    monkeypatch.setattr(
        lp, "sync_local_factor_to_staging",
        lambda **_kwargs: called.__setitem__("sync", True),
    )
    with pytest.raises(ValueError, match="cos publication"):
        lp.publish_factor_lake(
            factor_id="f", lake_root=tmp_path, approve=True,
            sync_from_local=True, reconcile=False,
            expected_staging_generation="g", expected_manifest_digest="d",
            expected_run_id="r", frequency="1min",
        )
    assert called == {"sync": False, "publish": False, "resolve": False}


def test_watermark_refuses_tail_without_complete_manifest_evidence():
    with pytest.raises(ValueError, match="coverage_complete"):
        lp.advance_published_watermark(
            factor_id="f", catalog=object(), store=object(),
            publish_identity={"coverage_complete": False}, frequency="1min",
        )


def test_complete_coverage_rejects_naive_time_and_gaps():
    proof = {
        "proof_version": "sqlite-axis-v1",
        "actual_axis_digest": "a", "expected_axis_digest": "a",
        "actual_key_count": 1, "expected_key_count": 1,
        "universe_snapshot": "u", "calendar_id": "c", "frequency": "1min",
        "market_timezone": "UTC",
        "inventory_digest": "d", "run_id": "r1",
    }
    with pytest.raises(ValueError, match="timezone-aware"):
        lp._validated_coverage({
            "coverage_complete": True,
            "manifest_digest": "d", "run_id": "r1", "coverage_proof": proof,
            "coverage_intervals": [{"start": "2025-01-01 09:30", "end": "2025-01-01 10:00", "expected_rows": 1, "observed_rows": 1}],
        }, "1min")
    with pytest.raises(ValueError, match="gap"):
        lp._validated_coverage({
            "coverage_complete": True,
            "manifest_digest": "d", "run_id": "r1", "coverage_proof": proof,
            "coverage_intervals": [
                {"start": "2025-01-01T09:30Z", "end": "2025-01-01T10:00Z", "expected_rows": 1, "observed_rows": 1},
                {"start": "2025-01-01T10:02Z", "end": "2025-01-01T11:00Z", "expected_rows": 1, "observed_rows": 1},
            ],
        }, "1min")


def test_watermark_uses_complete_manifest_intervals_and_keeps_intraday_precision(tmp_path):
    published = tmp_path / "published"
    _write_parquet(published)
    inventory = lp._factor_inventory(published)
    identity = {
        "generation_id": "g1",
        "run_id": "r1",
        "manifest_digest": lp._inventory_digest(inventory),
        "coverage_complete": True,
        "coverage_proof": _coverage_proof(published),
        "coverage_intervals": [{
            "start": "2025-01-01T09:30:00+00:00",
            "end": "2025-01-01T09:30:00+00:00",
            "expected_rows": 1, "observed_rows": 1,
        }],
    }

    class Catalog:
        def __init__(self):
            self.value = None
        def update_published_watermark(self, **kwargs):
            self.value = dict(kwargs)
            return dict(self.value)

    catalog = Catalog()
    store = SimpleNamespace(resolve_dataset_path=lambda *_a, **_k: published)
    result = lp.advance_published_watermark(
        factor_id="f", catalog=catalog, store=store,
        publish_identity=identity, frequency="1min", universe_snapshot="u1",
    )
    assert result["committed_tip"] == "2025-01-01T09:30:00+00:00"
    assert result["generation_id"] == "g1"
    assert result["frequency"] == "1min"


def test_watermark_does_not_skip_corrupt_manifest_member(tmp_path):
    published = tmp_path / "published"
    _write_parquet(published)
    inventory = lp._factor_inventory(published)
    identity = {
        "generation_id": "g1",
        "run_id": "r1",
        "manifest_digest": lp._inventory_digest(inventory),
        "coverage_complete": True,
        "coverage_proof": _coverage_proof(published),
        "coverage_intervals": [{"start": "2025-01-01T09:30:00Z", "end": "2025-01-01T09:30:00Z", "expected_rows": 1, "observed_rows": 1}],
    }
    (published / "part.parquet").write_bytes(b"corrupt")
    store = SimpleNamespace(resolve_dataset_path=lambda *_a, **_k: published)
    with pytest.raises(Exception):
        lp.advance_published_watermark(
            factor_id="f", catalog=object(), store=store,
            publish_identity=identity, frequency="1min",
        )


def test_real_local_publish_closure_binds_identity_and_persists_watermark(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    published = tmp_path / "published"
    workspace.mkdir()
    published.mkdir()
    config = tmp_path / "datasets.yaml"
    config.write_text(dedent(f"""
        factor_lake:
          kind: parametric
          access_mode: published
          layout: hive
          root_template: {published}/factors/{{factor_id}}
          glob_template: "year=*/*.parquet"
          params_schema: {{factor_id: str}}
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true
        factor_lake_staging:
          kind: parametric
          access_mode: staging
          layout: hive
          root_template: {workspace}/staging/ns/factors/{{factor_id}}
          glob_template: "year=*/*.parquet"
          params_schema: {{factor_id: str}}
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true
    """).strip() + "\n", encoding="utf-8")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(config))
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "ns")
    monkeypatch.setenv("QUANT_OPERATOR", "audit@test")
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    from data_access import get_store, reset_store
    reset_store()
    store = get_store()
    original_publish = store.publish_from_staging
    lock_observed = {"value": False}
    def publish_with_lock_assertion(*args, **kwargs):
        staging_dir = store.resolve_dataset_path("factor_lake_staging", factor_id="f")
        lock_observed["value"] = (staging_dir / ".data-access.mutation.lock").is_file()
        return original_publish(*args, **kwargs)
    monkeypatch.setattr(store, "publish_from_staging", publish_with_lock_assertion)
    lake_root = tmp_path / "lake"
    from factor_engine.tests.r39.test_perf_materialize_fast_2026_08 import (
        _engine, _factors,
    )
    engine = _engine(dates=3)
    factor = _factors()[0]
    expected_run = engine.run(factor)
    expected_result = expected_run["result"]
    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    fast = engine.materialize_many_fast(
        [factor], factor_ids=["f"], native_fusion=False, writer_threads=1,
        materialize_kwargs={
            "lake_root": str(lake_root), "write_target": "local",
            "writer_batch_size": 1,
        },
    )
    materialized = fast["materializations"][factor.name]
    receipt = materialized["write_receipt"]
    local_factor = lake_root / "factors" / "f"
    expected_digest = lp._inventory_digest(lp._factor_inventory(local_factor))
    empty_batch = mb.execute_materialize_batch(
        engine,
        [mb.MaterializeItem(
            factor=factor,
            factor_id="f",
            output={
                "factor": factor, "analysis": expected_run["analysis"],
                "result": expected_result.iloc[0:0],
            },
            options={"write_target": "local"},
        )],
        shared_options={"lake_root": lake_root, "write_target": "local"},
    )
    assert empty_batch["receipt"].items["f"].state is mb.WriteState.CREATED
    assert lp._inventory_digest(lp._factor_inventory(local_factor)) == expected_digest
    result = lp.publish_factor_lake(
        factor_id="f", lake_root=lake_root, approve=True,
        sync_from_local=True, reconcile=False,
        expected_staging_generation=receipt["generation_id"],
        expected_manifest_digest=expected_digest,
        expected_run_id=materialized["run_id"], frequency="1min", universe_snapshot="u1",
        materialization_receipt=receipt,
        coverage_intervals=[{
            "start": pd.Timestamp(expected_result.index[0][0]).tz_localize("UTC").isoformat(),
            "end": pd.Timestamp(expected_result.index[-1][0]).tz_localize("UTC").isoformat(),
            "expected_rows": len(expected_result), "observed_rows": len(expected_result),
        }],
        coverage_complete=True,
        expected_axis_keys=(
            (pd.Timestamp(ts).tz_localize("UTC"), instrument)
            for ts, instrument in expected_result.index
        ),
        calendar_id="XNYS-v1",
        market_timezone="UTC",
    )
    assert result["watermark"]["generation_id"] == receipt["generation_id"]
    assert result["watermark"]["committed_tip"].endswith("+00:00")
    assert (published / "factors" / "f").is_dir()
    assert lock_observed["value"], "expected-inventory check and publish must share staging lock"


def test_catalog_migration_columns_preserve_legacy_watermark(tmp_path):
    from factor_engine.storage.catalog import FactorCatalog
    path = tmp_path / "catalog.sqlite"
    catalog = FactorCatalog(path)
    catalog.register("legacy", "a", "1d", "h")
    catalog.update_watermark("legacy", "2024-01-01", "2024-01-31", row_count=2)
    catalog.close()
    reopened = FactorCatalog(path)
    legacy = reopened.get_watermark("legacy")
    assert legacy["start_date"] == "2024-01-01"
    assert legacy["end_date"] == "2024-01-31"
    assert legacy["generation_id"] is None
    assert reopened.get_published_watermark("legacy") is None


def _published_watermark(catalog, generation, start, end):
    return catalog.update_published_watermark(
        factor_id="f", start_date=start, end_date=end, row_count=1,
        generation_id=generation, frequency="1min", committed_tip=end,
        coverage_intervals=[{
            "start": start, "end": end, "expected_rows": 1, "observed_rows": 1,
        }],
        universe_snapshot="u1",
    )


def test_local_watermark_updates_cannot_mutate_published_authority(tmp_path):
    from factor_engine.storage.catalog import FactorCatalog

    catalog = FactorCatalog(tmp_path / "catalog.sqlite")
    catalog.register("f", "a", "1min", "h")
    published_a = _published_watermark(
        catalog, "generation-a", "2025-01-01T09:30:00+00:00",
        "2025-01-01T10:00:00+00:00",
    )
    catalog.update_watermark(
        "f", "2026-01-01T09:30:00+00:00", "2026-01-01T10:00:00+00:00",
        row_count=99,
    )
    assert catalog.get_watermark("f")["row_count"] == 99
    assert catalog.get_published_watermark("f") == published_a


def test_new_published_generation_replaces_older_tail_without_merging(tmp_path):
    from factor_engine.storage.catalog import FactorCatalog

    catalog = FactorCatalog(tmp_path / "catalog.sqlite")
    catalog.register("f", "a", "1min", "h")
    _published_watermark(
        catalog, "generation-old", "2025-01-01T09:30:00+00:00",
        "2025-01-31T10:00:00+00:00",
    )
    newer = _published_watermark(
        catalog, "generation-new", "2025-01-02T09:30:00+00:00",
        "2025-01-02T10:00:00+00:00",
    )
    assert newer["generation_id"] == "generation-new"
    assert newer["start_date"] == "2025-01-02T09:30:00+00:00"
    assert newer["end_date"] == "2025-01-02T10:00:00+00:00"
    assert newer["committed_tip"] == "2025-01-02T10:00:00+00:00"
