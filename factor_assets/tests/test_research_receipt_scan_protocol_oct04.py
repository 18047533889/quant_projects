"""Contract tests for the research-only receipt-to-FA scan audit."""

import hashlib
import importlib
import importlib.util
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.contracts.array_identity import authoritative_array_hash
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch


@pytest.mark.parametrize("container_type", [list, tuple])
def test_scan_audit_rejects_custom_receipt_container_before_iteration(container_type):
    # Custom iterators can lie about len and defeat the receipt admission cap.
    consumed = []

    class MisleadingContainer(container_type):
        def __len__(self):
            return 2

        def __iter__(self):
            consumed.append(True)
            yield from super().__iter__()

    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    with pytest.raises(TypeError):
        _run(MisleadingContainer(receipts))
    assert consumed == [], "admission must reject custom iterators before consuming them"

def test_scan_audit_accepts_exact_list_receipt_container():
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    report = _run(list(receipts))
    assert report["receipt_ids"] == ["factor-0", "factor-1", "factor-2"]
    assert report["member_ids"] == ["factor-0", "factor-1"]
    assert report["query_ids"] == ["factor-2"]
    assert report["winner_ids"] == ["factor-0"]

AUDIT_MODULE = "factor_assets.scripts.audit_research_receipt_scan_oct04"


def _audit_api():
    try:
        spec = importlib.util.find_spec(AUDIT_MODULE)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "research receipt scan audit must be implemented"
    module = importlib.import_module(AUDIT_MODULE)
    runner = getattr(module, "run_receipt_scan_audit", None)
    assert callable(runner), "research receipt scan audit runner must be callable"
    return runner


def _receipts(embeddings):
    from factor_assets.contracts.research_feature_receipt import (
        ResearchFeatureReceipt,
        ResearchSourceBinding,
    )

    factor_ids = tuple(f"factor-{i}" for i in range(len(embeddings)))
    times = np.asarray([101, 102], dtype=np.int64)
    assets = np.asarray(["asset-a", "asset-b"], dtype=object)
    time_axis = AxisRef("time", str(times.dtype), len(times), times)
    asset_axis = AxisRef("asset", str(assets.dtype), len(assets), assets)
    # Values are real fixture factor matrices. FA comparisons deliberately use
    # the explicit, caller-provided embeddings below, as the existing FA path does.
    values = np.arange(2 * 2 * len(factor_ids), dtype=np.float64).reshape(
        2, 2, len(factor_ids)
    )
    batch = FactorBatch(factor_ids, time_axis, asset_axis, values)
    manifest_sha256 = hashlib.sha256(b"candidate-pool-sample-fixture").hexdigest()
    manifest_uri = f"cos://research/{manifest_sha256}/landing_manifest.json"

    out = []
    for index, (factor_id, embedding) in enumerate(zip(factor_ids, embeddings)):
        child_values = np.ascontiguousarray(batch.values[:, :, index:index + 1])
        source_sha256 = hashlib.sha256(f"source-{factor_id}".encode()).hexdigest()
        source = ResearchSourceBinding(
            factor_id=factor_id,
            manifest_uri=manifest_uri,
            manifest_sha256=manifest_sha256,
            source_uri=f"cos://research/candidates/{source_sha256}.parquet",
            source_sha256=source_sha256,
            time_axis_hash=authoritative_array_hash(batch.time_axis.values),
            asset_axis_hash=authoritative_array_hash(batch.asset_axis.values),
            values_hash=authoritative_array_hash(child_values),
            validity_hash=None,
        )
        out.append(ResearchFeatureReceipt(
            source_binding=source,
            values_dtype=str(batch.values.dtype),
            validity_dtype=None,
            shape=tuple(child_values.shape),
            time_axis_name=batch.time_axis.name,
            time_axis_dtype=str(batch.time_axis.values.dtype),
            asset_axis_name=batch.asset_axis.name,
            asset_axis_dtype=str(batch.asset_axis.values.dtype),
            embedding=tuple(embedding),
            embedding_spec="research-scan-fixture-v1",
            embedding_model_version="fixture-model-v1",
        ))
    return tuple(out)


def _run(receipts, **overrides):
    options = {
        "paired_blocks": 3,
        "seed": 20261004,
        "max_chunk_rows": 4,
        "max_chunk_bytes": 1024 * 1024,
        "max_batch_queries": 16,
    }
    options.update(overrides)
    return _audit_api()(
        receipts,
        **options,
    )


def _receipt_tuple_with_change(receipts, index, **changes):
    updated = list(receipts)
    updated[index] = replace(updated[index], **changes)
    return tuple(updated)


def _scorer_sentinels(monkeypatch):
    module = importlib.import_module(AUDIT_MODULE)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid scan input reached the FA scorer")

    monkeypatch.setattr(module, "scan_exact_member_winner", forbidden)
    monkeypatch.setattr(module, "scan_exact_member_winners", forbidden)
    return calls


def test_scan_audit_preserves_first_exact_winner_and_matches_scalar_batch_paths():
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    report = _run(receipts)

    assert report["audit"] == "research-receipt-scan-oct04-v1"
    assert report["scope"] == "research_only"
    assert report["source_binding_status"] == "caller_supplied"
    assert report["receipt_ids"] == ["factor-0", "factor-1", "factor-2"]
    assert report["member_ids"] == ["factor-0", "factor-1"]
    assert report["query_ids"] == ["factor-2"]
    assert report["winner_ids"] == ["factor-0"]
    assert report["scores"] == [1.0]
    assert len(report["paired_blocks"]) == 3
    for block in report["paired_blocks"]:
        assert block["order"] in (
            ["scalar", "batch"],
            ["batch", "scalar"],
        )
        assert set(block["seconds"]) == {"scalar", "batch"}
        assert all(
            isinstance(seconds, (int, float))
            and math.isfinite(seconds)
            and seconds >= 0
            for seconds in block["seconds"].values()
        )
    json.dumps(report, allow_nan=False)


def test_scan_audit_tied_winners_keep_first_member_order():
    receipts = _receipts((
        (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0),
    ))
    report = _run(receipts)

    assert report["member_ids"] == ["factor-0", "factor-1"]
    assert report["query_ids"] == ["factor-2", "factor-3"]
    assert report["winner_ids"] == ["factor-0", "factor-0"]
    assert report["scores"] == [1.0, 0.0]


@pytest.mark.parametrize("field,changed", [
    ("embedding_spec", "different-spec"),
    ("embedding_model_version", "different-model"),
])
def test_mixed_embedding_configuration_rejected_before_scorer(monkeypatch, field, changed):
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    invalid = _receipt_tuple_with_change(receipts, 1, **{field: changed})
    calls = _scorer_sentinels(monkeypatch)

    with pytest.raises(ValueError):
        _run(invalid)
    assert calls == []


@pytest.mark.parametrize("binding_field,changed", [
    ("manifest_uri", "cos://research/other/landing_manifest.json"),
    ("manifest_sha256", "f" * 64),
    ("time_axis_hash", "e" * 64),
    ("asset_axis_hash", "d" * 64),
])
def test_mixed_source_or_axis_binding_rejected_before_scorer(
    monkeypatch, binding_field, changed,
):
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    binding = replace(
        receipts[1].source_binding, **{binding_field: changed}
    )
    invalid = _receipt_tuple_with_change(receipts, 1, source_binding=binding)
    calls = _scorer_sentinels(monkeypatch)

    with pytest.raises(ValueError):
        _run(invalid)
    assert calls == []


@pytest.mark.parametrize(("option", "value"), [
    ("paired_blocks", True),
    ("paired_blocks", 2),
    ("paired_blocks", 11),
    ("seed", True),
    ("seed", 2**32),
    ("max_chunk_rows", 0),
    ("max_chunk_rows", 257),
    ("max_chunk_rows", True),
    ("max_chunk_bytes", 0),
    ("max_chunk_bytes", 4 * 1024**2 + 1),
    ("max_chunk_bytes", True),
    ("max_batch_queries", 0),
    ("max_batch_queries", 17),
    ("max_batch_queries", True),
])
def test_invalid_protocol_options_rejected_before_scorer(monkeypatch, option, value):
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    calls = _scorer_sentinels(monkeypatch)

    with pytest.raises((TypeError, ValueError)):
        _run(receipts, **{option: value})
    assert calls == []


def test_query_count_cannot_exceed_batch_query_cap(monkeypatch):
    receipts = _receipts(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    calls = _scorer_sentinels(monkeypatch)

    with pytest.raises(ValueError):
        _run(receipts, max_batch_queries=1)
    assert calls == []


def test_duplicate_ids_and_zero_embeddings_are_rejected_before_scorer(monkeypatch):
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    duplicate_binding = replace(receipts[1].source_binding, factor_id="factor-0")
    duplicate_ids = _receipt_tuple_with_change(
        receipts, 1, source_binding=duplicate_binding
    )
    calls = _scorer_sentinels(monkeypatch)
    with pytest.raises(ValueError):
        _run(duplicate_ids)
    assert calls == []

    zero_embedding = _receipt_tuple_with_change(receipts, 1, embedding=(0.0, 0.0))
    with pytest.raises(ValueError):
        _run(zero_embedding)
    assert calls == []


def test_forged_content_hash_rejected_before_scorer(monkeypatch):
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    object.__setattr__(receipts[1], "content_hash", "0" * 64)
    calls = _scorer_sentinels(monkeypatch)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _run(receipts)
    assert calls == []


def test_scalar_batch_result_disagreement_fails_closed(monkeypatch):
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    module = importlib.import_module(AUDIT_MODULE)
    original = module.scan_exact_member_winners
    batch_calls = []

    def disagree(*args, **kwargs):
        actual = original(*args, **kwargs)
        batch_calls.append(actual)
        return [("wrong-factor", 0.5)]

    monkeypatch.setattr(module, "scan_exact_member_winners", disagree)
    with pytest.raises(RuntimeError):
        _run(receipts)
    assert batch_calls


def test_source_binding_drift_between_paired_scans_fails_closed(monkeypatch):
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    module = importlib.import_module(AUDIT_MODULE)
    original = module.scan_exact_member_winners
    calls = []

    def mutate_after_batch(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        object.__setattr__(receipts[0].source_binding, "source_sha256", "0" * 64)
        return result

    monkeypatch.setattr(module, "scan_exact_member_winners", mutate_after_batch)
    with pytest.raises(RuntimeError):
        _run(receipts)
    assert calls


def test_report_reproduces_exact_receipt_set_scan_config_and_shared_identity():
    receipts = _receipts(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    report = _run(
        receipts,
        paired_blocks=3,
        seed=17,
        max_chunk_rows=2,
        max_chunk_bytes=4096,
        max_batch_queries=4,
    )

    assert report["receipt_content_hashes"] == [
        receipt.content_hash for receipt in receipts
    ]
    assert report["scan_config"] == {
        "seed": 17,
        "paired_blocks_count": 3,
        "warmup_pairs": 2,
        "max_chunk_rows": 2,
        "max_chunk_bytes": 4096,
        "max_batch_queries": 4,
        "embedding_width": 2,
    }
    first = receipts[0]
    binding = first.source_binding
    assert report["shared_identity"] == {
        "manifest_uri": binding.manifest_uri,
        "manifest_sha256": binding.manifest_sha256,
        "time_axis_hash": binding.time_axis_hash,
        "asset_axis_hash": binding.asset_axis_hash,
        "shape": [2, 2, 1],
        "embedding_spec": first.embedding_spec,
        "embedding_model_version": first.embedding_model_version,
    }
    memory_scope = report["memory_scope"]
    assert isinstance(memory_scope, str) and memory_scope
    assert (
        "not total RSS" in memory_scope
        or memory_scope == "helper_chunk_arrays_only_not_total_rss"
    )
